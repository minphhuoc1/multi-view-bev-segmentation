import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 1. Import model, metrics, class weights, scheduler, palette
from unet_custom import UNetCustom
from metrics_unetthuan import compute_iou_dataset, pixel_accuracy
from palette_unetthuan import one_hot_palette_label

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mask_rgb_to_class(mask, palette):
    mask = np.array(mask)
    class_map = np.zeros(mask.shape[:2], dtype=np.int64)
    for idx, class_colors in enumerate(palette):
        for color in class_colors:
            color = np.array(color)
            matches = np.all(mask == color, axis=-1)
            class_map[matches] = idx
    return class_map

# 2. Dataset
class MultiCamBevDataset(Dataset):
    def __init__(self, front_dir, rear_dir, left_dir, right_dir, mask_dir, img_size=(128, 256)):
        self.front_paths = sorted(glob.glob(os.path.join(front_dir, "*.png")))
        self.rear_paths = sorted(glob.glob(os.path.join(rear_dir, "*.png")))
        self.left_paths = sorted(glob.glob(os.path.join(left_dir, "*.png")))
        self.right_paths = sorted(glob.glob(os.path.join(right_dir, "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        self.img_size = img_size
        self.img_transform = T.Compose([
            T.Resize(img_size, interpolation=Image.BILINEAR),
            T.ToTensor()
        ])
        self.mask_transform = T.Compose([
            T.Resize(img_size, interpolation=Image.NEAREST)
        ])
        self.palette = one_hot_palette_label

    def __len__(self):
        return len(self.front_paths)

    def __getitem__(self, idx):
        imgs = []
        for path in [self.front_paths[idx], self.rear_paths[idx], self.left_paths[idx], self.right_paths[idx]]:
            img = Image.open(path).convert('RGB')
            img = self.img_transform(img)
            imgs.append(img)
        img = torch.cat(imgs, dim=0)  # [12, H, W]
        mask = Image.open(self.mask_paths[idx]).convert('RGB')
        mask = self.mask_transform(mask)
        mask = mask_rgb_to_class(mask, self.palette)
        mask = torch.from_numpy(mask).long()
        return img, mask

# 3. Hyperparameters
num_classes = 7
batch_size = 8
num_epochs = 100
early_stop_patience = 7

# 4. DataLoader
train_dataset = MultiCamBevDataset(
    front_dir=r"D:\BEV\cam2bev-data-master\1_FRLR\train\front",
    rear_dir=r"D:\BEV\cam2bev-data-master\1_FRLR\train\rear",
    left_dir=r"D:\BEV\cam2bev-data-master\1_FRLR\train\left",
    right_dir=r"D:\BEV\cam2bev-data-master\1_FRLR\train\right",
    mask_dir=r"D:\BEV\cam2bev-data-master\1_FRLR\train\bev+occlusion"
)
val_dataset = MultiCamBevDataset(
    front_dir=r"D:\BEV\cam2bev-data-master\1_FRLR\val\front",
    rear_dir=r"D:\BEV\cam2bev-data-master\1_FRLR\val\rear",
    left_dir=r"D:\BEV\cam2bev-data-master\1_FRLR\val\left",
    right_dir=r"D:\BEV\cam2bev-data-master\1_FRLR\val\right",
    mask_dir=r"D:\BEV\cam2bev-data-master\1_FRLR\val\bev+occlusion"
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 5. Model, Loss, Optimizer, Scheduler
model = UNetCustom(in_channels=12, out_channels=num_classes).to(device)
class_weights = torch.tensor([1.107895, 2.361856, 4.913445, 7.074946, 2.187833, 0.94754, 2.696291], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# 6. Train/Val loop
def validate(model, loader):
    model.eval()
    val_loss = 0
    miou_total = []
    acc_total = []
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Validating", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            miou, _ = compute_iou_dataset(preds, masks, num_classes)
            acc = pixel_accuracy(preds, masks)
            miou_total.append(miou)
            acc_total.append(acc)
    n = len(loader.dataset)
    return val_loss / n, np.nanmean(miou_total), np.mean(acc_total)

def train():
    best_val_loss = float('inf')
    best_epoch = 0
    patience = 0
    for epoch in range(1, num_epochs+1):
        model.train()
        train_loss = 0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)

        val_loss, val_miou, val_acc = validate(model, val_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | mIoU {val_miou:.4f} | Acc {val_acc:.4f}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }, "best_checkpoint.pth")
            print(f"==> Saved best checkpoint at epoch {epoch} (val_loss={val_loss:.4f})")
        else:
            patience += 1

        # Early stopping
        if patience >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Save last checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss
    }, "last_checkpoint.pth")
    print(f"==> Saved last checkpoint at epoch {epoch} (val_loss={val_loss:.4f})")
    print(f"Best model at epoch {best_epoch} with val_loss {best_val_loss:.4f}")

if __name__ == "__main__":
    train()