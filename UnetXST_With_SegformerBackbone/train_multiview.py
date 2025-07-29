import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from multiview_segformer_unet import MultiViewSegformerUNet
from dataset_multiview import MultiViewBEVDataset
from metrics import compute_iou_dataset, pixel_accuracy
from palette import n_classes_label
import torch.nn as nn
import os

def save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, path)

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, total_miou, total_acc = 0, 0, 0
    for imgs, mask in tqdm(loader, desc="Train", leave=False):
        imgs = [img.to(device) for img in imgs]
        mask = mask.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = loss_fn(logits, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * mask.size(0)
        pred = torch.argmax(logits, dim=1)
        miou, _ = compute_iou_dataset(pred, mask, num_classes=logits.shape[1])
        acc = pixel_accuracy(pred, mask)
        total_miou += miou * mask.size(0)
        total_acc += acc * mask.size(0)
    n = len(loader.dataset)
    return total_loss/n, total_miou/n, total_acc/n

def validate_one_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss, total_miou, total_acc = 0, 0, 0
    with torch.no_grad():
        for imgs, mask in tqdm(loader, desc="Val", leave=False):
            imgs = [img.to(device) for img in imgs]
            mask = mask.to(device)
            logits = model(imgs)
            loss = loss_fn(logits, mask)
            total_loss += loss.item() * mask.size(0)
            pred = torch.argmax(logits, dim=1)
            miou, _ = compute_iou_dataset(pred, mask, num_classes=logits.shape[1])
            acc = pixel_accuracy(pred, mask)
            total_miou += miou * mask.size(0)
            total_acc += acc * mask.size(0)
    n = len(loader.dataset)
    return total_loss/n, total_miou/n, total_acc/n

if __name__ == "__main__":
    # ==== ĐIỀN ĐƯỜNG DẪN, CLASS WEIGHTS, ... ====
    train_dataset = MultiViewBEVDataset(
        front_dir= r"D:\KLTN\cam2bev-data-master\front",
        rear_dir= r"D:\KLTN\cam2bev-data-master\rear",
        left_dir= r"D:\KLTN\cam2bev-data-master\left",
        right_dir= r"D:\KLTN\cam2bev-data-master\right",
        mask_dir= r"D:\KLTN\cam2bev-data-master\bev+occlusion",
        image_shape=(128, 256),
        one_hot=False
    )
    val_dataset = MultiViewBEVDataset(
        front_dir= r"D:\KLTN\cam2bev-data-master\val\front",
        rear_dir= r"D:\KLTN\cam2bev-data-master\val\rear",
        left_dir= r"D:\KLTN\cam2bev-data-master\val\left",
        right_dir= r"D:\KLTN\cam2bev-data-master\val\right",
        mask_dir= r"D:\KLTN\cam2bev-data-master\val\bev+occlusion",
        image_shape=(128, 256),
        one_hot=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # === KHỞI TẠO MODEL MỚI (KHÔNG LOAD CHECKPOINT, KHÔNG double_skip_connection) ===
    model = MultiViewSegformerUNet(
        n_classes=n_classes_label,
        dropout=0.1
    ).to(device)

    # === Loss, optimizer, scheduler như cũ ===
    # Lưu ý: cập nhật lại class_weights nếu số lớp thay đổi!
    class_weights = torch.tensor([1.107895, 2.361856, 4.913445, 7.074946, 2.187833, 0.94754, 2.696291], dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_val_loss = float('inf')
    num_epochs = 100  # Đồng bộ số epoch
    patience = 7
    counter = 0
    save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints_newweights_4')
    os.makedirs(save_dir, exist_ok=True)
    log_miou_path = os.path.join(save_dir, 'best_model_miou_per_class_segformer_newweights_4.log')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss, train_miou, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_miou, val_acc = validate_one_epoch(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)
        print(f"Train loss: {train_loss:.4f} | mIoU: {train_miou:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f} | mIoU: {val_miou:.4f} | Acc: {val_acc:.4f}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, os.path.join(save_dir, 'best_model_segformer_newweights_4.pt'))
            print(f"Saved best model at epoch {epoch+1}!")
            counter = 0
            # --- Tính lại mIoU từng class trên val set ---
            model.eval()
            iou_sum = None
            total = 0
            with torch.no_grad():
                for imgs, mask in tqdm(val_loader, desc="Best model mIoU per class", leave=False):
                    imgs = [img.to(device) for img in imgs]
                    mask = mask.to(device)
                    logits = model(imgs)
                    pred = torch.argmax(logits, dim=1)
                    _, ious = compute_iou_dataset(pred, mask, num_classes=logits.shape[1])
                    ious_tensor = torch.tensor(ious, dtype=torch.float32)
                    if iou_sum is None:
                        iou_sum = ious_tensor
                    else:
                        iou_sum += ious_tensor
                    total += 1
            mean_ious = (iou_sum / total).tolist() if total > 0 else []
            # --- Lưu log ---
            with open(log_miou_path, 'w', encoding='utf-8') as f:
                f.write(f"Epoch: {epoch+1}\n")
                for idx, iou in enumerate(mean_ious):
                    f.write(f"Class {idx}: mIoU = {iou:.4f}\n")
            print("Saved best model and mIoU per class log!")
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered.")
                break

        # Save last checkpoint
        save_checkpoint(model, optimizer, epoch, val_loss, os.path.join(save_dir, 'last_model_segformer_newweights_4.pt'))
        torch.cuda.empty_cache()