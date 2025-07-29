import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from unet_custom import UNetCustom
from dataset_multiview import MultiViewBEVDataset
from metrics import compute_iou, pixel_accuracy
from palette import n_classes_label
from losses import Cam2BEVCrossEntropyLoss
import torch.nn as nn
import numpy as np
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
        miou, _ = compute_iou(pred, mask, num_classes=logits.shape[1])
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
            miou, _ = compute_iou(pred, mask, num_classes=logits.shape[1])
            acc = pixel_accuracy(pred, mask)
            total_miou += miou * mask.size(0)
            total_acc += acc * mask.size(0)
    n = len(loader.dataset)
    return total_loss/n, total_miou/n, total_acc/n

def main():
    # Đường dẫn dữ liệu (bạn cần chỉnh lại cho đúng)
    train_dirs = {
        'front_dir': r"D:\KLTN\cam2bev-data-master\front",
        'rear_dir': r"D:\KLTN\cam2bev-data-master\rear",
        'left_dir': r"D:\KLTN\cam2bev-data-master\left",
        'right_dir': r"D:\KLTN\cam2bev-data-master\right",
        'mask_dir': r"D:\KLTN\cam2bev-data-master\bev+occlusion",
    }
    val_dirs = {
        'front_dir': r"D:\KLTN\cam2bev-data-master\val\front",
        'rear_dir': r"D:\KLTN\cam2bev-data-master\val\rear",
        'left_dir': r"D:\KLTN\cam2bev-data-master\val\left",
        'right_dir': r"D:\KLTN\cam2bev-data-master\val\right",
        'mask_dir': r"D:\KLTN\cam2bev-data-master\val\bev+occlusion",
    }

    train_dataset = MultiViewBEVDataset(
        train_dirs['front_dir'], train_dirs['rear_dir'], train_dirs['left_dir'], train_dirs['right_dir'],
        train_dirs['mask_dir'], image_shape=(128, 256), one_hot=False
    )
    val_dataset = MultiViewBEVDataset(
        val_dirs['front_dir'], val_dirs['rear_dir'], val_dirs['left_dir'], val_dirs['right_dir'],
        val_dirs['mask_dir'], image_shape=(128, 256), one_hot=False
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
    print(device)
    model = UNetCustom(in_channels=3, out_channels=n_classes_label).to(device)
    class_weights = torch.tensor([
        0.98684351, 2.2481491, 10.47452063, 4.78351389, 7.01028204, 8.41360361,
        10.91633349, 2.38571558, 1.02473193, 2.79359197
    ], dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_val_loss = float('inf')
    n_epochs = 100
    patience = 7  # số epoch không cải thiện sẽ dừng sớm
    counter = 0
    save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    log_miou_path = os.path.join(save_dir, 'best_model_miou_per_class.log')
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        train_loss, train_miou, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        # Tính lại mIoU từng class trên tập val để log
        val_loss, val_miou, val_acc = validate_one_epoch(model, val_loader, loss_fn, device)
        print(f"Train Loss: {train_loss:.4f}, mIoU: {train_miou:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, mIoU: {val_miou:.4f}, Acc: {val_acc:.4f}")
        scheduler.step(val_loss)
        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # --- Save best checkpoint (dạng state_dict như cũ) ---
            save_checkpoint(model, optimizer, epoch, val_loss, os.path.join(save_dir, 'best_model_unet_custom.pt'))
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
                    _, ious = compute_iou(pred, mask, num_classes=logits.shape[1])
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
            print("Saved best model (full) and mIoU per class log!")
            counter = 0
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered.")
                break
        # Always save last checkpoint
        save_checkpoint(model, optimizer, epoch, val_loss, os.path.join(save_dir, 'last_model_unet_custom.pt'))

if __name__ == "__main__":
    main()
