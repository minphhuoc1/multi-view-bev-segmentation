import torch
from torch.utils.data import DataLoader
import numpy as np
from unet_custom import UNetCustom
from palette_unetthuan import one_hot_palette_label
from metrics_unetthuan import compute_iou_dataset, pixel_accuracy
from train_thuan import MultiCamBevDataset
from tqdm import tqdm

CHECKPOINT_PATH = r'C:\Users\admin\Downloads\4camoriginal_best_checkpoint.pth'
FRONT_DIR = r'D:\BEV\cam2bev-data-master\1_FRLR\val\front'
REAR_DIR  = r'D:\BEV\cam2bev-data-master\1_FRLR\val\rear'
LEFT_DIR  = r'D:\BEV\cam2bev-data-master\1_FRLR\val\left'
RIGHT_DIR = r'D:\BEV\cam2bev-data-master\1_FRLR\val\right'
MASK_DIR  = r'D:\BEV\cam2bev-data-master\1_FRLR\val\bev+occlusion'
BATCH_SIZE = 4
NUM_CLASSES = len(one_hot_palette_label)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    dataset = MultiCamBevDataset(FRONT_DIR, REAR_DIR, LEFT_DIR, RIGHT_DIR, MASK_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = UNetCustom(in_channels=12, out_channels=NUM_CLASSES)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Evaluating", leave=True):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    miou, ious = compute_iou_dataset(all_preds, all_targets, NUM_CLASSES)
    print("=== CLASS IOU SCORES (%) ON THE VALIDATION SET ===")
    for idx, iou in enumerate(ious):
        if np.isnan(iou):
            print(f"Class {idx}: mIoU = NaN")
        else:
            print(f"Class {idx}: mIoU = {iou:.4f}")
    print(f"Mean IoU: {miou:.4f}")

    acc = pixel_accuracy(all_preds, all_targets)
    print(f"Pixel accuracy: {acc:.4f}")