import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os

from multiview_segformer_unet import MultiViewSegformerUNet
from dataset_multiview import MultiViewBEVDataset
from palette import n_classes_label
from metrics import compute_iou_dataset

CHECKPOINT_PATH = r'C:\Users\admin\Downloads\best_model_unet_custom_new_weights_4.pt'
VALID_FRONT_DIR = r'D:\BEV\cam2bev-data-master\1_FRLR\val\front'
VALID_REAR_DIR  = r'D:\BEV\cam2bev-data-master\1_FRLR\val\rear'
VALID_LEFT_DIR  = r'D:\BEV\cam2bev-data-master\1_FRLR\val\left'
VALID_RIGHT_DIR = r'D:\BEV\cam2bev-data-master\1_FRLR\val\right'
VALID_MASK_DIR  = r'D:\BEV\cam2bev-data-master\1_FRLR\val\bev+occlusion'

BATCH_SIZE = 4
NUM_CLASSES = n_classes_label
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    valid_dataset = MultiViewBEVDataset(
        front_dir=VALID_FRONT_DIR,
        rear_dir=VALID_REAR_DIR,
        left_dir=VALID_LEFT_DIR,
        right_dir=VALID_RIGHT_DIR,
        mask_dir=VALID_MASK_DIR
    )
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = MultiViewSegformerUNet(n_classes=NUM_CLASSES)    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, masks in tqdm(valid_loader, desc="Validating"):
            imgs = [img.to(DEVICE) for img in imgs]
            masks = masks.to(DEVICE)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    class_idx = 4
    num_pred_class = (all_preds.numpy() == class_idx).sum()
    num_gt_class = (all_targets.numpy() == class_idx).sum()
    print(f"Pixels predicted as class {class_idx}: {num_pred_class}")
    print(f"Pixels groundtruth as class {class_idx}: {num_gt_class}")
    
    miou, ious = compute_iou_dataset(all_preds, all_targets, NUM_CLASSES)

    print("=== CLASS IOU SCORES (%) ON THE VALIDATION SET ===")
    for idx, iou in enumerate(ious):
        if np.isnan(iou):
            print(f"Class {idx}: mIoU = NaN")
        else:
            print(f"Class {idx}: mIoU = {iou:.4f}")
    print(f"Mean IoU: {miou:.4f}")
