import numpy as np
import torch

def compute_iou_dataset(preds, targets, num_classes):
    """
    preds, targets: (N, H, W) hoặc (B, H, W)
    num_classes: số lượng class
    """
    total_intersection = np.zeros(num_classes, dtype=np.float64)
    total_union = np.zeros(num_classes, dtype=np.float64)

    preds = preds.cpu().numpy() if torch.is_tensor(preds) else preds
    targets = targets.cpu().numpy() if torch.is_tensor(targets) else targets

    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        total_intersection[cls] += intersection
        total_union[cls] += union

    ious = []
    for cls in range(num_classes):
        if total_union[cls] == 0:
            ious.append(float('nan'))  # hoặc 0.0 nếu muốn
        else:
            ious.append(total_intersection[cls] / total_union[cls])
    miou = np.nanmean(ious)
    return miou, ious

def pixel_accuracy(pred, target):
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total