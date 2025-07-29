import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random
from unet_custom import UNetCustom
from palette_unetthuan import one_hot_palette_label
from train_thuan import MultiCamBevDataset, mask_rgb_to_class

CHECKPOINT_PATH = r'C:\Users\admin\Downloads\4camoriginal_best_checkpoint.pth'
FRONT_DIR = r'D:\BEV\cam2bev-data-master\1_FRLR\val\front'
REAR_DIR  = r'D:\BEV\cam2bev-data-master\1_FRLR\val\rear'
LEFT_DIR  = r'D:\BEV\cam2bev-data-master\1_FRLR\val\left'
RIGHT_DIR = r'D:\BEV\cam2bev-data-master\1_FRLR\val\right'
MASK_DIR  = r'D:\BEV\cam2bev-data-master\1_FRLR\val\bev+occlusion'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mask_index_to_color(mask, palette):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, class_colors in enumerate(palette):
        color = np.array(class_colors[0])
        color_mask[mask == idx] = color
    return color_mask

if __name__ == "__main__":
    dataset = MultiCamBevDataset(FRONT_DIR, REAR_DIR, LEFT_DIR, RIGHT_DIR, MASK_DIR)
    model = UNetCustom(in_channels=12, out_channels=7)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()

    num_samples = len(dataset)
    random_indices = random.sample(range(num_samples), 10)

    for idx in random_indices:
        img, mask_gt = dataset[idx]           # img: [12, H, W]
        img_input = img.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(img_input)
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]

        # Hiển thị ảnh front (3 kênh đầu)
        img_front = img[0:3].cpu().numpy().transpose(1,2,0)
        img_front = (img_front * 255).astype(np.uint8)
        mask_gt_np = mask_gt.numpy()
        mask_gt_color = mask_index_to_color(mask_gt_np, one_hot_palette_label)
        pred_color = mask_index_to_color(pred, one_hot_palette_label)

        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.imshow(img_front)
        plt.title('Front Image')
        plt.axis('off')
        plt.subplot(1,3,2)
        plt.imshow(mask_gt_color)
        plt.title('Ground Truth')
        plt.axis('off')
        plt.subplot(1,3,3)
        plt.imshow(pred_color)
        plt.title('Prediction')
        plt.axis('off')
        plt.tight_layout()
        plt.show()