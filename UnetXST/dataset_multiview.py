# dataset_multiview.py
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from palette import one_hot_palette_label, n_classes_label

class MultiViewBEVDataset(Dataset):
    def __init__(self, front_dir, rear_dir, left_dir, right_dir, mask_dir, image_shape=(128, 256), one_hot=False):
        self.front_dir = front_dir
        self.rear_dir = rear_dir
        self.left_dir = left_dir
        self.right_dir = right_dir
        self.mask_dir = mask_dir
        self.image_shape = image_shape
        self.one_hot = one_hot
        self.palette = one_hot_palette_label

        self.file_names = sorted([f for f in os.listdir(front_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.file_names)

    def _load_and_resize(self, path):
        img = Image.open(path).convert('RGB')
        img = img.resize(self.image_shape[::-1], Image.BILINEAR)
        return np.array(img)

    def _mask_to_class(self, mask):
        class_map = np.zeros(mask.shape[:2], dtype=np.int64)
        for idx, class_colors in enumerate(self.palette):
            for color in class_colors:
                color = np.array(color)
                matches = np.all(mask == color, axis=-1)
                class_map[matches] = idx
        return class_map

    def _mask_to_onehot(self, mask):
        h, w, _ = mask.shape
        onehot = np.zeros((len(self.palette), h, w), dtype=np.float32)
        for idx, class_colors in enumerate(self.palette):
            for color in class_colors:
                color = np.array(color)
                matches = np.all(mask == color, axis=-1)
                onehot[idx][matches] = 1.0
        return onehot

    def __getitem__(self, idx):
        fname = self.file_names[idx]
        imgs = []
        for cam_dir in [self.front_dir, self.rear_dir, self.left_dir, self.right_dir]:
            img_path = os.path.join(cam_dir, fname)
            img = self._load_and_resize(img_path)
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2,0,1)
            imgs.append(img)
        mask_path = os.path.join(self.mask_dir, fname)
        mask = self._load_and_resize(mask_path)
        if self.one_hot:
            mask_tensor = torch.from_numpy(self._mask_to_onehot(mask))
        else:
            mask_tensor = torch.from_numpy(self._mask_to_class(mask)).long()
        return imgs, mask_tensor