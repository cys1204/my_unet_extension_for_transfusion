import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, root, category, split="train", transform=None):
        """
        資料結構要求：
        root/category/images/train/*.png
        root/category/masks/train/*.png
        """
        # 正確路徑
        self.img_dir = os.path.join(root, category, "images", split)
        self.mask_dir = os.path.join(root, category, "masks", split)
        self.transform = transform

        # 檢查資料夾是否存在
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"找不到影像資料夾: {self.img_dir}")
        if not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(f"找不到標註資料夾: {self.mask_dir}")

        self.images = sorted(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # --- Load image ---
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

        # --- Load mask (二值化避免 DeepLabv3 loss 爆炸) ---
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype("float32")   # ★★★ 這行是 DeepLab 正常運作的關鍵！

        # --- Transform ---
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]

        # --- To tensor ---
        img = torch.tensor(img).permute(2, 0, 1).float()
        mask = torch.tensor(mask).unsqueeze(0).float()

        return img, mask
