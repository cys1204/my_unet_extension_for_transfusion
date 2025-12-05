import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        """
        root:  類別根目錄，例如: seg_dataset_visualization_GT/bottle
        split: "train" / "val" / "test"
        結構預期:
            root/images/train/*.png
            root/masks/train/*.png
        """
        self.img_dir = os.path.join(root, "images", split)
        self.mask_dir = os.path.join(root, "masks", split)
        self.transform = transform

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"找不到影像資料夾: {self.img_dir}")
        if not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(f"找不到標註資料夾: {self.mask_dir}")

        # 確保讀取順序一致
        self.images = sorted(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # --- Load Image ---
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"無法讀取圖片: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0  # 正規化到 [0, 1]

        # --- Load Mask ---
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype("float32")
        else:
            # 理論上不該發生，如果發生就給全黑
            mask = np.zeros((img.shape[0], img.shape[1]), dtype="float32")

        # --- Transform ---
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = aug["image"]
            mask = aug["mask"]

        # --- To Tensor ---
        img = torch.tensor(img).permute(2, 0, 1).float()
        mask = torch.tensor(mask).unsqueeze(0).float()

        return img, mask
