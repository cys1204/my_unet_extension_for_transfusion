import os
import shutil
import random
import numpy as np

# === 固定 Seed（讓所有人跑出完全相同結果） ===
random.seed(42)
np.random.seed(42)

# === Visualization 資料輸入 ===
ROOT = "experiments/transfusion_mvtec/visualizations/bottle"

# === 兩套輸出 ===
OUT_GT = "seg_dataset_visualization_GT/bottle"
OUT_PSEUDO = "seg_dataset_visualization_PSEUDO/bottle"

def prepare_dirs(base):
    IMG = os.path.join(base, "images")
    MASK = os.path.join(base, "masks")
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(IMG, split), exist_ok=True)
        os.makedirs(os.path.join(MASK, split), exist_ok=True)
    return IMG, MASK

IMG_GT, MASK_GT = prepare_dirs(OUT_GT)
IMG_PSEUDO, MASK_PSEUDO = prepare_dirs(OUT_PSEUDO)

# === 收集所有 index ===
all_ids = []
for fname in os.listdir(ROOT):
    if fname.endswith("_true.png"):
        idx = fname.split("_")[0]
        all_ids.append(idx)

all_ids = sorted(all_ids)
random.shuffle(all_ids)

# === split 7:1:2 ===
N = len(all_ids)
n_train = int(N * 0.7)
n_val   = int(N * 0.1)
n_test  = N - n_train - n_val

train_ids = all_ids[:n_train]
val_ids   = all_ids[n_train:n_train+n_val]
test_ids  = all_ids[n_train+n_val:]

def copy_pair(idx, split):
    # 原圖（true.png）
    src_img = os.path.join(ROOT, f"{idx}_true.png")

    # GT mask
    src_gt_mask = os.path.join(ROOT, f"{idx}_true_mask.png")

    # Pseudo mask
    src_ps_mask = os.path.join(ROOT, f"{idx}_mask.png")

    # 目標路徑
    dst_img_gt     = os.path.join(IMG_GT, split, f"{idx}.png")
    dst_mask_gt    = os.path.join(MASK_GT, split, f"{idx}.png")

    dst_img_ps     = os.path.join(IMG_PSEUDO, split, f"{idx}.png")
    dst_mask_ps    = os.path.join(MASK_PSEUDO, split, f"{idx}.png")

    # 複製原圖（兩套 dataset 共用）
    shutil.copy(src_img, dst_img_gt)
    shutil.copy(src_img, dst_img_ps)

    # 複製 GT mask
    shutil.copy(src_gt_mask, dst_mask_gt)

    # 複製 pseudo mask
    shutil.copy(src_ps_mask, dst_mask_ps)

for i in train_ids: copy_pair(i, "train")
for i in val_ids:   copy_pair(i, "val")
for i in test_ids:  copy_pair(i, "test")

print("========================================")
print(f"總數量: {N}")
print(f"Train: {len(train_ids)}")
print(f"Val:   {len(val_ids)}")
print(f"Test:  {len(test_ids)}")
print("GT dataset 輸出路徑:", OUT_GT)
print("Pseudo dataset 輸出路徑:", OUT_PSEUDO)
print("========================================")
