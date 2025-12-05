import os
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from typing import Optional

import albumentations as A
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
from scipy.ndimage import label

from dataset import SegmentationDataset
from model_unet import UNet


# ----------------------------
# Transforms
# ----------------------------
def get_transform(img_size: int = 256):
    """訓練用：Resize + Flip"""
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ]
    )


def get_val_transform(img_size: int = 256):
    """驗證用：只做 Resize"""
    return A.Compose(
        [
            A.Resize(img_size, img_size),
        ]
    )


# ----------------------------
# PRO function
# ----------------------------
def compute_pro(all_preds: np.ndarray, all_gts: np.ndarray, num_thresholds: int = 50) -> float:
    """
    計算 Per-Region Overlap (PRO) 分數
    all_preds: (N, H, W) 連續值 [0,1]
    all_gts:   (N, H, W) 0/1
    """
    all_preds = all_preds.astype(np.float32)
    all_gts = (all_gts > 0.5).astype(np.uint8)

    N, H, W = all_preds.shape
    thresholds = np.linspace(0, 1, num_thresholds)

    pro_values = []

    for thr in thresholds:
        preds_bin = all_preds >= thr  # bool, (N,H,W)
        region_overlaps = []

        for n in range(N):
            gt = all_gts[n]
            pred = preds_bin[n]

            labeled_gt, num_regions = label(gt)
            if num_regions == 0:
                continue

            for rid in range(1, num_regions + 1):
                region = labeled_gt == rid

                inter = np.logical_and(region, pred).sum()
                denom = region.sum()
                if denom == 0:
                    continue

                region_overlaps.append(inter / float(denom))

        if len(region_overlaps) == 0:
            pro_values.append(0.0)
        else:
            pro_values.append(float(np.mean(region_overlaps)))

    return float(np.mean(pro_values))


# ----------------------------
# Training Function
# ----------------------------
def train(
    category: str,
    dataset_root: str = "seg_dataset",
    num_epochs: int = 25,
    pro_every: int = 5,
    tag: Optional[str] = None,
):
    """
    category: MVTec 類別 (e.g. 'bottle')
    dataset_root: 資料集根目錄，例如：
        - 'seg_dataset'
        - 'seg_dataset_visualization_GT'
        - 'seg_dataset_visualization_PSEUDO'
    tag: 模型/log 的額外標籤，例如 'GT', 'PSEUDO'
    """

    model_name = f"unet_{category}"
    if tag:
        model_name += f"_{tag}"

    print("========================================")
    print(f"開始訓練 U-Net")
    print(f"Category     : {category}")
    print(f"Dataset root : {dataset_root}")
    print(f"Model name   : {model_name}")
    print(f"Epochs       : {num_epochs}")
    print(f"PRO every    : {pro_every}")
    print("========================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 建資料夾
    os.makedirs("checkpoints", exist_ok=True)
    log_dir = os.path.join("logs", model_name)
    os.makedirs(log_dir, exist_ok=True)

    # ----------------------------
    # Datasets (注意這裡傳的是類別根目錄 + split)
    # ----------------------------
    class_root = os.path.join(dataset_root, category)

    train_set = SegmentationDataset(class_root, split="train", transform=get_transform())
    val_set = SegmentationDataset(class_root, split="val", transform=get_val_transform())

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # ----------------------------
    # Model & Loss
    # ----------------------------
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    bce = nn.BCELoss()

    # Dice Loss（用 lambda 簡寫）
    dice = (
        lambda p, g: 1
        - (2 * (p * g).sum() + 1) / (p.sum() + g.sum() + 1)
    )

    # ----------------------------
    # Metric Logging
    # ----------------------------
    iou_history = []
    pro_history = []
    pro_epoch_index = []

    # ----------------------------
    # Training Loop
    # ----------------------------
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        # ---- Train ----
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)

            pred = model(img)
            loss = bce(pred, mask) + dice(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # ---- Validation IoU ----
        model.eval()
        ious = []

        with torch.no_grad():
            for img, mask in val_loader:
                img = img.to(device)
                pred = model(img)
                pred_bin = (pred > 0.5).float()

                iou = jaccard_score(
                    mask.cpu().numpy().flatten(),
                    pred_bin.cpu().numpy().flatten(),
                    zero_division=0,
                )
                ious.append(iou)

        mean_iou = float(np.mean(ious)) if len(ious) > 0 else 0.0
        iou_history.append(mean_iou)

        print(
            f"Epoch {epoch:03d}/{num_epochs} | "
            f"Loss: {running_loss:.4f} | Val IoU: {mean_iou:.4f}"
        )

        # ---- 每 pro_every 個 epoch 計算一次 PRO ----
        if epoch % pro_every == 0:
            preds, gts = [], []

            with torch.no_grad():
                for img, mask in val_loader:
                    img = img.to(device)
                    pred = model(img)

                    preds.append(pred.squeeze().cpu().numpy())
                    gts.append(mask.squeeze().cpu().numpy())

            if len(preds) > 0:
                preds = np.stack(preds)
                gts = np.stack(gts)

                pro_score = compute_pro(preds, gts)
                pro_history.append(pro_score)
                pro_epoch_index.append(epoch)

                print(f"  → PRO @ epoch {epoch}: {pro_score:.4f}")
            else:
                print("  → PRO: 驗證集為空，略過。")

        # ---- 每個 epoch 存一次（覆蓋式）----
        ckpt_path = os.path.join("checkpoints", f"{model_name}.pth")
        torch.save(model.state_dict(), ckpt_path)

    # ----------------------------
    # Plot IoU & PRO curve
    # ----------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), iou_history, label="IoU")

    if len(pro_history) > 0:
        plt.plot(pro_epoch_index, pro_history, label="PRO", marker="o")

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"U-Net Training Metrics ({model_name})")
    plt.legend()
    plt.grid(True)

    fig_path = os.path.join(log_dir, "metrics.png")
    plt.savefig(fig_path)

    print("========================================")
    print("訓練完成！")
    print(f"模型已儲存：{ckpt_path}")
    print(f"訓練曲線輸出：{fig_path}")
    print("========================================")


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        help="MVTec 類別名稱 (例如: bottle)",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="seg_dataset",
        help="資料集根目錄，例如 seg_dataset / seg_dataset_visualization_GT / seg_dataset_visualization_PSEUDO",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="訓練總 epoch 數",
    )
    parser.add_argument(
        "--pro_every",
        type=int,
        default=5,
        help="每幾個 epoch 計算一次 PRO",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="模型命名用的標籤（例如 GT, PSEUDO）",
    )

    args = parser.parse_args()

    tag = args.tag if args.tag != "" else None

    train(
        category=args.category,
        dataset_root=args.dataset_root,
        num_epochs=args.epochs,
        pro_every=args.pro_every,
        tag=tag,
    )
