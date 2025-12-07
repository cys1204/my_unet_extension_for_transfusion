import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import albumentations as A
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
from scipy.ndimage import label

from model_deeplab import DeepLabV3Plus
from dataset import SegmentationDataset


# -------------------------
# Transforms
# -------------------------
def get_transform(img_size=256):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ])


def get_val_transform(img_size=256):
    return A.Compose([
        A.Resize(img_size, img_size),
    ])


# -------------------------
# PRO 計算
# -------------------------
def compute_pro(all_preds, all_gts, num_thresholds=50):
    """
    all_preds: (N, H, W) 預測機率 [0,1]
    all_gts  : (N, H, W) 0/1
    """
    all_preds = all_preds.astype(np.float32)
    all_gts = (all_gts > 0.5).astype(np.uint8)

    thresholds = np.linspace(0, 1, num_thresholds)
    pro_values = []

    for thr in thresholds:
        preds_bin = all_preds >= thr
        region_overlaps = []

        for n in range(all_preds.shape[0]):
            gt = all_gts[n]
            pred = preds_bin[n]

            labeled_gt, num_regions = label(gt)
            if num_regions == 0:
                continue

            for rid in range(1, num_regions + 1):
                region = (labeled_gt == rid)
                inter = np.logical_and(region, pred).sum()
                denom = region.sum()
                if denom > 0:
                    region_overlaps.append(inter / float(denom))

        pro_values.append(np.mean(region_overlaps) if region_overlaps else 0.0)

    return float(np.mean(pro_values))


# -----------------------------------------------------
# TRAINING
# -----------------------------------------------------
def train(category,
          dataset_root,
          epochs=50,
          pro_every=5,
          img_size=256,
          batch_size=4,
          lr=1e-4,
          tag=None):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = f"deeplabv3_{category}"
    if tag:
        model_name += f"_{tag}"

    print("========================================")
    print("開始訓練 DeepLabv3+")
    print(f"Category     : {category}")
    print(f"Dataset root : {dataset_root}")
    print(f"Model name   : {model_name}")
    print(f"Epochs       : {epochs}")
    print(f"PRO every    : {pro_every}")
    print("========================================")

    # -----------------------------
    # Dataset
    # -----------------------------
    train_set = SegmentationDataset(
        root=dataset_root,
        category=category,
        split="train",
        transform=get_transform(img_size)
    )

    val_set = SegmentationDataset(
        root=dataset_root,
        category=category,
        split="val",
        transform=get_val_transform(img_size)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # -----------------------------
    # Model
    # -----------------------------
    model = DeepLabV3Plus(num_classes=1).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs("checkpoints", exist_ok=True)
    log_dir = os.path.join("logs", model_name)
    os.makedirs(log_dir, exist_ok=True)

    # Logs（每個 epoch 記一次）
    loss_history = []
    iou_history = []
    pro_history = []
    pro_epoch_index = []
    best_iou = -1.0
    best_path = None

    # -----------------------------
    # Training Loop
    # -----------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        # -------- Train --------
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)

            pred = model(img)              # logits
            loss = criterion(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # epoch 平均 loss
        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)

        # -----------------------------
        # Validation IoU
        # -----------------------------
        model.eval()
        ious = []

        with torch.no_grad():
            for img, mask in val_loader:
                img = img.to(device)
                pred = model(img)                # logits
                prob = torch.sigmoid(pred)       # 機率
                pred_bin = (prob > 0.5).float()  # threshold 0.5

                iou = jaccard_score(
                    mask.cpu().numpy().flatten(),
                    pred_bin.cpu().numpy().flatten(),
                    zero_division=0
                )
                ious.append(iou)

        mean_iou = float(np.mean(ious)) if ious else 0.0
        iou_history.append(mean_iou)

        print(f"Epoch {epoch:03d}/{epochs} | "
              f"Loss={epoch_loss:.4f} | Val IoU={mean_iou:.4f}")

        # -----------------------------
        # 儲存 best 模型（依照 IoU）
        # -----------------------------
        if mean_iou > best_iou:
            best_iou = mean_iou
            best_path = os.path.join("checkpoints", f"{model_name}_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"★ Best model updated (IoU={best_iou:.4f}) → {best_path}")

        # -----------------------------
        # PRO（每 pro_every 個 epoch）
        # -----------------------------
        if epoch % pro_every == 0:
            preds, gts = [], []
            with torch.no_grad():
                for img, mask in val_loader:
                    img = img.to(device)
                    pred = model(img)
                    prob = torch.sigmoid(pred)

                    preds.append(prob.squeeze().cpu().numpy())
                    gts.append(mask.squeeze().cpu().numpy())

            preds = np.stack(preds)
            gts = np.stack(gts)

            pro_score = compute_pro(preds, gts)
            pro_history.append(pro_score)
            pro_epoch_index.append(epoch)

            print(f"→ PRO @ epoch {epoch}: {pro_score:.4f}")

        # -----------------------------
        # 每個 epoch 更新 latest
        # -----------------------------
        latest_path = os.path.join("checkpoints", f"{model_name}_latest.pth")
        torch.save(model.state_dict(), latest_path)

        # -----------------------------
        # 每 50 epoch 存 milestone
        # -----------------------------
        if epoch % 50 == 0:
            milestone_path = os.path.join(
                "checkpoints", f"{model_name}_epoch{epoch:03d}.pth"
            )
            torch.save(model.state_dict(), milestone_path)
            print(f"✔ Milestone saved: {milestone_path}")

    # ============================================================
    # Plot 1: Loss curve
    # ============================================================
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), loss_history, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"DeepLabv3+ Loss ({model_name})")
    plt.grid(True)
    plt.legend()
    loss_fig_path = os.path.join(log_dir, "loss_curve.png")
    plt.savefig(loss_fig_path)
    plt.close()

    # ============================================================
    # Plot 2: IoU + PRO
    # ============================================================
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), iou_history, label="Val IoU", color="orange")
    if pro_history:
        plt.plot(pro_epoch_index, pro_history, "o-", label="PRO", color="green")

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"DeepLabv3+ Metrics ({model_name})")
    plt.grid(True)
    plt.legend()
    metrics_fig_path = os.path.join(log_dir, "metrics_curve.png")
    plt.savefig(metrics_fig_path)
    plt.close()

    print("========================================")
    print("訓練完成！")
    print(f"最新模型：{latest_path}")
    if best_path is not None:
        print(f"最佳模型：{best_path}")
    print(f"Loss 曲線：{loss_fig_path}")
    print(f"IoU + PRO 曲線：{metrics_fig_path}")
    print("========================================")


# -----------------------------------------------------
# CLI
# -----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--pro_every", type=int, default=5)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    train(
        category=args.category,
        dataset_root=args.dataset_root,
        epochs=args.epochs,
        pro_every=args.pro_every,
        img_size=args.img_size,
        batch_size=args.batch_size,
        lr=args.lr,
        tag=args.tag
    )
