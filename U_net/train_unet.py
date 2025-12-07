import os
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import albumentations as A
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
from scipy.ndimage import label

from dataset import SegmentationDataset
from model_unet import UNet


# ----------------------------
# Transforms
# ----------------------------
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


# ----------------------------
# PRO 計算
# ----------------------------
def compute_pro(all_preds, all_gts, num_thresholds=50):

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
                    region_overlaps.append(inter / denom)

        pro_values.append(np.mean(region_overlaps) if region_overlaps else 0.0)

    return float(np.mean(pro_values))


# ----------------------------
# Training Function
# ----------------------------
def train(category,
          dataset_root="seg_dataset",
          num_epochs=25,
          pro_every=5,
          tag=None):

    model_name = f"unet_{category}"
    if tag:
        model_name += f"_{tag}"

    print("========================================")
    print(f"開始訓練 U-Net")
    print(f"Category     : {category}")
    print(f"Dataset root : {dataset_root}")
    print(f"Model name   : {model_name}")
    print(f"Epochs       : {num_epochs}")
    print("========================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs("checkpoints", exist_ok=True)
    log_dir = os.path.join("logs", model_name)
    os.makedirs(log_dir, exist_ok=True)

    # Dataset
    class_root = os.path.join(dataset_root, category)

    train_set = SegmentationDataset(class_root, split="train", transform=get_transform())
    val_set = SegmentationDataset(class_root, split="val", transform=get_val_transform())

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # Model
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    bce = nn.BCELoss()
    dice = lambda p, g: 1 - (2 * (p * g).sum() + 1) / (p.sum() + g.sum() + 1)

    # Logs
    loss_history = []
    iou_history = []
    pro_history = []
    pro_epoch_index = []

    best_iou = -1.0
    best_path = None

    # -------------------------
    # Training Loop
    # -------------------------
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

        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)

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

        mean_iou = float(np.mean(ious)) if ious else 0.0
        iou_history.append(mean_iou)

        print(f"Epoch {epoch}/{num_epochs} | Loss={epoch_loss:.4f} | IoU={mean_iou:.4f}")

        # ---- Save best ----
        if mean_iou > best_iou:
            best_iou = mean_iou
            best_path = os.path.join("checkpoints", f"{model_name}_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"★ Best IoU 更新：{best_iou:.4f} → 已儲存 {best_path}")

        # ---- PRO ----
        if epoch % pro_every == 0:
            preds, gts = [], []
            with torch.no_grad():
                for img, mask in val_loader:
                    img = img.to(device)
                    pred = model(img)

                    preds.append(pred.squeeze().cpu().numpy())
                    gts.append(mask.squeeze().cpu().numpy())

            preds = np.stack(preds)
            gts = np.stack(gts)

            pro_score = compute_pro(preds, gts)
            pro_history.append(pro_score)
            pro_epoch_index.append(epoch)

            print(f" → PRO@{epoch}: {pro_score:.4f}")

        # ---- Save latest ----
        latest_path = os.path.join("checkpoints", f"{model_name}_latest.pth")
        torch.save(model.state_dict(), latest_path)

        # ---- Save milestone ----
        if epoch % 50 == 0:
            milestone_path = os.path.join("checkpoints", f"{model_name}_epoch{epoch:03d}.pth")
            torch.save(model.state_dict(), milestone_path)
            print(f"✔ Milestone saved: {milestone_path}")

    # -------------------------
    # Plot: Loss Curve
    # -------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"U-Net Loss ({model_name})")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(log_dir, "loss_curve.png"))
    plt.close()

    # -------------------------
    # Plot: IoU + PRO
    # -------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), iou_history, label="Val IoU", color="orange")
    if pro_history:
        plt.plot(pro_epoch_index, pro_history, "o-", label="PRO", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"U-Net Metrics ({model_name})")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(log_dir, "metrics_curve.png"))
    plt.close()

    print("========================================")
    print("訓練完成！")
    print("========================================")


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, default="seg_dataset")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--pro_every", type=int, default=5)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    tag = args.tag if args.tag != "" else None

    train(
        category=args.category,
        dataset_root=args.dataset_root,
        num_epochs=args.epochs,
        pro_every=args.pro_every,
        tag=tag,
    )
