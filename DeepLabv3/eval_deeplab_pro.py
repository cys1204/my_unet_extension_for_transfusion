import os
import argparse
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from scipy.ndimage import label

from dataset import SegmentationDataset
from model_deeplab import DeepLabV3Plus


# --------------------------
# Transform
# --------------------------
def get_eval_transform(img_size=256):
    import albumentations as A
    return A.Compose([
        A.Resize(img_size, img_size),
    ])


# --------------------------
# PRO 計算
# --------------------------
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


# --------------------------
# Visualization
# --------------------------
def visualize(output_dir, img, gt, pred, name):
    os.makedirs(output_dir, exist_ok=True)

    img = (img * 255).astype(np.uint8)
    pred_bin = (pred > 0.5).astype(np.uint8) * 255
    gt = (gt > 0.5).astype(np.uint8) * 255

    overlay = img.copy()
    overlay[pred > 0.5] = [255, 0, 0]  # 紅色區域為預測結果

    cv2.imwrite(os.path.join(output_dir, f"{name}_img.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, f"{name}_gt.png"), gt)
    cv2.imwrite(os.path.join(output_dir, f"{name}_pred.png"), pred_bin)
    cv2.imwrite(os.path.join(output_dir, f"{name}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


# --------------------------
# Main
# --------------------------
def main(category, dataset_root, tag=""):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = f"deeplabv3_{category}"
    if tag:
        model_name += f"_{tag}"

    model_path = f"checkpoints/{model_name}.pth"

    print(f"載入模型：{model_path}")
    if not os.path.exists(model_path):
        print("❌ 找不到模型！")
        return

    # 載入模型
    model = DeepLabV3Plus().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 正確 dataset 呼叫方式
    dataset = SegmentationDataset(
        root=dataset_root,
        category=category,
        split="test",
        transform=get_eval_transform()
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"測試資料筆數：{len(dataset)}")

    all_preds, all_gts = [], []

    vis_dir = f"deeplab_eval_{category}_{tag}"
    os.makedirs(vis_dir, exist_ok=True)

    # --------------------------
    # inference
    # --------------------------
    with torch.no_grad():
        for idx, (img, mask) in enumerate(loader):
            img = img.to(device)
            pred = model(img)  # logits

            pred = pred.squeeze().cpu().numpy()
            gt = mask.squeeze().cpu().numpy()
            raw_img = img.squeeze().permute(1, 2, 0).cpu().numpy()

            all_preds.append(pred)
            all_gts.append(gt)

            visualize(vis_dir, raw_img, gt, pred, f"{idx}")

    all_preds = np.stack(all_preds)
    all_gts = np.stack(all_gts)

    # Pixel AUROC
    try:
        auroc = roc_auc_score(all_gts.flatten(), all_preds.flatten())
    except ValueError:
        auroc = float("nan")

    # PRO
    pro_score = compute_pro(all_preds, all_gts)

    print("\n===== 評估結果 =====")
    print(f"Category : {category}")
    print(f"Pixel AUROC : {auroc:.4f}")
    print(f"PRO : {pro_score:.4f}")
    print(f"Visualization directory: {vis_dir}")
    print("=================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    main(args.category, args.dataset_root, args.tag)
