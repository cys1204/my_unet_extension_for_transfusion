import os
import sys
import argparse

import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

# ---------------------------------------------------------
# 讓 Python 看得到專案根目錄 (ECCV_TransFusion)
# ---------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from dataset import SegmentationDataset
from model_unet import UNet
from utils import EvaluationUtils, VisualizationUtils


def normalize_like_experiment(pred_np):
    """模仿 Experiment.py 的 normalize 行為，使視覺化公平"""
    p_min = pred_np.min()
    p_max = pred_np.max()
    pred_np = (pred_np - p_min) / (p_max - p_min + 1e-6)
    return pred_np


def evaluate_unet(model, dataset, device, out_dir):
    """
     - model: UNet
     - dataset: SegmentationDataset
     - out_dir: 例如 seg_results/bottle/unet
    """
    os.makedirs(out_dir, exist_ok=True)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    preds_list, gts_list = [], []

    model.eval()
    with torch.no_grad():
        for idx, (img, mask) in enumerate(loader):
            img = img.to(device)                    # [1,3,H,W]
            gt = mask.squeeze().cpu().numpy()      # [H,W] in {0,1}

            # ---- 推論 ----
            logits = model(img)                    # [1,1,H,W] or [1,H,W]
            if logits.dim() == 4:
                logits = logits[:, 0, :, :]

            pred = torch.sigmoid(logits)
            pred_np = pred.squeeze().cpu().numpy()

            # ---- ★ normalize 與 Experiment 一致 ----
            pred_np_norm = normalize_like_experiment(pred_np)

            preds_list.append(pred_np_norm)
            gts_list.append(gt)

            # ---- 視覺化輸出 ----
            img_np = img[0].cpu().numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np, 0, 1)

            base_name = f"{idx}"

            # GT mask
            VisualizationUtils.save_img_cv2(
                os.path.join(out_dir, f"{base_name}_true_mask.png"), gt
            )

            # 預測 mask（與 TransFusion 相同 normalize 規則）
            VisualizationUtils.save_img_cv2(
                os.path.join(out_dir, f"{base_name}_mask.png"), pred_np_norm
            )

            # 原圖
            VisualizationUtils.save_img_cv2(
                os.path.join(out_dir, f"{base_name}_true.png"), img_np
            )

            # Heatmap 疊加（與 Experiment 相同）
            VisualizationUtils.save_heatmap_over_img(
                os.path.join(out_dir, f"{base_name}_heatmap.png"),
                img_np,
                pred_np_norm,
            )

    preds = np.stack(preds_list)
    gts = np.stack(gts_list)

    # ---- 指標計算 ----
    img_scores_pred = preds.reshape(preds.shape[0], -1).max(axis=1)
    img_scores_true = gts.reshape(gts.shape[0], -1).max(axis=1)

    auc_image = roc_auc_score(img_scores_true, img_scores_pred)
    auc_pixel = roc_auc_score(gts.flatten(), preds.flatten())
    ap_pixel = average_precision_score(gts.flatten(), preds.flatten())
    pro = EvaluationUtils.compute_pro(preds, gts)

    print("==============================")
    print("UNet evaluation results")
    print(f"Image AUROC : {auc_image:.4f}")
    print(f"Pixel AUROC : {auc_pixel:.4f}")
    print(f"Pixel AP    : {ap_pixel:.4f}")
    print(f"PRO         : {pro:.4f}")
    print("==============================")

    return auc_pixel, ap_pixel, auc_image, pro


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--out_root", type=str, default="seg_results")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 若未指定 category → 自動從路徑最後一層推斷
    if args.category is None:
        category = os.path.basename(os.path.normpath(args.dataset_root))
    else:
        category = args.category

    # 创建 dataset
    dataset = SegmentationDataset(
        root=args.dataset_root,
        split="test",
        transform=None,
    )

    model = UNet().to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)

    out_dir = os.path.join(args.out_root, category, "unet")

    auc_pixel, ap_pixel, auc_image, pro = evaluate_unet(
        model, dataset, device, out_dir
    )

    print("==== Summary (UNet) ====")
    print(f"Pixel AUROC : {auc_pixel:.4f}")
    print(f"Pixel AP    : {ap_pixel:.4f}")
    print(f"Image AUROC : {auc_image:.4f}")
    print(f"PRO         : {pro:.4f}")
    print("========================")


if __name__ == "__main__":
    main()
