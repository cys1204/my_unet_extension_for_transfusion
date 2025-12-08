import os
import argparse
import numpy as np
import torch
import cv2

from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys
sys.path.insert(0, ROOT)

from dataset import SegmentationDataset
from model_deeplab import DeepLabV3Plus
from utils import EvaluationUtils, VisualizationUtils

def normalize_like_experiment(pred_np):
    p_min = pred_np.min()
    p_max = pred_np.max()
    return (pred_np - p_min) / (p_max - p_min + 1e-6)

def evaluate_deeplab(model, dataset, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    preds_list, gts_list = [], []

    model.eval()
    with torch.no_grad():
        for idx, (img, mask) in enumerate(loader):
            img = img.to(device)
            gt = mask.squeeze().cpu().numpy().astype(np.uint8)

            out = model(img)
            logits = out["out"] if isinstance(out, dict) else out

            if logits.dim() == 4:
                logits = logits[:, 0, :, :]

            pred = torch.sigmoid(logits)
            pred_np = pred.squeeze().cpu().numpy()

            pred_np = normalize_like_experiment(pred_np)

            preds_list.append(pred_np)
            gts_list.append(gt)

            img_np = img[0].cpu().numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np, 0, 1)

            base = f"{idx}"

            VisualizationUtils.save_img_cv2(os.path.join(out_dir, f"{base}_true_mask.png"), gt)
            VisualizationUtils.save_img_cv2(os.path.join(out_dir, f"{base}_mask.png"), pred_np)
            VisualizationUtils.save_img_cv2(os.path.join(out_dir, f"{base}_true.png"), img_np)
            VisualizationUtils.save_heatmap_over_img(
                os.path.join(out_dir, f"{base}_heatmap.png"),
                img_np,
                pred_np,
            )

    preds = np.stack(preds_list)
    gts = np.stack(gts_list)

    auc_image = roc_auc_score(gts.reshape(len(gts), -1).max(axis=1),
                              preds.reshape(len(preds), -1).max(axis=1))
    auc_pixel = roc_auc_score(gts.flatten(), preds.flatten())
    ap_pixel = average_precision_score(gts.flatten(), preds.flatten())
    pro = EvaluationUtils.compute_pro(preds, gts)

    print("==============================")
    print("DeepLabV3+ evaluation results")
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
    parser.add_argument("--category", required=True)  # ★ DeepLab 需要 category
    parser.add_argument("--out_root", default="seg_results")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = SegmentationDataset(
        root=args.dataset_root,
        category=args.category,
        split="test",
        transform=None,
    )

    model = DeepLabV3Plus().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    out_dir = os.path.join(args.out_root, args.category, "deeplab")

    evaluate_deeplab(model, dataset, device, out_dir)

if __name__ == "__main__":
    main()
