import os
import argparse
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.ndimage import label

from U_net.dataset import SegmentationDataset
from U_net.model_unet import UNet
from DeepLabv3.model_deeplab import DeepLabV3Plus
from utils import ParsingUtils, EvaluationUtils   # for TransFusion

# ==========================================
# PRO（與原論文一致）
# ==========================================
def compute_pro(preds, gts, num_thresholds=50):
    preds = preds.astype(np.float32)
    gts = (gts > 0.5).astype(np.uint8)

    thresholds = np.linspace(0, 1, num_thresholds)
    all_scores = []

    for thr in thresholds:
        pred_bin = preds >= thr
        per_thr = []

        for i in range(len(gts)):
            gt = gts[i]
            pb = pred_bin[i]

            labeled, num_regions = label(gt)
            if num_regions == 0:
                continue

            regions = []
            for rid in range(1, num_regions + 1):
                region = (labeled == rid)
                inter = np.logical_and(region, pb).sum()
                denom = region.sum()
                if denom > 0:
                    regions.append(inter / denom)

            per_thr.append(np.mean(regions) if regions else 0)

        all_scores.append(np.mean(per_thr))

    return float(np.mean(all_scores))


# ==========================================
# Predictors
# ==========================================
def predict_unet(model, img):
    pred = model(img)
    return pred.squeeze().cpu().numpy()

def predict_deeplab(model, img):
    pred = model(img)['out']
    return pred.squeeze().cpu().numpy()

def predict_transfusion(model, idx_tensor, testset):
    recon, disc, recon2 = EvaluationUtils.calculate_transfussion_results(
        model, testset, idx_tensor
    )
    w = 0.5
    return (w * disc.squeeze().cpu().numpy() +
            (1 - w) * recon2.squeeze().cpu().numpy())


# ==========================================
# Main Evaluation (核心)
# ==========================================
def evaluate(model_type, model, testset, device):

    loader = DataLoader(testset, batch_size=1, shuffle=False)

    preds_list, gts_list = [], []
    img_pred_scores, img_true_labels = [], []

    model.eval()

    with torch.no_grad():
        for idx, (img, mask) in enumerate(loader):

            img = img.to(device)
            gt = mask.squeeze().cpu().numpy()

            # ------------------------------
            # Prediction
            # ------------------------------
            if model_type == "unet":
                pred = predict_unet(model, img)

            elif model_type == "deeplab":
                pred = predict_deeplab(model, img)

            elif model_type == "transfusion":
                pred = predict_transfusion(
                    model,
                    idx_tensor=torch.tensor([idx]).to(device),
                    testset=testset
                )

            preds_list.append(pred)
            gts_list.append(gt)

            # ------------------------------
            # Image-level AUROC labels
            # ------------------------------
            img_pred_scores.append(np.max(pred))
            img_true_labels.append(1 if np.max(gt) > 0.5 else 0)

    preds = np.stack(preds_list)
    gts = np.stack(gts_list)

    # ===============================================
    # Pixel AUROC
    # ===============================================
    auc_pixel = roc_auc_score(gts.flatten(), preds.flatten())

    # ===============================================
    # Pixel AP
    # ===============================================
    ap_pixel = average_precision_score(gts.flatten(), preds.flatten())

    # ===============================================
    # Image-level AUROC
    # ===============================================
    auc_image = roc_auc_score(img_true_labels, img_pred_scores)

    # ===============================================
    # PRO
    # ===============================================
    pro = compute_pro(preds, gts)

    return auc_pixel, ap_pixel, auc_image, pro


# ==========================================
# CLI
# ==========================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", choices=["unet", "deeplab", "transfusion"])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--mode", type=str, default="rgb")  # transfusion

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------
    # Load dataset
    # ------------------------------------------
    testset = SegmentationDataset(
        os.path.join(args.dataset_root),
        split="test",
        transform=None
    )

    # ------------------------------------------
    # Load model by type
    # ------------------------------------------
    if args.model_type == "unet":
        model = UNet().to(device)
        model.load_state_dict(torch.load(args.model_path))

    elif args.model_type == "deeplab":
        model = DeepLabV3Plus().to(device)
        model.load_state_dict(torch.load(args.model_path))

    elif args.model_type == "transfusion":
        dummy_args = argparse.Namespace()
        dummy_args.category = args.category
        dummy_args.mode = args.mode
        dummy_args.dataset = "mvtec"

        _, _, _, model, _, _, _ = ParsingUtils.parse_args(dummy_args)
        state = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state["net_state_dict"])
        model = model.to(device)

    # ------------------------------------------
    # Run evaluator
    # ------------------------------------------
    auc_pixel, ap_pixel, auc_image, pro = evaluate(
        args.model_type, model, testset, device
    )

    print("==========================")
    print(f"Model: {args.model_type}")
    print(f"Pixel AUROC   : {auc_pixel:.4f}")
    print(f"Pixel AP      : {ap_pixel:.4f}")
    print(f"Image AUROC   : {auc_image:.4f}")
    print(f"PRO score     : {pro:.4f}")
    print("==========================")



if __name__ == "__main__":
    main()
