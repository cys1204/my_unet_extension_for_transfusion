import os
import argparse
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score
from scipy.ndimage import label

import albumentations as A
import cv2

from dataset import SegmentationDataset
from model_unet import UNet


# ----------------------------
# Transform
# ----------------------------
def get_eval_transform(img_size: int = 256):
    """
    評估時不做增強，只做 Resize，需跟訓練時的大小一致
    """
    return A.Compose([
        A.Resize(img_size, img_size),
    ])


# ----------------------------
# PRO 計算函式
# ----------------------------
def compute_pro(all_preds: np.ndarray,
                all_gts: np.ndarray,
                num_thresholds: int = 50) -> float:
    """
    計算 Per-Region Overlap (PRO) 分數

    all_preds: (N, H, W) 浮點數 anomaly map, range [0, 1]
    all_gts:   (N, H, W) 0/1 ground truth mask
    """
    all_preds = all_preds.astype(np.float32)
    all_gts = (all_gts > 0.5).astype(np.uint8)

    N, H, W = all_preds.shape
    thresholds = np.linspace(0, 1, num_thresholds)

    pro_values = []

    for thr in thresholds:
        # 二值化 prediction
        preds_bin = all_preds >= thr  # bool, shape (N, H, W)
        region_overlaps = []

        for n in range(N):
            gt = all_gts[n]
            pred = preds_bin[n]

            # 對 GT 做連通區域標記（每個 defect region 一個 label）
            labeled_gt, num_regions = label(gt)

            if num_regions == 0:
                continue

            for rid in range(1, num_regions + 1):
                region = (labeled_gt == rid)

                inter = np.logical_and(region, pred).sum()
                denom = region.sum()
                if denom == 0:
                    continue

                overlap = inter / float(denom)
                region_overlaps.append(overlap)

        if len(region_overlaps) == 0:
            pro_values.append(0.0)
        else:
            pro_values.append(float(np.mean(region_overlaps)))

    # 對所有 threshold 的 PRO 取平均
    pro_score = float(np.mean(pro_values))
    return pro_score


# ----------------------------
# Main Evaluation + Visualization
# ----------------------------
def main(category: str,
         dataset_root: str = "seg_dataset",
         tag: Optional[str] = None,
         img_size: int = 256,
         num_thresholds: int = 50,
         save_vis: bool = True,
         vis_root: str = "eval_results"):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 確定模型名稱與路徑 ---
    model_name = f"unet_{category}"
    if tag:
        model_name += f"_{tag}"
    model_path = os.path.join("checkpoints", f"{model_name}.pth")

    if not os.path.exists(model_path):
        print(f"[錯誤] 找不到模型：{model_path}")
        return

    print(f"載入模型：{model_path}")
    model = UNet().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # --- 建立 test dataset + dataloader ---
    # 結構預期：
    #   dataset_root/category/images/test/*.png
    #   dataset_root/category/masks/test/*.png
    class_root = os.path.join(dataset_root, category)
    transform = get_eval_transform(img_size=img_size)
    dataset = SegmentationDataset(class_root, split="test", transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"測試資料數量：{len(dataset)}")

    # --- 可視化輸出資料夾 ---
    run_name = category if tag is None else f"{category}_{tag}"
    out_dir = os.path.join(vis_root, run_name)
    if save_vis:
        os.makedirs(out_dir, exist_ok=True)
        print(f"可視化結果將輸出到: {out_dir}")

    all_preds = []
    all_gts = []

    with torch.no_grad():
        for idx, (img, mask) in enumerate(loader):
            # img: (1,3,H,W), mask: (1,1,H,W)
            img = img.to(device)
            mask = mask.to(device)

            pred = model(img)                    # (1,1,H,W)
            pred_np = pred.squeeze().cpu().numpy()   # (H,W)
            gt_np = mask.squeeze().cpu().numpy()     # (H,W)

            all_preds.append(pred_np)
            all_gts.append(gt_np)

            if save_vis:
                # 從 tensor 還原原圖 (RGB, [0,1]) -> uint8 BGR
                img_np = img[0].detach().cpu().permute(1, 2, 0).numpy()
                img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                # GT mask: 0/1 -> 0/255
                gt_vis = (gt_np * 255).astype(np.uint8)

                # Pred binary mask: threshold 0.5
                pred_bin = (pred_np > 0.5).astype(np.uint8) * 255

                # Heatmap (根據 pred_bin 或 pred_np 都可以，這裡用 pred_np 連續值)
                # 先把 pred_np 正規化到 0~255
                pred_norm = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-6)
                pred_norm_u8 = (pred_norm * 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(pred_norm_u8, cv2.COLORMAP_JET)

                # 疊合圖片 (Original 60% + Heatmap 40%)
                overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)

                idx_str = f"{idx:03d}"
                cv2.imwrite(os.path.join(out_dir, f"{idx_str}_image.png"), img_bgr)
                cv2.imwrite(os.path.join(out_dir, f"{idx_str}_gt.png"), gt_vis)
                cv2.imwrite(os.path.join(out_dir, f"{idx_str}_pred.png"), pred_bin)
                cv2.imwrite(os.path.join(out_dir, f"{idx_str}_heatmap.png"), heatmap)
                cv2.imwrite(os.path.join(out_dir, f"{idx_str}_overlay.png"), overlay)

    all_preds = np.stack(all_preds, axis=0)  # (N,H,W)
    all_gts = np.stack(all_gts, axis=0)      # (N,H,W)

    # --- Pixel-wise AUROC ---
    try:
        roc_pixel = roc_auc_score(all_gts.flatten() > 0.5,
                                  all_preds.flatten())
    except ValueError:
        # 如果全部都是 good（沒有 positive），roc_auc_score 會報錯
        roc_pixel = float("nan")

    # --- PRO ---
    pro_score = compute_pro(all_preds, all_gts, num_thresholds=num_thresholds)

    print("\n===== 評估結果 =====")
    print(f"Category      : {category}")
    if tag:
        print(f"Model tag     : {tag}")
    print(f"Dataset root  : {dataset_root}")
    print(f"Pixel AUROC   : {roc_pixel:.4f}")
    print(f"PRO (region)  : {pro_score:.4f}")
    if save_vis:
        print(f"Visualization : {out_dir}")
    print("====================\n")


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
        "--tag",
        type=str,
        default="",
        help="模型命名用的標籤（例如 GT, PSEUDO）",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="輸入影像 resize 大小（需與訓練時一致）",
    )
    parser.add_argument(
        "--num_thresholds",
        type=int,
        default=50,
        help="計算 PRO 時使用的 threshold 數量",
    )
    parser.add_argument(
        "--no_vis",
        action="store_true",
        help="如果加上此旗標，就只算指標、不輸出圖",
    )
    parser.add_argument(
        "--vis_root",
        type=str,
        default="eval_results",
        help="可視化輸出根目錄",
    )

    args = parser.parse_args()
    tag = args.tag if args.tag != "" else None

    main(
        category=args.category,
        dataset_root=args.dataset_root,
        tag=tag,
        img_size=args.img_size,
        num_thresholds=args.num_thresholds,
        save_vis=not args.no_vis,
        vis_root=args.vis_root,
    )
