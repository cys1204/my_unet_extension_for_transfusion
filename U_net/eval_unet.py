import os
import argparse
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import albumentations as A

from dataset import SegmentationDataset
from model_unet import UNet


def get_eval_transform(img_size=256):
    # 評估時不做 flip，只做 resize，跟訓練的大小一致
    return A.Compose([
        A.Resize(img_size, img_size),
    ])


def save_image(path, img):
    """img: numpy array in [0,255] or [0,1]"""
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    cv2.imwrite(path, img)


def main(category):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load model ---
    model_path = f"checkpoints/unet_{category}.pth"
    if not os.path.exists(model_path):
        print(f"找不到模型：{model_path}")
        return

    print(f"載入模型：{model_path}")
    model = UNet().to(device)

    # 這裡 PyTorch 給的 warning 可以先無視，因為你載的是自己訓練的權重
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # --- Load test dataset (加上 Resize transform) ---
    test_root = f"seg_dataset/{category}/test"
    test_transform = get_eval_transform(img_size=256)
    dataset = SegmentationDataset(test_root, transform=test_transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # --- Output folder ---
    out_dir = f"eval_results/{category}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"開始推論，共 {len(loader)} 張測試圖片...")

    with torch.no_grad():
        for idx, (img, mask) in enumerate(loader):
            img_tensor = img.to(device)
            pred = model(img_tensor)
            pred = pred.squeeze().cpu().numpy()

            # Binarize the prediction
            pred_bin = (pred > 0.5).astype(np.uint8) * 255

            # Original image (經過 resize 過後的版本)
            img_np = img.squeeze().permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)

            # GT mask (可能是全 0：good 圖)
            gt_mask = mask.squeeze().numpy()
            gt_mask = (gt_mask * 255).astype(np.uint8)

            # Save files
            save_image(f"{out_dir}/{idx:03d}_image.png", img_np)
            save_image(f"{out_dir}/{idx:03d}_gt.png", gt_mask)
            save_image(f"{out_dir}/{idx:03d}_pred.png", pred_bin)

            # heatmap overlay
            heatmap = cv2.applyColorMap(pred_bin, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
            save_image(f"{out_dir}/{idx:03d}_overlay.png", overlay)

            print(f"輸出：{idx:03d}_image.png / gt.png / pred.png / overlay.png")

    print(f"\n全部完成！結果存放在： {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True)
    args = parser.parse_args()

    main(args.category)
