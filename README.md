# 生成式期末專案
本專案基於 TransFusion (ECCV 2022) 框架進行延伸，整合了 U-Net 與 DeepLabv3+ 的影像瑕疵分割流程。我們使用了兩種訓練方式以作為比較：使用 Ground Truth (GT) 以及使用模型預測的 Pseudo Mask 進行監督學習與評估。
本專案包含：

- MVTec dataset 前處理（RGB→可視化 GT、Pseudo Mask）
- U-Net segmentation（GT / PSEUDO）
- DeepLabv3+ segmentation（GT / PSEUDO）
- IoU、PRO 指標
- 訓練與推論可視化工具
---
# 📖 參考與致謝 (Acknowledgment)
本專案的基礎架構參考自以下研究成果，特此致謝：

TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Normalizing Flows (ECCV 2022)
* [📄 Paper](https://arxiv.org/abs/2311.09999) | [💻 Original Code](https://github.com/MaticFuc/ECCV_TransFusion)

註：本專案將其核心特徵提取邏輯應用於 MVTec 影像瑕疵檢測任務。

---

# 📦 專案架構
```
ECCV_TransFusion/
│
├── dataset/
|  
├── U_net/
│ ├── dataset.py
│ ├── model_unet.py
│ ├── train_unet.py
│ ├── eval_unet_pro.py
│ └── split_visualization_dataset.py
│
├── DeepLabv3/
│ ├── dataset.py
│ ├── model_deeplab.py
│ ├── train_deeplab.py
│ └── eval_deeplab_pro.py
│
└── README.md
```
---

# 1️⃣ 產生可視化資料（TransFusion）

先做
    
    dataset/mvtec/bottle/train/good 的資料夾複製一份到 test 資料夾中，名字改成 good_train
    (這邊以bottle為例 記得改成自己要跑得類別)

然後執行

    python Experiment.py \
        -c test \
        -r transfusion_mvtec \
        -d ./dataset/mvtec/ \
        -ds mvtec \
        --mode rgb \
        --visualize \
        --category bottle


# 2️⃣ Split 可視化 dataset（7:1:2）

重要！！！
因為一些問題 請自己把 ECCV_TransFusion裡面的 utils 資料夾複製到 U_net 和 DeepLabv3 中各一份！！！
打開 split_visualization_dataset.py 改第10行
ROOT = "experiments/transfusion_mvtec/visualizations/bottle"
把 bottle 改成自己要跑得類別


執行：
    
    python split_visualization_dataset.py

產生：
    
    seg_dataset_visualization_GT
        images
            rain/
            val/
            test/
        masks
            train/
            val/
            test/
            
    seg_dataset_visualization_PSEUDO
        images
            rain/
            val/
            test/
        masks
            train/
            val/
            test/

來源包含 Ground Truth / Pseudo Mask：

輸出格式如下：
```
seg_dataset_visualization_GT/bottle/
seg_dataset_visualization_PSEUDO/bottle/
```
# 3️⃣ U-Net 訓練與評估
## U-Net（使用 Ground Truth）
### 訓練
```
python U_net/train_unet.py \
    --category bottle \
    --dataset_root seg_dataset_visualization_GT \
    --epochs 50 \
    --pro_every 5 \
    --tag GT
```
### 評估
```
 python U_net/eval_unet_pro.py     --model_path checkpoints/unet_bottle_GT.pth     --dataset_root seg_dataset_visualization_GT/bottle
```
## U-Net（使用 Pseudo Mask）
### 訓練
```
python U_net/train_unet.py \
    --category bottle \
    --dataset_root seg_dataset_visualization_PSEUDO \
    --epochs 50 \
    --pro_every 5 \
    --tag PSEUDO
```
### 評估
```
 python U_net/eval_unet_pro.py     --model_path checkpoints/unet_bottle_PSEUDO.pth     --dataset_root seg_dataset_visualization_PSEUDO/bottle
```
# 4️⃣ DeepLabv3+ 訓練與評估
## DeepLabv3+（使用 Ground Truth）
### 訓練
```
python DeepLabv3/train_deeplab.py
--category bottle
--dataset_root seg_dataset_visualization_PSEUDO
--epochs 50
--pro_every 5
--tag PSEUDO
```

### 評估
```
python DeepLabv3/eval_deeplab_pro.py     --model_path checkpoints/deeplabv3_bottle_GT.pth     --dataset_root seg_dataset_visualization_GT     --category bottle
```
## DeepLabv3+（使用 Pseudo Mask）
### 訓練
```
python DeepLabv3/train_deeplab.py
--category bottle
--dataset_root seg_dataset_visualization_PSEUDO
--epochs 50
--pro_every 5
--tag PSEUDO
```

### 評估
```
python DeepLabv3/eval_deeplab_pro.py     --model_path checkpoints/deeplabv3_bottle_PSEUDO.pth     --dataset_root seg_dataset_visualization_PSEUDO     --category bottle
```

# 5️⃣ TransFusion 評估
## 使用 Ground Truth
### 評估
```
python experiment.py     -c test     --custom_seg_dataset seg_dataset_visualization_GT     --category bottle     --run-name transfusion_mvtec     --visualize     --mode rgb
```


# 📊 評估指標
指標	說明
⭐ 1. Image AUROC（影像層級 AUROC）
👉 衡量模型「判斷整張圖片是否異常」的能力

不是看每個像素，而是看整張圖是否有異常。
如何計算：

    每張圖取 模型預測 anomaly map 的最大值（max score）

    與該圖是否異常（0/1 標籤）比較

    計算 ROC 曲線下面積（AUC）

代表意義：

越接近 1 → 模型越能區分正常圖與異常圖。
⭐ 2. Pixel AUROC（像素層級 AUROC）
👉 衡量模型是否能在「像素層級」區分異常與正常

每個像素都視為一個二分類問題。
如何計算：

    將所有像素展平

    anomaly score vs GT mask（0/1）比較

    計算 ROC‑AUC

代表意義：

越接近 1 → 模型越能清楚區分哪個像素是異常的。
⭐ 3. Pixel AP（Pixel Average Precision）
👉 衡量模型「像素層級」預測準確度的另一種方式

特別適合檢測「異常區域較小」的任務。
如何計算：

    使用 precision-recall 曲線

    計算面積（Average Precision，AP）

代表意義：

越高 → 模型對異常區域的偵測越精準、越不會亂亮整張圖。
⭐ 4. PRO（Per-Region Overlap）
👉 評估模型在「異常區域輪廓」的匹配程度

比 Pixel AUROC 更能反映「模型是否畫對形狀」。
如何計算（概念化）：

    對不同 threshold 切出 binary mask

    計算每個異常區域的 overlap（類似 IoU）

    對多個 threshold 取平均 → 得到 PRO score

代表意義：

越高 → 預測到的異常形狀越準，越接近 GT 的輪廓。

這是 anomaly segmentation 中最重要、但也最敏感的指標。


Loss	BCE（DeepLabv3+）或 BCE+Dice（U-Net）


作者

      CYS

       生成式 AI 期末專案

       模型：U-Net, DeepLabv3+

       任務：MVTec 瑕疵分割
