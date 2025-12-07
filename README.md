# 生成式期末專案
本專案建立在 TransFusion + U-Net + DeepLabv3+ 的影像瑕疵分割流程之上，  
提供 **GT（Ground Truth）版本** 以及 **Pseudo Mask（模型預測遮罩）版本** 的訓練與評估管線。

本專案包含：

- MVTec dataset 前處理（RGB→可視化 GT、Pseudo Mask）
- U-Net segmentation（GT / PSEUDO）
- DeepLabv3+ segmentation（GT / PSEUDO）
- IoU、PRO 指標
- 訓練與推論可視化工具

---

# 📦 專案架構
```
transfusion/
│
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
由 TransFusion 產生可視化 RGB：

```
python Experiment.py \
    -c test \
    -r transfusion_mvtec \
    -d ./dataset/mvtec/ \
    -ds mvtec \
    --mode rgb \
    --visualize \
    --category bottle
```
bottle 類別共 292 張影像。
# 2️⃣ Split 可視化 dataset（7:2:1）

執行以下 script，產生：
    
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

python U_net/split_visualization_dataset.py

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
python U_net/eval_unet_pro.py \
    --category bottle \
    --dataset_root seg_dataset_visualization_GT \
    --tag GT
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
python U_net/eval_unet_pro.py \
    --category bottle \
    --dataset_root seg_dataset_visualization_PSEUDO \
    --tag PSEUDO
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
python DeepLabv3/eval_deeplab_pro.py
--category bottle
--dataset_root seg_dataset_visualization_PSEUDO
--tag PSEUDO
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
python DeepLabv3/eval_deeplab_pro.py
--category bottle
--dataset_root seg_dataset_visualization_PSEUDO
--tag PSEUDO
```

# 📊 評估指標
指標	說明
IoU	segmentation overlap（越高越好）
PRO	region-level overlap（適合 anomaly segmentation）
Loss	BCE（DeepLabv3+）或 BCE+Dice（U-Net）

訓練完成後會輸出：

logs/<model_name>/training_curve.png

包含三條曲線：

    Train Loss

    Validation IoU

    Validation PRO

作者

  CYS

   生成式 AI 期末專案

   模型：U-Net, DeepLabv3+

   任務：MVTec 瑕疵分割
