#!/bin/bash

echo "========================================"
echo "   DeepLabv3+ (GT) Training Started     "
echo "========================================"

python DeepLabv3/train_deeplab.py \
    --category bottle \
    --dataset_root seg_dataset_visualization_GT \
    --epochs 200 \
    --pro_every 5 \
    --tag GT

echo "========================================"
echo "        DeepLabv3+ (GT) Evaluating      "
echo "========================================"

python DeepLabv3/eval_deeplab_pro.py \
    --category bottle \
    --dataset_root seg_dataset_visualization_GT \
    --tag GT


echo "=========================================="
echo "  DeepLabv3+ (PSEUDO) Training Started    "
echo "=========================================="

python DeepLabv3/train_deeplab.py \
    --category bottle \
    --dataset_root seg_dataset_visualization_PSEUDO \
    --epochs 200 \
    --pro_every 5 \
    --tag PSEUDO

echo "=========================================="
echo "       DeepLabv3+ (PSEUDO) Evaluating     "
echo "=========================================="

python DeepLabv3/eval_deeplab_pro.py \
    --category bottle \
    --dataset_root seg_dataset_visualization_PSEUDO \
    --tag PSEUDO


echo "======================================================="
echo "    ALL DEEPLABV3+ TRAINING + EVALUATION COMPLETED!    "
echo "======================================================="
