#!/bin/bash

echo "==============================="
echo "  U-Net (GT) Training Started  "
echo "==============================="

python U_net/train_unet.py \
    --category bottle \
    --dataset_root seg_dataset_visualization_GT \
    --epochs 200 \
    --pro_every 5 \
    --tag GT

echo "==============================="
echo "     U-Net (GT) Evaluating     "
echo "==============================="

python U_net/eval_unet_pro.py \
    --category bottle \
    --dataset_root seg_dataset_visualization_GT \
    --tag GT

echo "==============================="
echo " U-Net (PSEUDO) Training Started "
echo "==============================="

python U_net/train_unet.py \
    --category bottle \
    --dataset_root seg_dataset_visualization_PSEUDO \
    --epochs 200 \
    --pro_every 5 \
    --tag PSEUDO

echo "==============================="
echo "   U-Net (PSEUDO) Evaluating    "
echo "==============================="

python U_net/eval_unet_pro.py \
    --category bottle \
    --dataset_root seg_dataset_visualization_PSEUDO \
    --tag PSEUDO

echo "======================================"
echo "   ALL TRAINING + EVALUATION DONE!    "
echo "======================================"
