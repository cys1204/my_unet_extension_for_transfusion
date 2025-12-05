import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()

        intersection = (pred * target).sum()
        denom = pred.sum() + target.sum() + self.eps
        return 1 - (2 * intersection / denom)

def iou_score(pred, target):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return intersection / union.clamp(min=1e-6)
