import torch
import torch.nn as nn
import torchvision.models as models


# ------------------------------------------------------
# ASPP 模組
# ------------------------------------------------------
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        p1 = self.block1(x)
        p2 = self.block2(x)
        p3 = self.block3(x)
        p4 = self.block4(x)

        gp = self.global_pool(x)
        gp = nn.functional.interpolate(gp, size=(h, w), mode="bilinear", align_corners=False)

        x = torch.cat([p1, p2, p3, p4, gp], dim=1)
        return self.conv_out(x)


# ------------------------------------------------------
# DeepLabV3+ 主模型
# ------------------------------------------------------
class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        # Backbone: ResNet50(pretrained=True)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1   # 256 channels
        self.layer2 = resnet.layer2   # 512 channels
        self.layer3 = resnet.layer3   # 1024 channels
        self.layer4 = resnet.layer4   # 2048 channels

        # ASPP
        self.aspp = ASPP(2048, 256)

        # 縮放後的低階特徵
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # 最後的 Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        h, w = x.shape[2:]

        x0 = self.layer0(x)
        x1 = self.layer1(x0)    # low-level (256 ch)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)    # high-level (2048 ch)

        aspp = self.aspp(x4)
        aspp = nn.functional.interpolate(aspp, size=x1.shape[2:], mode="bilinear", align_corners=False)

        low = self.low_level_conv(x1)
        concat = torch.cat([aspp, low], dim=1)

        out = self.decoder(concat)
        out = nn.functional.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)

        return out
