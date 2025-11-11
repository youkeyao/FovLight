import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

class BlendingNetwork(nn.Module):
    def __init__(self, level=4):
        super().__init__()
        self.level = level
        self.conv = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.encoder1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            # C256→128
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.decoder1 = nn.Sequential(
            # C128→64
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.weight1_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Softplus(),
        )
        self.weight2_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Softplus(),
        )

    def forward(self, hdr_env, ldr_env, mask):
        input = torch.cat([hdr_env, ldr_env, mask, 1 - mask], dim=0)
        x = self.conv(input)
        features = x
        if self.level > 1:
            features = self.encoder1(features)
        if self.level > 2:
            features = self.encoder2(features)
        if self.level > 3:
            features = self.encoder3(features)
            features = self.decoder3(features)
        if self.level > 2:
            features = self.decoder2(features)
        if self.level > 1:
            features = self.decoder1(features)
        x = features
        # 各参数预测
        w1 = self.weight1_head(x)
        w2 = self.weight2_head(x)
        return w1 * hdr_env + w2 * ldr_env