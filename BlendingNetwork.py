import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

class BlendingNetwork(nn.Module):
    def __init__(self, use_full=True):
        super().__init__()
        self.use_full = use_full
        self.conv = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.encoder1 = nn.Sequential(
            # C64→128
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.encoder2 = nn.Sequential(
            # C128→256
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            # C256→128
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.decoder1 = nn.Sequential(
            # C128→64
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.weight_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, hdr_env, ldr_env, mask):
        input = torch.cat([hdr_env, ldr_env, mask, 1 - mask], dim=0)
        x = self.conv(input)
        features = self.encoder1(x)
        if self.use_full:
            features = self.encoder2(features)
            features = self.decoder2(features)
        x = self.decoder1(features) + x
        # 各参数预测
        w = self.weight_head(x)
        return (1 - w) * hdr_env + w * ldr_env