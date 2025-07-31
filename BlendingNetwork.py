import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

class BlendingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.encoder = nn.Sequential(
            # C64→128
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            # C128→256
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            # C256→128
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
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
        features = self.encoder(x)
        x = self.decoder(features) + x
        # 各参数预测
        w = self.weight_head(x)
        return (1 - w) * hdr_env + w * ldr_env