import torch
import torch.nn as nn
import torch.nn.functional as F

# 3D编码解码器
class SGLVEncoderDecoder(nn.Module):
    def __init__(self, level=4):
        super().__init__()
        self.level = level
        self.conv = nn.Sequential(
            nn.Conv3d(5, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.encoder1 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.decoder1 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # 各参数预测头
        self.color_head = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Softplus()
        )
        self.alpha_head = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # 约束到[0,1]
        )
        self.w_head = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Softplus()
        )
        self.lamda_head = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Softplus()  # 确保正数
        )
        self.s_head = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 3, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, volume):
        Ve = volume[4, :, :]
        x = self.conv(volume)
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
        color = self.color_head(x) * (Ve+1)
        alpha = self.alpha_head(x) * (Ve+1)
        w = self.w_head(x) * (Ve+1)
        lamda = self.lamda_head(x) * (Ve+1)
        s = self.s_head(x) * (Ve+1)
        s = F.normalize(s, p=2, dim=0)
        return torch.cat([color, alpha, w, lamda, s], dim=0)