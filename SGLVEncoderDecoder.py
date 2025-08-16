import torch
import torch.nn as nn
import torch.nn.functional as F

# 3D编码解码器
class SGLVEncoderDecoder(nn.Module):
    def __init__(self, use_full=True):
        super().__init__()
        self.use_full = use_full
        self.conv = nn.Sequential(
            nn.Conv3d(5, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.encoder1 = nn.Sequential(
            # C64→128
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.encoder2 = nn.Sequential(
            # C128→256
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            # C256→128
            nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.decoder1 = nn.Sequential(
            # C128→64
            nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # 各参数预测头
        self.color_head = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 3, kernel_size=3, padding=1),
            nn.Softplus()
        )
        self.alpha_head = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # 约束到[0,1]
        )
        self.w_head = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 3, kernel_size=3, padding=1),
        )
        self.lamda_head = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 1, kernel_size=3, padding=1),
            nn.Softplus()  # 确保正数
        )
        self.s_head = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 3, kernel_size=3, padding=1),
        )

    def forward(self, volume):
        Ve = volume[4, :, :]
        x = self.conv(volume)
        features = self.encoder1(x)
        if self.use_full:
            features = self.encoder2(features)
            features = self.decoder2(features)
        x = self.decoder1(features) + x
        # 各参数预测
        color = self.color_head(x) * (Ve+1)
        alpha = self.alpha_head(x) * (Ve+1)
        w = self.w_head(x) * (Ve+1)
        lamda = self.lamda_head(x) * (Ve+1)
        s = self.s_head(x) * (Ve+1)
        s = F.normalize(s, p=2, dim=1)
        return torch.cat([color, alpha, w, lamda, s], dim=0)