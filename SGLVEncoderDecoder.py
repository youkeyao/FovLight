import torch
import torch.nn as nn
import torch.nn.functional as F

# 3D编码解码器
class SGLVEncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(5, 64, kernel_size=3, stride=1, padding=1)
        self.encoder = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            # C64→128
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            # C128→256
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(256 * 84 * 60 * 64, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 256 * 84 * 60 * 64),
        #     nn.ReLU(),
        # )
        self.decoder = nn.Sequential(
            # C256→128
            nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            # C128→64
            nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
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
        x = self.conv(volume.unsqueeze(0))
        features = self.encoder(x)
        # features = features.view(1, -1)
        # features = self.fc(features)
        # features = features.view(1, 256, 84, 60, 64)
        x = self.decoder(features) + x
        # 各参数预测
        color = self.color_head(x) * (Ve + 1)
        alpha = self.alpha_head(x) * (Ve + 1)
        w = self.w_head(x) * (Ve + 1)
        lamda = self.lamda_head(x) * (Ve + 1)
        s = self.s_head(x) * (Ve + 1)
        s = F.normalize(s, p=2, dim=1)
        return torch.cat([color, alpha, w, lamda, s], dim=1).squeeze(0)