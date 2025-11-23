import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU3D(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv_z = nn.Conv3d(channels * 2, channels, kernel_size, padding=padding)
        self.conv_r = nn.Conv3d(channels * 2, channels, kernel_size, padding=padding)
        self.conv_h = nn.Conv3d(channels * 2, channels, kernel_size, padding=padding)

    def forward(self, x, h_prev):
        if h_prev is None:
            h_prev = torch.zeros_like(x, device=x.device)
        combined = torch.cat([x, h_prev], dim=0)
        z = torch.sigmoid(self.conv_z(combined))
        r = torch.sigmoid(self.conv_r(combined))
        combined_r = torch.cat([x, r * h_prev], dim=0)
        h_tilde = torch.tanh(self.conv_h(combined_r))
        h_new = (1 - z) * h_prev + z * h_tilde
        return h_new

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
        self.gru0 = GRU3D(32)

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

        if self.level == 2:
            self.gru1 = GRU3D(64)
        elif self.level == 3:
            self.gru1 = GRU3D(128)
        elif self.level == 4:
            self.gru1 = GRU3D(256)

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
        self.u_head = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # 约束到[0,1]
        )

        self.h0 = None
        self.h1 = None

    def forward(self, volume):
        if self.h0 is not None:
            self.h0 = self.h0.detach()
        if self.h1 is not None:
            self.h1 = self.h1.detach()

        Ve = volume[4, :, :]
        x = self.conv(volume)
        features = x
        self.h0 = self.gru0(features, self.h0)
        if self.level > 1:
            features = self.encoder1(features)
        if self.level > 2:
            features = self.encoder2(features)
        if self.level > 3:
            features = self.encoder3(features)
            self.h1 = self.gru1(features, self.h1)
            features = self.decoder3(self.h1)
        else:
            self.h1 = self.gru1(features, self.h1)
            features = self.h1
        if self.level > 2:
            features = self.decoder2(features)
        else:
            self.h1 = self.gru1(features, self.h1)
            features = self.h1
        if self.level > 1:
            features = self.decoder1(features)
        x = features + self.h0
        # 各参数预测
        color = self.color_head(x) * (Ve+1)
        alpha = self.alpha_head(x) * (Ve+1)
        w = self.w_head(x) * (Ve+1)
        lamda = self.lamda_head(x) * (Ve+1)
        s = self.s_head(x) * (Ve+1)
        s = F.normalize(s, p=2, dim=0)
        u = self.u_head(x)

        return torch.cat([color, alpha, w, lamda, s], dim=0), u
    
    def reset(self):
        self.h0 = None
        self.h1 = None