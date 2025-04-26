import torch
import torch.nn as nn
import torch.nn.functional as F

# 3D编码解码器
class SGLVEncoderDecoder(nn.Module):
    def __init__(self):
        super(SGLVEncoderDecoder, self).__init__()
        self.encoder = nn.Sequential(
            # C64-K4-S2-P1
            nn.Conv3d(5, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # C64-K3-S1-P1
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # C128-K4-S2-P1
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # C128-K3-S1-P1
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # C256-K4-S2-P1
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # C256-K3-S1-P1
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # C256-K3-S1-P1
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # C128-K3-S1-P1
            nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # C128-K3-S1-P1
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # C64-K3-S1-P1
            nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.color_decoder = nn.Sequential(
            # C32-K3-S1-P1
            nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # C3-K3-S1-P1
            nn.Conv3d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.alpha_decoder = nn.Sequential(
            # C32-K3-S1-P1
            nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # C1-K3-S1-P1
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.w_decoder = nn.Sequential(
            # C32-K3-S1-P1
            nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # C3-K3-S1-P1
            nn.Conv3d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.lamda_decoder = nn.Sequential(
            # C32-K3-S1-P1
            nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # C1-K3-S1-P1
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.s_decoder = nn.Sequential(
            # C32-K3-S1-P1
            nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # C3-K3-S1-P1
            nn.Conv3d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, volume):
        features = self.encoder(volume)
        color = self.color_decoder(features)
        alpha = self.alpha_decoder(features)
        w = self.w_decoder(features)
        lamda = self.lamda_decoder(features)
        s = self.s_decoder(features)
        print(color.shape)
        return torch.cat([color, alpha, w, lamda, s], dim=1)