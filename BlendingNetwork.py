import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU2D(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.reset_gate = nn.Conv2d(channels * 2, channels, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(channels * 2, channels, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(channels * 2, channels, kernel_size, padding=padding)

    def forward(self, x, h_prev):
        if h_prev is None:
            h_prev = torch.zeros_like(x, device=x.device)
        combined = torch.cat([x, h_prev], dim=0)
        r = torch.sigmoid(self.reset_gate(combined))
        z = torch.sigmoid(self.update_gate(combined))
        combined_r = torch.cat([x, r * h_prev], dim=0)
        h_tilde = torch.tanh(self.out_gate(combined_r))
        h_next = (1 - z) * h_prev + z * h_tilde
        return h_next

class BlendingNetwork(nn.Module):
    def __init__(self, level=4):
        super().__init__()
        self.level = level
        self.conv = nn.Sequential(
            nn.Conv2d(13, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.gru0 = GRU2D(32)

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

        if self.level == 2:
            self.gru1 = GRU2D(64)
        elif self.level == 3:
            self.gru1 = GRU2D(128)
        elif self.level == 4:
            self.gru1 = GRU2D(256)

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

        self.weight_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Softplus(),
        )

        self.h0 = None
        self.h1 = None

    def forward(self, prev_hdr, hdr_env, ldr_env, mask, depth_pano, depth_accum, weight_accum):
        if self.h0 is not None:
            self.h0 = self.h0.detach()
        if self.h1 is not None:
            self.h1 = self.h1.detach()

        input = torch.cat([prev_hdr, hdr_env, ldr_env, mask, 1 - mask, depth_pano, depth_accum], dim=0)
        x = self.conv(input)
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
        x = features
        # 各参数预测
        w = self.weight_head(x)
        indicator = ((depth_accum - depth_pano) < 0.25).float()
        Lm = torch.clamp(w - indicator, min=0.0)
        one = torch.ones_like(Lm)
        invisible_comb = (1.0 - weight_accum) * hdr_env + weight_accum * prev_hdr

        Li = Lm * ldr_env + (one - Lm) * invisible_comb
        L_hat_D = Lm * depth_pano + (one - Lm) * depth_accum
        L_hat_M = torch.clamp(weight_accum + Lm, 0.0, 1.0)
        return Li, L_hat_D, L_hat_M
    
    def reset(self):
        self.h0 = None
        self.h1 = None