"""SGLV 体渲染器。

负责将体素参数沿球面方向采样，生成等距柱状环境贴图。
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

class SGLVRenderer():
    """SGLV 体采样与光照合成实现。"""

    def __init__(self, resolution=(160, 320), sample_num=100):
        self.resolution = resolution
        self.sample_num = sample_num

    def render(self, origin, SGLV, voxel_range):
        """渲染环境图入口，处理批维并调用体采样。"""
        # 第一段：统一输入维度，兼容单样本和批量输入。
        # SGLV 可能是 (11, D, H, W) 或 (B, 11, D, H, W)。
        if SGLV.dim() == 4:
            SGLV = SGLV.unsqueeze(0)
        device = SGLV.device

        # 第二段：构建等距柱状图每个像素对应的球面方向。
        v_coords = torch.arange(self.resolution[0], device=device)
        u_coords = torch.arange(self.resolution[1], device=device)
        v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing='ij')
        # 计算球面坐标
        theta = 2 * torch.pi * u_grid / self.resolution[1]
        phi = torch.pi * v_grid / self.resolution[0]
        # 计算笛卡尔坐标方向向量
        x = torch.sin(phi) * torch.sin(theta)
        y = torch.cos(phi)
        z = -torch.sin(phi) * torch.cos(theta)
        # 方向向量（单位球面）
        directions = torch.stack([x, y, z], dim=-1)

        # 第三段：对齐体素范围和相机原点的批维。
        if voxel_range.dim() == 2:
            voxel_range = voxel_range.unsqueeze(0).expand(SGLV.shape[0], -1, -1)
        elif voxel_range.dim() == 3 and voxel_range.shape[0] != SGLV.shape[0]:
            raise ValueError("voxel_range batch dimension must match SGLV batch dimension")

        if origin.dim() == 1:
            origin = origin.unsqueeze(0)
        if origin.shape[0] == 1 and SGLV.shape[0] > 1:
            origin = origin.expand(SGLV.shape[0], -1)
        elif origin.shape[0] != SGLV.shape[0]:
            raise ValueError("origin batch dimension must match SGLV batch dimension")

        # 第四段：沿每条光线体采样并输出环境图。
        envmap = self.sample_volume(origin.to(device).contiguous(), directions, voxel_range.to(device).contiguous(), SGLV)
        return envmap

    def sample_volume(self, ray_origin, ray_directions, voxel_range, SGLV):
        """沿每条光线在体素中采样并累积颜色/球高斯项。"""
        # 第一段：计算每条光线与体素包围盒的相交区间。
        t0, t1 = self.ray_box_intersection(ray_origin, ray_directions, voxel_range)

        # 第二段：准备采样参数与输出缓存。
        # SGLV 形状为 (B, C, D, H, W)。
        B = SGLV.shape[0]
        H = self.resolution[0]
        W = self.resolution[1]

        # 生成采样点
        envmap = torch.zeros(B, 3, H, W, device=SGLV.device)
        t_values_base = torch.linspace(0, 1, steps=self.sample_num, device=SGLV.device).reshape(1, 1, -1)

        # t0、t1 形状为 (B, H, W)，在相交区间内做均匀采样。
        t0_expanded = t0.unsqueeze(-1)
        t1_expanded = t1.unsqueeze(-1)
        t_values = t0_expanded + t_values_base * (t1_expanded - t0_expanded)  # (B,H,W,S)

        # 第三段：展开光线起点和方向，形成 5D 采样点张量。
        ray_directions_expanded = ray_directions.unsqueeze(2).unsqueeze(0).expand(B, -1, -1, self.sample_num, -1)
        ray_origin_expanded = ray_origin.view(B, 1, 1, 3).unsqueeze(3).expand(-1, H, W, self.sample_num, -1)

        # 计算采样点坐标
        points = ray_origin_expanded + t_values.unsqueeze(-1) * ray_directions_expanded  # (B,H,W,S,3)
        # 归一化点坐标
        vr0 = voxel_range[:, 0].view(B, 1, 1, 3)
        vr1 = voxel_range[:, 1].view(B, 1, 1, 3)
        points = (points - vr0) / (vr1 - vr0) * 2 - 1
        points = points[..., [2, 1, 0]]  # 调整到 grid_sample 需要的 z,y,x 顺序

        # 第四段：使用 3D 双线性插值采样各参数场。
        # grid_sample 期望网格为 (N, D_out, H_out, W_out, 3)，这里对应 (B, H, W, S, 3)。
        grid = points

        # 按参数通道分组采样：颜色、alpha、球高斯权重/宽度/方向。
        c = F.grid_sample(SGLV[:, :3, ...], grid, mode='bilinear', padding_mode='zeros', align_corners=True)  # (B,3,H,W,S)
        alpha = F.grid_sample(SGLV[:, 3:4, ...], grid, mode='bilinear', padding_mode='zeros', align_corners=True)  # (B,1,H,W,S)
        w = F.grid_sample(SGLV[:, 4:7, ...], grid, mode='bilinear', padding_mode='zeros', align_corners=True)  # (B,3,H,W,S)
        lamb = F.grid_sample(SGLV[:, 7:8, ...], grid, mode='bilinear', padding_mode='zeros', align_corners=True)  # (B,1,H,W,S)
        s = F.grid_sample(SGLV[:, 8:, ...], grid, mode='bilinear', padding_mode='zeros', align_corners=True)  # (B,3,H,W,S)

        # 第五段：体渲染前向合成，计算透射率与体素权重。
        # 沿采样维 S 累积不透明度。
        transmittance = torch.cumprod(1 - alpha, dim=-1)
        weights = alpha * transmittance

        # 累积颜色
        accumulated_color = torch.sum(weights * c, dim=-1)  # (B,3,H,W)
        # 累积球面高斯参数
        accumulated_w = torch.sum(weights * w, dim=-1)  # (B,3,H,W)
        accumulated_lamb = torch.sum(weights * lamb, dim=-1)  # (B,1,H,W)
        accumulated_s = torch.sum(weights * s, dim=-1)  # (B,3,H,W)

        # 第六段：根据方向与球高斯参数恢复最终环境图。
        # accumulated_s: (B,3,H,W) -> (B,H,W,3)
        accumulated_s_perm = accumulated_s.permute(0, 2, 3, 1)
        # ray_directions: (H, W, 3) -> (1,H,W,3)
        rd = ray_directions.unsqueeze(0)
        accumulated_s_dot_dir = (accumulated_s_perm * rd).sum(-1)  # (B,H,W)

        envmap = accumulated_color + accumulated_w * torch.exp(accumulated_lamb * (accumulated_s_dot_dir.unsqueeze(1) - 1))

        return envmap

    def ray_box_intersection(self, ray_origin, ray_directions, voxel_range):
        """计算光线与轴对齐包围盒的入射/离开距离。"""
        # 第一段：统一 ray_origin 维度。
        # ray_origin: (B,3) 或 (3,)
        if ray_origin.dim() == 1:
            ray_origin = ray_origin.view(1, 3)
        B = ray_origin.shape[0]

        device = ray_directions.device
        inv_dir = 1.0 / ray_directions  # (H,W,3)

        # 第二段：广播到批量射线，便于向量化求交。
        ro = ray_origin.view(B, 1, 1, 3)
        vr0 = voxel_range[:, 0].view(B, 1, 1, 3)
        vr1 = voxel_range[:, 1].view(B, 1, 1, 3)
        inv = inv_dir.unsqueeze(0)

        t_min = (vr0 - ro) * inv
        t_max = (vr1 - ro) * inv

        # 第三段：仅保留正向交点并得到有效采样区间。
        inf = torch.tensor(float('inf'), device=device)
        t0 = torch.min(torch.where(t_min > 0, t_min, inf), dim=-1).values
        t1 = torch.min(torch.where(t_max > 0, t_max, inf), dim=-1).values
        t_start = torch.zeros_like(t0)
        t_end = torch.min(t0, t1)

        return t_start, t_end

if __name__ == "__main__":
    sglv_renderer = SGLVRenderer()
    origin = torch.tensor([0, 0, 0])
    SGLV = torch.randn(11, 84, 60, 64)
    voxel_range = torch.stack([torch.tensor([-5, -5, -5]), torch.tensor([5, 5, 5])])

    envmap = sglv_renderer.render(origin, SGLV, voxel_range)
    envmap_np = envmap.permute(1, 2, 0).numpy()[:, :, ::-1]
    cv2.imshow("envmap", envmap_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()