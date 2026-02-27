import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

class SGLVRenderer():
    def __init__(self, resolution=(160, 320), sample_num=100):
        self.resolution = resolution
        self.sample_num = sample_num

    def render(self, origin, SGLV, voxel_range):
        # 生成像素坐标网格
        # SGLV may be (11, D, H, W) or (B, 11, D, H, W)
        if SGLV.dim() == 4:
            SGLV = SGLV.unsqueeze(0)
        device = SGLV.device
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
        # 方向向量并归一化
        directions = torch.stack([x, y, z], dim=-1)
        # 调整输入形状以处理批量体素范围
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

        # 采样体积并分配到环境贴图
        envmap = self.sample_volume(origin.to(device).contiguous(), directions, voxel_range.to(device).contiguous(), SGLV)
        return envmap

    def sample_volume(self, ray_origin, ray_directions, voxel_range, SGLV):
        # 计算光线与体积的相交点
        t0, t1 = self.ray_box_intersection(ray_origin, ray_directions, voxel_range)

        # SGLV shape: (B, C, D, H, W)
        B = SGLV.shape[0]
        H = self.resolution[0]
        W = self.resolution[1]

        # 生成采样点
        envmap = torch.zeros(B, 3, H, W, device=SGLV.device)
        t_values_base = torch.linspace(0, 1, steps=self.sample_num, device=SGLV.device).reshape(1, 1, -1)

        # t0, t1 shapes: (B, H, W)
        t0_expanded = t0.unsqueeze(-1)
        t1_expanded = t1.unsqueeze(-1)
        t_values = t0_expanded + t_values_base * (t1_expanded - t0_expanded)  # (B,H,W,S)

        # 扩展光线方向和起始点 -> produce (B, H, W, S, 3)
        ray_directions_expanded = ray_directions.unsqueeze(2).unsqueeze(0).expand(B, -1, -1, self.sample_num, -1)
        ray_origin_expanded = ray_origin.view(B, 1, 1, 3).unsqueeze(3).expand(-1, H, W, self.sample_num, -1)

        # 计算采样点坐标
        points = ray_origin_expanded + t_values.unsqueeze(-1) * ray_directions_expanded  # (B,H,W,S,3)
        # 归一化点坐标
        vr0 = voxel_range[:, 0].view(B, 1, 1, 3)
        vr1 = voxel_range[:, 1].view(B, 1, 1, 3)
        points = (points - vr0) / (vr1 - vr0) * 2 - 1
        points = points[..., [2, 1, 0]]  # reorder to z,y,x

        # grid_sample expects grid shape (N, D_out, H_out, W_out, 3) -> we use (B, H, W, S, 3)
        grid = points

        # 三线性插值 for each channel group
        c = F.grid_sample(SGLV[:, :3, ...], grid, mode='bilinear', padding_mode='zeros', align_corners=True)  # (B,3,H,W,S)
        alpha = F.grid_sample(SGLV[:, 3:4, ...], grid, mode='bilinear', padding_mode='zeros', align_corners=True)  # (B,1,H,W,S)
        w = F.grid_sample(SGLV[:, 4:7, ...], grid, mode='bilinear', padding_mode='zeros', align_corners=True)  # (B,3,H,W,S)
        lamb = F.grid_sample(SGLV[:, 7:8, ...], grid, mode='bilinear', padding_mode='zeros', align_corners=True)  # (B,1,H,W,S)
        s = F.grid_sample(SGLV[:, 8:, ...], grid, mode='bilinear', padding_mode='zeros', align_corners=True)  # (B,3,H,W,S)

        # 累积不透明度 (along sample dimension S at dim=-1)
        transmittance = torch.cumprod(1 - alpha, dim=-1)
        weights = alpha * transmittance

        # 累积颜色
        accumulated_color = torch.sum(weights * c, dim=-1)  # (B,3,H,W)
        # 累积球面高斯参数
        accumulated_w = torch.sum(weights * w, dim=-1)  # (B,3,H,W)
        accumulated_lamb = torch.sum(weights * lamb, dim=-1)  # (B,1,H,W)
        accumulated_s = torch.sum(weights * s, dim=-1)  # (B,3,H,W)

        # 计算环境贴图
        # accumulated_s: (B,3,H,W) -> (B,H,W,3)
        accumulated_s_perm = accumulated_s.permute(0, 2, 3, 1)
        # ray_directions: (H, W, 3) -> (1,H,W,3)
        rd = ray_directions.unsqueeze(0)
        accumulated_s_dot_dir = (accumulated_s_perm * rd).sum(-1)  # (B,H,W)

        envmap = accumulated_color + accumulated_w * torch.exp(accumulated_lamb * (accumulated_s_dot_dir.unsqueeze(1) - 1))

        return envmap

    def ray_box_intersection(self, ray_origin, ray_directions, voxel_range):
        # 计算光线与体积边界盒的相交点
        # ray_origin: (B,3) or (3,)
        if ray_origin.dim() == 1:
            ray_origin = ray_origin.view(1, 3)
        B = ray_origin.shape[0]

        device = ray_directions.device
        inv_dir = 1.0 / ray_directions  # (H,W,3)

        # Broadcast to (B, H, W, 3)
        ro = ray_origin.view(B, 1, 1, 3)
        vr0 = voxel_range[:, 0].view(B, 1, 1, 3)
        vr1 = voxel_range[:, 1].view(B, 1, 1, 3)
        inv = inv_dir.unsqueeze(0)

        t_min = (vr0 - ro) * inv
        t_max = (vr1 - ro) * inv

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