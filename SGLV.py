import torch
import torch.nn as nn
import torch.nn.functional as F

class SGLV(nn.Module):
    def __init__(self, size=(100, 100, 100), voxel_resolution=(50, 50, 50)):
        super(SGLV, self).__init__()
        self.size = torch.tensor(size)
        self.voxel_range = [-self.size / 2, self.size / 2]
        self.voxel_resolution = voxel_resolution
        self.channels = 11  # (c: 3, α: 1, w: 3, λ: 1, s: 3)

        # init
        self.volume = nn.Parameter(
            torch.randn(self.channels, *voxel_resolution),
            requires_grad=True
        )

    def forward(self, ray_origins):
        # 根据光线方向和起点采样体积
        sampled_params = self.sample_volume(ray_origins)
        return self.render(sampled_params)

    def sample_volume(self, ray_origins, ray_directions):
        # 计算光线与体积的相交点
        t0, t1 = self.ray_box_intersection(ray_origins, ray_directions)
        if t0 is None:
            return None  # 光线不与体积相交

        # 生成采样点
        t_values = torch.linspace(t0, t1, steps=100).unsqueeze(1)  # 100 个采样点
        points = ray_origins.unsqueeze(0) + t_values * ray_directions.unsqueeze(0)

        # 将点坐标归一化
        points = (points - self.voxel_range[0]) / (self.voxel_range[1] - self.voxel_range[0]) * 2 - 1
        points = points.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # 三线性插值
        c = F.grid_sample(self.volume[:3, ...].unsqueeze(0), points, mode='bilinear', align_corners=True).squeeze()
        alpha = F.grid_sample(self.volume[3:4, ...].unsqueeze(0), points, mode='bilinear', align_corners=True).squeeze()
        w = F.grid_sample(self.volume[4:7, ...].unsqueeze(0), points, mode='bilinear', align_corners=True).squeeze()
        lamb = F.grid_sample(self.volume[7:8, ...].unsqueeze(0), points, mode='bilinear', align_corners=True).squeeze()
        s = F.grid_sample(self.volume[8:, ...].unsqueeze(0), points, mode='bilinear', align_corners=True).squeeze()

        # 累积不透明度
        transmittance = torch.cumprod(1 - alpha + 1e-10, dim=0)
        weights = alpha * transmittance

        # 累积颜色
        accumulated_color = torch.sum(weights.unsqueeze(0) * c, dim=1)

        # 累积球面高斯参数
        accumulated_w = torch.sum(weights.unsqueeze(0) * w, dim=1)
        accumulated_lamb = torch.sum(weights * lamb)
        accumulated_s = torch.sum(weights.unsqueeze(0) * s, dim=1)

        return accumulated_color + accumulated_w * torch.exp(accumulated_lamb * (torch.dot(accumulated_s, ray_directions) - 1))

    def ray_box_intersection(self, ray_origins, ray_directions):
        # 计算光线与体积边界盒的相交点
        inv_dir = 1.0 / ray_directions
        t_min = (self.voxel_range[0] - ray_origins) * inv_dir
        t_max = (self.voxel_range[1] - ray_origins) * inv_dir

        t0 = torch.max(torch.max(t_min), torch.tensor(0))
        t1 = torch.min(t_max)

        if t0 > t1:
            return None, None  # 光线不与体积相交
        return t0, t1

if __name__ == "__main__":
    sglv = SGLV()
    ray_directions = torch.tensor([0.1, 0.2, 0.3])
    ray_origins = torch.tensor([0.0, 0.0, 0.0])

    L = sglv.sample_volume(ray_origins, ray_directions)
    print(L)