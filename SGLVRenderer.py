import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

class SGLVRenderer(nn.Module):
    def __init__(self):
        super().__init__()
        self.resolution = (16, 32)

    def forward(self, origin, SGLV, voxel_range):
        # 生成像素坐标网格
        v_coords = torch.arange(self.resolution[0], device=SGLV.device)
        u_coords = torch.arange(self.resolution[1], device=SGLV.device)
        v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing='ij')
        # 计算球面坐标
        phi = 2 * torch.pi * u_grid / self.resolution[1]
        theta = torch.pi * v_grid / self.resolution[0]
        # 计算笛卡尔坐标方向向量
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.cos(theta)
        z = torch.sin(theta) * torch.sin(phi)
        # 方向向量并归一化
        directions = torch.stack([x, y, z], dim=-1)
        directions = F.normalize(directions, p=2, dim=-1)
        # 采样体积并分配到环境贴图
        envmap = self.sample_volume(origin.to(SGLV.device), directions, voxel_range, SGLV)
        return envmap

    def sample_volume(self, ray_origin, ray_directions, voxel_range, SGLV):
        # 计算光线与体积的相交点
        t0, t1 = self.ray_box_intersection(ray_origin, ray_directions, voxel_range)

        # 生成采样点
        envmap = torch.zeros(3, *self.resolution, device=SGLV.device)
        t_values = torch.linspace(0, 1, steps=100, device=SGLV.device).reshape(1, 1, -1)
        t0_expanded = t0.unsqueeze(-1)
        t1_expanded = t1.unsqueeze(-1)
        t_values = t0_expanded + t_values * (t1_expanded - t0_expanded)
        # 扩展光线方向和起始点
        ray_directions_expanded = ray_directions.unsqueeze(2)
        ray_origin_expanded = ray_origin.view(1, 1, 1, 3).expand(*ray_directions.shape[:2], 100, 3)
        # 计算采样点坐标
        points = ray_origin_expanded + t_values.unsqueeze(-1) * ray_directions_expanded
        # 归一化点坐标
        points = (points - voxel_range[0]) / (voxel_range[1] - voxel_range[0]) * 2 - 1
        points = points.unsqueeze(0)
        # 三线性插值
        c = F.grid_sample(SGLV[:3, ...].unsqueeze(0), points, mode='bilinear', align_corners=True).squeeze()
        alpha = F.grid_sample(SGLV[3:4, ...].unsqueeze(0), points, mode='bilinear', align_corners=True).squeeze()
        w = F.grid_sample(SGLV[4:7, ...].unsqueeze(0), points, mode='bilinear', align_corners=True).squeeze()
        lamb = F.grid_sample(SGLV[7:8, ...].unsqueeze(0), points, mode='bilinear', align_corners=True).squeeze()
        s = F.grid_sample(SGLV[8:, ...].unsqueeze(0), points, mode='bilinear', align_corners=True).squeeze()
        # 累积不透明度
        transmittance = torch.cumprod(1 - alpha + 1e-10, dim=2)
        weights = alpha * transmittance
        # 累积颜色
        accumulated_color = torch.sum(weights.unsqueeze(0) * c, dim=3)
        # 累积球面高斯参数
        accumulated_w = torch.sum(weights.unsqueeze(0) * w, dim=3)
        accumulated_lamb = torch.sum(weights * lamb, dim=2)
        accumulated_s = torch.sum(weights.unsqueeze(0) * s, dim=3)
        # 计算环境贴图
        accumulated_s_dot_dir = torch.einsum('ijk, ijk->ij', ray_directions, accumulated_s.permute(1, 2, 0))
        envmap = accumulated_color + accumulated_w * torch.exp(accumulated_lamb * (accumulated_s_dot_dir - 1))

        # for i in range(t0.shape[0]):
        #     for j in range(t0.shape[1]):
        #         t_values = torch.linspace(t0[i][j], t1[i][j], steps=100).unsqueeze(1)  # 100 个采样点
        #         points = ray_origin.unsqueeze(0) + t_values * ray_directions[i][j].unsqueeze(0)

        #         # 将点坐标归一化
        #         points = (points - voxel_range[0]) / (voxel_range[1] - voxel_range[0]) * 2 - 1
        #         points = points.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        #         # 三线性插值
        #         c = F.grid_sample(SGLV[:3, ...].unsqueeze(0), points, mode='bilinear', align_corners=True).squeeze()
        #         alpha = F.grid_sample(SGLV[3:4, ...].unsqueeze(0), points, mode='bilinear', align_corners=True).squeeze()
        #         w = F.grid_sample(SGLV[4:7, ...].unsqueeze(0), points, mode='bilinear', align_corners=True).squeeze()
        #         lamb = F.grid_sample(SGLV[7:8, ...].unsqueeze(0), points, mode='bilinear', align_corners=True).squeeze()
        #         s = F.grid_sample(SGLV[8:, ...].unsqueeze(0), points, mode='bilinear', align_corners=True).squeeze()

        #         # 累积不透明度
        #         transmittance = torch.cumprod(1 - alpha + 1e-10, dim=0)
        #         weights = alpha * transmittance

        #         # 累积颜色
        #         accumulated_color = torch.sum(weights.unsqueeze(0) * c, dim=1)

        #         # 累积球面高斯参数
        #         accumulated_w = torch.sum(weights.unsqueeze(0) * w, dim=1)
        #         accumulated_lamb = torch.sum(weights * lamb)
        #         accumulated_s = torch.sum(weights.unsqueeze(0) * s, dim=1)

        #         envmap[:, i, j] = accumulated_color + accumulated_w * torch.exp(accumulated_lamb * (torch.dot(accumulated_s, ray_directions[i][j]) - 1))

        return envmap

    def ray_box_intersection(self, ray_origin, ray_directions, voxel_range):
        # 计算光线与体积边界盒的相交点
        inv_dir = 1.0 / ray_directions
        t_min = (voxel_range[0] - ray_origin) * inv_dir
        t_max = (voxel_range[1] - ray_origin) * inv_dir

        t0 = torch.min(torch.where(t_min > 0, t_min, torch.tensor(float('inf'))), dim=-1).values
        t1 = torch.min(torch.where(t_max > 0, t_max, torch.tensor(float('inf'))), dim=-1).values
        t_start = torch.zeros_like(t0)
        t_end = torch.min(t0, t1)

        return t_start, t_end

if __name__ == "__main__":
    sglv_renderer = SGLVRenderer()
    origin = torch.tensor([0, 0, 0])
    SGLV = torch.randn(11, 84, 60, 64)
    voxel_range = [torch.tensor([-5, -5, -5]), torch.tensor([5, 5, 5])]

    envmap = sglv_renderer(origin, SGLV, voxel_range)
    envmap_np = envmap.permute(1, 2, 0).numpy()[:, :, ::-1]
    cv2.imshow("envmap", envmap_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()