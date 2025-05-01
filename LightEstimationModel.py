import cv2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_free_gpu, create_projection_matrix, visualize_voxel_data
from Datasets import OpenRoomsDataset
from SGLVEncoderDecoder import SGLVEncoderDecoder

# 混合网络
class BlendingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, hdr_env, ldr_env, mask, depth_panorama):
        x = torch.cat([hdr_env, ldr_env, mask, depth_panorama], dim=1)
        features = self.encoder(x)
        blend_weight = self.decoder(features)
        return blend_weight

# 主模型
class LightingEstimationModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.voxel_range = [torch.tensor([-5, -5, -5]), torch.tensor([5, 5, 5])]
        self.voxel_resolution = (84, 60, 64)

        # c, a, e
        self.register_buffer('volume', torch.randn(5, *self.voxel_resolution))
        self.register_buffer('sglv_volume', torch.randn(11, *self.voxel_resolution))

        self.sglv_encoder_decoder = SGLVEncoderDecoder()
        self.blending_network = BlendingNetwork()
        # self.sglv = SGLV()
        # self.renderer = MonteCarloRenderer()
        # self.sgru = SGGRU(input_channels=128, hidden_channels=128)

    def forward(self, camera_matrix, input_image, depth_map):
        # 单图像输入
        self.initialize_volume(camera_matrix, input_image, depth_map)
        self.sglv_volume = self.sglv_encoder_decoder(self.volume)
        # sglv_prediction = self.encoder_decoder(*init_volume)
        # blended_env = self.blending_network(sglv_prediction, input_image, depth_map)
        # rendered_sphere = self.renderer(blended_env)
        # return blended_env, rendered_sphere

    def initialize_volume(self, camera_matrix, input_image, depth_map):
        # _, height, width = input_image.shape
        D_max = torch.max(depth_map)
        self.voxel_range = [
            torch.tensor([-1.1 * D_max, -0.8 * D_max, -1.2 * D_max]),
            torch.tensor([1.1 * D_max, 0.8 * D_max, 0.5 * D_max])
        ]
        voxel_size = (self.voxel_range[1] - self.voxel_range[0]) / torch.tensor(self.voxel_resolution)
        # 创建体积
        self.volume = torch.zeros(5, *self.voxel_resolution, device=self.volume.device)

        # 将图像投影到3D空间
        x = torch.arange(self.voxel_resolution[0], device=self.volume.device)
        y = torch.arange(self.voxel_resolution[1], device=self.volume.device)
        z = torch.arange(self.voxel_resolution[2], device=self.volume.device)
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
        # 向量化计算体素空间坐标
        voxel_pos_x = self.voxel_range[0][0] + grid_x * (self.voxel_range[1][0] - self.voxel_range[0][0]) / self.voxel_resolution[0]
        voxel_pos_y = self.voxel_range[0][1] + grid_y * (self.voxel_range[1][1] - self.voxel_range[0][1]) / self.voxel_resolution[1]
        voxel_pos_z = self.voxel_range[0][2] + grid_z * (self.voxel_range[1][2] - self.voxel_range[0][2]) / self.voxel_resolution[2]
        # 构建齐次坐标 (4D)
        points = torch.stack([
            voxel_pos_x.flatten(),
            voxel_pos_y.flatten(),
            voxel_pos_z.flatten(),
            torch.ones_like(voxel_pos_z.flatten())
        ], dim=1)
        # 批量投影
        proj_pos = torch.mm(points, camera_matrix.T.to(self.volume.device))  # [N,4]
        proj_pos = proj_pos / proj_pos[:, -1:]  # 透视除法
        # 转换为grid_sample需要的坐标格式 (注意Y轴翻转)
        grid_sample = torch.stack([
            proj_pos[:, 0],  # X坐标
            -proj_pos[:, 1]   # Y坐标取反
        ], dim=1).view(1, 1, -1, 2)  # 形状调整为[1,1,N,2]
        # 批量采样颜色和深度
        input_color = F.grid_sample(input_image.unsqueeze(0).to(self.volume.device),
                                grid_sample,
                                mode='bilinear',
                                padding_mode='zeros',
                                align_corners=True).squeeze().view(3, -1)
        input_depth = F.grid_sample(depth_map.unsqueeze(0).to(self.volume.device),
                                grid_sample,
                                mode='bilinear',
                                padding_mode='zeros',
                                align_corners=True).squeeze().view(1, -1)
        # 重新组织数据为三维体素结构
        input_color_grid = input_color.view(3, *self.voxel_resolution)
        input_depth_grid = input_depth.view(*self.voxel_resolution)
        current_depth = -voxel_pos_z
        depth_diff = input_depth_grid - current_depth
        # 计算alpha值
        alpha = torch.where(
            depth_diff > 0,
            4 * (-depth_diff / voxel_size[2] + 1),
            4 * (depth_diff / voxel_size[2] + 5)
        )
        alpha = torch.clamp(alpha, 0, 1)
        # 处理投影边界外的体素
        proj_mask = (proj_pos[:, 0].view_as(alpha) < -1) | \
                    (proj_pos[:, 0].view_as(alpha) > 1) | \
                    (proj_pos[:, 1].view_as(alpha) < -1) | \
                    (proj_pos[:, 1].view_as(alpha) > 1)
        alpha[proj_mask] = 0
        # 计算e通道
        e_channel = torch.where(
            (current_depth > 0) & (depth_diff > 3 * voxel_size[2]),
            -1.0,
            0.0
        )
        # 更新volume张量
        self.volume[:3, ...] = alpha.unsqueeze(0) * input_color_grid
        self.volume[3, ...] = alpha
        self.volume[4, ...] = e_channel

        # for x in range(self.voxel_resolution[0]):
        #     for y in range(self.voxel_resolution[1]):
        #         for z in range(self.voxel_resolution[2]):
        #             # 将体素投影到图像空间
        #             voxel_pos_x = self.voxel_range[0][0] + x * (self.voxel_range[1][0] - self.voxel_range[0][0]) / self.voxel_resolution[0]
        #             voxel_pos_y = self.voxel_range[0][1] + y * (self.voxel_range[1][1] - self.voxel_range[0][1]) / self.voxel_resolution[1]
        #             voxel_pos_z = self.voxel_range[0][2] + z * (self.voxel_range[1][2] - self.voxel_range[0][2]) / self.voxel_resolution[2]
        #             point_homogeneous = torch.tensor([voxel_pos_x, voxel_pos_y, voxel_pos_z, 1.0], dtype=torch.float32)
        #             proj_pos = camera_matrix @ point_homogeneous
        #             proj_pos = proj_pos / proj_pos[3]
        #             proj_x = proj_pos[0]
        #             proj_y = -proj_pos[1]
        #             # 采样颜色和深度
        #             grid = torch.tensor([proj_x, proj_y])
        #             grid = grid.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        #             input_color = F.grid_sample(input_image.unsqueeze(0), grid, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
        #             input_depth = F.grid_sample(depth_map.unsqueeze(0), grid, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()

        #             current_depth = -voxel_pos_z
        #             depth_diff = input_depth - current_depth
        #             # 根据深度差值计算α值
        #             if depth_diff > 0:
        #                 self.volume[3, x, y, z] = 4 * (-depth_diff / voxel_size[2] + 1)
        #             else:
        #                 self.volume[3, x, y, z] = 4 * (depth_diff / voxel_size[2] + 5)
        #             self.volume[3, x, y, z] = torch.clamp(self.volume[3, x, y, z], 0, 1)
        #             # 在相机外
        #             if proj_x < -1 or proj_x > 1 or proj_y < -1 or proj_y > 1:
        #                 self.volume[3, x, y, z] = 0
        #             # 设置e通道
        #             if current_depth > 0 and depth_diff > 3 * voxel_size[2]:
        #                 self.volume[4, x, y, z] = -1
        #             else:
        #                 self.volume[4, x, y, z] = 0
        #             # 计算颜色值
        #             self.volume[:3, x, y, z] = self.volume[3, x, y, z] * input_color

if __name__ == "__main__":
    selected_gpu = get_free_gpu()
    device = torch.device("cpu" if selected_gpu is None else f"cuda:{selected_gpu}")
    print(f"Using device: {device}")
    # 训练
    model = LightingEstimationModel().to(device)
    projection_matrix = create_projection_matrix(57.9516, 640/480, 0.1, 100)
    dataset = OpenRoomsDataset(root_dir='/mnt/data/youkeyao/Datasets/OpenRooms/releasingData')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in dataloader:
        image = batch['image'][0]
        depth = batch['depth'][0]
        lighting = batch['lighting'][0]
        image_np = image.permute(1, 2, 0).numpy()[:, :, ::-1]
        cv2.imshow("Image and Depth Map", image_np)
        cv2.waitKey(0)
        print("Start")

        start_time = time.time()
        model(projection_matrix, image, depth)
        end_time = time.time()
        print(f"代码执行时间：{end_time - start_time} 秒")
        visualize_voxel_data(model.volume, model.voxel_range)
        break