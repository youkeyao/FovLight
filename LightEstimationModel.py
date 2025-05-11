import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_free_gpu, create_projection_matrix, visualize_voxel_data
from KePanoLighting import KePanoLighting
from SGLVEncoderDecoder import SGLVEncoderDecoder
from SGLVRenderer import SGLVRenderer

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
        self.sglv_renderer = SGLVRenderer()
        # self.blending_network = BlendingNetwork()
        # self.sglv = SGLV()
        # self.renderer = MonteCarloRenderer()
        # self.sgru = SGGRU(input_channels=128, hidden_channels=128)

    def forward(self, origin, camera_matrix, input_image, depth_map):
        # 单图像输入
        self.initialize_volume(camera_matrix, input_image, depth_map)
        self.sglv_volume = self.sglv_encoder_decoder(self.volume)
        envmap = self.sglv_renderer(origin, self.sglv_volume, self.voxel_range)
        # blended_env = self.blending_network(sglv_prediction, input_image, depth_map)
        # rendered_sphere = self.renderer(blended_env)
        # return blended_env, rendered_sphere
        return envmap

    def initialize_volume(self, camera_matrix, input_image, depth_map):
        device = self.volume.device  # 统一设备管理
    
        # --- 1. 体素范围初始化优化 ---
        D_max = torch.max(depth_map)
        self.voxel_range = torch.stack([
            torch.tensor([-1.1, -0.8, -1.2], device=device) * D_max,
            torch.tensor([1.1, 0.8, 0.5], device=device) * D_max
        ])
        voxel_size = (self.voxel_range[1] - self.voxel_range[0]) / torch.tensor(self.voxel_resolution, device=device)
        
        # --- 2. 体素坐标生成优化 ---
        # 使用linspace直接生成坐标网格
        coords = [torch.linspace(self.voxel_range[0, i], 
              self.voxel_range[1, i], 
              self.voxel_resolution[i], 
              device=device) 
              for i in range(3)]
        voxel_pos_x, voxel_pos_y, voxel_pos_z = torch.meshgrid(*coords, indexing='ij')  # 使用ij索引
        
        # --- 3. 齐次坐标构建优化 ---
        points = torch.stack([
            voxel_pos_x.flatten(),
            voxel_pos_y.flatten(),
            voxel_pos_z.flatten(),
            torch.ones_like(voxel_pos_z.flatten())
        ], dim=1)
        
        # --- 4. 投影计算优化 ---
        camera_matrix = camera_matrix.to(device)  # 确保矩阵在正确设备
        proj_pos = torch.matmul(points, camera_matrix.T)
        proj_pos = proj_pos / proj_pos[:, 2:3]  # 透视除法
        
        # --- 5. 坐标归一化优化 ---
        # 假设输入图像尺寸为(H,W)，需要转换为grid_sample的归一化坐标
        uv_normalized = torch.empty_like(proj_pos[:, :2])
        uv_normalized[:, 0] = proj_pos[:, 0]
        uv_normalized[:, 1] = -proj_pos[:, 1]  # Y翻转
        
        # --- 6. 采样优化 ---
        grid_sample = uv_normalized.view(1, 1, -1, 2)  # 直接使用归一化坐标
        input_color = F.grid_sample(
            input_image.unsqueeze(0).to(device),
            grid_sample,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        ).squeeze().view(3, *self.voxel_resolution)
        
        input_depth = F.grid_sample(
            depth_map.unsqueeze(0).to(device),
            grid_sample,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        ).squeeze().view(1, *self.voxel_resolution)

        current_depth = -voxel_pos_z  # 使用预先生成的网格坐标
        depth_diff = input_depth - current_depth
        
        # --- 8. Alpha计算向量化 ---
        alpha = torch.where(
            depth_diff > 0,
            4 * (-depth_diff / voxel_size[2] + 1),
            4 * (depth_diff / voxel_size[2] + 5)
        )
        alpha = torch.clamp(alpha, 0, 1)
        
        # --- 9. 投影掩码优化 ---
        proj_mask = (uv_normalized[:, 0].view_as(alpha) < -1) | \
                    (uv_normalized[:, 0].view_as(alpha) > 1) | \
                    (uv_normalized[:, 1].view_as(alpha) < -1) | \
                    (uv_normalized[:, 1].view_as(alpha) > 1)
        alpha[proj_mask] = 0
        
        # --- 10. e通道计算优化 ---
        e_channel = torch.where(
            (current_depth > 0) & (depth_diff > 3 * voxel_size[2]),
            -1.0,
            0.0
        )

        # --- 11. Volume更新优化 ---
        self.volume = torch.zeros(5, *self.voxel_resolution, device=device)
        self.volume[:3] = alpha * input_color
        self.volume[3] = alpha
        self.volume[4] = e_channel

if __name__ == "__main__":
    selected_gpu = get_free_gpu()
    device = torch.device("cpu" if selected_gpu is None else f"cuda:{selected_gpu}")
    print(f"Using device: {device}")
    # 训练
    model = LightingEstimationModel().to(device)
    model.load_state_dict({k.replace("module.", ""): v for k, v in torch.load("checkpoints/model_checkpoint_4.pth", map_location=device)['model_state_dict'].items()})
    origin = torch.tensor([1, 0.5, -1])
    projection_matrix = create_projection_matrix(57.9516, 640/480, 0.1, 100)
    dataset = KePanoLighting(root='/mnt/data/youkeyao/Datasets/FutureHouse/KePanoLight')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in dataloader:
        image = batch['image'][0]
        albedo = batch['albedo'][0]
        depth = batch['depth'][0]
        image_np = image.permute(1, 2, 0).numpy()[:, :, ::-1]
        albedo_np = albedo.permute(1, 2, 0).numpy()[:, :, ::-1]
        depth_np = depth.repeat(3, 1, 1).permute(1, 2, 0).numpy()[:, :, ::-1]
        depth_np /= np.max(depth_np)
        combined_image = np.hstack((image_np, depth_np))
        cv2.imshow("Albedo", albedo_np)
        cv2.imshow("Image and Depth Map", combined_image)
        cv2.waitKey(0)
        print("Start")

        start_time = time.time()
        envmap = model(origin, projection_matrix, image, depth)
        end_time = time.time()
        print(f"代码执行时间：{end_time - start_time} 秒")
        # visualize_voxel_data(model.volume, model.voxel_range)
        print("End")

        envmap_np = envmap.permute(1, 2, 0).detach().cpu().numpy()[:, :, ::-1]
        cv2.imshow("envmap", envmap_np)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break