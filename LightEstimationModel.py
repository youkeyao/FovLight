import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from utils import get_free_gpu, create_projection_matrix, visualize_voxel_data, linear_to_srgb
from EnvMapDatasets import EnvMapDatasets
from EnvMapVideoDatasets import EnvMapVideoDatasets
from SGLVEncoderDecoder import SGLVEncoderDecoder
from SGLVRenderer import SGLVRenderer
from DetailedRenderer import DetailedRenderer
from BlendingNetwork import BlendingNetwork

# 主模型
class LightingEstimationModel(nn.Module):
    def __init__(self, voxel_resolution=(84, 60, 64), output_resolution=(320, 640), sample_num=100, level=4):
        super().__init__()

        self.voxel_range = None
        self.voxel_resolution = voxel_resolution
        self.output_resolution = output_resolution

        self.sglv_encoder_decoder = SGLVEncoderDecoder(level)
        self.sglv_renderer = SGLVRenderer(output_resolution, sample_num)
        self.detailed_rednerer = DetailedRenderer(output_resolution)
        self.blending_network = BlendingNetwork(level)

        # RNN 状态缓存
        self.volume = torch.zeros(5, *self.voxel_resolution)
        self.sglv_volume = torch.zeros(11, *self.voxel_resolution)
        self.prev_envmap = None
        self.depth_accum = None
        self.weight_accum = None

    def process_volume(self, P, V, input_image, depth_map):
        device = next(self.sglv_encoder_decoder.parameters()).device
    
        # --- 1. 体素范围初始化优化 ---
        with torch.no_grad():
            if self.voxel_range is None:
                D_max = torch.max(depth_map)
                self.voxel_range = torch.stack([
                    torch.tensor([-1.1, -0.8, -1.2], device=device) * D_max,
                    torch.tensor([1.1, 0.8, 0.5], device=device) * D_max
                ])
                self.sglv_volume = torch.zeros(11, *self.voxel_resolution, device=device)
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
            proj_pos = torch.matmul(points, (P @ V).T)
            proj_pos = proj_pos / proj_pos[:, 2:3]  # 透视除法

            # --- 5. 坐标归一化优化 ---
            # 假设输入图像尺寸为(H,W)，需要转换为grid_sample的归一化坐标
            uv_normalized = torch.empty_like(proj_pos[:, :2])
            uv_normalized[:, 0] = proj_pos[:, 0]
            uv_normalized[:, 1] = -proj_pos[:, 1]  # Y翻转

            # --- 6. 采样优化 ---
            grid_sample = uv_normalized.view(1, 1, -1, 2)  # 直接使用归一化坐标
            input_color = F.grid_sample(
                input_image.unsqueeze(0),
                grid_sample,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            ).squeeze().view(3, *self.voxel_resolution)
            
            input_depth = F.grid_sample(
                depth_map.unsqueeze(0),
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
            # alpha_mask = alpha > 0
            # self.volume[:3] = torch.where(alpha_mask, alpha * input_color, self.volume[:3])
            # self.volume[3] = torch.where(alpha_mask, alpha, self.volume[3])
            # self.volume[4] = torch.where(alpha_mask, e_channel, self.volume[4])
            self.volume = torch.zeros(5, *self.voxel_resolution, device=device)
            self.volume[:3] = alpha * input_color
            self.volume[3] = alpha
            self.volume[4] = e_channel

        new_volume, u = self.sglv_encoder_decoder(self.volume.detach())
        self.sglv_volume = u * self.sglv_volume.detach() + (1 - u) * new_volume

    def forward(self, origin, P, V, input_image, depth_map, use_detailed=True, reset_model=False):
        device = next(self.sglv_encoder_decoder.parameters()).device
        origin = origin.to(device)
        P = P.to(device)
        V = V.to(device)
        input_image = input_image.to(device)
        depth_map = depth_map.to(device)
        if reset_model:
            self.reset()
        if self.prev_envmap is None:
            self.prev_envmap = torch.zeros(3, *self.output_resolution, device=device)
        self.process_volume(P, V, input_image, depth_map)
        envmap = self.sglv_renderer.render(origin, self.sglv_volume, self.voxel_range)
        last = self.prev_envmap.detach()
        if use_detailed:
            if self.depth_accum is None:
                self.depth_accum = torch.zeros(1, *self.output_resolution, device=device)
                self.weight_accum = torch.zeros(*self.output_resolution, device=device)
            detailed_envmap, depth_pano = self.detailed_rednerer.render(origin, P, V, input_image, depth_map)
            mask = (detailed_envmap > 0).any(dim=0, keepdim=True).float()
            envmap, self.depth_accum, self.weight_accum = self.blending_network(last, envmap, detailed_envmap, mask, depth_pano, self.depth_accum.detach(), self.weight_accum.detach())
        self.prev_envmap = envmap
        return envmap, last

    def reset(self):
        self.voxel_range = None
        self.prev_envmap = None
        self.depth_accum = None
        self.weight_accum = None
        self.sglv_encoder_decoder.reset()
        self.blending_network.reset()

if __name__ == "__main__":
    accelerator = Accelerator()
    device = accelerator.device
    print(f"Using device: {device}")
    # 推理
    model = LightingEstimationModel()
    model = accelerator.prepare(model)
    accelerator.load_state("checkpoints/voxel_5")
    projection_matrix = create_projection_matrix(120, 832/400, 0.1, 20)
    dataset = EnvMapVideoDatasets(root_dir='/mnt/data/youkeyao/Datasets/EnvMapVideo')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    model.eval()
    for batch in dataloader:
        rgb = batch['rgb'][0]
        depth = batch['depth'][0]
        pose = batch['pose'][0]
        lighting = batch["lighting"][0]
        position = batch["position"][0]

        n_frames = rgb.shape[0]
        reset_model = True
        for i in range(n_frames):
            rgb_np = rgb[i].permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
            depth_np = depth[i].repeat(3, 1, 1).permute(1, 2, 0).cpu().numpy()
            depth_np /= np.max(depth_np)
            lighting_np = lighting.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
            print(pose[i])

            result, _ = model(position, projection_matrix, torch.inverse(pose[i]), rgb[i], depth[i], False, reset_model)
            reset_model = False

            cv2.imshow("rgb", rgb_np)
            cv2.imshow("depth", depth_np)
            cv2.imshow("lighting", linear_to_srgb(lighting_np))
            cv2.imshow("result", linear_to_srgb(result.permute(1, 2, 0).detach().cpu().numpy()[:, :, ::-1]))
            key = cv2.waitKey(0) & 0xFF
            while key != ord('n') and key != ord('q'):
                key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
        if key == ord('q'):
            break
    cv2.destroyAllWindows()