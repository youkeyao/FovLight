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

        # RNN 状态缓存 (will be created per-batch during first forward)
        self.volume = None
        self.sglv_volume = None
        self.prev_envmap = None
        self.depth_accum = None
        self.weight_accum = None

    def process_volume(self, P, V, input_image, depth_map):
        device = next(self.sglv_encoder_decoder.parameters()).device
        B = input_image.shape[0]

        with torch.no_grad():
            # Initialize voxel range and stored volumes per-batch
            if self.voxel_range is None or self.voxel_range.shape[0] != B:
                template_range = torch.tensor(
                    [[-1.1, -0.8, -1.2], [1.1, 0.8, 0.5]],
                    device=device,
                    dtype=depth_map.dtype,
                )
                depth_max = depth_map.reshape(B, -1).max(dim=1).values
                self.voxel_range = template_range.unsqueeze(0) * depth_max.view(B, 1, 1)
            if self.sglv_volume is None or self.sglv_volume.shape[0] != B:
                self.sglv_volume = torch.zeros(B, 11, *self.voxel_resolution, device=device)

            resolution_tensor = torch.tensor(self.voxel_resolution, device=device, dtype=depth_map.dtype)

            # Prepare per-batch volumes
            volumes = []

            for b in range(B):
                vr = self.voxel_range[b]
                voxel_size = (vr[1] - vr[0]) / resolution_tensor

                coords = [
                    torch.linspace(vr[0, i], vr[1, i], self.voxel_resolution[i], device=device, dtype=depth_map.dtype)
                    for i in range(3)
                ]
                voxel_pos_x, voxel_pos_y, voxel_pos_z = torch.meshgrid(*coords, indexing='ij')

                # Homogeneous coordinates for points: (N_points, 4)
                points = torch.stack([
                    voxel_pos_x.flatten(),
                    voxel_pos_y.flatten(),
                    voxel_pos_z.flatten(),
                    torch.ones_like(voxel_pos_z.flatten())
                ], dim=1)

                # Determine projection matrix for this batch element
                P_b = P[b] if P.dim() == 3 else P
                V_b = V[b] if V.dim() == 3 else V
                # Project points
                proj_pos = torch.matmul(points, (P_b @ V_b).T)
                proj_pos = proj_pos / proj_pos[:, 2:3]

                uv_normalized = torch.empty_like(proj_pos[:, :2], device=device)
                uv_normalized[:, 0] = proj_pos[:, 0]
                uv_normalized[:, 1] = -proj_pos[:, 1]

                # grid for sampling: (1, 1, N_points, 2)
                grid_sample = uv_normalized.view(1, 1, -1, 2)

                in_img = input_image[b].unsqueeze(0)
                in_depth = depth_map[b].unsqueeze(0)

                input_color = F.grid_sample(
                    in_img,
                    grid_sample,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=True
                ).squeeze().view(3, *self.voxel_resolution)

                input_depth = F.grid_sample(
                    in_depth,
                    grid_sample,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=True
                ).squeeze().view(1, *self.voxel_resolution)

                current_depth = -voxel_pos_z
                depth_diff = input_depth - current_depth

                alpha = torch.where(
                    depth_diff > 0,
                    4 * (-depth_diff / voxel_size[2] + 1),
                    4 * (depth_diff / voxel_size[2] + 5)
                )
                alpha = torch.clamp(alpha, 0, 1)

                proj_mask = (uv_normalized[:, 0].view_as(alpha) < -1) | \
                            (uv_normalized[:, 0].view_as(alpha) > 1) | \
                            (uv_normalized[:, 1].view_as(alpha) < -1) | \
                            (uv_normalized[:, 1].view_as(alpha) > 1)
                alpha[proj_mask] = 0

                e_channel = torch.where(
                    (current_depth > 0) & (depth_diff > 3 * voxel_size[2]),
                    -1.0,
                    0.0
                )

                vol = torch.zeros(5, *self.voxel_resolution, device=device)
                vol[:3] = alpha * input_color
                vol[3] = alpha
                vol[4] = e_channel
                volumes.append(vol)

            # Stack volumes into (B,5,D,H,W)
            volumes = torch.stack(volumes, dim=0)

        new_volume, u = self.sglv_encoder_decoder(volumes.detach())
        # Update running sglv_volume per batch
        self.sglv_volume = u * self.sglv_volume.detach() + (1 - u) * new_volume

    def forward(self, origin, P, V, input_image, depth_map, use_detailed=True, reset_model=True):
        device = next(self.sglv_encoder_decoder.parameters()).device
        origin = origin.to(device)
        P = P.to(device)
        V = V.to(device)
        input_image = input_image.to(device)
        depth_map = depth_map.to(device)
        if reset_model:
            self.reset()
        # Ensure batch dims for inputs
        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)
            depth_map = depth_map.unsqueeze(0)
        B = input_image.shape[0]
        if origin.dim() == 1:
            origin = origin.unsqueeze(0).expand(B, -1)

        if self.prev_envmap is None or self.prev_envmap.shape[0] != B:
            self.prev_envmap = torch.zeros(B, 3, *self.output_resolution, device=device)
        self.process_volume(P, V, input_image, depth_map)
        envmap = self.sglv_renderer.render(origin, self.sglv_volume, self.voxel_range)
        last = self.prev_envmap.detach()
        if use_detailed:
            if self.depth_accum is None or self.depth_accum.shape[0] != B:
                self.depth_accum = torch.zeros(B, 1, *self.output_resolution, device=device)
                self.weight_accum = torch.zeros(B, 1, *self.output_resolution, device=device)
            detailed_envmap, depth_pano = self.detailed_rednerer.render(origin, P, V, input_image, depth_map)
            # mask shape (B,1,H,W)
            mask = (detailed_envmap > 0).any(dim=1, keepdim=True).float()
            envmap, self.depth_accum, self.weight_accum = self.blending_network(
                last, envmap, detailed_envmap, mask, depth_pano, self.depth_accum.detach(), self.weight_accum.detach()
            )
        self.prev_envmap = envmap
        return envmap, last

    def reset(self):
        self.voxel_range = None
        self.sglv_volume = None
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
        rgb = batch['rgb'].permute(1, 0, 2, 3, 4)
        depth = batch['depth'].permute(1, 0, 2, 3, 4)
        pose = batch['pose'].permute(1, 0, 2, 3)
        lighting = batch["lighting"]
        position = batch["position"]

        n_frames = rgb.shape[0]
        reset_model = True
        for i in range(n_frames):
            rgb_np = rgb[i, 0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
            depth_np = depth[i, 0].repeat(3, 1, 1).permute(1, 2, 0).cpu().numpy()
            depth_np /= np.max(depth_np)
            lighting_np = lighting[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
            print(pose[i])

            result, _ = model(position, projection_matrix, torch.inverse(pose[i]), rgb[i], depth[i], False, reset_model)
            reset_model = False

            cv2.imshow("rgb", rgb_np)
            cv2.imshow("depth", depth_np)
            cv2.imshow("lighting", linear_to_srgb(lighting_np))
            cv2.imshow("result", linear_to_srgb(result[0].permute(1, 2, 0).detach().cpu().numpy()[:, :, ::-1]))
            key = cv2.waitKey(0) & 0xFF
            while key != ord('n') and key != ord('q'):
                key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
        if key == ord('q'):
            break
    cv2.destroyAllWindows()