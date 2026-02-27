import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.io import save_obj
from pytorch3d.renderer import (
    look_at_view_transform,
    TexturesUV,
    BlendParams,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardFlatShader,
)
from EnvMapDatasets import EnvMapDatasets
from EnvMapVideoDatasets import EnvMapVideoDatasets
from utils import create_projection_matrix, linear_to_srgb

def create_mesh_from_depth(depth_map, P, V, depth_threshold=0.1):
    _, H, W = depth_map.shape
    device = depth_map.device

    # Back Projection
    v_coords = (torch.arange(H, device=device) - (H-1) / 2) / (H-1) * 2
    u_coords = (torch.arange(W, device=device) - (W-1) / 2) / (W-1) * 2
    v, u = torch.meshgrid(v_coords, u_coords, indexing='ij')
    directions = torch.stack([u / P[0, 0], -v / P[1, 1], -torch.ones_like(v)], dim=-1)
    vertices = directions * depth_map[0, :, :].unsqueeze(-1)
    ones = torch.ones((H, W, 1), device=device)
    vertices = torch.cat([vertices, ones], dim=-1)
    vertices = torch.einsum("hwk, kj -> hwj", vertices, torch.inverse(V).T)
    vertices = vertices[..., :3]

    # Generate triangular faces
    depth_map_2d = depth_map[0]
    d00 = depth_map_2d[:-1, :-1]
    d01 = depth_map_2d[:-1, 1:]
    d10 = depth_map_2d[1:, :-1]
    d11 = depth_map_2d[1:, 1:]

    valid_mask = (
        (d00 > 0) & (d01 > 0) & (d10 > 0) & (d11 > 0) &
        (torch.abs(d00 - d01) <= depth_threshold) &
        (torch.abs(d00 - d10) <= depth_threshold) &
        (torch.abs(d00 - d11) <= depth_threshold)
    )

    valid_i, valid_j = torch.where(valid_mask)
    num_valid = valid_i.size(0)

    if num_valid == 0:
        faces = torch.zeros((0, 3), dtype=torch.int64, device=device)
    else:
        v00 = valid_i * W + valid_j
        v01 = v00 + 1
        v10 = (valid_i + 1) * W + valid_j
        v11 = v10 + 1

        triangles = torch.stack([
            torch.stack([v00, v01, v10], dim=1),
            torch.stack([v10, v01, v11], dim=1)
        ], dim=0)

        faces = triangles.reshape(-1, 3)

    # Normalized texture coordinates
    tex_u = (u + 1) / 2
    tex_v = 1 - (v + 1) / 2
    tex_coords = torch.stack((tex_u, tex_v), dim=-1)

    return vertices.reshape(-1, 3), faces.reshape(-1, 3), tex_coords.reshape(-1, 2)

def render_cube_maps(mesh, origin, device, resolution=512, fov=90):
    camera_pos = origin.repeat(6, 1)

    # Define 6 directions for cubemap (Right, Left, Up, Down, Front, Back)
    # Note: PyTorch3D uses different coordinate system than Open3D
    directions = torch.tensor([
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ], dtype=torch.float32, device=device)

    ups = torch.tensor([
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, -1],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
    ], dtype=torch.float32, device=device)

    # Setup rasterizer
    raster_settings = RasterizationSettings(
        image_size=resolution,
        blur_radius=0.0,
    )

    # Setup renderer
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=None,
            raster_settings=raster_settings
        ),
        shader=HardFlatShader(
            device=device,
            cameras=None,
            blend_params=BlendParams(background_color=(0, 0, 0)),
        )
    )

    target = camera_pos + directions
    R, T = look_at_view_transform(
        eye=camera_pos,
        at=target,
        up=ups,
        device=device
    )
    znear = 0.01
    zfar = 20
    cameras = FoVPerspectiveCameras(
        znear=znear,
        zfar=zfar,
        R=R,
        T=T,
        fov=fov,
        device=device
    )
    renderer.rasterizer.cameras = cameras
    renderer.shader.cameras = cameras

    # Render the scene
    with torch.no_grad():
        fragments = renderer.rasterizer(mesh)
        depth_world = fragments.zbuf[..., 0].unsqueeze(1)
        images = renderer.shader(fragments, mesh)
        rgb = images[..., :3].permute(0, 3, 1, 2)

    return rgb, depth_world

def create_equirectangular_from_cubemap(cubemap_images, resolution=(160, 320)):
    height, width = resolution
    device = cubemap_images.device
    C = cubemap_images.shape[1]

    # 创建坐标网格 (向量化)
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing='ij'
    )

    # 转换为球面坐标
    theta = 2 * np.pi * x / width  # 经度 [0, 2π]
    phi = np.pi * y / height      # 纬度 [0, π]

    # 转换为笛卡尔坐标 (右手坐标系)
    sin_phi = torch.sin(phi)
    x_cart = sin_phi * torch.sin(theta)  # X: 右为正
    y_cart = torch.cos(phi)              # Y: 上为正
    z_cart = -sin_phi * torch.cos(theta)  # Z: 前为正

    xyz = torch.stack([x_cart, y_cart, z_cart], dim=-1)  # [H, W, 3]

    # 计算绝对值和主导轴
    abs_xyz = torch.abs(xyz)
    max_val, max_axis = torch.max(abs_xyz, dim=-1)  # [H, W]

    # 初始化面索引和UV坐标
    face_idx = torch.full((height, width), -1, dtype=torch.long, device=device)
    u_map = torch.zeros_like(x_cart)
    v_map = torch.zeros_like(x_cart)

    # 计算每个面的UV坐标 (向量化)
    # 右面 (X+)
    mask = (max_axis == 0) & (xyz[..., 0] >= 0)
    if mask.any():
        scale = torch.reciprocal(xyz[..., 0][mask].abs())
        u_map[mask] = xyz[..., 2][mask] * scale
        v_map[mask] = -xyz[..., 1][mask] * scale
        face_idx[mask] = 0

    # 左面 (X-)
    mask = (max_axis == 0) & (xyz[..., 0] < 0)
    if mask.any():
        scale = torch.reciprocal(xyz[..., 0][mask].abs())
        u_map[mask] = -xyz[..., 2][mask] * scale
        v_map[mask] = -xyz[..., 1][mask] * scale
        face_idx[mask] = 1

    # 上面 (Y+)
    mask = (max_axis == 1) & (xyz[..., 1] >= 0)
    if mask.any():
        scale = torch.reciprocal(xyz[..., 1][mask].abs())
        u_map[mask] = -xyz[..., 0][mask] * scale
        v_map[mask] = xyz[..., 2][mask] * scale
        face_idx[mask] = 2

    # 下面 (Y-)
    mask = (max_axis == 1) & (xyz[..., 1] < 0)
    if mask.any():
        scale = torch.reciprocal(xyz[..., 1][mask].abs())
        u_map[mask] = -xyz[..., 0][mask] * scale
        v_map[mask] = -xyz[..., 2][mask] * scale
        face_idx[mask] = 3

    # 前面 (Z+)
    mask = (max_axis == 2) & (xyz[..., 2] >= 0)
    if mask.any():
        scale = torch.reciprocal(xyz[..., 2][mask].abs())
        u_map[mask] = -xyz[..., 0][mask] * scale
        v_map[mask] = -xyz[..., 1][mask] * scale
        face_idx[mask] = 4

    # 后面 (Z-)
    mask = (max_axis == 2) & (xyz[..., 2] < 0)
    if mask.any():
        scale = torch.reciprocal(xyz[..., 2][mask].abs())
        u_map[mask] = xyz[..., 0][mask] * scale
        v_map[mask] = -xyz[..., 1][mask] * scale
        face_idx[mask] = 5

    # 创建采样网格
    grid = torch.stack([u_map, v_map], dim=-1)  # [H, W, 2]

    # 初始化输出等距柱状图
    equirect = torch.zeros((C, height, width), device=device)

    # 为每个面采样
    for face in range(6):
        mask = (face_idx == face)
        if not mask.any():
            continue

        # 创建当前面的采样网格
        face_grid = grid.clone()
        face_grid[~mask] = -2  # 将非当前面的点移出采样范围

        # 调整网格维度 [H, W, 2] -> [1, H, W, 2]
        face_grid = face_grid.unsqueeze(0)

        # 采样当前面
        sampled = F.grid_sample(
            cubemap_images[face:face+1],  # [1, C, H_face, W_face]
            face_grid,
            mode='bilinear',
            align_corners=True,
            padding_mode='zeros'
        )  # [1, C, H, W]

        # 更新等距柱状图
        equirect[:, mask] = sampled[0, :, mask]

    return equirect

class DetailedRenderer:
    def __init__(self, resolution=(160, 320)):
        self.resolution = resolution

    def render(self, origin, P, V, input_image, depth_map):
        with torch.no_grad():
            # Support batched input: input_image (B, C, H, W) or (C, H, W)
            single = False
            if input_image.dim() == 3:
                input_image = input_image.unsqueeze(0)
                depth_map = depth_map.unsqueeze(0)
                origin = origin.unsqueeze(0)
                single = True

            B = input_image.shape[0]
            envmaps = []
            depth_panos = []
            for b in range(B):
                P_b = P[b] if P.dim() == 3 else P
                V_b = V[b] if V.dim() == 3 else V
                vertices, faces, tex_coords = create_mesh_from_depth(
                    depth_map[b], P_b, V_b, depth_threshold=0.5
                )

                # Create mesh
                textures = TexturesUV(
                    maps=[input_image[b].permute(1, 2, 0)] * 6,
                    faces_uvs=[faces] * 6,
                    verts_uvs=[tex_coords] * 6
                )
                mesh = Meshes(
                    verts=[vertices] * 6,
                    faces=[faces] * 6,
                    textures=textures
                )
                cubemap_images, depth_world = render_cube_maps(
                    mesh, origin[b], input_image.device, resolution=512, fov=90
                )
                envmap = create_equirectangular_from_cubemap(cubemap_images, self.resolution)
                depth_pano = create_equirectangular_from_cubemap(depth_world, self.resolution)
                envmaps.append(envmap)
                depth_panos.append(depth_pano)

            envmaps = torch.stack(envmaps, dim=0)
            depth_panos = torch.stack(depth_panos, dim=0)

            if single:
                return envmaps[0], depth_panos[0]
            return envmaps, depth_panos

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    detailed_renderer = DetailedRenderer()
    projection_matrix = create_projection_matrix(120, 832/400, 0.1, 20)
    dataset = EnvMapVideoDatasets(root_dir='/mnt/data/youkeyao/Datasets/EnvMapVideo')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

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
            envmap_np = lighting[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1]

            # for j in range(n_objects):
            result, depth_pano = detailed_renderer.render(position[0], projection_matrix, torch.inverse(pose[i, 0]), rgb[i, 0], depth[i, 0])

            cv2.imshow("rgb", rgb_np)
            cv2.imshow("depth", depth_np)
            cv2.imshow("lighting", envmap_np)
            cv2.imshow("envmap", result.permute(1, 2, 0).detach().cpu().numpy()[:, :, ::-1])
            key = cv2.waitKey(0) & 0xFF
            while key != ord('n') and key != ord('q'):
                key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
    cv2.destroyAllWindows()