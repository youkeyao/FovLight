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
from LightEnvDatasets import LightEnvDatasets
from utils import create_projection_matrix

def get_cubemap_uv(x, y, z):
    """
    Convert 3D direction to cubemap face and UV coordinates.
    """
    abs_x, abs_y, abs_z = abs(x), abs(y), abs(z)

    if abs_x >= abs_y and abs_x >= abs_z:
        if x > 0:  # Right face
            return 0, (z / abs_x + 1) / 2, (-y / abs_x + 1) / 2
        else:  # Left face
            return 1, (-z / abs_x + 1) / 2, (-y / abs_x + 1) / 2
    elif abs_y >= abs_x and abs_y >= abs_z:
        if y > 0:  # Up face
            return 2, (-x / abs_y + 1) / 2, (z / abs_y + 1) / 2
        else:  # Down face
            return 3, (-x / abs_y + 1) / 2, (-z / abs_y + 1) / 2
    else:
        if z > 0:  # Front face
            return 4, (-x / abs_z + 1) / 2, (-y / abs_z + 1) / 2
        else:  # Back face
            return 5, (x / abs_z + 1) / 2, (-y / abs_z + 1) / 2

def create_mesh_from_depth(depth_map, camera_matrix, depth_threshold=0.1):
    _, H, W = depth_map.shape
    f = camera_matrix[0, 0]
    device = depth_map.device

    # Back Projection
    v_coords = (torch.arange(H, device=device) - (H-1) / 2) / (H-1) * 2
    u_coords = (torch.arange(W, device=device) - (W-1) / 2) / (W-1) * 2
    v, u = torch.meshgrid(v_coords, u_coords, indexing='ij')
    directions = torch.stack([u / camera_matrix[0, 0], -v / camera_matrix[1, 1], -torch.ones_like(v)], dim=-1)
    vertices = directions * depth_map[0, :, :].unsqueeze(-1)

    # Generate triangular faces
    faces = []
    for i in range(H - 1):
        for j in range(W - 1):
            # Get depth values for current quad
            d00 = depth_map[0, i, j]
            d01 = depth_map[0, i, j+1]
            d10 = depth_map[0, i+1, j]
            d11 = depth_map[0, i+1, j+1]
            
            # Skip invalid depths or large discontinuities
            if (d00 <= 0 or d01 <= 0 or d10 <= 0 or d11 <= 0 or
                torch.abs(d00 - d01) > depth_threshold or
                torch.abs(d00 - d10) > depth_threshold or
                torch.abs(d00 - d11) > depth_threshold):
                continue
            
            # Vertex indices
            v00 = i * W + j
            v01 = i * W + j + 1
            v10 = (i+1) * W + j
            v11 = (i+1) * W + j + 1
            
            # Create two triangles per quad
            faces.append([v00, v01, v10])
            faces.append([v10, v01, v11])
    
    if not faces:
        faces = torch.zeros((0, 3), dtype=torch.int64, device=device)
    else:
        faces = torch.tensor(faces, dtype=torch.int64, device=device)
    
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
    cameras = FoVPerspectiveCameras(
        R=R,
        T=T,
        fov=fov,
        device=device
    )
    renderer.rasterizer.cameras = cameras
    renderer.shader.cameras = cameras
        
    # Render the scene
    with torch.no_grad():
        images = renderer(mesh)
        return images

def create_equirectangular_from_cubemap(cubemap_images, resolution=(160, 320)):
    height, width = resolution
    equirect = torch.zeros((3, height, width), device=cubemap_images.device)
    
    for y in range(height):
        for x in range(width):
            # Convert pixel coordinates to spherical coordinates
            theta = 2 * np.pi * x / width  # longitude
            phi = np.pi * y / height   # latitude
            
            # Convert spherical to cartesian
            dx = np.sin(phi) * np.cos(theta)
            dy = np.cos(phi)
            dz = np.sin(phi) * np.sin(theta)
            
            # Determine which face and UV coordinates
            face_idx, u, v = get_cubemap_uv(dx, dy, dz)
            
            if 0 <= face_idx < 6:
                face_img = cubemap_images[face_idx, :]
                face_h, face_w = face_img.shape[:2]
                
                # Convert UV to pixel coordinates
                px = int(u * (face_w - 1))
                py = int(v * (face_h - 1))

                equirect[:, y, x] = face_img[py, px, :3]
    
    return equirect

class DetailedRenderer:
    def __init__(self, resolution=(160, 320)):
        self.resolution = resolution

    def render(self, origin, camera_matrix, input_image, depth_map):
        vertices, faces, tex_coords = create_mesh_from_depth(
            depth_map, camera_matrix, depth_threshold=0.5
        )
        
        # Create mesh
        textures = TexturesUV(
            maps=[input_image.permute(1, 2, 0)] * 6,
            faces_uvs=[faces] * 6,
            verts_uvs=[tex_coords] * 6
        )
        mesh = Meshes(
            verts=[vertices] * 6,
            faces=[faces] * 6,
            textures=textures
        )
        cubemap_images = render_cube_maps(
            mesh, origin, input_image.device, resolution=512, fov=90
        )
        envmap = create_equirectangular_from_cubemap(cubemap_images, self.resolution)

        # save_obj("output.obj", verts=vertices, faces=faces, verts_uvs=tex_coords, faces_uvs=faces, texture_map=input_image.permute(1, 2, 0))

        return envmap

if __name__ == "__main__":
    detailed_renderer = DetailedRenderer()
    projection_matrix = create_projection_matrix(39.6, 640/480, 0.1, 20)
    dataset = LightEnvDatasets(root_dir='/mnt/data/youkeyao/Datasets/LightEnv')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    for batch in dataloader:
        image = batch['image'][0]
        depth = batch['depth'][0]
        lighting = batch['lighting'][0]
        pos = batch["pos"][0]

        print(pos)
        envmap = detailed_renderer.render(pos, projection_matrix, image, depth)
        envmap_np = envmap.permute(1, 2, 0).numpy()[:, :, ::-1]
        cv2.imshow("envmap", envmap_np)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()