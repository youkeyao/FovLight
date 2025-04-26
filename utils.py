import torch
import math

def get_free_gpu():
    """
    选择最大空闲内存gpu
    """
    num_gpus = torch.cuda.device_count()
    free_gpu = None
    free_load = 0
    for i in range(num_gpus):
        torch.cuda.set_device(i)
        memory_free = torch.cuda.mem_get_info()[0] / 1e9  # 转换为GB
        if memory_free > free_load:
            free_load = memory_free
            free_gpu = i
    return free_gpu

def create_projection_matrix(fov_deg, aspect_ratio, near, far):
    """
    创建透视投影矩阵。
    
    参数:
        fov_deg: 视场角（以度为单位）
        aspect_ratio: 宽高比 (width / height)
        near: 近裁剪面
        far: 远裁剪面
    
    返回:
        4x4 透视投影矩阵
    """
    # 将视场角从度转换为弧度
    fov_rad = math.radians(fov_deg)
    
    # 计算焦距 (f)
    f = 1.0 / math.tan(fov_rad / 2)
    
    # 创建投影矩阵
    proj_matrix = torch.zeros((4, 4), dtype=torch.float32)
    
    proj_matrix[0, 0] = f / aspect_ratio
    proj_matrix[1, 1] = f
    proj_matrix[2, 2] = -(far + near) / (far - near)
    proj_matrix[2, 3] = -2 * far * near / (far - near)
    proj_matrix[3, 2] = -1.0
    
    return proj_matrix

def visualize_voxel_data(voxel_data, voxel_range):
    """
    可视化体素数据，包括颜色和透明度。
    
    参数:
        voxel_data: 体素数据，形状为 (5, volume_size, volume_size, volume_size)
        voxel_range: 体素范围，[[x_min, y_min, z_min], [x_max, y_max, z_max]]
    """
    _, volume_size_x, volume_size_y, volume_size_z = voxel_data.shape
    points = []
    colors = []
    alphas = []

    for i in range(volume_size_x):
        for j in range(volume_size_y):
            for k in range(volume_size_z):
                if voxel_data[4, i, j, k] == 0:  # 只保存需要渲染的点
                    voxel_pos_x = voxel_range[0][0] + i * (voxel_range[1][0] - voxel_range[0][0]) / volume_size_x
                    voxel_pos_y = voxel_range[0][1] + j * (voxel_range[1][1] - voxel_range[0][1]) / volume_size_y
                    voxel_pos_z = voxel_range[0][2] + k * (voxel_range[1][2] - voxel_range[0][2]) / volume_size_z
                    r, g, b = voxel_data[0, i, j, k], voxel_data[1, i, j, k], voxel_data[2, i, j, k]
                    r = max(0, min(r, 1))
                    g = max(0, min(g, 1))
                    b = max(0, min(b, 1))
                    alpha = voxel_data[3, i, j, k]

                    points.append((voxel_pos_x, voxel_pos_y, voxel_pos_z))
                    colors.append((r, g, b))
                    alphas.append(alpha)

    with open("voxel.ply", 'w') as f:
        # PLY 文件头部
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float r\n")
        f.write("property float g\n")
        f.write("property float b\n")
        f.write("property float a\n")
        f.write("end_header\n")

        # 写入点数据
        for (x, y, z), (r, g, b), alpha in zip(points, colors, alphas):
            f.write(f"{x} {y} {z} {r} {g} {b} {alpha}\n")