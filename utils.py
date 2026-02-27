import torch
import math
import numpy as np
import cv2

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
    
    proj_matrix[0, 0] = f
    proj_matrix[1, 1] = f * aspect_ratio
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
    if isinstance(voxel_range, torch.Tensor) and voxel_range.dim() == 3:
        voxel_range = voxel_range[0]
    _, volume_size_x, volume_size_y, volume_size_z = voxel_data.shape
    points = []
    colors = []
    alphas = []

    for i in range(volume_size_x):
        for j in range(volume_size_y):
            for k in range(volume_size_z):
                # if voxel_data[4, i, j, k] == 0:  # 只保存需要渲染的点
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

def linear_to_srgb(linear_rgb_array, exposure=0):
    x = linear_rgb_array.astype(np.float32) * (2.0 ** exposure)
    
    # 为了防止负值导致的计算错误 (虽然线性光不应有负值)
    x = np.maximum(x, 0.0)

    # 2. Filmic Tone Mapping (ACES 近似算法)
    # 这是一个被广泛用于模拟 Filmic look 的 Narkowicz ACES 拟合曲线。
    # 它能很好地模拟高动态范围压缩和高光去饱和。
    # 公式: y = (x * (a*x + b)) / (x * (c*x + d) + e)
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    
    mapped = (x * (a * x + b)) / (x * (c * x + d) + e)

    # 截断到 0-1 范围
    mapped = np.clip(mapped, 0.0, 1.0)

    # 3. Gamma 矫正 (Linear -> sRGB)
    # ACES 拟合曲线输出的结果通常被认为是“适合显示的线性值”，
    # 但为了在普通显示器(sRGB)上看起来正确，通常还需要应用 Gamma 1/2.2
    # 注意：Blender 内部流程复杂，但数学模拟通常包含这一步。
    # 如果你发现画面太白，可以去掉这一步，或者改用 sRGB 标准公式。
    
    # 简单的 Gamma 2.2 近似:
    result = np.power(mapped, 1.0 / 2.2)

    return result

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = max(np.max(img1), np.max(img2))
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def compute_peli_pyramid(image):
    # 1. 预处理：转灰度 + 归一化
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    else:
        img_gray = image.astype(np.float64) / 255.0
    
    rows, cols = img_gray.shape
    
    # 2. 构建频率坐标网格 (Frequency Grid)
    u = np.arange(1, cols + 1)
    v = np.arange(1, rows + 1)
    U, V = np.meshgrid(u, v)
    center_x = (cols + 1) / 2.0
    center_y = (rows + 1) / 2.0
    distances = np.sqrt((U - center_x)**2 + (V - center_y)**2)
    distances[distances == 0] = 1e-10 # 避免 log(0)
    log2_distances = np.log2(distances)
    
    # 3. FFT 变换到频域
    fourier_image = np.fft.fftshift(np.fft.fft2(img_gray))
    
    # 4. 定义倍频程 (Octave Bands)
    min_dim = min(rows, cols)
    octave_bands = [2.0, 4.0] 
    while True:
        next_val = octave_bands[-1] * 2
        if next_val > (min_dim / 2):
            break
        octave_bands.append(next_val)
    octave_bands = np.array(octave_bands)
    
    # 5. 构建滤波器并生成空域图像
    spatial_bands = []
    
    for level in range(len(octave_bands)):
        if level == 0:
            # 低频残差 / Low frequency residual
            # MATLAB: mask = (dist > band[0]) & (dist <= band[1]) ... then 1-filter
            mask_trans = (distances > octave_bands[level]) & (distances <= octave_bands[level+1])
            term = 0.5 * (1 + np.cos(np.pi * log2_distances - np.log2(octave_bands[level+1]) * np.pi))
            
            # 组合过渡带和极低频部分
            filt = np.ones_like(distances)
            filt[mask_trans] = (1.0 - term)[mask_trans] # 过渡带取反
            filt[distances > octave_bands[level+1]] = 0.0 # 高频部分切除
            
        elif level == len(octave_bands) - 1:
            # 高频残差 / High frequency residual
            mask = distances > octave_bands[level-1]
            term = 0.5 * (1 + np.cos(np.pi * log2_distances - np.log2(octave_bands[level-1]) * np.pi))
            filt = (1.0 - term) * mask
            
        else:
            # 标准带通 / Standard Band-pass
            mask = (distances > octave_bands[level-1]) & (distances <= octave_bands[level+1])
            term = 0.5 * (1 + np.cos(np.pi * log2_distances - np.log2(octave_bands[level]) * np.pi))
            filt = term * mask
            
        # --- 频域滤波并逆变换回空域 ---
        # 结果保留实部 (Real part of IFFT)
        band_freq = fourier_image * filt
        band_spatial = np.real(np.fft.ifft2(np.fft.ifftshift(band_freq)))
        
        spatial_bands.append(band_spatial)
        
    return spatial_bands, octave_bands

def calculate_spatial_metrics(img, mask, fov=120):
    spatial_bands, octave_bands = compute_peli_pyramid(img)
    cpd_values = octave_bands / fov
    
    # 确保 mask 是布尔类型
    mask_bool = mask.astype(bool)

    contrast_metrics = []
    current_background_l = spatial_bands[0].copy()
    
    valid_band_indices = range(1, len(spatial_bands) - 1)
    
    for i in range(1, len(spatial_bands) - 1):
        band_a = spatial_bands[i]

        denom = np.abs(current_background_l) + 1e-7
        local_contrast_map = band_a / denom
        
        roi_contrast = local_contrast_map[mask_bool]
        
        rms_contrast = np.sqrt(np.mean(roi_contrast ** 2))
        # rms_contrast = np.mean(np.abs(roi_contrast))

        contrast_metrics.append(rms_contrast)
        current_background_l += band_a

    if len(contrast_metrics) > 0:
        best_idx_local = 4
        best_band_global_idx = valid_band_indices[best_idx_local]
        
        dominant_rms = contrast_metrics[best_idx_local]
        dominant_cpd = cpd_values[best_band_global_idx]
        vis_band = spatial_bands[best_band_global_idx] + spatial_bands[0]
        vis_band[~mask_bool] = vis_band.max()
    else:
        dominant_rms = 0.0
        dominant_cpd = 0.0
        vis_band = np.zeros_like(img, dtype=np.float64)

    vis_min, vis_max = vis_band.min(), vis_band.max()
    if vis_max - vis_min > 1e-9:
        vis_img = ((vis_band - vis_min) / (vis_max - vis_min) * 255).astype(np.uint8)
    else:
        vis_img = np.zeros(vis_band.shape, dtype=np.uint8)

    return dominant_rms, dominant_cpd, vis_img