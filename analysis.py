import torch
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import time
import re
from skimage.metrics import peak_signal_noise_ratio as psnr
from pytorch_msssim import ms_ssim
import lpips
import matplotlib.pyplot as plt

def lpips_input(arr: np.ndarray):
    tensor = torch.from_numpy(arr).float()

    min_val = tensor.min()
    max_val = tensor.max()
    tensor = (tensor - min_val) / (max_val - min_val)
    tensor = tensor * 2 - 1

    return tensor.permute(2, 0, 1).unsqueeze(0)

def ms_ssim_input(arr: np.ndarray):
    arr = cv2.resize(arr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(arr).float()

    return tensor.permute(2, 0, 1).unsqueeze(0)

loss_fn_alex = lpips.LPIPS(net='alex')

root_dir = "/mnt/data/youkeyao/Datasets/LightEnv"
image_paths = []
pattern = re.compile(
    r'^'                # 开头
    r'([-+]?\d*\.?\d+)' # 第一个浮点数（可选符号，可选整数部分，可选小数部分）
    r'_'                # 下划线分隔符
    r'([-+]?\d*\.?\d+)' # 第二个浮点数
    r'_'                # 下划线分隔符
    r'([-+]?\d*\.?\d+)' # 第三个浮点数
    r'\.exr$'           # .exr 扩展名
)
for scene in os.listdir(root_dir):
    for filename in os.listdir(os.path.join(root_dir, scene)):
        if not filename.lower().endswith('.exr'):
            continue
        match = pattern.match(filename)
        if match:
            try:
                x = float(match.group(1))
                y = float(match.group(2))
                z = float(match.group(3))
                lighting_path = os.path.join(root_dir, scene, filename)
                image_paths.append(lighting_path)
            except ValueError:
                continue

checkpoint_paths = [
    "/mnt/data/youkeyao/FovLight/checkpoints/voxel_0",
    "/mnt/data/youkeyao/FovLight/checkpoints/voxel_1",
    "/mnt/data/youkeyao/FovLight/checkpoints/voxel_2",
    "/mnt/data/youkeyao/FovLight/checkpoints/voxel_3",
    "/mnt/data/youkeyao/FovLight/checkpoints/voxel_4",
    "/mnt/data/youkeyao/FovLight/checkpoints/voxel_5",
    "/mnt/data/youkeyao/FovLight/checkpoints/voxel_6",
    "/mnt/data/youkeyao/FovLight/checkpoints/voxel_7",
    "/mnt/data/youkeyao/FovLight/checkpoints/voxel_8",
    # "/mnt/data/youkeyao/FovLight/checkpoints/voxel_9",
    # "/mnt/data/youkeyao/FovLight/checkpoints/network_0",
    # "/mnt/data/youkeyao/FovLight/checkpoints/network_1",
    # "/mnt/data/youkeyao/FovLight/checkpoints/network_2",
]

psnrs = []
ms_ssims = []
lpipss = []
with torch.no_grad():
    for checkpoint_path in checkpoint_paths:
        psnr_score = 0
        ssim_score = 0
        lpips_score = 0
        for image_path in image_paths:
            origin = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0:3][:, :, ::-1].astype(np.float32)

            image = cv2.imread(os.path.join(checkpoint_path, os.path.basename(image_path)), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0:3][:, :, ::-1].astype(np.float32)

            # range = max(np.max(origin), np.max(image))
            range = 1.0
            psnr_score += psnr(origin, image, data_range=range)
            ssim_score += ms_ssim(ms_ssim_input(origin), ms_ssim_input(image), data_range=range)
            lpips_score += loss_fn_alex(lpips_input(origin), lpips_input(image)).item()
        psnr_score /= len(image_paths)
        ssim_score /= len(image_paths)
        lpips_score /= len(image_paths)
        psnrs.append(psnr_score)
        ms_ssims.append(ssim_score)
        lpipss.append(lpips_score)

psnrs = np.array(psnrs)
# psnrs = (psnrs) / (psnrs.max())
ms_ssims = np.array(ms_ssims)
# ssims = (ssims) / (ssims.max())
lpipss = np.array(lpipss)
# lpipss = (lpipss) / (lpipss.max())

fig, ax1 = plt.subplots()
x = np.linspace(0, len(psnrs)-1, len(psnrs))
ax1.plot(x, psnrs, label='psnr')
ax1.legend()
ax2 = ax1.twinx()
ax2.plot(x, ms_ssims, label='ms_ssim')
ax2.plot(x, lpipss, label='lpips')

plt.title("quality")
plt.xlabel("params")
plt.ylabel("value")
ax2.legend()
plt.grid(True)

# 保存图片
plt.savefig("quality.png", dpi=300, bbox_inches='tight')