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
from accelerate import Accelerator
from LightEstimationModel import LightingEstimationModel
from EnvMapDatasets import EnvMapDatasets
from ObjectRenderer import ObjectRenderer
from utils import create_projection_matrix

loss_fn_alex = lpips.LPIPS(net='alex')

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

def analysis_envmap(checkpoint_dir, model_param):
    accelerator = Accelerator()
    device = accelerator.device
    model = LightingEstimationModel(*model_param)
    model = accelerator.prepare(model)
    accelerator.load_state(checkpoint_dir)
    projection_matrix = create_projection_matrix(120, 440/385, 0.1, 20)
    dataset = EnvMapDatasets(root_dir='/mnt/data/youkeyao/Datasets/LightEnv', image_resolution=(385, 440), lighting_resolution=model.output_resolution)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    with torch.no_grad():
        psnr_score = 0
        ssim_score = 0
        lpips_score = 0
        count = 0
        for batch in dataloader:
            image = batch['image'][0]
            depth = batch['depth'][0]
            lighting = batch['lighting'][0]
            pos = batch["pos"][0]

            envmap = model(pos.to(device), projection_matrix.to(device), image.to(device), depth.to(device)).permute(1, 2, 0).detach().cpu().numpy()
            origin = lighting.permute(1, 2, 0).detach().cpu().numpy()

            # range = max(np.max(origin), np.max(envmap))
            range = 1.0
            psnr_score += psnr(origin, envmap, data_range=range)
            ssim_score += ms_ssim(ms_ssim_input(origin), ms_ssim_input(envmap), data_range=range)
            lpips_score += loss_fn_alex(lpips_input(origin), lpips_input(envmap)).item()
            count += 1
        psnr_score /= count
        ssim_score /= count
        lpips_score /= count
        
        return (psnr_score, ssim_score, lpips_score)
    
def analysis_render(checkpoint_dir, model_param):
    accelerator = Accelerator()
    device = accelerator.device
    model = LightingEstimationModel(*model_param)
    model = accelerator.prepare(model)
    accelerator.load_state(checkpoint_dir)
    projection_matrix = create_projection_matrix(120, 440/385, 0.1, 20)
    dataset = EnvMapDatasets(root_dir='/mnt/data/youkeyao/Datasets/LightEnv', image_resolution=(385, 440), lighting_resolution=model.output_resolution)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    renderer = ObjectRenderer((385, 440), 120)
    render_pos = torch.tensor([0, 0, -0.15])

    with torch.no_grad():
        psnr_score = 0
        ssim_score = 0
        lpips_score = 0
        count = 0
        for batch in dataloader:
            image = batch['image'][0]
            depth = batch['depth'][0]
            lighting = batch['lighting'][0]
            pos = batch["pos"][0]

            envmap = model(pos.to(device), projection_matrix.to(device), image.to(device), depth.to(device))
            origin = lighting

            sphere = renderer.render_sphere(render_pos, envmap)
            origin = renderer.render_sphere(render_pos, origin)

            range = 1.0
            psnr_score += psnr(origin, sphere, data_range=range)
            ssim_score += ms_ssim(ms_ssim_input(origin), ms_ssim_input(sphere), data_range=range)
            lpips_score += loss_fn_alex(lpips_input(origin), lpips_input(sphere)).item()
            count += 1
        psnr_score /= count
        ssim_score /= count
        lpips_score /= count
        
        return (psnr_score, ssim_score, lpips_score)
    
def main():
    result = []
    # result.append(analysis_envmap("/mnt/data/youkeyao/FovLight/checkpoints/voxel_0", ((168, 120, 128), (320, 640), 100, 4)))
    # result.append(analysis_envmap("/mnt/data/youkeyao/FovLight/checkpoints/voxel_1", ((151, 108, 115), (320, 640), 100, 4)))
    # result.append(analysis_envmap("/mnt/data/youkeyao/FovLight/checkpoints/voxel_2", ((134, 96, 102), (320, 640), 100, 4)))
    # result.append(analysis_envmap("/mnt/data/youkeyao/FovLight/checkpoints/voxel_3", ((118, 84, 90), (320, 640), 100, 4)))
    # result.append(analysis_envmap("/mnt/data/youkeyao/FovLight/checkpoints/voxel_4", ((101, 72, 77), (320, 640), 100, 4)))
    # result.append(analysis_envmap("/mnt/data/youkeyao/FovLight/checkpoints/voxel_5", ((84, 60, 64), (320, 640), 100, 4)))
    # result.append(analysis_envmap("/mnt/data/youkeyao/FovLight/checkpoints/voxel_6", ((67, 48, 51), (320, 640), 100, 4)))
    # result.append(analysis_envmap("/mnt/data/youkeyao/FovLight/checkpoints/voxel_7", ((50, 36, 38), (320, 640), 100, 4)))
    # result.append(analysis_envmap("/mnt/data/youkeyao/FovLight/checkpoints/voxel_8", ((34, 24, 26), (320, 640), 100, 4)))
    # result.append(analysis_render("/mnt/data/youkeyao/FovLight/checkpoints/voxel_0", ((168, 120, 128), (320, 640), 100, 4)))
    # result.append(analysis_render("/mnt/data/youkeyao/FovLight/checkpoints/voxel_1", ((151, 108, 115), (320, 640), 100, 4)))
    # result.append(analysis_render("/mnt/data/youkeyao/FovLight/checkpoints/voxel_2", ((134, 96, 102), (320, 640), 100, 4)))
    # result.append(analysis_render("/mnt/data/youkeyao/FovLight/checkpoints/voxel_3", ((118, 84, 90), (320, 640), 100, 4)))
    # result.append(analysis_render("/mnt/data/youkeyao/FovLight/checkpoints/voxel_4", ((101, 72, 77), (320, 640), 100, 4)))
    # result.append(analysis_render("/mnt/data/youkeyao/FovLight/checkpoints/voxel_5", ((84, 60, 64), (320, 640), 100, 4)))
    # result.append(analysis_render("/mnt/data/youkeyao/FovLight/checkpoints/voxel_6", ((67, 48, 51), (320, 640), 100, 4)))
    # result.append(analysis_render("/mnt/data/youkeyao/FovLight/checkpoints/voxel_7", ((50, 36, 38), (320, 640), 100, 4)))
    # result.append(analysis_render("/mnt/data/youkeyao/FovLight/checkpoints/voxel_8", ((34, 24, 26), (320, 640), 100, 4)))
    result.append(analysis_envmap("/mnt/data/youkeyao/FovLight/checkpoints_new2/voxel_0", ((168, 120, 128), (320, 640), 100, 4)))
    result.append(analysis_envmap("/mnt/data/youkeyao/FovLight/checkpoints_new2/network_0", ((168, 120, 128), (320, 640), 100, 3)))
    result.append(analysis_envmap("/mnt/data/youkeyao/FovLight/checkpoints_new2/network_1", ((168, 120, 128), (320, 640), 100, 2)))
    result.append(analysis_envmap("/mnt/data/youkeyao/FovLight/checkpoints_new2/network_2", ((168, 120, 128), (320, 640), 100, 1)))

    result = np.array(result)
    psnrs = result[:, 0]
    # psnrs = (psnrs) / (psnrs.max())
    ms_ssims = result[:, 1]
    # ssims = (ssims) / (ssims.max())
    lpipss = result[:, 2]
    # lpipss = (lpipss) / (lpipss.max())

    fig, ax1 = plt.subplots()
    x = np.linspace(0, len(psnrs)-1, len(psnrs))
    ax1.plot(x, psnrs, label='psnr')
    ax1.legend()
    ax2 = ax1.twinx()
    ax2.plot(x, ms_ssims, label='ms_ssim', color='red')
    ax2.plot(x, lpipss, label='lpips')

    plt.title("quality")
    plt.xlabel("params")
    plt.ylabel("value")
    ax2.legend()
    plt.grid(True)

    # 保存图片
    plt.savefig("quality.png", dpi=300, bbox_inches='tight')
    
if __name__ == "__main__":
    # main()

    gt = cv2.imread("/mnt/data/youkeyao/FovLight/pilot/scene_001/gt/sphere.png", -1)[:, :, ::-1].astype(np.float32) / 255
    im = cv2.imread("/mnt/data/youkeyao/FovLight/pilot/scene_001/qint8/sphere.png", -1)[:, :, ::-1].astype(np.float32) / 255
    print(psnr(gt, im, data_range=1))
    print(ms_ssim(ms_ssim_input(gt), ms_ssim_input(im), data_range=1))
    print(loss_fn_alex(lpips_input(gt), lpips_input(im)).item())

    # gt = cv2.imread("/mnt/data/youkeyao/FovLight/pilot/0/gt.png", -1)[1700:2150, 2200:2500, 0:3][:, :, ::-1].astype(np.float32) / 255
    # im = cv2.imread("/mnt/data/youkeyao/FovLight/pilot/0/voxel_8.png", -1)[1700:2150, 2200:2500, 0:3][:, :, ::-1].astype(np.float32) / 255
    # cv2.imshow("test", gt)
    # cv2.waitKey(0)
    # print(psnr(gt, im, data_range=1))

    # gt = cv2.imread("/mnt/data/youkeyao/FovLight/pilot/1/gt.png", -1)[1700:2150, 2400:2800, 0:3][:, :, ::-1].astype(np.float32) / 255
    # im = cv2.imread("/mnt/data/youkeyao/FovLight/pilot/1/voxel_8.png", -1)[1700:2150, 2400:2800, 0:3][:, :, ::-1].astype(np.float32) / 255
    # # cv2.imshow("test", gt)
    # # cv2.waitKey(0)
    # print(psnr(gt, im, data_range=1))

    # gt = cv2.imread("/mnt/data/youkeyao/FovLight/pilot/2/gt.png", -1)[1700:2150, 2800:3200, 0:3][:, :, ::-1].astype(np.float32) / 255
    # im = cv2.imread("/mnt/data/youkeyao/FovLight/pilot/2/voxel_8.png", -1)[1700:2150, 2800:3200, 0:3][:, :, ::-1].astype(np.float32) / 255
    # # cv2.imshow("test", gt)
    # # cv2.waitKey(0)
    # print(psnr(gt, im, data_range=1))