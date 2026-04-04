"""单样例推理脚本。

用于读取固定输入图像与深度图，预测环境贴图，并完成物体插入渲染演示。
"""

import torch
import mitsuba as mi
from torch.utils.data import ConcatDataset, DataLoader
import cv2
import numpy as np
from accelerate import Accelerator
from skimage.metrics import structural_similarity as ssim
from EnvMapDatasets import EnvMapDatasets
from LightEstimationModel import LightingEstimationModel
from ObjectRenderer import ObjectRenderer
from utils import create_projection_matrix, linear_to_srgb, psnr

def main(checkpoint_dir, model_param):
    """加载检查点并执行一次端到端推理。"""
    accelerator = Accelerator()
    device = accelerator.device
    print(f"Using device: {device}")
    # 推理
    model = LightingEstimationModel(*model_param)
    model = accelerator.prepare(model)
    accelerator.load_state(checkpoint_dir)
    projection_matrix = create_projection_matrix(120, 832/400, 0.1, 20)
    model.eval()
    
    # 第一段：读取输入图像、深度和参考环境图。
    input_img = cv2.imread("/mnt/data/youkeyao/FovLight/input.png", -1)[:, :, 0:3][:, :, ::-1].astype(np.float32) / 255
    input_img = torch.from_numpy(input_img).permute(2, 0, 1)
    depth_img = torch.from_numpy(cv2.imread("/mnt/data/youkeyao/FovLight/depth.exr", -1)[:, :, 0:1].astype(np.float32)).permute(2, 0, 1)
    gt_envmap = torch.from_numpy(cv2.imread("/mnt/data/youkeyao/Datasets/EnvMapVideo/scene_003/0.36_0_-2.5.exr", -1).astype(np.float32)).permute(2, 0, 1)
    pos = torch.tensor([0, 0, -1.0])
    envmap, _ = model(pos, projection_matrix, torch.eye(4), input_img, depth_img, True, True)

    # 第二段：执行物体插入渲染与阴影合成。
    bg_img = torch.from_numpy(cv2.imread("/mnt/data/youkeyao/Sig25_4DLighting/exp1/scene_006/bg.png", -1)[:, :, 0:3][:, :, ::-1].astype(np.float32) / 255.0).permute(2, 0, 1)
    renderer = ObjectRenderer((bg_img.shape[1], bg_img.shape[2]), 120)
    def _build_transform(pos, rot_deg, scale=None):
        pitch, yaw, roll = rot_deg
        t = mi.ScalarTransform4f().translate(pos) @ mi.ScalarTransform4f().rotate([1, 0, 0], pitch) @ mi.ScalarTransform4f().rotate([0, 1, 0], yaw) @ mi.ScalarTransform4f().rotate([0, 0, 1], roll)
        if scale is not None:
            t = t @ mi.ScalarTransform4f().scale(scale)
        return t
    plane_pos = [0,-0.7,-2]
    plane_rot = [-90,0,0]
    plane_scale = 2.0
    renderer.plane_transform = _build_transform(plane_pos, plane_rot, scale=plane_scale)
    render_pos = torch.tensor([0, -0.5, -1.5])
    sphere = renderer.render_sphere(render_pos, envmap[0], bg_img, 0.2, 0.0, 0.0, [0.5, 0.5, 0.5])
    sphere = renderer.render_shadow(render_pos, envmap[0], bg_img, torch.tensor(sphere).permute(2, 0, 1), 0.2, 0.0, 0.0, [0.5, 0.5, 0.5])
    
    # 第三段：保存预测环境图与合成结果。
    cv2.imwrite("output_envmap.exr", cv2.cvtColor(envmap[0].detach().permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_envmap.png", (sphere[:, :, ::-1] * 255).astype(np.uint8))

if __name__ == "__main__":
    # main("checkpoints/network_2", ((84, 60, 64), (320, 640), 100, 1))
    main("checkpoints/network_1", ((84, 60, 64), (320, 640), 100, 2))
    # main("checkpoints/voxel_7", ((50, 36, 38), (320, 640), 100, 4))