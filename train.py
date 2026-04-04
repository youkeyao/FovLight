"""训练入口脚本。

该文件负责单帧数据训练流程：构建数据集、执行两阶段训练、保存与恢复检查点。
"""

import os
import torch
from torch.utils.data import ConcatDataset
from torchvision.utils import save_image
import json
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
from EnvMapDatasets import EnvMapDatasets
from LightEstimationModel import LightingEstimationModel
from ObjectRenderer import ObjectRenderer
from utils import create_projection_matrix

class LogL2Loss(torch.nn.Module):
    """基于 log1p 的加权 L2 损失。

    对高亮区域与普通区域使用不同权重，提升强光区域拟合稳定性。
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        log_input = torch.log1p(pred)
        log_target = torch.log1p(target)
        # 亮区掩码：任一通道强度超过阈值即视作高亮区域。
        mask = (target > 0.9).any(dim=0)
        w = 0.7

        return torch.sum(w * mask * ((log_input - log_target) ** 2) + (1 - w) * (~mask) * ((log_input - log_target) ** 2))

def train(model, projection_matrix, train_loader, criterion, optimizer, accelerator, use_detailed):
    """执行一个训练轮次并返回平均损失。"""
    total_loss = 0
    object_renderer = ObjectRenderer((256, 256))
    render_pos = torch.tensor([0, 0, -0.5])
    # 逐批训练：网络预测 + 渲染一致性约束。
    for batch in train_loader:
        image = batch['image']
        depth = batch['depth']
        lighting = batch['lighting']
        pos = batch["pos"]

        # 前向推理并计算联合损失。
        optimizer.zero_grad()
        output, _ = model(pos, projection_matrix, torch.eye(4), image, depth, use_detailed, True)
        loss = 0
        for b in range(output.shape[0]):
            render_old = torch.from_numpy(object_renderer.render_sphere(render_pos, lighting[b]))
            render_new = torch.from_numpy(object_renderer.render_sphere(render_pos, output[b]))
            loss += criterion(output, lighting) + criterion(render_new, render_old)

        accelerator.backward(loss)
        optimizer.step()

        total_loss += loss
    return total_loss / len(train_loader)

def freeze_parameters(model, layer_names):
    """冻结指定名字子串匹配到的参数。"""
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = False
            if hasattr(param, 'data'):
                param.data = param.data.detach()

def main(checkpoint_dir, model_param):
    """训练主函数：初始化组件、加载断点、分阶段训练。"""
    batch_size = 1
    learning_rate = 1e-4
    pre_epochs = 150
    num_epochs = 300
    saving_interval = 100

    accelerator = Accelerator(
        log_with="tensorboard",
        project_dir="./logs",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    )
    accelerator.init_trackers(project_name=os.path.basename(checkpoint_dir))
    device = accelerator.device
    print(f"Using device: {device}")

    projection_matrix = create_projection_matrix(120, 832/400, 0.1, 20).to(device)
    model = LightingEstimationModel(*model_param).to(device)
    # 创建数据集
    dataset = EnvMapDatasets(root_dir='/mnt/data/youkeyao/Datasets/LightEnv', image_resolution=(400, 832), lighting_resolution=(320, 640))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = LogL2Loss()
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # 加载检查点
    start_epoch = 0
    if os.path.exists(checkpoint_dir):
        accelerator.load_state(checkpoint_dir)
        with open(os.path.join(checkpoint_dir, "extra_state.json"), "r") as f:
            extra_state = json.load(f)
            start_epoch = extra_state["epoch"]
        accelerator.print(f"load from {checkpoint_dir} {start_epoch}")
    else:
        accelerator.print(f"start from {checkpoint_dir} 0")

    # 训练低频
    model.train()
    for epoch in range(start_epoch, pre_epochs):
        train_loss = train(model, projection_matrix, tqdm(dataloader, desc=f"{device} {checkpoint_dir} Train Epoch {epoch + 1}/{num_epochs}", unit="batch"), criterion, optimizer, accelerator, False)
        if accelerator.is_main_process:
            accelerator.print(f'{device} {checkpoint_dir} Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}')
            accelerator.log({"train/epoch_loss": train_loss}, step=epoch+1)
            # 保存检查点
            if (epoch+1) % saving_interval == 0:
                accelerator.save_state(checkpoint_dir)
                with open(os.path.join(checkpoint_dir, "extra_state.json"), "w") as f:
                    json.dump({"epoch": epoch+1}, f)

    # 训练高频
    start_epoch = max(start_epoch, pre_epochs)
    freeze_parameters(model, ["sglv_encoder_decoder"])
    for epoch in range(start_epoch, num_epochs):
        train_loss = train(model, projection_matrix, tqdm(dataloader, desc=f"{device} Train Epoch {epoch + 1}/{num_epochs}", unit="batch"), criterion, optimizer, accelerator, True)
        if accelerator.is_main_process:
            accelerator.print(f'{device} {checkpoint_dir} Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}')
            accelerator.log({"train/epoch_loss": train_loss}, step=epoch+1)
            # 保存检查点
            if (epoch+1) % saving_interval == 0:
                accelerator.save_state(checkpoint_dir)
                with open(os.path.join(checkpoint_dir, "extra_state.json"), "w") as f:
                    json.dump({"epoch": epoch+1}, f)

if __name__ == "__main__":
    # main("checkpoints/voxel_0", ((168, 120, 128), (320, 640), 100, 4))
    # main("checkpoints/voxel_1", ((151, 108, 115), (320, 640), 100, 4))
    # main("checkpoints/voxel_2", ((134, 96, 102), (320, 640), 100, 4))
    # main("checkpoints/voxel_3", ((118, 84, 90), (320, 640), 100, 4))
    # main("checkpoints/voxel_4", ((101, 72, 77), (320, 640), 100, 4))
    # main("checkpoints/voxel_5", ((84, 60, 64), (320, 640), 100, 4))
    # main("checkpoints/voxel_6", ((67, 48, 51), (320, 640), 100, 4))
    # main("checkpoints/voxel_7", ((50, 36, 38), (320, 640), 100, 4))
    # main("checkpoints/voxel_8", ((34, 24, 26), (320, 640), 100, 4))
    # main("checkpoints/voxel_9", ((17, 12, 13), (320, 640), 100, 4))

    main("checkpoints/network_2", ((84, 60, 64), (320, 640), 100, 1))
    main("checkpoints/network_1", ((84, 60, 64), (320, 640), 100, 2))
    main("checkpoints/network_0", ((84, 60, 64), (320, 640), 100, 3))

    main("checkpoints/voxel_5", ((84, 60, 64), (320, 640), 100, 4))
    main("checkpoints/voxel_6", ((67, 48, 51), (320, 640), 100, 4))
    main("checkpoints/voxel_7", ((50, 36, 38), (320, 640), 100, 4))
    main("checkpoints/voxel_8", ((34, 24, 26), (320, 640), 100, 4))
    main("checkpoints/voxel_9", ((17, 12, 13), (320, 640), 100, 4))

    # main("checkpoints/voxel", ((84, 60, 64), (320, 640), 100, True))
    # main("checkpoints/resolution", ((168, 120, 128), (80, 160), 100, True))
    # main("checkpoints/sample", ((168, 120, 128), (320, 640), 50, True))
    # main("checkpoints/network", ((168, 120, 128), (320, 640), 100, False))