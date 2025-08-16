import os
import torch
import json
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
from EnvMapDatasets import EnvMapDatasets
from LightEstimationModel import LightingEstimationModel
from ObjectRenderer import ObjectRenderer
from utils import create_projection_matrix

class LogL2Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        log_input = torch.log1p(pred)
        log_target = torch.log1p(target)

        return torch.sum((log_input - log_target) ** 2)

def train(model, projection_matrix, train_loader, criterion, optimizer, accelerator, use_detailed):
    total_loss = 0
    object_renderer = ObjectRenderer((256, 256))
    render_pos = torch.tensor([0, 0, -0.5])
    for batch in train_loader:
        image = batch['image'][0]
        depth = batch['depth'][0]
        lighting = batch['lighting'][0]
        pos = batch["pos"][0]

        optimizer.zero_grad()
        output = model(pos, projection_matrix, image, depth, use_detailed)
        render_old, mask_old = object_renderer.render_sphere(render_pos, lighting)
        render_new, mask_new = object_renderer.render_sphere(render_pos, output)
        loss = criterion(output, lighting) + criterion(render_new * mask_new, render_old * mask_old)

        accelerator.backward(loss)
        optimizer.step()

        total_loss += loss
    return total_loss / len(train_loader)

def freeze_parameters(model, layer_names):
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = False
            if hasattr(param, 'data'):
                param.data = param.data.detach()

def main(checkpoint_dir, model_param):
    batch_size = 1
    learning_rate = 2e-4
    num_epochs = 1000
    detailed_epochs = 1500

    accelerator = Accelerator(
        log_with="tensorboard",
        project_dir="./logs",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    )
    accelerator.init_trackers(project_name=os.path.basename(checkpoint_dir))
    device = accelerator.device
    print(f"Using device: {device}")

    projection_matrix = create_projection_matrix(39.6, 640/480, 0.1, 20).to(device)
    model = LightingEstimationModel(*model_param).to(device)
    # 创建数据集
    dataset = EnvMapDatasets(root_dir='/mnt/data/youkeyao/Datasets/LightEnv', image_resolution=(240, 320), lighting_resolution=model.output_resolution)
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
        print(f"load from {start_epoch}")

    # 训练低频
    model.train()
    for epoch in range(start_epoch, num_epochs):
        train_loss = train(model, projection_matrix, tqdm(dataloader, desc=f"{device} Train Epoch {epoch + 1}/{num_epochs}", unit="batch"), criterion, optimizer, accelerator, False)
        if accelerator.is_main_process:
            accelerator.print(f'{device} Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}')
            accelerator.log({"train/epoch_loss": train_loss}, step=epoch)
            # 保存检查点
            if (epoch+1) % 100 == 0:
                accelerator.save_state(checkpoint_dir)
                with open(os.path.join(checkpoint_dir, "extra_state.json"), "w") as f:
                    json.dump({"epoch": epoch+1}, f)

    # 训练高频
    start_epoch = max(start_epoch, num_epochs)
    freeze_parameters(model, ["sglv_encoder_decoder"])
    for epoch in range(start_epoch, detailed_epochs):
        train_loss = train(model, projection_matrix, tqdm(dataloader, desc=f"{device} Train Epoch {epoch + 1}/{detailed_epochs}", unit="batch"), criterion, optimizer, accelerator, True)
        if accelerator.is_main_process:
            accelerator.print(f'{device} Epoch [{epoch + 1}/{detailed_epochs}], Train Loss: {train_loss:.4f}')
            accelerator.log({"train/epoch_loss": train_loss}, step=epoch)
            # 保存检查点
            if (epoch+1) % 100 == 0:
                accelerator.save_state(checkpoint_dir)
                with open(os.path.join(checkpoint_dir, "extra_state.json"), "w") as f:
                    json.dump({"epoch": epoch+1}, f)

    accelerator.end_training()

if __name__ == "__main__":
    main("checkpoints/base", ((168, 120, 128), (160, 320), 100, True))
    main("checkpoints/voxel", ((84, 60, 64), (160, 320), 100, True))
    main("checkpoints/resolution", ((168, 120, 128), (80, 160), 100, True))
    main("checkpoints/sample", ((168, 120, 128), (160, 320), 50, True))
    main("checkpoints/network", ((168, 120, 128), (160, 320), 100, False))