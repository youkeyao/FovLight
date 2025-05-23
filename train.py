import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from KePanoLighting import KePanoLighting
from LightEstimationModel import LightingEstimationModel
from utils import get_free_gpu, create_projection_matrix

class LogL2Loss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        l2 = torch.sum((pred - target)**2)
        return torch.log(l2 + self.eps)

# 保存检查点
def save_checkpoint(model, optimizer, epoch, save_dir):
    checkpoint_filename = f"model_checkpoint_{epoch}.pth"
    checkpoint_path = os.path.join(save_dir, checkpoint_filename)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

# 加载检查点
def load_checkpoint(model, optimizer, load_dir, device):
    checkpoint_files = [f for f in os.listdir(load_dir) if f.startswith("model_checkpoint_") and f.endswith(".pth")]
    if not checkpoint_files:
        print(f"No checkpoints found in {load_dir}")
        return 0
    
    # 获取最新的检查点文件
    checkpoint_epochs = [int(f.split('_')[-1].split('.')[0]) for f in checkpoint_files]
    latest_epoch = max(checkpoint_epochs)
    latest_checkpoint_file = f"model_checkpoint_{latest_epoch}.pth"
    load_path = os.path.join(load_dir, latest_checkpoint_file)

    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded checkpoint from {load_path}")
    return latest_epoch

# 训练函数
def train(model, projection_matrix, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        image = batch['image'][0].to(device)
        depth = batch['depth'][0].to(device)
        lighting = batch['lighting'][0].to(device)
        lighting_x = batch['lighting_x'][0]
        lighting_y = batch['lighting_y'][0]

        # 计算3D坐标
        fx = projection_matrix[0, 0]
        fy = projection_matrix[1, 1]
        cx = projection_matrix[0, 2]
        cy = projection_matrix[1, 2]
        Z = depth[:, lighting_y, lighting_x]
        X = (lighting_x - cx) * Z / fx
        Y = (lighting_y - cy) * Z / fy
        origin = torch.stack([X, Y, -Z], dim=-1)
        
        optimizer.zero_grad()
        output = model(origin, projection_matrix, image, depth)
        loss = criterion(output, lighting)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(train_loader)

# 验证函数
def validate(model, projection_matrix, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            image = batch['image'][0].to(device)
            depth = batch['depth'][0].to(device)
            lighting = batch['lighting'][0].to(device)
            lighting_x = batch['lighting_x'][0]
            lighting_y = batch['lighting_y'][0]

            # 计算3D坐标
            fx = projection_matrix[0, 0]
            fy = projection_matrix[1, 1]
            cx = projection_matrix[0, 2]
            cy = projection_matrix[1, 2]
            Z = depth[:, lighting_y, lighting_x]
            X = (lighting_x - cx) * Z / fx
            Y = (lighting_y - cy) * Z / fy
            origin = torch.stack([X, Y, -Z], dim=-1)

            output = model(origin, projection_matrix, image, depth)
            loss = criterion(output, lighting)
            total_loss += loss.item()
    return total_loss / len(val_loader)

if __name__ == "__main__":
    batch_size = 1
    learning_rate = 5e-4
    val_split = 0.2
    num_epochs = 500
    checkpoint_dir = './checkpoints'
    multi_gpu = True
    # 创建检查点目录
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # 选择GPU
    if multi_gpu:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ['LOCAL_RANK'])
        device = torch.device(f'cuda:{local_rank}')
    else:
        selected_gpu = get_free_gpu()
        device = torch.device("cpu" if selected_gpu is None else f"cuda:{selected_gpu}")
    print(f"Using device: {device}")
    # 训练
    model = LightingEstimationModel().to(device)
    if multi_gpu:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device)
    criterion = LogL2Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    projection_matrix = create_projection_matrix(57.9516, 640/480, 0.1, 100)
    # 创建数据集
    dataset = KePanoLighting(root='/mnt/data/youkeyao/Datasets/FutureHouse/KePanoLight')
    # 划分训练集和验证集
    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    # 创建数据加载器
    if multi_gpu:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    # 训练和验证
    start_epoch = load_checkpoint(model, optimizer, checkpoint_dir, device)
    for epoch in range(start_epoch, num_epochs):
        if multi_gpu:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
        train_loss = train(model, projection_matrix, tqdm(train_loader, desc=f"{device} Train Epoch {epoch + 1}/{num_epochs}", unit="batch"), criterion, optimizer, device)
        val_loss = validate(model, projection_matrix, tqdm(val_loader, desc=f"{device} Val Epoch {epoch + 1}/{num_epochs}", unit="batch"), criterion, device)

        print(f'{device} Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # 保存检查点
        if (epoch+1) % 1 == 0:
            if multi_gpu:
                if torch.distributed.get_rank() == 0:
                    save_checkpoint(model.module, optimizer, epoch + 1, checkpoint_dir)
            else:
                save_checkpoint(model, optimizer, epoch + 1, checkpoint_dir)

        if multi_gpu:
            torch.distributed.barrier()
    if multi_gpu:
        torch.distributed.destroy_process_group()