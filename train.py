import os
import torch
from tqdm import tqdm
from KePanoLighting import KePanoLighting
from LightEstimationModel import LightingEstimationModel
from utils import get_free_gpu, create_projection_matrix

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
def train(model, projection_matrix, train_loader, criterion, optimizer):
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
            image = batch['image'][0]
            depth = batch['depth'][0]
            lighting = batch['lighting'][0]
            lighting_x = batch['lighting_x'][0]
            lighting_y = batch['lighting_y'][0]

            # 计算3D坐标
            fx = projection_matrix[0, 0]
            fy = projection_matrix[1, 1]
            cx = projection_matrix[0, 2]
            cy = projection_matrix[1, 2]
            Z = depth[:, lighting_x, lighting_y]
            X = (lighting_x - cx) * Z / fx
            Y = (lighting_y - cy) * Z / fy
            origin = torch.stack([X, Y, -Z], dim=-1)

            output = model(origin, projection_matrix, image, depth)
            loss = criterion(output, lighting)
            total_loss += loss.item()
    return total_loss / len(val_loader)

if __name__ == "__main__":
    batch_size = 1
    learning_rate = 0.001
    val_split = 0.2
    num_epochs = 500
    checkpoint_dir = './checkpoints'
    # 创建检查点目录
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # 选择GPU
    selected_gpu = get_free_gpu()
    device = torch.device("cpu" if selected_gpu is None else f"cuda:{selected_gpu}")
    print(f"Using device: {device}")
    # 训练
    model = LightingEstimationModel().to(device)
    criterion = torch.nn.MSELoss()
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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    start_epoch = load_checkpoint(model, optimizer, checkpoint_dir, device)
    # 训练和验证
    for epoch in range(start_epoch, num_epochs):
        train_loss = train(model, projection_matrix, tqdm(train_loader, desc=f"Train Epoch {epoch + 1}/{num_epochs}", unit="batch"), criterion, optimizer)
        val_loss = validate(model, tqdm(val_loader, desc=f"Val Epoch {epoch + 1}/{num_epochs}", unit="batch"), criterion)
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 保存检查点
        if (epoch+1) % 1 == 0:
            save_checkpoint(model, optimizer, epoch + 1, checkpoint_dir)