"""单帧环境光数据集读取模块。"""

import os
import re
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class EnvMapDatasets(Dataset):
    """读取静态场景图像、深度和环境贴图。"""

    def __init__(self, root_dir, image_resolution=(240, 320), lighting_resolution=(160, 320)):
        """初始化数据集路径与输出分辨率。"""
        self.image_resolution = image_resolution
        self.lighting_resolution = lighting_resolution
        self.root_dir = root_dir
        self.data_paths = self.load_data_paths()

    def load_data_paths(self):
        """扫描目录并收集样本路径与相机位置。"""
        data_paths = []
        pattern = re.compile(
            r'^'                # 开头
            r'([-+]?\d*\.?\d+)' # 第一个浮点数（可选符号，可选整数部分，可选小数部分）
            r'_'                # 下划线分隔符
            r'([-+]?\d*\.?\d+)' # 第二个浮点数
            r'_'                # 下划线分隔符
            r'([-+]?\d*\.?\d+)' # 第三个浮点数
            r'\.exr$'           # exr 扩展名
        )
        for scene in os.listdir(self.root_dir):
            image_path = os.path.join(self.root_dir, scene, "image.png")
            depth_path = os.path.join(self.root_dir, scene, "depth.exr")
            for filename in os.listdir(os.path.join(self.root_dir, scene)):
                if not filename.lower().endswith('.exr'):
                    continue
                match = pattern.match(filename)
                if match:
                    try:
                        x = float(match.group(1))
                        y = float(match.group(2))
                        z = float(match.group(3))
                        lighting_path = os.path.join(self.root_dir, scene, filename)
                        data_paths.append((image_path, depth_path, lighting_path, (x, y, z)))
                    except ValueError:
                        continue
        return data_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        """按索引返回一个训练样本字典。"""
        scene_data = self.data_paths[idx]
        image_path = scene_data[0]
        depth_path = scene_data[1]
        lighting_path = scene_data[2]
        pos = scene_data[3]

        # 第一段：读取并缩放输入图像。
        image = cv2.imread(image_path, -1)[:, :, 0:3][:, :, ::-1].astype(np.float32) / 255
        image = cv2.resize(image, (self.image_resolution[1], self.image_resolution[0]))
        # 可选：sRGB 转线性空间。
        # image = np.where(
        #     image <= 0.04045,
        #     image / 12.92,
        #     ((image + 0.055) / 1.055) ** 2.4
        # )

        # 第二段：读取并缩放深度图。
        depth = cv2.imread(depth_path, -1)[:, :, 0:1].astype(np.float32)
        depth = cv2.resize(depth, (self.image_resolution[1], self.image_resolution[0])).reshape(self.image_resolution[0], self.image_resolution[1], 1)

        # 第三段：读取并缩放环境贴图。
        lighting = cv2.imread(lighting_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0:3][:, :, ::-1].astype(np.float32)
        lighting = cv2.resize(lighting, (self.lighting_resolution[1], self.lighting_resolution[0]))

        # 第四段：组装为训练样本字典（通道优先格式）。
        # image/depth/lighting 统一为 (C, H, W)。
        sample = {
            "name": os.path.join(*lighting_path.split(os.sep)[-2:]),
            'image': torch.from_numpy(image).permute(2, 0, 1),
            'depth': torch.from_numpy(depth).permute(2, 0, 1),
            'lighting': torch.from_numpy(lighting).permute(2, 0, 1),
            "pos": torch.tensor(pos)
        }

        return sample

# 使用示例
if __name__ == "__main__":
    dataset = EnvMapDatasets(root_dir='/mnt/data/youkeyao/Datasets/LightEnv', image_resolution=(400, 832), lighting_resolution=(320, 640))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    for batch in dataloader:
        name = batch['name'][0]
        image = batch['image'][0]
        depth = batch['depth'][0]
        lighting = batch['lighting'][0]
        pos = batch["pos"][0]

        image_np = image.permute(1, 2, 0).numpy()[:, :, ::-1]
        depth_np = depth.repeat(3, 1, 1).permute(1, 2, 0).numpy()
        depth_np /= np.max(depth_np)
        combined_image = np.hstack((image_np, depth_np))
        lighting_np = lighting.permute(1, 2, 0).numpy()[:, :, ::-1]

        cv2.imshow("Image and Depth Map", combined_image)
        cv2.imshow("Lighting map", lighting_np)
        print(name, pos)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()