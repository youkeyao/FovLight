import os
import re
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class EnvMapDatasets(Dataset):
    def __init__(self, root_dir, image_resolution=(240, 320), lighting_resolution=(160, 320)):
        """
        Args:
            root_dir (string): Directory with all the data.
        """
        self.image_resolution = image_resolution
        self.lighting_resolution = lighting_resolution
        self.root_dir = root_dir
        self.data_paths = self.load_data_paths()

    def load_data_paths(self):
        """Load paths to all data files for the specified scenes."""
        data_paths = []
        pattern = re.compile(
            r'^'                # 开头
            r'([-+]?\d*\.?\d+)' # 第一个浮点数（可选符号，可选整数部分，可选小数部分）
            r'_'                # 下划线分隔符
            r'([-+]?\d*\.?\d+)' # 第二个浮点数
            r'_'                # 下划线分隔符
            r'([-+]?\d*\.?\d+)' # 第三个浮点数
            r'\.exr$'           # .png 扩展名
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
        scene_data = self.data_paths[idx]
        image_path = scene_data[0]
        depth_path = scene_data[1]
        lighting_path = scene_data[2]
        pos = scene_data[3]

        # Load image
        image = cv2.imread(image_path, -1)[:, :, 0:3][:, :, ::-1].astype(np.float32) / 255
        image = cv2.resize(image, (self.image_resolution[1], self.image_resolution[0]))
        # srgb to linear
        image = np.where(
            image <= 0.04045,
            image / 12.92,
            ((image + 0.055) / 1.055) ** 2.4
        )

        # Load depth
        depth = cv2.imread(depth_path, -1)[:, :, 0:1].astype(np.float32)
        depth = cv2.resize(depth, (self.image_resolution[1], self.image_resolution[0])).reshape(self.image_resolution[0], self.image_resolution[1], 1)

        # Load lighting environment map
        lighting = cv2.imread(lighting_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0:3][:, :, ::-1].astype(np.float32)
        lighting = cv2.resize(lighting, (self.lighting_resolution[1], self.lighting_resolution[0]))

        # Create a sample dictionary
        # image: channel, height, width
        # depth: channel, height, width
        # lighting: channel, height, width
        sample = {
            'image': torch.from_numpy(image).permute(2, 0, 1),
            'depth': torch.from_numpy(depth).permute(2, 0, 1),
            'lighting': torch.from_numpy(lighting).permute(2, 0, 1),
            "pos": torch.tensor(pos)
        }

        return sample

# Example usage
if __name__ == "__main__":
    dataset = EnvMapDatasets(root_dir='/mnt/data/youkeyao/Datasets/LightEnv', image_resolution=(480, 640), lighting_resolution=(80, 160))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    for batch in dataloader:
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
        print(pos)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()