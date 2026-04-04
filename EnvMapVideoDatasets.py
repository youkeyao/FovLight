"""视频环境光数据集读取模块。"""

import os
from glob import glob
import re
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class EnvMapVideoDatasets(Dataset):
    """读取场景视频帧、深度序列、位姿与目标环境图。"""

    def __init__(self, root_dir):
        self.root_dir = root_dir
        scenes = sorted(glob(os.path.join(root_dir, "scene_*")))
        self.data_paths = []
        for scene_path in scenes:
            pattern = re.compile(
                r'^'                # 开头
                r'([-+]?\d*\.?\d+)' # 第一个浮点数（可选符号，可选整数部分，可选小数部分）
                r'_'                # 下划线分隔符
                r'([-+]?\d*\.?\d+)' # 第二个浮点数
                r'_'                # 下划线分隔符
                r'([-+]?\d*\.?\d+)' # 第三个浮点数
                r'\.exr$'           # .png 扩展名
            )
            for filename in os.listdir(scene_path):
                if not filename.lower().endswith('.exr'):
                    continue
                match = pattern.match(filename)
                if match:
                    try:
                        x = float(match.group(1))
                        y = float(match.group(2))
                        z = float(match.group(3))
                        self.data_paths.append((scene_path, filename, np.array([x, y, z], dtype=np.float32)))
                    except ValueError:
                        continue

    def __len__(self):
        return len(self.data_paths)

    def load_camera_matrices(self, path):
        """从文本文件读取逐帧 4x4 相机矩阵。"""
        matrices = []
        current = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    if current:
                        matrices.append(np.array(current, dtype=np.float32))
                        current = []
                    continue
                row = [float(x) for x in line.split()]
                current.append(row)
            if current:
                matrices.append(np.array(current, dtype=np.float32))
        return np.stack(matrices)

    def __getitem__(self, idx):
        """按索引返回一个完整视频样本。"""
        scene_path, filename, position = self.data_paths[idx]

        # --- 读取相机姿态矩阵 ---
        cam_file = os.path.join(scene_path, "camera_matrix.txt")
        poses = self.load_camera_matrices(cam_file)

        # --- 打开视频 ---
        rgb_cap = cv2.VideoCapture(os.path.join(scene_path, "video.mp4"))
        depth_paths = sorted(glob(os.path.join(scene_path, "depth", "frame*")))

        n_rgb = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_depth = len(depth_paths)
        n_frames = min(len(poses), n_rgb, n_depth)

        # --- 逐帧读取 ---
        rgb_frames, depth_frames = [], []
        for i in range(n_frames):
            ret_rgb, rgb = rgb_cap.read()
            if not ret_rgb:
                break

            # 深度
            depth = cv2.imread(depth_paths[i], -1)[:, :, 0:1].astype(np.float32)
            depth = torch.from_numpy(depth).permute(2, 0, 1)

            # 读取并转换 RGB 帧。
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

            rgb_frames.append(rgb)
            depth_frames.append(depth)

        rgb_cap.release()

        # --- 环境图 ---
        lighting = cv2.imread(os.path.join(scene_path, filename), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0:3][:, :, ::-1].astype(np.float32)

        # --- 堆叠为张量序列 ---
        rgb_tensor = torch.stack(rgb_frames)     # [N,3,H,W]
        depth_tensor = torch.stack(depth_frames) # [N,1,H,W]
        pose_tensor = torch.from_numpy(poses[:len(rgb_frames)]).float()  # [N,4,4]
        lighting = torch.from_numpy(lighting).permute(2, 0, 1)  # [N,3,H,W]
        position = torch.from_numpy(position)                  # [N,3]

        sample = {
            "rgb": rgb_tensor,
            "depth": depth_tensor,
            "pose": pose_tensor,
            "lighting": lighting,
            "position": position
        }
        return sample

if __name__ == "__main__":
    dataset = EnvMapVideoDatasets("/mnt/data/youkeyao/Datasets/EnvMapVideo")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    for batch in dataloader:
        rgb = batch['rgb'][0]
        depth = batch['depth'][0]
        pose = batch['pose'][0]
        lighting = batch["lighting"][0]
        position = batch["position"][0]

        n_frames = rgb.shape[0]
        for i in range(n_frames):
            rgb_np = rgb[i].permute(1, 2, 0).numpy()[:, :, ::-1]
            depth_np = depth[i].repeat(3, 1, 1).permute(1, 2, 0).numpy()
            depth_np /= np.max(depth_np)
            lighting_np = lighting.permute(1, 2, 0).numpy()[:, :, ::-1]
            print(pose[i])
            cv2.imshow("rgb", rgb_np)
            cv2.imshow("depth", depth_np)
            cv2.imshow("lighting", lighting_np)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
        if key == ord('q'):
                break
    cv2.destroyAllWindows()