import os
import cv2
import struct
import numpy as np
import torch
from torch.utils.data import Dataset

class InteriorverseDatasets(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the data.
        """
        self.root_dir = root_dir
        self.scenes = self._load_scene_list()
        self.data_paths = self._load_data_paths()

    def _load_scene_list(self):
        """Load the list of scenes"""
        scenes = os.listdir(self.root_dir)
        return scenes

    def _load_data_paths(self):
        """Load paths to all data files for the specified scenes."""
        data_paths = []
        for scene in self.scenes:
            scene_dir = os.path.join(self.root_dir, scene)
            batch_num = len(os.listdir(scene_dir)) / 6
            for i in range(batch_num):
                image_path = os.path.join(scene_dir, f'im_{i}.png')
                depth_path = os.path.join(scene_dir, f'imdepth_{i}.dat')
                lighting_path = os.path.join(scene_dir, f'imenv_{i}.hdr')
                data_paths.append((image_path, depth_path, lighting_path))
        return data_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        scene_data = self.data_paths[idx]
        image_path = scene_data[0]
        depth_path = scene_data[1]
        lighting_path = scene_data[2]

        # Load image
        image = cv2.imread(image_path)[:, :, ::-1].astype(np.float32) / 255

        # Load geometry depth
        with open(depth_path, 'rb') as f:
            height = struct.unpack('i', f.read(4))[0]
            width = struct.unpack('i', f.read(4))[0]
            depth_buffer = struct.unpack('f' * height * width, f.read(4 * height * width))
            depth = np.array(depth_buffer, dtype=np.float32).reshape(height, width)

        # Load lighting environment map
        lighting = cv2.imread(lighting_path, -1)[:, :, ::-1].astype(np.float32)
        lighting = lighting.reshape(120, 16, 160, 32, 3)
        lighting = lighting.transpose(0, 2, 4, 1, 3)

        # Create a sample dictionary
        # image: channel, height, width
        # depth: channel, height, width
        # lighting: spatial_height, spatial_width, channel, height, width
        sample = {
            'image': torch.from_numpy(image).permute(2, 0, 1),
            'depth': torch.from_numpy(depth).unsqueeze(0),
            'lighting': torch.from_numpy(lighting)
        }

        return sample

# Example usage
if __name__ == "__main__":
    dataset = InteriorverseDatasets(root_dir='/mnt/data/youkeyao/Datasets/OpenRooms/releasingData')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    for batch in dataloader:
        images = batch['image']
        depths = batch['depth']
        lightings = batch['lighting']

        image_np = images[0].permute(1, 2, 0).numpy()[:, :, ::-1]
        depth_np = depths[0].repeat(3, 1, 1).permute(1, 2, 0).numpy()[:, :, ::-1]
        depth_np /= np.max(depth_np)
        combined_image = np.hstack((image_np, depth_np))

        lighting_nps = []
        for x in range(120):
            for y in range(160):
                lighting_nps.append(lightings[0][x][y].permute(1, 2, 0).numpy()[:, :, ::-1])

        cv2.imshow("Image and Depth Map", combined_image)
        for i in range(len(lighting_nps)):
            cv2.imshow("Lighting map", lighting_nps[i])
            cv2.waitKey(0)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()