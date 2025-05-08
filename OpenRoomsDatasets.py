import os
import cv2
import struct
import numpy as np
import torch
from torch.utils.data import Dataset

def tonemapper(exr,mode=0):
    if mode ==0:
        return torch.pow(torch.clamp(exr,0.0,1.0), (1/2.2))
    elif mode == 1:
        A = 2.51
        B = 0.03
        C = 2.43
        D = 0.59
        E = 0.14
        return torch.pow(torch.clamp(( (exr * (A * exr + B)) / (exr * (C * exr + D) + E) ),0.0,1.0), (1/2.2))

class OpenRoomsDataset(Dataset):
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
        scenes_root_dir = os.path.join(self.root_dir, "Image", "main_xml")
        scenes = os.listdir(scenes_root_dir)
        return scenes

    def _load_data_paths(self):
        """Load paths to all data files for the specified scenes."""
        data_paths = []
        image_xmls = ["main_xml", "main_xml1", "mainDiffLight_xml", "mainDiffLight_xml1", "mainDiffMat_xml", "mainDiffMat_xml1"]
        depth_xmls = ["main_xml", "main_xml1", "main_xml", "main_xml1", "main_xml", "main_xml1"]
        lighting_xmls = ["main_xml", "main_xml1", "mainDiffLight_xml", "mainDiffLight_xml1", "mainDiffMat_xml", "mainDiffMat_xml1"]
        for xml in range(len(image_xmls)):
            for scene in self.scenes:
                image_dir = os.path.join(self.root_dir, 'ImageSDR', image_xmls[xml], scene)
                depth_dir = os.path.join(self.root_dir, 'Geometry', depth_xmls[xml], scene)
                lighting_dir = os.path.join(self.root_dir, 'SVLighting', lighting_xmls[xml], scene)
                if os.path.exists(image_dir):
                    for i in range(1, len(os.listdir(lighting_dir))+1):
                        image_path = os.path.join(image_dir, f'im_{i}.png')
                        depth_path = os.path.join(depth_dir, f'imdepth_{i}.dat')
                        lighting_path = os.path.join(lighting_dir, f'imenv_{i}.hdr')
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
            'lighting': torch.from_numpy(lighting) * torch.pow(torch.tensor(2.0), -1.0)
        }

        return sample

# Example usage
if __name__ == "__main__":
    dataset = OpenRoomsDataset(root_dir='/mnt/data/youkeyao/Datasets/OpenRooms/releasingData')
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
                lighting_nps.append(lightings[0][x, y, :, :, :].permute(1, 2, 0).numpy()[:, :, ::-1])

        cv2.imshow("Image and Depth Map", combined_image)
        for i in range(len(lighting_nps)):
            cv2.imshow("Lighting map", lighting_nps[i])
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()