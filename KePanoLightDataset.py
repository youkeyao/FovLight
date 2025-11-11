import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

def fov_to_focal_length(fov_deg, sensor_width_pixels):
    fov_rad = np.deg2rad(fov_deg)
    return sensor_width_pixels / (2 * np.tan(fov_rad / 2))

def equirectangular_to_perspective(input_shape=(512, 256), target_shape=(440, 385), h_fov_deg=120, yaw=0, pitch=0, roll=0):
    # 输入参数检查
    # assert len(img.shape) == 3, "输入图像需为RGB格式"
    W, H = input_shape
    assert W == 2 * H, "全景图宽高比需为2:1"

    # 目标图像尺寸 (宽, 高)
    out_w, out_h = target_shape
    focal = fov_to_focal_length(h_fov_deg, out_w)

    # 生成目标图像的像素网格
    x = np.arange(out_w) - out_w / 2
    y = np.arange(out_h) - out_h / 2
    x, y = np.meshgrid(x, y)
    z = focal * np.ones_like(x)

    # 构建3D坐标并归一化
    xyz = np.stack([x, y, z], axis=-1)
    xyz = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)

    # 应用旋转矩阵 (Yaw-Pitch-Roll顺序)
    yaw_rad = np.deg2rad(yaw)
    pitch_rad = np.deg2rad(pitch)
    roll_rad = np.deg2rad(roll)
    
    # Yaw (绕Y轴旋转)
    R_yaw = np.array([
        [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
        [0, 1, 0],
        [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
    ])
    
    # Pitch (绕X轴旋转)
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])
    
    # Roll (绕Z轴旋转)
    R_roll = np.array([
        [np.cos(roll_rad), -np.sin(roll_rad), 0],
        [np.sin(roll_rad), np.cos(roll_rad), 0],
        [0, 0, 1]
    ])
    
    # 组合旋转矩阵
    R = R_roll @ R_pitch @ R_yaw
    xyz_rot = xyz @ R.T

    # 转换为球面坐标 (经度, 纬度)
    x_rot, y_rot, z_rot = xyz_rot[..., 0], xyz_rot[..., 1], xyz_rot[..., 2]
    lon = np.arctan2(x_rot, z_rot)
    lat = np.arcsin(y_rot)

    # 将经纬度映射到全景图像素坐标
    u = (lon + np.pi) / (2 * np.pi) * W
    v = (lat + np.pi/2) / np.pi * H

    # 使用双线性插值采样
    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)

    return (map_x, map_y, xyz_rot)

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

def probe_img2channel(probe_hdr, env_h, pano_h, pano_w):
    """translate probe tensor 2 channel tensor

    Args:
        probe_hdr ([type]): [shape: (3,env_h*pano_h, env_w*pano_w)]
        env_h ([type]): [height of every probe]
        pano_h ([type]): [height of rendered panorama]

    Returns:
        [type]: [shape: (3*env_h*env_w, pano_h, pano_w)]
    """
    c,h,w = probe_hdr.shape
    probe_hdr = probe_hdr.reshape(3, pano_h, env_h, pano_w, env_h*2)
    probe_hdr = probe_hdr.permute(0,2,4,1,3)

    return probe_hdr

def probe_channel2img(probe_channels, env_h, pano_h, pano_w):
    """translate channel tensor 2 probe tensor

    Args:
        probe_channels ([type]): [shape: (3*env_h*env_w, pano_h, pano_w)]
        env_h ([type]): [height of every probe]
        pano_h ([type]): [height of rendered panorama]

    Returns:
        [type]: [shape: (3,env_h*pano_h, env_w*pano_w)]
    """
    c,h,w = probe_channels.shape
    probe_channels = probe_channels.reshape(3,env_h, int(env_h*2), pano_h, pano_w)
    probe_channels = probe_channels.permute(0,3,1,4,2).reshape(3, env_h*pano_h, int(env_h*2)*pano_w)

    return probe_channels

class KePanoLightDataset(Dataset):
    """read pano data, include pano lighting.
    it's format is .hdr.
    example['image'],
    example['albedo'],
    example['normal'],
    example['roughness'],
    example['metallic'],
    example['depth'],
    example['mask']
    """
    def __init__(self, root, fov=120, image_resolution=(385, 440), lighting_resolution=(160, 320)):
        super().__init__()
        
        self.root = root
        self.pano_h = 256
        self.pano_w = int(self.pano_h*2)
        self.probes_h = 128
        self.probes_w = int(self.probes_h*2)
        self.env_h = 16
        self.pos_size = (2, 3)
        self.image_resolution = image_resolution
        self.lighting_resolution = lighting_resolution

        self.map_x, self.map_y, self.xyz_rot = equirectangular_to_perspective((self.pano_w, self.pano_h), (image_resolution[1], image_resolution[0]), fov)

        self.max_depth = 5   # 10 m, norm [0,1]

        self.all_item = self.read_all_item(root)
        self.is_random_exposure = False

    def __getitem__(self, index):
        one_item = self.all_item[index]
        one_path = one_item[0]
        iindex = one_item[1]
        xindex, yindex = one_item[2]
        
        # range: [-2,-0.5)
        # if self.is_random_exposure:
        #     random_exposure = torch.rand(1)*1.5 - 2.0
        # else:
        #     random_exposure = -1.0

        lighting = cv2.imread(os.path.join(one_path,str(iindex)+'_light.exr'),-1)[..., ::-1].astype(np.float32)
        lighting = torch.from_numpy(lighting).permute(2,0,1)
        lighting = probe_img2channel(lighting, self.env_h, self.probes_h, self.probes_w)
        # light = light * torch.pow(torch.tensor(2.0),random_exposure)
        lighting_x = int(self.map_x[yindex, xindex] / self.pano_w * self.probes_w)
        lighting_y = int(self.map_y[yindex, xindex] / self.pano_h * self.probes_h)
        lighting = lighting[:, :, :, lighting_y, lighting_x]
        lighting = torch.roll(lighting, self.env_h, dims=2)
        lighting = F.interpolate(lighting.unsqueeze(0), size=self.lighting_resolution, mode='bilinear', align_corners=False)[0]
        lighting = np.where(
            lighting <= 0.04045,
            lighting / 12.92,
            ((lighting + 0.055) / 1.055) ** 2.4
        )
        # light = tonemapper(light)

        one_path = one_path.replace('LightProbeData','CubemapData').replace('KePanoLight','KePanoData')

        image = cv2.imread(os.path.join(one_path,str(iindex)+'_image.hdr'),-1)[:,:,0:3][:, :, ::-1].astype(np.float32)
        image = cv2.resize(image,(self.pano_w,self.pano_h))
        image = cv2.remap(image, self.map_x, self.map_y, cv2.INTER_LINEAR)
        image = np.where(
            image <= 0.04045,
            image / 12.92,
            ((image + 0.055) / 1.055) ** 2.4
        )
        # image = torch.from_numpy(image)
        # image = image.permute(2,0,1)
        # image = image * torch.pow(torch.tensor(2.0),random_exposure)
        # image = tonemapper(image,mode=1)    # ACES tonemapping

        # albedo = cv2.imread(os.path.join(one_path,str(iindex)+'_albedo.hdr'),-1)[:,:,0:3]
        # albedo = cv2.resize(albedo,(self.pano_w,self.pano_h))
        # albedo = np.asarray(albedo,dtype=np.float32)
        # albedo = albedo[...,::-1].copy()
        # albedo = torch.from_numpy(albedo)
        # albedo = albedo.permute(2,0,1)
        # albedo = tonemapper(albedo)

        # roughness = cv2.imread(os.path.join(one_path,str(iindex)+'_roughness.hdr'),-1)[:,:,0:1]
        # roughness = cv2.resize(roughness,(self.pano_w,self.pano_h))
        # roughness = np.asarray(roughness,dtype=np.float32)
        # roughness = torch.from_numpy(roughness)
        # roughness = roughness.unsqueeze(0)

        # metallic = cv2.imread(os.path.join(one_path,str(iindex)+'_metallic.hdr'),-1)[:,:,0:1]
        # metallic = cv2.resize(metallic,(self.pano_w,self.pano_h))
        # metallic = np.asarray(metallic,dtype=np.float32)
        # metallic = torch.from_numpy(metallic)
        # metallic = metallic.unsqueeze(0)

        # mask = cv2.imread(os.path.join(one_path,str(iindex)+'_mask.hdr'),-1)[:,:,0:1]
        # mask = cv2.resize(mask,(self.pano_w,self.pano_h))
        # mask = np.asarray(mask,dtype=np.float32)
        # mask = torch.from_numpy(mask)
        # mask = mask.unsqueeze(0)

        # normal = cv2.imread(os.path.join(one_path,str(iindex)+'_normal.hdr'),-1)[:,:,0:3]
        # normal = cv2.resize(normal,(self.pano_w,self.pano_h))
        # normal = np.asarray(normal,dtype=np.float32)
        # normal = normal[...,::-1].copy()
        # normal = (normal*2.0)-1.0
        # normal = torch.from_numpy(normal)
        # normal = normal.permute(2,0,1)

        depth = cv2.imread(os.path.join(one_path,str(iindex)+'_depth.hdr'),-1)[:,:,0:1].astype(np.float32)
        depth = cv2.resize(depth,(self.pano_w,self.pano_h))
        depth = cv2.remap(depth, self.map_x, self.map_y, cv2.INTER_LINEAR)
        pos = np.array([self.xyz_rot[yindex, xindex] * depth[yindex, xindex]])
        pos[0, 2] *= -1
        depth *= self.xyz_rot[..., 2]
        depth = depth.reshape(1, self.image_resolution[0], self.image_resolution[1])
        # depth = torch.from_numpy(depth)
        # depth = depth.unsqueeze(0)

        # depth_mask = (depth>0) & (depth<=self.max_depth) & (~torch.isnan(depth))

        name = one_path.split('/')[-3]+"_"+str(iindex)

        batchDict = {
            'name': name,
            'image': torch.from_numpy(image).permute(2, 0, 1),
            'depth': torch.from_numpy(depth),
            'lighting': lighting,
            'pos': torch.tensor(pos, dtype=torch.float32)
            # 'uv':torch.tensor([2 * (xindex / self.out_w) - 1, -2 * (yindex / self.out_h) + 1]),
        }
        
        return batchDict

    def __len__(self):
        return len(self. all_item)

    def read_all_item(self, root):
        all_item = []
        for id in os.listdir(root):
            if not os.path.exists(os.path.join(root,id)):
                continue
            if not os.path.exists(os.path.join(root.replace('KePanoLight','KePanoData'),id)):
                continue
            whole_path = os.path.join(root,id,'ue4_result','LightProbeData')
            # -----------filter----------------------------
            data_path = whole_path.replace('LightProbeData','CubemapData').replace('KePanoLight','KePanoData')
            depth = cv2.imread(os.path.join(data_path,'0_depth.hdr'),-1)[:,:,0:1]
            depth = cv2.resize(depth,(self.pano_w,self.pano_h))
            depth = np.asarray(depth,dtype=np.float32)
            depth = cv2.remap(depth, self.map_x, self.map_y, cv2.INTER_LINEAR)
            depth *= self.xyz_rot[..., 2]
            if np.max(depth) > self.max_depth or np.min(depth) < 0.15:
                continue
            # if np.var(depth) < 0.01:
            #     continue
            # -----------------------------------------------
            items = os.listdir(whole_path)
            if (len(items)) != 1:
                print(id)
                continue
            num = len(items)
            for i in range(num):
                for x in range(2, self.pos_size[1] + 2):
                    for y in range(2, self.pos_size[0] + 2):
                        one_item = []
                        one_item.append(whole_path)
                        one_item.append(i)
                        one_item.append(((int)(x * self.image_resolution[1] / (self.pos_size[1] + 2)), (int)(y * self.image_resolution[0] / (self.pos_size[0] + 2))))
                        all_item.append(one_item)
        return all_item

# Example usage
if __name__ == "__main__":
    dataset = KePanoLightDataset(root='/mnt/data/youkeyao/Datasets/FutureHouse/KePanoLight')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    for batch in dataloader:
        image = batch['image'][0]
        depth = batch['depth'][0]
        lighting = batch['lighting'][0]
        pos = batch["pos"][0]

        image_np = image.permute(1, 2, 0).numpy()[:, :, ::-1]
        depth_np = depth.repeat(3, 1, 1).permute(1, 2, 0).numpy()[:, :, ::-1]
        print(np.max(depth_np))
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