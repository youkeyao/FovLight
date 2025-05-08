import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

def fov_to_focal_length(fov_deg, sensor_width_pixels):
    fov_rad = np.deg2rad(fov_deg)
    return sensor_width_pixels / (2 * np.tan(fov_rad / 2))

def equirectangular_to_perspective(input_shape=(256, 512), target_shape=(320, 240), h_fov_deg=57.9516, yaw=0, pitch=0, roll=0):
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

    return (map_x, map_y, z_rot)

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

class KePanoLighting(Dataset):
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
    def __init__(self, root, pano_h=256, env_h=16, probes_h=128, out_w=320, out_h=240, is_random_exposure=True):
        super().__init__()
        
        self.root = root
        self.pano_h = pano_h
        self.pano_w = int(pano_h*2)
        self.probes_h = probes_h
        self.probes_w = int(probes_h*2)
        self.env_h = env_h
        self.env_w = int(env_h*2)
        self.out_h=out_h
        self.out_w=out_w

        self.max_depth = 10   # 10 m, norm [0,1]

        self.all_item = self.read_all_item(root)
        self.is_random_exposure = is_random_exposure

    def __getitem__(self, index):
        one_item = self.all_item[index]
        one_path = one_item[0]
        iindex = one_item[1]
        
        # range: [-2,-0.5)
        if self.is_random_exposure:
            random_exposure = torch.rand(1)*1.5 - 2.0
        else:
            random_exposure = -1.0

        map_x, map_y, z_rot = equirectangular_to_perspective((self.pano_w, self.pano_h), (self.out_w, self.out_h))

        light = cv2.imread(os.path.join(one_path,str(iindex)+'_light.exr'),-1)
        light = np.asarray(light,dtype=np.float32)
        light = light[...,::-1].copy()
        light = torch.from_numpy(light)
        light = light.permute(2,0,1)
        light = probe_img2channel(light,self.env_h,self.probes_h,self.probes_w)
        light = light * torch.pow(torch.tensor(2.0),random_exposure)

        one_path = one_path.replace('LightProbeData','CubemapData').replace('KePanoLight','KePanoData')

        image = cv2.imread(os.path.join(one_path,str(iindex)+'_image.hdr'),-1)[:,:,0:3]
        image = cv2.resize(image,(self.pano_w,self.pano_h))
        image = np.asarray(image,dtype=np.float32)
        image = image[...,::-1].copy()
        image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        image = torch.from_numpy(image)
        image = image.permute(2,0,1)
        image = image * torch.pow(torch.tensor(2.0),random_exposure)
        image = tonemapper(image,mode=1)    # ACES tonemapping

        albedo = cv2.imread(os.path.join(one_path,str(iindex)+'_albedo.hdr'),-1)[:,:,0:3]
        albedo = cv2.resize(albedo,(self.pano_w,self.pano_h))
        albedo = np.asarray(albedo,dtype=np.float32)
        albedo = albedo[...,::-1].copy()
        albedo = torch.from_numpy(albedo)
        albedo = albedo.permute(2,0,1)

        roughness = cv2.imread(os.path.join(one_path,str(iindex)+'_roughness.hdr'),-1)[:,:,0:1]
        roughness = cv2.resize(roughness,(self.pano_w,self.pano_h))
        roughness = np.asarray(roughness,dtype=np.float32)
        roughness = torch.from_numpy(roughness)
        roughness = roughness.unsqueeze(0)

        metallic = cv2.imread(os.path.join(one_path,str(iindex)+'_metallic.hdr'),-1)[:,:,0:1]
        metallic = cv2.resize(metallic,(self.pano_w,self.pano_h))
        metallic = np.asarray(metallic,dtype=np.float32)
        metallic = torch.from_numpy(metallic)
        metallic = metallic.unsqueeze(0)

        mask = cv2.imread(os.path.join(one_path,str(iindex)+'_mask.hdr'),-1)[:,:,0:1]
        mask = cv2.resize(mask,(self.pano_w,self.pano_h))
        mask = np.asarray(mask,dtype=np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0)

        normal = cv2.imread(os.path.join(one_path,str(iindex)+'_normal.hdr'),-1)[:,:,0:3]
        normal = cv2.resize(normal,(self.pano_w,self.pano_h))
        normal = np.asarray(normal,dtype=np.float32)
        normal = normal[...,::-1].copy()
        normal = (normal*2.0)-1.0
        normal = torch.from_numpy(normal)
        normal = normal.permute(2,0,1)

        depth = cv2.imread(os.path.join(one_path,str(iindex)+'_depth.hdr'),-1)[:,:,0:1]
        depth = cv2.resize(depth,(self.pano_w,self.pano_h))
        depth = np.asarray(depth,dtype=np.float32)
        depth = cv2.remap(depth, map_x, map_y, cv2.INTER_LINEAR)
        depth *= z_rot
        depth = torch.from_numpy(depth)
        depth = depth.unsqueeze(0)

        depth_mask = (depth>0) & (depth<=self.max_depth) & (~torch.isnan(depth))

        name = one_path.split('/')[-3]+"_"+str(iindex)
        
        batchDict = {
            'image':image,
            'albedo':albedo,
            'normal':normal,
            'roughness':roughness,
            'metallic':metallic,
            'depth':depth,
            'depth_mask':depth_mask,
            'mask':mask,
            'lighting':light,
            'map_x':map_x,
            'map_y':map_y,
            'name':name
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
            items = os.listdir(whole_path)
            if (len(items)) != 1:
                print(id)
                continue
            num = len(items)
            for i in range(0, num):
                one_item = []
                one_item.append(whole_path)
                one_item.append(i)
                all_item.append(one_item)
        return all_item

if __name__ == "__main__":
    dataset = KePanoLighting(root='/mnt/data/youkeyao/Datasets/FutureHouse/KePanoLight')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    for batch in dataloader:
        image = batch['image'][0]
        albedo = batch['albedo'][0]
        depth = batch['depth'][0]
        lighting = batch['lighting'][0]
        map_x = batch['map_x'][0]
        map_y = batch['map_y'][0]

        albedo_np = albedo.permute(1, 2, 0).numpy()[:, :, ::-1]
        image_np = image.permute(1, 2, 0).numpy()[:, :, ::-1]
        depth_np = depth.repeat(3, 1, 1).permute(1, 2, 0).numpy()[:, :, ::-1]
        # depth_np /= np.max(depth_np)
        depth_np /= 5
        combined_image = np.hstack((image_np, depth_np))

        lighting_nps = []
        for x in range(dataset.out_w):
            for y in range(dataset.out_h):
                lighting_x = int(map_x[y, x] / dataset.pano_w * dataset.probes_w)
                lighting_y = int(map_y[y, x] / dataset.pano_h * dataset.probes_h)
                lighting_np = lighting[:, :, :, lighting_y, lighting_x].permute(1, 2, 0).numpy()[:, :, ::-1]
                lighting_nps.append(lighting_np)

        cv2.imshow("Albedo", albedo_np)
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