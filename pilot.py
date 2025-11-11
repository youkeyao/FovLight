import mitsuba as mi
import numpy as np
import torch
import cv2
import os
from accelerate import Accelerator
from EnvMapDatasets import EnvMapDatasets
from LightEstimationModel import LightingEstimationModel
from ObjectRenderer import ObjectRenderer
from utils import get_free_gpu, create_projection_matrix, linear_to_srgb

def main(checkpoint_dir, model_param, dataset_dir, background_image_path, positions, radius, offset, camera_transform, plane_transform, sphere_image_paths=None):
    accelerator = Accelerator()
    device = accelerator.device
    print(f"Using device: {device}")
    model = LightingEstimationModel(*model_param)
    model = accelerator.prepare(model)
    accelerator.load_state("checkpoints_old/" + checkpoint_dir)
    projection_matrix = create_projection_matrix(120, 440/385, 0.1, 20)
    renderer = ObjectRenderer((3850, 4400), 120)
    renderer.camera_transform = camera_transform
    renderer.plane_transform = plane_transform

    model.eval()
    image = cv2.imread(dataset_dir + "/image.png", -1)[:, :, 0:3][:, :, ::-1].astype(np.float32) / 255
    image = np.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055) ** 2.4
    )
    image = torch.from_numpy(image).permute(2, 0, 1)

    depth = cv2.imread(dataset_dir + "/depth.exr", -1)[:, :, 0:1].astype(np.float32)
    depth = torch.from_numpy(depth).permute(2, 0, 1)

    background_image = cv2.imread(background_image_path, -1)[:, :, 0:3][:, :, ::-1].astype(np.float32) / 255
    background_image = np.where(
            background_image <= 0.04045,
            background_image / 12.92,
            ((background_image + 0.055) / 1.055) ** 2.4
    )
    background_image = torch.from_numpy(background_image).permute(2, 0, 1)

    for i in range(len(positions)):
        pos = torch.tensor(positions[i])
        envmap = model(pos.to(device), projection_matrix.to(device), image.to(device), depth.to(device))

        if sphere_image_paths is None:
            result = renderer.render_sphere(pos, envmap, background_image, radius[i])
        else:
            sphere_image = cv2.imread(sphere_image_paths[i], -1)[:, :, 0:3][:, :, ::-1].astype(np.float32) / 255
            sphere_image = np.where(
                    sphere_image <= 0.04045,
                    sphere_image / 12.92,
                    ((sphere_image + 0.055) / 1.055) ** 2.4
            )
            sphere_image = torch.from_numpy(sphere_image).permute(2, 0, 1)
            result = renderer.render_shadow(pos, envmap, background_image, sphere_image, radius[i])

        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        if not os.path.exists("/mnt/data/youkeyao/FovLight/pilot/" + str(i+offset)):
            os.mkdir("/mnt/data/youkeyao/FovLight/pilot/" + str(i+offset))
        cv2.imwrite("/mnt/data/youkeyao/FovLight/pilot/" + str(i+offset) + "/" + checkpoint_dir + ".png",
                    (linear_to_srgb(result) * 255).astype(np.uint8))

if __name__ == "__main__":
    ObjectRenderer((3850, 4400), 120)

    # main("voxel_0", ((168, 120, 128), (160, 320), 100, 3), "/mnt/data/youkeyao/Datasets/LightEnv/scene_001", "/mnt/data/youkeyao/FovLight/pilot/scene_001.png",
    #     [[0.26, 0, -2.5], [0.78, 0, -2.5], [1.57, 0, -2.5]], [0.1, 0.147, 0.197], 0,
    #     mi.ScalarTransform4f().look_at(origin=[0, 0, 0],target=[0, 0, -1],up=[0, 1, 0]),
    #     mi.ScalarTransform4f().translate(np.array([0, -1.233, 0])) @ mi.ScalarTransform4f().rotate([1, 0, 0], -90) @ mi.ScalarTransform4f().scale([10, 10, 1]),)
    # main("voxel_2", ((134, 96, 102), (160, 320), 100, 3), "/mnt/data/youkeyao/Datasets/LightEnv/scene_001", "/mnt/data/youkeyao/FovLight/pilot/scene_001.png",
    #     [[0.26, 0, -2.5], [0.78, 0, -2.5], [1.57, 0, -2.5]], [0.1, 0.147, 0.197], 0,
    #     mi.ScalarTransform4f().look_at(origin=[0, 0, 0],target=[0, 0, -1],up=[0, 1, 0]),
    #     mi.ScalarTransform4f().translate(np.array([0, -1.233, 0])) @ mi.ScalarTransform4f().rotate([1, 0, 0], -90) @ mi.ScalarTransform4f().scale([10, 10, 1]),)
    # main("voxel_4", ((101, 72, 77), (160, 320), 100, 3), "/mnt/data/youkeyao/Datasets/LightEnv/scene_001", "/mnt/data/youkeyao/FovLight/pilot/scene_001.png",
    #     [[0.26, 0, -2.5], [0.78, 0, -2.5], [1.57, 0, -2.5]], [0.1, 0.147, 0.197], 0,
    #     mi.ScalarTransform4f().look_at(origin=[0, 0, 0],target=[0, 0, -1],up=[0, 1, 0]),
    #     mi.ScalarTransform4f().translate(np.array([0, -1.233, 0])) @ mi.ScalarTransform4f().rotate([1, 0, 0], -90) @ mi.ScalarTransform4f().scale([10, 10, 1]),)
    # main("voxel_6", ((67, 48, 51), (160, 320), 100, 3), "/mnt/data/youkeyao/Datasets/LightEnv/scene_001", "/mnt/data/youkeyao/FovLight/pilot/scene_001.png",
    #     [[0.26, 0, -2.5], [0.78, 0, -2.5], [1.57, 0, -2.5]], [0.1, 0.147, 0.197], 0,
    #     mi.ScalarTransform4f().look_at(origin=[0, 0, 0],target=[0, 0, -1],up=[0, 1, 0]),
    #     mi.ScalarTransform4f().translate(np.array([0, -1.233, 0])) @ mi.ScalarTransform4f().rotate([1, 0, 0], -90) @ mi.ScalarTransform4f().scale([10, 10, 1]),)
    # main("voxel_8", ((34, 24, 26), (160, 320), 100, 3), "/mnt/data/youkeyao/Datasets/LightEnv/scene_001", "/mnt/data/youkeyao/FovLight/pilot/scene_001.png",
    #     [[0.26, 0, -2.5], [0.78, 0, -2.5], [1.57, 0, -2.5]], [0.1, 0.147, 0.197], 0,
    #     mi.ScalarTransform4f().look_at(origin=[0, 0, 0],target=[0, 0, -1],up=[0, 1, 0]),
    #     mi.ScalarTransform4f().translate(np.array([0, -1.233, 0])) @ mi.ScalarTransform4f().rotate([1, 0, 0], -90) @ mi.ScalarTransform4f().scale([10, 10, 1]),)
    # main("network_1", ((168, 120, 128), (160, 320), 100, 2), "/mnt/data/youkeyao/Datasets/LightEnv/scene_001", "/mnt/data/youkeyao/FovLight/pilot/scene_001.png",
    #     [[0.26, 0, -2.5], [0.78, 0, -2.5], [1.57, 0, -2.5]], [0.1, 0.147, 0.197], 0,
    #     mi.ScalarTransform4f().look_at(origin=[0, 0, 0],target=[0, 0, -1],up=[0, 1, 0]),
    #     mi.ScalarTransform4f().translate(np.array([0, -1.233, 0])) @ mi.ScalarTransform4f().rotate([1, 0, 0], -90) @ mi.ScalarTransform4f().scale([10, 10, 1]),)
    # main("network_2", ((168, 120, 128), (160, 320), 100, 1), "/mnt/data/youkeyao/Datasets/LightEnv/scene_001", "/mnt/data/youkeyao/FovLight/pilot/scene_001.png",
    #     [[0.26, 0, -2.5], [0.78, 0, -2.5], [1.57, 0, -2.5]], [0.1, 0.147, 0.197], 0,
    #     mi.ScalarTransform4f().look_at(origin=[0, 0, 0],target=[0, 0, -1],up=[0, 1, 0]),
    #     mi.ScalarTransform4f().translate(np.array([0, -1.233, 0])) @ mi.ScalarTransform4f().rotate([1, 0, 0], -90) @ mi.ScalarTransform4f().scale([10, 10, 1]),)
    
    main("voxel_0", ((168, 120, 128), (160, 320), 100, 3), "/mnt/data/youkeyao/Datasets/LightEnv/scene_002", "/mnt/data/youkeyao/FovLight/pilot/scene_002.png",
        [[0.26, -1.0168, -2.2839], [0.78, -1.0168, -2.2839], [1.57, -1.0168, -2.2839]], [0.1, 0.147, 0.197], 3,
        mi.ScalarTransform4f().look_at(origin=[0, 0, 0],target=[0, -1.2202, -2.7406],up=[0, 1, 0]),
        mi.ScalarTransform4f().translate(np.array([0, -1.233, 0])) @ mi.ScalarTransform4f().rotate([1, 0, 0], -90) @ mi.ScalarTransform4f().scale([10, 10, 1]),
        ["/mnt/data/youkeyao/FovLight/pilot/scene_002_3.png", "/mnt/data/youkeyao/FovLight/pilot/scene_002_4.png", "/mnt/data/youkeyao/FovLight/pilot/scene_002_5.png"])
    main("voxel_2", ((134, 96, 102), (160, 320), 100, 3), "/mnt/data/youkeyao/Datasets/LightEnv/scene_002", "/mnt/data/youkeyao/FovLight/pilot/scene_002.png",
        [[0.26, -1.0168, -2.2839], [0.78, -1.0168, -2.2839], [1.57, -1.0168, -2.2839]], [0.1, 0.147, 0.197], 3,
        mi.ScalarTransform4f().look_at(origin=[0, 0, 0],target=[0, -1.2202, -2.7406],up=[0, 1, 0]),
        mi.ScalarTransform4f().translate(np.array([0, -1.233, 0])) @ mi.ScalarTransform4f().rotate([1, 0, 0], -90) @ mi.ScalarTransform4f().scale([10, 10, 1]),
        ["/mnt/data/youkeyao/FovLight/pilot/scene_002_3.png", "/mnt/data/youkeyao/FovLight/pilot/scene_002_4.png", "/mnt/data/youkeyao/FovLight/pilot/scene_002_5.png"])
    main("voxel_4", ((101, 72, 77), (160, 320), 100, 3), "/mnt/data/youkeyao/Datasets/LightEnv/scene_002", "/mnt/data/youkeyao/FovLight/pilot/scene_002.png",
        [[0.26, -1.0168, -2.2839], [0.78, -1.0168, -2.2839], [1.57, -1.0168, -2.2839]], [0.1, 0.147, 0.197], 3,
        mi.ScalarTransform4f().look_at(origin=[0, 0, 0],target=[0, -1.2202, -2.7406],up=[0, 1, 0]),
        mi.ScalarTransform4f().translate(np.array([0, -1.233, 0])) @ mi.ScalarTransform4f().rotate([1, 0, 0], -90) @ mi.ScalarTransform4f().scale([10, 10, 1]),
        ["/mnt/data/youkeyao/FovLight/pilot/scene_002_3.png", "/mnt/data/youkeyao/FovLight/pilot/scene_002_4.png", "/mnt/data/youkeyao/FovLight/pilot/scene_002_5.png"])
    main("voxel_6", ((67, 48, 51), (160, 320), 100, 3), "/mnt/data/youkeyao/Datasets/LightEnv/scene_002", "/mnt/data/youkeyao/FovLight/pilot/scene_002.png",
        [[0.26, -1.0168, -2.2839], [0.78, -1.0168, -2.2839], [1.57, -1.0168, -2.2839]], [0.1, 0.147, 0.197], 3,
        mi.ScalarTransform4f().look_at(origin=[0, 0, 0],target=[0, -1.2202, -2.7406],up=[0, 1, 0]),
        mi.ScalarTransform4f().translate(np.array([0, -1.233, 0])) @ mi.ScalarTransform4f().rotate([1, 0, 0], -90) @ mi.ScalarTransform4f().scale([10, 10, 1]),
        ["/mnt/data/youkeyao/FovLight/pilot/scene_002_3.png", "/mnt/data/youkeyao/FovLight/pilot/scene_002_4.png", "/mnt/data/youkeyao/FovLight/pilot/scene_002_5.png"])
    main("voxel_8", ((34, 24, 26), (160, 320), 100, 3), "/mnt/data/youkeyao/Datasets/LightEnv/scene_002", "/mnt/data/youkeyao/FovLight/pilot/scene_002.png",
        [[0.26, -1.0168, -2.2839], [0.78, -1.0168, -2.2839], [1.57, -1.0168, -2.2839]], [0.1, 0.147, 0.197], 3,
        mi.ScalarTransform4f().look_at(origin=[0, 0, 0],target=[0, -1.2202, -2.7406],up=[0, 1, 0]),
        mi.ScalarTransform4f().translate(np.array([0, -1.233, 0])) @ mi.ScalarTransform4f().rotate([1, 0, 0], -90) @ mi.ScalarTransform4f().scale([10, 10, 1]),
        ["/mnt/data/youkeyao/FovLight/pilot/scene_002_3.png", "/mnt/data/youkeyao/FovLight/pilot/scene_002_4.png", "/mnt/data/youkeyao/FovLight/pilot/scene_002_5.png"])
    main("network_1", ((168, 120, 128), (160, 320), 100, 2), "/mnt/data/youkeyao/Datasets/LightEnv/scene_002", "/mnt/data/youkeyao/FovLight/pilot/scene_002.png",
        [[0.26, -1.0168, -2.2839], [0.78, -1.0168, -2.2839], [1.57, -1.0168, -2.2839]], [0.1, 0.147, 0.197], 3,
        mi.ScalarTransform4f().look_at(origin=[0, 0, 0],target=[0, -1.2202, -2.7406],up=[0, 1, 0]),
        mi.ScalarTransform4f().translate(np.array([0, -1.233, 0])) @ mi.ScalarTransform4f().rotate([1, 0, 0], -90) @ mi.ScalarTransform4f().scale([10, 10, 1]),
        ["/mnt/data/youkeyao/FovLight/pilot/scene_002_3.png", "/mnt/data/youkeyao/FovLight/pilot/scene_002_4.png", "/mnt/data/youkeyao/FovLight/pilot/scene_002_5.png"])
    main("network_2", ((168, 120, 128), (160, 320), 100, 1), "/mnt/data/youkeyao/Datasets/LightEnv/scene_002", "/mnt/data/youkeyao/FovLight/pilot/scene_002.png",
        [[0.26, -1.0168, -2.2839], [0.78, -1.0168, -2.2839], [1.57, -1.0168, -2.2839]], [0.1, 0.147, 0.197], 3,
        mi.ScalarTransform4f().look_at(origin=[0, 0, 0],target=[0, -1.2202, -2.7406],up=[0, 1, 0]),
        mi.ScalarTransform4f().translate(np.array([0, -1.233, 0])) @ mi.ScalarTransform4f().rotate([1, 0, 0], -90) @ mi.ScalarTransform4f().scale([10, 10, 1]),
        ["/mnt/data/youkeyao/FovLight/pilot/scene_002_3.png", "/mnt/data/youkeyao/FovLight/pilot/scene_002_4.png", "/mnt/data/youkeyao/FovLight/pilot/scene_002_5.png"])