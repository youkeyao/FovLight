import torch
from torch.utils.data import ConcatDataset, DataLoader
import cv2
import numpy as np
import time
from accelerate import Accelerator
from skimage.metrics import structural_similarity as ssim
from EnvMapDatasets import EnvMapDatasets
from KePanoLightDataset import KePanoLightDataset
from LightEstimationModel import LightingEstimationModel
from ObjectRenderer import ObjectRenderer
from utils import create_projection_matrix, linear_to_srgb, psnr

def main(checkpoint_dir, model_param):
    accelerator = Accelerator()
    device = accelerator.device
    print(f"Using device: {device}")
    # 推理
    model = LightingEstimationModel(*model_param)
    model = accelerator.prepare(model)
    accelerator.load_state(checkpoint_dir)
    projection_matrix = create_projection_matrix(120, 832/400, 0.1, 20)
    dataset = EnvMapDatasets(root_dir='/mnt/data/youkeyao/Datasets/LightEnv', image_resolution=(400, 832), lighting_resolution=model.output_resolution)
    new_dataset = KePanoLightDataset(root='/mnt/data/youkeyao/Datasets/FutureHouse/KePanoLight')
    # dataloader = DataLoader(ConcatDataset([dataset, new_dataset]), batch_size=1, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    renderer = ObjectRenderer((256, 256))
    render_pos = torch.tensor([0, 0, -0.5])

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            image = batch['image'][0]
            depth = batch['depth'][0]
            lighting = batch['lighting'][0]
            pos = batch["pos"][0]

            lighting_np = lighting.permute(1, 2, 0).numpy()[:, :, ::-1]
            print(pos)

            print("Start")
            start_time = time.time()
            envmap, _ = model(pos, projection_matrix, torch.eye(4), image, depth, True, True)
            end_time = time.time()
            print(f"代码执行时间：{end_time - start_time} 秒")
            print("End")

            image_np = image.permute(1, 2, 0).detach().cpu().numpy()[:, :, ::-1]
            envmap_np = envmap[0].permute(1, 2, 0).detach().cpu().numpy()[:, :, ::-1]
            render_old = renderer.render_sphere(render_pos, lighting)[:, :, ::-1]
            render_new = renderer.render_sphere(render_pos, envmap[0])[:, :, ::-1]

            print(f"Envmap PSNR: {psnr(lighting_np, envmap_np)}")
            print(f"Envmap SSIM: {ssim(lighting_np, envmap_np, channel_axis=-1, data_range=1.0)}")
            print(f"Shadow PSNR: {psnr(render_old, render_new)}")
            print(f"Shadow SSIM: {ssim(render_old, render_new, channel_axis=-1, data_range=1.0)}")
            cv2.imshow("image", image_np)
            cv2.imshow("Lighting map", linear_to_srgb(np.hstack((lighting_np, envmap_np))))

            # mask = (lighting_np > 0.9).any(axis=2)[:, :, None]
            # cv2.imwrite("test.exr", envmap_np * mask)
            cv2.imshow("render", np.hstack((render_old, render_new)))
            key = cv2.waitKey(0) & 0xFF
            while key != ord('n') and key != ord('q'):
                key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # main("checkpoints/voxel_0", ((168, 120, 128), (320, 640), 100, 4))
    # main("checkpoints/voxel_1", ((151, 108, 115), (320, 640), 100, 4))
    # main("checkpoints/voxel_2", ((134, 96, 102), (320, 640), 100, 4))
    # main("checkpoints/voxel_3", ((118, 84, 90), (320, 640), 100, 4))
    # main("checkpoints/voxel_4", ((101, 72, 77), (320, 640), 100, 4))
    # main("checkpoints/voxel_5", ((84, 60, 64), (320, 640), 100, 4))
    # main("checkpoints/voxel_6", ((67, 48, 51), (320, 640), 100, 4))
    main("checkpoints/voxel_7", ((50, 36, 38), (320, 640), 100, 4))
    # main("checkpoints/voxel_8", ((34, 24, 26), (320, 640), 100, 4))
    # main("checkpoints/voxel_9", ((17, 12, 13), (320, 640), 100, 4))

    # main("checkpoints/network_0", ((84, 60, 64), (320, 640), 100, 3))
    # main("checkpoints/network_1", ((84, 60, 64), (320, 640), 100, 2))
    # main("checkpoints/network_2", ((84, 60, 64), (320, 640), 100, 1))

    # main("checkpoints/base", ((168, 120, 128), (160, 320), 100, True))
    # main("checkpoints/voxel", ((84, 60, 64), (160, 320), 100, True))
    # main("checkpoints/resolution", ((168, 120, 128), (80, 160), 100, True))
    # main("checkpoints/sample", ((168, 120, 128), (160, 320), 50, True))
    # main("checkpoints/network", ((168, 120, 128), (160, 320), 100, 3))