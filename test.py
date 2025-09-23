import torch
import cv2
import numpy as np
import time
from accelerate import Accelerator
from skimage.metrics import structural_similarity as ssim
from EnvMapDatasets import EnvMapDatasets
from LightEstimationModel import LightingEstimationModel
from ObjectRenderer import ObjectRenderer
from utils import get_free_gpu, create_projection_matrix, linear_to_srgb, psnr

def main(checkpoint_dir, model_param):
    accelerator = Accelerator()
    device = accelerator.device
    print(f"Using device: {device}")
    # 推理
    model = LightingEstimationModel(*model_param)
    model = accelerator.prepare(model)
    accelerator.load_state(checkpoint_dir)
    projection_matrix = create_projection_matrix(39.6, 640/480, 0.1, 20)
    dataset = EnvMapDatasets(root_dir='/mnt/data/youkeyao/Datasets/LightEnv', image_resolution=(240, 320), lighting_resolution=model.output_resolution)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    renderer = ObjectRenderer((256, 256))
    render_pos = torch.tensor([0, 0, -1])

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
            envmap = model(pos.to(device), projection_matrix.to(device), image.to(device), depth.to(device))
            end_time = time.time()
            print(f"代码执行时间：{end_time - start_time} 秒")
            print("End")

            envmap_np = envmap.permute(1, 2, 0).detach().cpu().numpy()[:, :, ::-1]
            old, old_mask = renderer.render_shadow(render_pos, lighting)
            old_np = old.permute(1, 2, 0).numpy()[:, :, ::-1]
            new, new_mask = renderer.render_shadow(render_pos, envmap)
            new_np = new.permute(1, 2, 0).numpy()[:, :, ::-1]

            old_plane = renderer.render_plane(render_pos, lighting)
            new_plane = renderer.render_plane(render_pos, envmap)
            old_plane_np = old_plane.permute(1, 2, 0).numpy()[:, :, ::-1]
            new_plane_np = new_plane.permute(1, 2, 0).numpy()[:, :, ::-1]
            shadow_mask = np.expand_dims(np.where(old_mask < 1.5, 1, 0), 2)
            old_shadow = np.abs((old_plane_np - old_np) * shadow_mask)
            new_shadow = np.abs((new_plane_np - new_np) * shadow_mask)

            print(f"Envmap PSNR: {psnr(lighting_np, envmap_np)}")
            print(f"Envmap SSIM: {ssim(lighting_np, envmap_np, channel_axis=-1, data_range=1.0)}")
            print(f"Shadow PSNR: {psnr(old_shadow, new_shadow)}")
            print(f"Shadow SSIM: {ssim(old_shadow, new_shadow, channel_axis=-1, data_range=1.0)}")
            cv2.imshow("Lighting map", linear_to_srgb(np.hstack((lighting_np, envmap_np))))
            cv2.imshow("render", linear_to_srgb(np.hstack((old_np, new_np))))
            cv2.imshow("shadow", linear_to_srgb(np.hstack((old_shadow, new_shadow))))
            key = cv2.waitKey(0) & 0xFF
            while key != ord('n') and key != ord('q'):
                key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # main("checkpoints/voxel_0", ((168, 120, 128), (160, 320), 100, 3))
    # main("checkpoints/voxel_1", ((151, 108, 115), (160, 320), 100, 3))
    # main("checkpoints/voxel_2", ((134, 96, 102), (160, 320), 100, 3))
    # main("checkpoints/voxel_3", ((118, 84, 90), (160, 320), 100, 3))
    # main("checkpoints/voxel_4", ((101, 72, 77), (160, 320), 100, 3))
    # main("checkpoints/voxel_5", ((84, 60, 64), (160, 320), 100, 3))
    # main("checkpoints/voxel_6", ((67, 48, 51), (160, 320), 100, 3))
    # main("checkpoints/voxel_7", ((50, 36, 38), (160, 320), 100, 3))
    # main("checkpoints/voxel_8", ((34, 24, 26), (160, 320), 100, 3))
    # main("checkpoints/voxel_9", ((17, 12, 13), (160, 320), 100, 3))

    main("checkpoints/network_0", ((168, 120, 128), (160, 320), 100, 3))
    # main("checkpoints/network_1", ((168, 120, 128), (160, 320), 100, 2))
    # main("checkpoints/network_2", ((168, 120, 128), (160, 320), 100, 1))

    # main("checkpoints/base", ((168, 120, 128), (160, 320), 100, True))
    # main("checkpoints/voxel", ((84, 60, 64), (160, 320), 100, True))
    # main("checkpoints/resolution", ((168, 120, 128), (80, 160), 100, True))
    # main("checkpoints/sample", ((168, 120, 128), (160, 320), 50, True))
    # main("checkpoints/network", ((168, 120, 128), (160, 320), 100, 3))