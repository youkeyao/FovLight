import mitsuba as mi
import numpy as np
import torch
import cv2
from EnvMapDatasets import EnvMapDatasets
from LightEstimationModel import LightingEstimationModel
from utils import get_free_gpu, create_projection_matrix


class ObjectRenderer:
    def __init__(self, resolution=(480, 640)):
        mi.set_variant('cuda_ad_rgb')
        self.resolution = resolution

    def render(self, origin, envmap):
        origin = origin.numpy()

        sensor = mi.load_dict({
            'type': 'perspective',
            'fov': 39.6,
            'to_world': mi.ScalarTransform4f().look_at(
                origin=[0, 0, 0],
                target=origin,
                up=[0, 1, 0]
            ),
            'sampler': {
                'type': 'independent',
                'sample_count': 16
            },
            'film': {
                'type': 'hdrfilm',
                'width': self.resolution[1],
                'height': self.resolution[0],
                'rfilter': {
                    'type': 'tent',
                },
                'pixel_format': 'rgb',
            },
        })

        emitter = {
            'type': 'envmap',
            'bitmap': mi.Bitmap(envmap.permute(1, 2, 0).detach().cpu()),
        }

        scene = mi.load_dict({
            'type': 'scene',
            'integrator': {'type': 'path'},
            'emitter': emitter,
            'sphere': {
                'type': 'sphere',
                'radius': 1,
                'to_world': mi.ScalarTransform4f().translate(origin),
                'bsdf': {
                    'type': 'roughconductor',
                    'material': 'Ag',
                    'alpha': 0.2
                }
            },
        })

        img = mi.render(scene, sensor=sensor)

        return torch.from_numpy(img.numpy()).permute(2, 0, 1)

# 使用示例
if __name__ == "__main__":
    selected_gpu = get_free_gpu()
    device = torch.device("cpu" if selected_gpu is None else f"cuda:{selected_gpu}")
    model = LightingEstimationModel().to(device)
    model.load_state_dict(torch.load("checkpoints_new/model_checkpoint_2000.pth", map_location=device, weights_only=True)['model_state_dict'])
    projection_matrix = create_projection_matrix(39.6, 640/480, 0.1, 20)
    renderer = ObjectRenderer((256, 256))
    dataset = EnvMapDatasets(root_dir='/mnt/data/youkeyao/Datasets/LightEnv')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    for batch in dataloader:
        image = batch['image'][0]
        depth = batch['depth'][0]
        lighting = batch['lighting'][0]
        pos = batch["pos"][0]

        print(pos)
        envmap = model(pos.to(device), projection_matrix.to(device), image.to(device), depth.to(device))
        envmap_np = envmap.permute(1, 2, 0).detach().cpu().numpy()[:, :, ::-1]
        lighting_np = lighting.permute(1, 2, 0).numpy()[:, :, ::-1]
        old = renderer.render(pos, lighting)
        old_np = old.permute(1, 2, 0).numpy()[:, :, ::-1]
        new = renderer.render(pos, envmap)
        new_np = new.permute(1, 2, 0).numpy()[:, :, ::-1]
        cv2.imshow("render", np.hstack((old_np, new_np)))
        cv2.imshow("Lighting map", np.hstack((lighting_np, envmap_np)))
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()