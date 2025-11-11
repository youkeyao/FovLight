import mitsuba as mi
import numpy as np
import torch
import cv2
from accelerate import Accelerator
from EnvMapDatasets import EnvMapDatasets
from LightEstimationModel import LightingEstimationModel
from utils import get_free_gpu, create_projection_matrix, linear_to_srgb

class ObjectRenderer:
    def __init__(self, resolution=(480, 640), fov=39.6):
        mi.set_variant('cuda_ad_rgb')
        self.resolution = resolution
        self.fov = fov
        self.camera_transform = mi.ScalarTransform4f().look_at(origin=[0, 0, 0],target=[0, 0, -1],up=[0, 1, 0])
        self.plane_transform = mi.ScalarTransform4f().translate(np.array([0, -1.233, 0])) @ mi.ScalarTransform4f().rotate([1, 0, 0], -90) @ mi.ScalarTransform4f().scale([10, 10, 1])

    def render_sphere(self, origin, envmap, background=None, radius=0.1):
        origin = origin.cpu().numpy()

        sensor = {
            'type': 'perspective',
            'fov': self.fov,
            'to_world': self.camera_transform,
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
        }

        emitter = {
            'type': 'envmap',
            'bitmap': mi.Bitmap(envmap.permute(1, 2, 0).detach().cpu()),
        }

        scene_dict = {
            'type': 'scene',
            'integrator': {
                'type': 'aov',
                'aovs': 'mask:shape_index',
                'output': {
                    'type': 'path',
                }
            },
            'sensor': sensor,
            'emitter': emitter,
            'sphere': {
                'type': 'sphere',
                'radius': radius,
                'to_world': mi.ScalarTransform4f().translate(origin),
                'bsdf': {
                    'type': 'roughconductor',
                    'material': 'Ag',
                    'alpha': 0.01
                }
            },
        }

        scene = mi.load_dict(scene_dict)
        imgs = mi.render(scene).numpy()
        sphere = imgs[:, :, 0:3]
        sphere_mask = imgs[:, :, 3:]

        result = sphere * sphere_mask
        if background is not None:
            background = background.permute(1, 2, 0).detach().cpu().numpy()
            result += background * (1-sphere_mask)
        return result
    
    def render_shadow(self, origin, envmap, background, sphere, radius=0.1):
        origin = origin.cpu().numpy()
        background = background.permute(1, 2, 0).detach().cpu().numpy()
        sphere = sphere.permute(1, 2, 0).detach().cpu().numpy()

        sensor = {
            'type': 'perspective',
            'fov': self.fov,
            'to_world': self.camera_transform,
            'sampler': {
                'type': 'independent',
                'sample_count': 128
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
        }

        emitter = {
            'type': 'envmap',
            'bitmap': mi.Bitmap(envmap.permute(1, 2, 0).detach().cpu()),
        }

        scene_dict = {
            'type': 'scene',
            'integrator': {
                'type': 'aov',
                'aovs': 'mask:shape_index',
                'output': {
                    'type': 'path',
                }
            },
            'sensor': sensor,
            'emitter': emitter,
            'sphere': {
                'type': 'sphere',
                'radius': radius,
                'to_world': mi.ScalarTransform4f().translate(origin),
                'bsdf': {
                    'type': 'roughconductor',
                    'material': 'Ag',
                    'alpha': 0.01
                }
            },
            'plane': {
                'type': 'rectangle',
                'to_world': self.plane_transform,
                'bsdf': {
                    'type': 'diffuse',
                }
            }
        }

        scene = mi.load_dict(scene_dict)
        imgs = mi.render(scene).numpy()
        denoised_all = cv2.medianBlur(imgs[:, :, 0:3], 3)
        sphere_mask = np.where(imgs[:, :, 3:] > 1.2, 1, 0)

        scene_dict = {
            'type': 'scene',
            'integrator': {
                'type': 'aov',
                'aovs': 'mask:shape_index',
                'output': {
                    'type': 'path',
                }
            },
            'sensor': sensor,
            'emitter': emitter,
            'plane': {
                'type': 'rectangle',
                'to_world': self.plane_transform,
                'bsdf': {
                    'type': 'diffuse',
                }
            }
        }

        scene = mi.load_dict(scene_dict)
        imgs = mi.render(scene).numpy()
        denoised_plane = cv2.medianBlur(imgs[:, :, 0:3], 3)

        shadow = (denoised_all / denoised_plane)
        return (background * shadow * (1 - sphere_mask) + sphere * sphere_mask).astype(np.float32)

# 使用示例
if __name__ == "__main__":
    accelerator = Accelerator()
    device = accelerator.device
    model = LightingEstimationModel((168, 120, 128), (160, 320), 100, 3)
    model = accelerator.prepare(model)
    accelerator.load_state("checkpoints_old/voxel_0")
    projection_matrix = create_projection_matrix(120, 440/385, 0.1, 20)
    dataset = EnvMapDatasets(root_dir='/mnt/data/youkeyao/Datasets/LightEnv', image_resolution=(385, 440))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    renderer = ObjectRenderer((385, 440), 120)

    model.eval()
    for batch in dataloader:
        image = batch['image'][0]
        depth = batch['depth'][0]
        lighting = batch['lighting'][0]
        pos = batch["pos"][0]

        print(pos)
        envmap = model(pos.to(device), projection_matrix.to(device), image.to(device), depth.to(device))
        envmap_np = envmap.permute(1, 2, 0).detach().cpu().numpy()[:, :, ::-1]
        lighting_np = lighting.permute(1, 2, 0).numpy()
        sphere = renderer.render_sphere(pos, envmap, image)[:, :, ::-1]
        shadow = renderer.render_shadow(pos, envmap, image, image)[:, :, ::-1]
        cv2.imshow("sphere", linear_to_srgb(sphere))
        cv2.imshow("shadow", linear_to_srgb(shadow))
        cv2.imshow("Lighting map", linear_to_srgb(envmap_np))
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()