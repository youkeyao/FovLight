import mitsuba as mi
import numpy as np
import torch
import cv2
from accelerate import Accelerator
from EnvMapDatasets import EnvMapDatasets
from LightEstimationModel import LightingEstimationModel
from utils import get_free_gpu, create_projection_matrix, linear_to_srgb, calculate_spatial_metrics

class ObjectRenderer:
    def __init__(self, resolution=(480, 640), fov=39.6):
        mi.set_variant('cuda_ad_rgb')
        self.resolution = resolution
        self.fov = fov
        self.camera_transform = mi.ScalarTransform4f().look_at(origin=[0, 0, 0],target=[0, 0, -1],up=[0, 1, 0])
        self.plane_transform = mi.ScalarTransform4f().translate(np.array([0, -1.5, -3.5])) @ mi.ScalarTransform4f().rotate([1, 0, 0], -90) @ mi.ScalarTransform4f().scale([2, 2, 1])

    def render_sphere(self, origin, envmap, background=None, radius=0.1, roughness=0.0, metallic=1.0, color=[1.0,1.0,1.0]):
        origin = origin.cpu().numpy()

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
                    'type': 'box',
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
                    "type": "principled",
                    "base_color": {
                        "type": "rgb",
                        "value": color
                    },
                    "metallic": metallic,
                    "roughness": roughness
                }
            },
        }

        scene = mi.load_dict(scene_dict)
        imgs = mi.render(scene).numpy()
        denoiser = mi.OptixDenoiser(
            input_size=(imgs.shape[1], imgs.shape[0]),
            albedo=False,
            normals=False,
            temporal=False
        )
        sphere = linear_to_srgb(denoiser(imgs[:, :, 0:3]).numpy())
        sphere_mask = imgs[:, :, 3:]

        result = sphere * sphere_mask
        if background is not None:
            background = background.permute(1, 2, 0).detach().cpu().numpy()
            result += background * (1-sphere_mask)
        return result

    def render_bunny(self, origin, envmap, background=None, mesh_path=None, scale=1.0, color=[0.7, 0.7, 0.7], metallic=0.0, roughness=0.0):
        origin = origin.cpu().numpy()

        if mesh_path is None:
            raise ValueError("`mesh_path` is required for render_bunny. Provide a PLY/OBJ mesh file path.")

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
                    'type': 'box',
                },
                'pixel_format': 'rgb',
            },
        }

        emitter = {
            'type': 'envmap',
            'bitmap': mi.Bitmap(envmap.permute(1, 2, 0).detach().cpu()),
        }

        mesh_ext = mesh_path.split('.')[-1].lower()
        if mesh_ext in ['ply']:
            mesh_type = 'ply'
        elif mesh_ext in ['obj']:
            mesh_type = 'obj'
        else:
            mesh_type = 'ply'

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
            'bunny': {
                'type': mesh_type,
                'filename': mesh_path,
                'to_world': (
                    mi.ScalarTransform4f().translate(origin)
                    @ mi.ScalarTransform4f().rotate([1, 0, 0], 90) @ mi.ScalarTransform4f().rotate([0, 1, 0], 0) @ mi.ScalarTransform4f().rotate([0, 0, 1], 90)
                    @ mi.ScalarTransform4f().scale([scale, scale, scale])
                ),
                'bsdf': {
                    'type': 'principled',
                    'base_color': {
                        'type': 'rgb',
                        'value': color
                    },
                    'metallic': metallic,
                    'roughness': roughness
                }
            },
        }

        scene = mi.load_dict(scene_dict)
        imgs = mi.render(scene).numpy()
        denoiser = mi.OptixDenoiser(
            input_size=(imgs.shape[1], imgs.shape[0]),
            albedo=False,
            normals=False,
            temporal=False
        )
        bunny = linear_to_srgb(denoiser(imgs[:, :, 0:3]).numpy())
        bunny_mask = imgs[:, :, 3:]

        result = bunny * bunny_mask
        if background is not None:
            background = background.permute(1, 2, 0).detach().cpu().numpy()
            result += background * (1-bunny_mask)
        return result

    def render_plane(self, origin, envmap, background=None, radius=0.1, roughness=0, metallic=1.0, color=[1.0,1.0,1.0]):
        origin = origin.cpu().numpy()

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
                    'type': 'box',
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
                    "type": "principled",
                    "base_color": {
                        "type": "rgb",
                        "value": color
                    },
                    "metallic": metallic,
                    "roughness": roughness,
                }
            },
            'plane': {
                'type': 'rectangle',
                'to_world': self.plane_transform,
                'bsdf': {
                    'type': 'diffuse',
                    "reflectance": {
                        "type": "rgb",
                        "value": [1.0, 1.0, 1.0]
                    }
                }
            }
        }

        scene = mi.load_dict(scene_dict)
        imgs = mi.render(scene).numpy()
        denoiser = mi.OptixDenoiser(
            input_size=(imgs.shape[1], imgs.shape[0]),
            albedo=False,
            normals=False,
            temporal=False
        )
        denoised_all = linear_to_srgb(denoiser(imgs[:, :, 0:3]).numpy())
        mask = np.where(imgs[:, :, 3:] > 0, 1, 0)

        result = denoised_all * mask
        if background is not None:
            background = background.permute(1, 2, 0).detach().cpu().numpy()
            result += background * (1-mask)

        return result
    
    def render_shadow(self, origin, envmap, background, sphere, radius=0.1, roughness=0, metallic=1.0, color=[1.0,1.0,1.0]):
        origin = origin.cpu().numpy()
        background = background.permute(1, 2, 0).detach().cpu().numpy()
        sphere = sphere.permute(1, 2, 0).detach().cpu().numpy()

        sensor = {
            'type': 'perspective',
            'fov': self.fov,
            'to_world': self.camera_transform,
            'sampler': {
                'type': 'independent',
                'sample_count': 200
            },
            'film': {
                'type': 'hdrfilm',
                'width': self.resolution[1],
                'height': self.resolution[0],
                'rfilter': {
                    'type': 'box',
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
                    'type': 'principled',
                    'base_color': {
                        'type': 'rgb',
                        'value': color
                    },
                    'metallic': 1.0,
                    'roughness': 1.0,
                }
            },
            'plane': {
                'type': 'rectangle',
                'to_world': self.plane_transform,
                'bsdf': {
                    'type': 'diffuse',
                    "reflectance": {
                        "type": "rgb",
                        "value": [1.0, 1.0, 1.0]
                    }
                }
            }
        }

        scene = mi.load_dict(scene_dict)
        imgs = mi.render(scene).numpy()
        denoiser = mi.OptixDenoiser(
            input_size=(imgs.shape[1], imgs.shape[0]),
            albedo=False,
            normals=False,
            temporal=False
        )
        denoised_all = denoiser(imgs[:, :, 0:3]).numpy()
        mask_idx = np.rint(imgs[:, :, 3]).astype(np.int32)
        # calculate object mask
        unique, counts = np.unique(mask_idx, return_counts=True)
        pairs = [(int(u), int(c)) for u, c in zip(unique, counts) if int(u) != 0]

        if len(pairs) == 0:
            sphere_mask = np.zeros(mask_idx.shape, dtype=np.uint8)
            plane_mask  = np.zeros(mask_idx.shape, dtype=np.uint8)
        else:
            pairs_sorted = sorted(pairs, key=lambda x: x[1])
            if len(pairs_sorted) == 1:
                sphere_idx = pairs_sorted[0][0]
                plane_idx = None
            else:
                sphere_idx = pairs_sorted[0][0]
                plane_idx  = pairs_sorted[-1][0]
            sphere_mask = (mask_idx == sphere_idx).astype(np.uint8)
            plane_mask  = (mask_idx == plane_idx).astype(np.uint8) if plane_idx is not None else np.zeros_like(sphere_mask)
        sphere_mask = np.expand_dims(sphere_mask, axis=-1)
        plane_mask  = np.expand_dims(plane_mask, axis=-1)

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
                    "reflectance": {
                        "type": "rgb",
                        "value": [1.0, 1.0, 1.0]
                    }
                }
            }
        }
        sphere_mask = np.where(sphere > 0, 1, 0) & sphere_mask

        scene = mi.load_dict(scene_dict)
        imgs = mi.render(scene).numpy()
        denoised_plane = denoiser(imgs[:, :, 0:3]).numpy()

        # shadow = np.clip(denoised_all.sum(axis=-1, keepdims=True) / (denoised_plane.sum(axis=-1, keepdims=True) + 1e-7), 0, 1)
        # return (1 - plane_mask - sphere_mask) * background + np.clip(shadow * plane_mask * background, 0, 1) + sphere_mask * sphere
        
        shadow = (denoised_all.sum(axis=-1, keepdims=True) / (denoised_plane.sum(axis=-1, keepdims=True) + 1e-7))
        return np.clip((1 - sphere_mask) * shadow * background, 0, 1) + sphere_mask * sphere

    def render_bunny_shadow(self, origin, envmap, background, bunny, mesh_path=None, scale=1.0, color=[1.0,1.0,1.0]):
        origin = origin.cpu().numpy()
        background = background.permute(1, 2, 0).detach().cpu().numpy()
        bunny = bunny.permute(1, 2, 0).detach().cpu().numpy()

        if mesh_path is None:
            raise ValueError("`mesh_path` is required for render_bunny_shadow. Provide a PLY/OBJ mesh file path.")

        sensor = {
            'type': 'perspective',
            'fov': self.fov,
            'to_world': self.camera_transform,
            'sampler': {
                'type': 'independent',
                'sample_count': 200
            },
            'film': {
                'type': 'hdrfilm',
                'width': self.resolution[1],
                'height': self.resolution[0],
                'rfilter': {
                    'type': 'box',
                },
                'pixel_format': 'rgb',
            },
        }

        emitter = {
            'type': 'envmap',
            'bitmap': mi.Bitmap(envmap.permute(1, 2, 0).detach().cpu()),
        }

        mesh_ext = mesh_path.split('.')[-1].lower()
        if mesh_ext in ['ply']:
            mesh_type = 'ply'
        elif mesh_ext in ['obj']:
            mesh_type = 'obj'
        else:
            mesh_type = 'ply'

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
            'bunny': {
                'type': mesh_type,
                'filename': mesh_path,
                'to_world': (
                    mi.ScalarTransform4f().translate(origin)
                    @ mi.ScalarTransform4f().rotate([1, 0, 0], 90) @ mi.ScalarTransform4f().rotate([0, 1, 0], 0) @ mi.ScalarTransform4f().rotate([0, 0, 1], 90)
                    @ mi.ScalarTransform4f().scale([scale, scale, scale])
                ),
                'bsdf': {
                    'type': 'principled',
                    'base_color': {
                        'type': 'rgb',
                        'value': color
                    },
                    'metallic': 1.0,
                    'roughness': 1.0,
                }
            },
            'plane': {
                'type': 'rectangle',
                'to_world': self.plane_transform,
                'bsdf': {
                    'type': 'diffuse',
                    "reflectance": {
                        "type": "rgb",
                        "value": [1.0, 1.0, 1.0]
                    }
                }
            }
        }

        scene = mi.load_dict(scene_dict)
        imgs = mi.render(scene).numpy()
        denoiser = mi.OptixDenoiser(
            input_size=(imgs.shape[1], imgs.shape[0]),
            albedo=False,
            normals=False,
            temporal=False
        )
        denoised_all = denoiser(imgs[:, :, 0:3]).numpy()
        mask_idx = np.rint(imgs[:, :, 3]).astype(np.int32)
        # calculate object mask
        unique, counts = np.unique(mask_idx, return_counts=True)
        pairs = [(int(u), int(c)) for u, c in zip(unique, counts) if int(u) != 0]

        if len(pairs) == 0:
            bunny_mask = np.zeros(mask_idx.shape, dtype=np.uint8)
            plane_mask  = np.zeros(mask_idx.shape, dtype=np.uint8)
        else:
            pairs_sorted = sorted(pairs, key=lambda x: x[1])
            if len(pairs_sorted) == 1:
                bunny_idx = pairs_sorted[0][0]
                plane_idx = None
            else:
                bunny_idx = pairs_sorted[0][0]
                plane_idx  = pairs_sorted[-1][0]
            bunny_mask = (mask_idx == bunny_idx).astype(np.uint8)
            plane_mask  = (mask_idx == plane_idx).astype(np.uint8) if plane_idx is not None else np.zeros_like(bunny_mask)
        bunny_mask = np.expand_dims(bunny_mask, axis=-1)
        plane_mask  = np.expand_dims(plane_mask, axis=-1)

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
                    "reflectance": {
                        "type": "rgb",
                        "value": [1.0, 1.0, 1.0]
                    }
                }
            }
        }
        plane_mask = np.where(imgs[:, :, 3:] > 0, 1, 0) & plane_mask

        scene = mi.load_dict(scene_dict)
        imgs = mi.render(scene).numpy()
        denoised_plane = denoiser(imgs[:, :, 0:3]).numpy()

        shadow = (denoised_all.sum(axis=-1, keepdims=True) / (denoised_plane.sum(axis=-1, keepdims=True) + 1e-7))
        return np.clip((1 - bunny_mask) * shadow * background, 0, 1) + bunny_mask * bunny

# 使用示例 — now accepts CLI args:
if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Render an object with an envmap and background')
    parser.add_argument('--background', '-b', required=True, help='Path to background image')
    parser.add_argument('--envmap', '-e', required=True, help='Path to envmap (EXR/HDRE)')
    parser.add_argument('--out', '-o', required=True, help='Output image path')
    parser.add_argument('--resolution', '-r', type=int, nargs=2, default=None, help='Target resolution as WIDTHxHEIGHT or WIDTH,HEIGHT (optional)')

    parser.add_argument('--camera_pos', type=float, nargs=3, default=[0,0,0], help='Camera position as x,y,z')
    parser.add_argument('--camera_rot', type=float, nargs=3, default=[0,180,0], help='Camera rotation as pitch,yaw,roll in degrees')
    parser.add_argument('--fov', type=float, default=120.0, help='Camera FOV in degrees')

    parser.add_argument('--plane_pos', type=float, nargs=3, default=[0,-1.5,-3.5], help='Plane position as x,y,z')
    parser.add_argument('--plane_rot', type=float, nargs=3, default=[-90,0,0], help='Plane rotation as pitch,yaw,roll in degrees')
    parser.add_argument('--plane_scale', type=float, default=2.0, help='Plane scale factor')

    parser.add_argument('--sphere_pos', type=float, nargs=3, default=[0,0,-1], help='Sphere position as x,y,z')
    parser.add_argument('--sphere_radius', type=float, default=0.5, help='Sphere radius')
    parser.add_argument('--sphere_roughness', type=float, default=0.01, help='Roughness alpha for the sphere material')
    parser.add_argument('--sphere_metallic', type=float, default=1.0, help='Metallic value for the sphere material')
    parser.add_argument('--sphere_color', type=float, nargs=3, default=[0.3,0.3,0.3], help='Sphere base color as r,g,b (0-1)')

    parser.add_argument('--flip', type=bool, default=False, help='Flip the background and output image horizontally')
    args = parser.parse_args()

    def _build_transform(pos, rot_deg, scale=None):
        pitch, yaw, roll = rot_deg
        t = mi.ScalarTransform4f().translate(pos) @ mi.ScalarTransform4f().rotate([1, 0, 0], pitch) @ mi.ScalarTransform4f().rotate([0, 1, 0], yaw) @ mi.ScalarTransform4f().rotate([0, 0, 1], roll)
        if scale is not None:
            t = t @ mi.ScalarTransform4f().scale(scale)
        return t

    # Load background
    bg_img = cv2.imread(args.background, -1)
    if bg_img is None:
        raise FileNotFoundError(f"Background not found: {args.background}")
    background = bg_img[:, :, 0:3][:, :, ::-1].astype(np.float32) / 255.0

    # Resize if resolution provided
    if args.resolution:
        width, height = args.resolution
        background = cv2.resize(background, (width, height))

    # Load envmap (keep HDR information)
    env = cv2.imread(args.envmap, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if env is None:
        raise FileNotFoundError(f"Envmap not found: {args.envmap}")
    envmap = env[:, :, 0:3][:, :, ::-1].astype(np.float32)

    if args.flip:
        background = cv2.flip(background, 1)

    # Create renderer with resolution matching background
    renderer = ObjectRenderer((background.shape[0], background.shape[1]), args.fov)

    # Update camera transform from CLI args
    cam_pos = args.camera_pos
    cam_rot = args.camera_rot
    renderer.camera_transform = _build_transform(cam_pos, cam_rot)

    # Update plane transform from CLI args (apply scale as before)
    plane_pos = args.plane_pos
    plane_rot = args.plane_rot
    plane_scale = [args.plane_scale, args.plane_scale, 1]
    renderer.plane_transform = _build_transform(plane_pos, plane_rot, scale=plane_scale)

    # Render plane (default demo parameters)
    sphere_pos = torch.tensor(args.sphere_pos)
    img = renderer.render_sphere(
        sphere_pos,
        torch.tensor(envmap).permute(2, 0, 1),
        torch.tensor(background).permute(2, 0, 1),
        args.sphere_radius,
        args.sphere_roughness,
        args.sphere_metallic,
        color=args.sphere_color
    )
    mask = renderer.render_sphere(
        sphere_pos,
        torch.tensor(envmap).permute(2, 0, 1),
        None,
        args.sphere_radius,
        args.sphere_roughness,
        args.sphere_metallic,
        color=args.sphere_color
    ).sum(axis=-1) > 0
    # img = renderer.render_plane(
    #     sphere_pos,
    #     torch.tensor(envmap).permute(2, 0, 1),
    #     None,
    #     args.sphere_radius,
    #     args.sphere_roughness,
    #     args.sphere_metallic
    # )
    img = renderer.render_shadow(
        sphere_pos,
        torch.tensor(envmap).permute(2, 0, 1),
        torch.tensor(background).permute(2, 0, 1),
        torch.tensor(img).permute(2, 0, 1),
        args.sphere_radius,
        args.sphere_roughness,
        args.sphere_metallic,
        color=args.sphere_color
    )

    if args.flip:
        img = cv2.flip(img, 1)
        mask = np.fliplr(mask)
    rms_contrast, mean_cpd, dominant_band_img = calculate_spatial_metrics((img * 255).astype(np.uint8), mask)
    print(f"Spatial Metrics - RMS Contrast: {rms_contrast}, Mean CPD: {mean_cpd}")

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(args.out, (img[:, :, ::-1] * 255).astype(np.uint8))
    cv2.imwrite(args.out.replace('.png', '_dominant_band.png'), dominant_band_img)