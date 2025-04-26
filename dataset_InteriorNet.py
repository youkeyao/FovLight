import torch
import numpy as np
from collections import namedtuple

def world_to_camera_projection(p_world, intrinsics, world_to_camera):
    """Project world coordinates to camera coordinates."""
    height, width = p_world.shape[:2]
    p_world_homogeneous = torch.cat([p_world, torch.ones(height, width, 1, device=p_world.device)], dim=-1)
    intrinsics = intrinsics.unsqueeze(0).unsqueeze(0).expand(height, width, -1, -1)
    world_to_camera = world_to_camera.unsqueeze(0).unsqueeze(0).expand(height, width, -1, -1)
    p_camera = torch.bmm(world_to_camera.view(-1, 4, 4), p_world_homogeneous.view(-1, 4, 1)).view(height, width, 4)
    p_camera_z = p_camera * torch.tensor([1., 1., -1.], device=p_world.device)
    p_image = torch.bmm(intrinsics.view(-1, 3, 3), p_camera_z[:, :, :3].view(-1, 3, 1)).view(height, width, 3)
    return p_image[:, :, :2] / (p_image[:, :, -1:] + 1e-8), p_image[:, :, -1]

def camera_to_world_projection(depth, intrinsics, camera_to_world):
    """Project camera coordinates to world coordinates."""
    height, width = depth.shape[:2]
    y, x = torch.meshgrid(torch.arange(height, device=depth.device), torch.arange(width, device=depth.device), indexing='ij')
    p_pixel = torch.stack([x, y], dim=-1).float()
    p_pixel_homogeneous = torch.cat([p_pixel, torch.ones(height, width, 1, device=depth.device)], dim=-1)
    camera_to_world = camera_to_world.unsqueeze(0).unsqueeze(0).expand(height, width, -1, -1)
    intrinsics = intrinsics.unsqueeze(0).unsqueeze(0).expand(height, width, -1, -1)
    p_image = torch.bmm(torch.inverse(intrinsics).view(-1, 3, 3), p_pixel_homogeneous.view(-1, 3, 1)).view(height, width, 3)
    lookat_axis = torch.tensor([0., 0., 1.], device=depth.device).unsqueeze(0).unsqueeze(0).expand(height, width, -1)
    z = depth * torch.sum(torch.nn.functional.normalize(p_image, dim=-1) * lookat_axis, dim=-1, keepdim=True)
    p_camera = z * p_image
    p_camera = p_camera * torch.tensor([1., 1., -1.], device=depth.device)
    p_camera_homogeneous = torch.cat([p_camera, torch.ones(height, width, 1, device=depth.device)], dim=-1)
    p_world = torch.bmm(camera_to_world.view(-1, 4, 4), p_camera_homogeneous.view(-1, 4, 1)).view(height, width, 4)
    return p_world[:, :, :3]

def image_overlap(depth1, pose1_c2w, depth2, pose2_c2w, intrinsics):
    """Determines the overlap of two images."""
    pose1_w2c = torch.inverse(torch.cat([pose1_c2w, torch.tensor([[0., 0., 0., 1.]], device=pose1_c2w.device)], dim=0))[:3]
    pose2_w2c = torch.inverse(torch.cat([pose2_c2w, torch.tensor([[0., 0., 0., 1.]], device=pose2_c2w.device)], dim=0))[:3]

    p_world1 = camera_to_world_projection(depth1, intrinsics, pose1_c2w)
    p_image1_in_2, z1_c2 = world_to_camera_projection(p_world1, intrinsics, pose2_w2c)

    p_world2 = camera_to_world_projection(depth2, intrinsics, pose2_c2w)
    p_image2_in_1, z2_c1 = world_to_camera_projection(p_world2, intrinsics, pose1_w2c)

    height, width = depth1.shape[:2]
    height = float(height)
    width = float(width)

    mask_h2_in_1 = (p_image2_in_1[:, :, 1] <= height) & (p_image2_in_1[:, :, 1] >= 0)
    mask_w2_in_1 = (p_image2_in_1[:, :, 0] <= width) & (p_image2_in_1[:, :, 0] >= 0)
    mask2_in_1 = (mask_h2_in_1 & mask_w2_in_1) & (z2_c1 > 0)

    mask_h1_in_2 = (p_image1_in_2[:, :, 1] <= height) & (p_image1_in_2[:, :, 1] >= 0)
    mask_w1_in_2 = (p_image1_in_2[:, :, 0] <= width) & (p_image1_in_2[:, :, 0] >= 0)
    mask1_in_2 = (mask_h1_in_2 & mask_w1_in_2) & (z1_c2 > 0)

    return mask1_in_2, mask2_in_1

class ViewTrip(namedtuple('ViewTrip', [
    'scene_id', 'sequence_id', 'timestamp', 'rgb', 'pano', 'depth',
    'normal', 'mask', 'pose', 'intrinsics', 'resolution'
])):
    """A class for handling a trip of views."""
    
    def overlap_mask(self):
        """Generate overlap masks for the views."""
        intrinsics = self.intrinsics * torch.tensor([[1., 1., 1.], [1., -1., 1.], [1., 1., 1.]])
        mask1_in_2, mask2_in_1 = image_overlap(self.depth[0], self.pose[0], self.depth[1], self.pose[1], intrinsics)
        masks = torch.stack([mask1_in_2, mask2_in_1], dim=0)
        return ViewTrip(
            self.scene_id, self.sequence_id, self.timestamp, self.rgb,
            self.pano, self.depth, self.normal, masks, self.pose,
            self.intrinsics, self.resolution
        )
    
    def reverse(self):
        """Returns the reverse of the trip."""
        return ViewTrip(
            self.scene_id, self.sequence_id,
            self.timestamp.flip(0), self.rgb.flip(0), self.pano.flip(0),
            self.depth.flip(0), self.normal.flip(0), self.mask.flip(0),
            self.pose.flip(0), self.intrinsics, self.resolution
        )
    
    def random_reverse(self):
        """Returns either the trip or its reverse, with equal probability."""
        if np.random.rand() < 0.5:
            return self
        else:
            return self.reverse()
    
    def deterministic_reverse(self):
        """Returns either the trip or its reverse, based on the sequence id."""
        hash_bucket = hash(self.scene_id) % 2
        if hash_bucket == 0:
            return self
        else:
            return self.reverse()
    
    def hash_in_range(self, buckets, base, limit):
        """Return true if the hashing key falls in the range [base, limit)."""
        hash_bucket = hash(self.scene_id) % buckets
        return base <= hash_bucket < limit

class ViewSequence(namedtuple('ViewSequence', [
    'scene_id', 'sequence_id', 'timestamp', 'rgb', 'pano', 'depth',
    'normal', 'pose', 'intrinsics', 'resolution'
])):
    """A class for handling a sequence of views."""
    
    def subsequence(self, stride):
        """Extract a subsequence with a given stride."""
        return ViewSequence(
            self.scene_id, self.sequence_id,
            self.timestamp[::stride],
            self.rgb[::stride],
            self.pano[::stride],
            self.depth[::stride],
            self.normal[::stride],
            self.pose[::stride],
            self.intrinsics[::stride],
            self.resolution[::stride]
        )
    
    def random_subsequence(self, min_stride, max_stride):
        """Extract a random subsequence with a random stride."""
        random_stride = np.random.randint(min_stride, max_stride + 1)
        return self.subsequence(random_stride)
    
    def generate_trips(self, min_gap=1, max_gap=5):
        """Generate a dataset of training triplets with an offset between three frames."""
        def mapper(timestamp_trips, rgb_trips, pano_trips, depth_trips, normal_trips, pose_trips):
            """A function mapping a data tuple to ViewTrip."""
            return ViewTrip(
                self.scene_id, self.sequence_id, timestamp_trips,
                rgb_trips, pano_trips, depth_trips, normal_trips,
                torch.zeros(1), pose_trips, self.intrinsics[0], self.resolution[0]
            )
        
        length = len(self.timestamp)
        assert max_gap < length, f"max_gap ({max_gap}) must be less than sequence length ({length})"
        
        timestamp_trips = []
        rgb_trips = []
        pano_trips = []
        depth_trips = []
        normal_trips = []
        pose_trips = []
        
        for stride in range(min_gap, max_gap + 1):
            inds = np.arange(stride, length - stride)
            inds_jitter = np.random.randint(-40, 40, size=len(inds))
            rand_inds = np.clip(inds + inds_jitter, 0, length - 1)
            
            timestamp = np.stack([
                self.timestamp[:-2 * stride], self.timestamp[2 * stride:],
                self.timestamp[stride:-stride], self.timestamp[rand_inds]
            ], axis=1)
            
            rgb = np.stack([
                self.rgb[:-2 * stride], self.rgb[2 * stride:],
                self.rgb[stride:-stride], self.rgb[rand_inds]
            ], axis=1)
            
            pano = np.stack([
                self.pano[:-2 * stride], self.pano[2 * stride:],
                self.pano[stride:-stride], self.pano[rand_inds]
            ], axis=1)
            
            depth = np.stack([
                self.depth[:-2 * stride], self.depth[2 * stride:],
                self.depth[stride:-stride], self.depth[rand_inds]
            ], axis=1)
            
            normal = np.stack([
                self.normal[:-2 * stride], self.normal[2 * stride:],
                self.normal[stride:-stride], self.normal[rand_inds]
            ], axis=1)
            
            pose = np.stack([
                self.pose[:-2 * stride], self.pose[2 * stride:],
                self.pose[stride:-stride], self.pose[rand_inds]
            ], axis=1)
            
            timestamp_trips.append(timestamp)
            rgb_trips.append(rgb)
            pano_trips.append(pano)
            depth_trips.append(depth)
            normal_trips.append(normal)
            pose_trips.append(pose)
        
        timestamp_trips = np.concatenate(timestamp_trips, axis=0)
        rgb_trips = np.concatenate(rgb_trips, axis=0)
        pano_trips = np.concatenate(pano_trips, axis=0)
        depth_trips = np.concatenate(depth_trips, axis=0)
        normal_trips = np.concatenate(normal_trips, axis=0)
        pose_trips = np.concatenate(pose_trips, axis=0)
        
        dataset = list(zip(timestamp_trips, rgb_trips, pano_trips, depth_trips, normal_trips, pose_trips))
        return [mapper(*trip) for trip in dataset]
    
    def length(self):
        """Returns the length of the sequence."""
        return len(self.timestamp)
    
    def reverse(self):
        """Returns the reverse of the sequence."""
        return ViewSequence(
            self.scene_id, self.sequence_id,
            self.timestamp[::-1],
            self.rgb[::-1], self.pano[::-1], self.depth[::-1],
            self.normal[::-1], self.pose[::-1],
            self.intrinsics[::-1], self.resolution[::-1]
        )
    
    def random_reverse(self):
        """Returns either the sequence or its reverse, with equal probability."""
        if np.random.rand() < 0.5:
            return self
        else:
            return self.reverse()
    
    def deterministic_reverse(self):
        """Returns either the sequence or its reverse, based on the sequence id."""
        hash_bucket = hash(self.scene_id) % 2
        if hash_bucket == 0:
            return self
        else:
            return self.reverse()
    
    def hash_in_range(self, buckets, base, limit):
        """Return true if the hashing key falls in the range [base, limit)."""
        hash_bucket = hash(self.scene_id) % buckets
        return base <= hash_bucket < limit