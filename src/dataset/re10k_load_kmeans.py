import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
import json
from packaging import version as pver
from tqdm import tqdm
# from src.utils.camera import load_metadata
from einops import rearrange, repeat
import hydra
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix, Transform3d
from io import BytesIO
from src.utils.step_tracker import StepTracker
from src.dataset.view_sampler.view_sampler_evaluation import ViewSamplerEvaluation
from src.dataset.view_sampler.view_sampler_evaluation import ViewSamplerEvaluationCfg
from tqdm import tqdm
from src.utils.shims import apply_crop_shim
import torchvision.transforms as tf
import random

class Re10kDataset(Dataset):
    def __init__(self, cfg, split='train', view_sampler=None):
        self.cfg = cfg
        self.data_dir = cfg.data.data_dir
        self.split = split
        self.overfit = cfg.data.overfit
        self.image_size = cfg.data.image_size
        self.to_tensor = tf.ToTensor()

        self.kmeans_dir = cfg.data.kmeans_dir
        self.data_dir = os.path.join(self.data_dir, self.split)

        if self.split == 'train':
            # Load sequence data from JSON
            sequence_file = f'assets/sequences_{split}.json'
            with open(sequence_file, 'r') as f:
                sequences = json.load(f)
                # Filter sequences with length > 45 and extract their keys
                valid_sequences = {seq['key'] for seq in sequences if seq['length'] > 45}

            # Create a list of valid paths that exist
            self.data_lists = []
            for path in os.listdir(self.data_dir):
                if path.endswith(".torch") and os.path.splitext(path)[0] in valid_sequences:
                    full_path = os.path.join(self.data_dir, path)
                    if os.path.exists(full_path):
                        self.data_lists.append(full_path)
                    else:
                        print(f"Warning: File not found: {full_path}")
                                
            self.data_lists.sort()
        elif self.split == 'test':
            # import pdb; pdb.set_trace()
            with open(f'assets/evaluation_index_re10k_video_context4_data_100.json', 'r') as f:
                data = json.load(f)
                data = {k: v for k, v in data.items() if k is not None and v is not None}
                self.data_lists = sorted([os.path.join(self.data_dir, path+".torch") for path in data.keys()])


    def convert_images(self, images):
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)
    

    def normalize_camera_params(self, context_extrinsics, target_extrinsics):
        n_context = len(context_extrinsics)        

        # 2. Extract rotations (R) and translations (t) from context extrinsics
        R_context = context_extrinsics[:, :3, :3]  # Shape: (N_context, 3, 3)
        t_context = context_extrinsics[:, :3, 3]   # Shape: (N_context, 3)

        # 3. Calculate t_mean (average translation)
        t_mean = t_context.mean(dim=0).unsqueeze(0)  # Shape: (1, 3)

        # 4. Calculate r_mean (average rotation) using PyTorch3D quaternions
        quaternions = matrix_to_quaternion(R_context)  # Shape: (N_context, 4)
        mean_quaternion = quaternions.mean(dim=0)      # Shape: (4,)
        mean_quaternion = mean_quaternion / torch.norm(mean_quaternion)  # Normalize
        r_mean = quaternion_to_matrix(mean_quaternion).unsqueeze(0)  # Shape: (1, 3, 3)

        world_transform = torch.eye(4)  # Shape: (4, 4)
        world_transform[:3, :3] = r_mean
        world_transform[:3, 3] = t_mean

        world_transform_inv = torch.linalg.inv(world_transform) 

        all_extrinsics = torch.cat([context_extrinsics, target_extrinsics], dim=0)  # Shape: (N_all, 4, 4)
       
        transformed_extrinsics = torch.bmm(world_transform_inv.unsqueeze(0).expand(all_extrinsics.size(0), -1, -1),
                                        all_extrinsics)  # Shape: (N_all, 4, 4)
        
        # Extract translations from transformed extrinsics
        translations = transformed_extrinsics[:, :3, 3]  # Shape: (N_all, 3)

        # Find the maximum absolute coordinate value
        max_coord = translations.abs().max()  # A single scalar value
        scale = 1 / max_coord
        transformed_extrinsics[:, :3, 3] *= scale

        # save_extrinsics_to_txt(transformed_extrinsics, 'test.txt')

        return transformed_extrinsics[:n_context], transformed_extrinsics[n_context:]


    def get_images(self, scene_dir):
        images_path = os.path.join(scene_dir, 'images_8')
        image_path_list = []
        for image_path in os.listdir(images_path):
            if image_path.endswith('.png'):
                image_path_list.append(os.path.join(images_path, image_path))
        # Sort based on frame number (extract number between 'frame_' and '.png')
        image_path_list.sort(key=lambda x: int(os.path.basename(x).split('frame_')[1].split('.')[0]))
        return image_path_list


    def __len__(self):
        return len(self.data_lists)


    def load_images(self, path):
        """Load and normalize an image to [-1, 1] range."""

        img = np.array(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0

        return img
        
    def load_metadata(self, scene_id):
        """Load metadata.json file for a given scene_id"""
        metadata_path = os.path.join(self.kmeans_dir, scene_id, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            
            # Check if metadata contains numbered keys (e.g., "1", "2", "3")
            # If so, sample one randomly
            numeric_keys = [k for k in metadata_dict.keys() if k.isdigit()]
            if numeric_keys:
                # Randomly sample one metadata entry
                sampled_key = random.choice(numeric_keys)
                return metadata_dict[sampled_key]
            else:
                # Return the entire metadata if no numbered keys
                return metadata_dict
        else:
            print(f"Warning: Metadata not found for scene {scene_id}")
            return None
            
    def pad_anchor_idx(self, anchor_idx, max_length=4096):
        """Pad anchor_idx to fixed length with zeros"""
        if anchor_idx is None:
            return torch.zeros(max_length, dtype=torch.long)
        
        anchor_idx_tensor = torch.tensor(anchor_idx, dtype=torch.long)
        padded = torch.zeros(max_length, dtype=torch.long)
        length = min(len(anchor_idx_tensor), max_length)
        padded[:length] = anchor_idx_tensor[:length]
        return padded


    def convert_poses(self, poses):
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style C2W matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32),
                     "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)

        c2w = w2c.inverse()

        return c2w, intrinsics
    
    def scale_intrinsics_to_pixel_coords(self, intrinsics):
        h, w = self.image_size  # (height, width)

        fx = intrinsics[:, 0, 0] * w
        fy = intrinsics[:, 1, 1] * h
        cx = intrinsics[:, 0, 2] * w
        cy = intrinsics[:, 1, 2] * h

        return torch.stack([fx, fy, cx, cy], dim=1)
    
    def sample_target_idx(self, condition_view_idx):
        """Sample target view indices within the range of condition_view_idx"""
        if len(condition_view_idx) < 2:
            return condition_view_idx  # Not enough views to sample, return the same

        min_idx = min(condition_view_idx)
        max_idx = max(condition_view_idx)
        
        # Sample random indices within the range, excluding the condition views
        candidate_indices = set(range(min_idx, max_idx + 1)) - set(condition_view_idx)
        
        if not candidate_indices:  # If there are no candidates, use condition views
            return torch.tensor(random.sample(condition_view_idx.tolist(), min(4, len(condition_view_idx))))
        
        # Convert to list and sample random indices
        candidate_indices = list(candidate_indices)
        num_targets = self.cfg.view_sampler.cfg.num_target_views  # Sample up to 4 target views
        target_indices = random.sample(candidate_indices, num_targets)
        
        return torch.tensor(target_indices)

    def __getitem__(self, idx):
        data = torch.load(self.data_lists[idx], weights_only=False)
        data_dict = {}

        images = data['images']
        timestamps = data['timestamps']
        key = data['key']
        
        poses = data['cameras']
        # extrinsics is a opencv c2w matrix
        extrinsics, intrinsics = self.convert_poses(poses)

        metadata = self.load_metadata(key)
        anchor_idx = metadata.get('anchor_idx', []) if metadata else []
        
        num_tokens = len(anchor_idx)
        padded_anchor_idx = self.pad_anchor_idx(anchor_idx)
        
        labels = metadata.get('labels', None) if metadata else None
        
        condition_view_idx = torch.tensor(metadata.get('condition_view_idx', []), dtype=torch.long) if metadata else torch.tensor([])
        if len(condition_view_idx) == 0:
            # Fallback if no condition_view_idx in metadata
            condition_view_idx = torch.tensor([0, len(images)//4, len(images)//2, 3*len(images)//4], dtype=torch.long)
            
        # Make sure condition_view_idx is within valid range
        condition_view_idx = condition_view_idx[condition_view_idx < len(images)]
        if len(condition_view_idx) == 0:
            condition_view_idx = torch.tensor([0], dtype=torch.long)
            
        target_indices = self.sample_target_idx(condition_view_idx)
        
        # Convert to indices that can be used to index into images
        context_indices = condition_view_idx

        # Load images for context and target views
        context_images = self.convert_images([images[idx] for idx in context_indices.tolist()])
        target_images = self.convert_images([images[idx] for idx in target_indices.tolist()])

        context_image_invalid = context_images.shape[1:] != (3, 360, 640)
        target_image_invalid = target_images.shape[1:] != (3, 360, 640)
        if context_image_invalid or target_image_invalid:
            print(
                f"Skipped bad example {key}. Context shape was "
                f"{context_images.shape} and target shape was "
                f"{target_images.shape}."
            )

        context_timestamps = timestamps[context_indices]
        target_timestamps = timestamps[target_indices]

        data_dict.update({
            'context_timestamps': context_timestamps,
            'target_timestamps': target_timestamps,
            'data_id': key,
        })
        
        if self.split == 'train':
            data_dict.update({
                'labels': torch.tensor(labels),
                'num_tokens': num_tokens,
                'anchor_idx': padded_anchor_idx,
            })
        
        context_extrinsics, target_extrinsics = self.normalize_camera_params(
            extrinsics[context_indices],
            extrinsics[target_indices],
        )

        example = {
            "context": {
                "extrinsics": context_extrinsics,
                "intrinsics": intrinsics[context_indices],
                "image": context_images,
                "index": context_indices,
            },
            "target": {
                "extrinsics": target_extrinsics,
                "intrinsics": intrinsics[target_indices],
                "image": target_images,
                "index": target_indices,
            },
            "scene": key,
        }

        example = apply_crop_shim(example, tuple(self.image_size))

        condition_views_intrinsics = self.scale_intrinsics_to_pixel_coords(
            example['context']['intrinsics'],
        )
        target_views_intrinsics = self.scale_intrinsics_to_pixel_coords(
            example['target']['intrinsics'],
        )

        data_dict.update({
            'condition_views': example['context']['image'].permute(0, 2, 3, 1),
            'condition_views_extrinsics': example['context']['extrinsics'],
            'condition_views_intrinsics': condition_views_intrinsics,
            'sampled_views': example['target']['image'].permute(0, 2, 3, 1),
            'sampled_views_extrinsics': example['target']['extrinsics'],
            'sampled_views_intrinsics': target_views_intrinsics,
            "condition_view_idx": context_indices,
        })
        return data_dict


def build_re10k_dataloader(cfg, step_tracker):
    train_dataset = Re10kDataset(cfg, split='train')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.data.batch_size, 
        shuffle=True, 
        num_workers=cfg.data.num_workers,
        pin_memory=False
    )

    index_path = "assets/evaluation_index_re10k_video_context4_data_100.json"

    val_view_sampler = ViewSamplerEvaluation(
        ViewSamplerEvaluationCfg(
            name="evaluation",
            index_path=index_path,
            num_context_views=cfg.view_sampler.cfg.num_context_views,
        ),
        stage="test",
        step_tracker=step_tracker
    )
    
    val_dataset = Re10kDataset(cfg, split='test', view_sampler=val_view_sampler)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.data.val_batch_size, 
        shuffle=False, 
        num_workers=cfg.data.num_workers,
        pin_memory=False
    )

    return train_loader, val_loader
