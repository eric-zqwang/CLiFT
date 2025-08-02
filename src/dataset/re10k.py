import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
import json
from einops import rearrange, repeat
import hydra
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix, Transform3d
from io import BytesIO
from src.utils.step_tracker import StepTracker
from src.dataset.view_sampler.view_sampler_evaluation import ViewSamplerEvaluation
from src.dataset.view_sampler.view_sampler_evaluation import ViewSamplerEvaluationCfg
from src.utils.shims import apply_crop_shim
import torchvision.transforms as tf

class Re10kDataset(Dataset):
    def __init__(self, cfg, split='train', view_sampler=None):
        self.cfg = cfg
        self.data_dir = cfg.data.data_dir
        self.split = split
        self.overfit = cfg.data.overfit
        self.image_size = cfg.data.image_size
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()


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
            with open(f'{self.view_sampler.cfg.index_path}', 'r') as f:
                data = json.load(f)
                data = {k: v for k, v in data.items() if k is not None and v is not None}
                self.data_lists = sorted([os.path.join(self.data_dir, path+".torch") for path in data.keys()])
        

        if self.overfit > 0:
            self.data_lists = [os.path.join(cfg.data.data_dir, 'test', '5aca87f95a9412c6.torch')] * 200
            if self.split == 'train':
                self.data_lists = self.data_lists[:180]
            elif self.split == 'test':
                self.data_lists = self.data_lists[180:]


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


    
    def __getitem__(self, idx):
        data = torch.load(self.data_lists[idx], weights_only=False)
        data_dict = {}
        
        poses = data['cameras']

        # extrinsics is a opencv c2w matrix
        extrinsics, intrinsics = self.convert_poses(poses)

        images = data['images']
        timestamps = data['timestamps']
        # url = data['url']
        key = data['key']


        out_data = self.view_sampler.sample(
            scene=key,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        )

        if isinstance(out_data, tuple):
            context_indices, target_indices = out_data[:2]

        # context_indices = torch.tensor([58, 70, 124, 133])
        # target_indices = torch.tensor([84, 102, 129])

        context_images = self.convert_images([images[idx] for idx in context_indices.tolist()])
        target_images = self.convert_images([images[idx] for idx in target_indices.tolist()])

        context_image_invalid = context_images.shape[1:] != (3, 360, 640)
        target_image_invalid = target_images.shape[1:] != (3, 360, 640)
        if context_image_invalid or target_image_invalid:
            print(
                f"Skipped bad example {data['url']}. Context shape was "
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
        
        # data_dict.update({
        #     'condition_views': context_images,
        #     'condition_views_extrinsics': extrinsics[context_indices],
        #     'condition_views_extrinsics_original': context_extrinsics_original,
        #     'condition_views_intrinsics': K_vec[context_indices],
        #     'sampled_views': target_images,
        #     'sampled_views_extrinsics': extrinsics[target_indices],
        #     'sampled_views_extrinsics_original': target_extrinsics_original,
        #     'sampled_views_intrinsics': K_vec[target_indices],
        #     'data_id': key,
        #     'condition_timestamps': context_timestamps,
        #     'sampled_timestamps': target_timestamps
        # })

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
            'sampled_views_intrinsics': target_views_intrinsics
        })
        return data_dict


def build_re10k_dataloader(cfg, step_tracker):

    # Initialize view samplers
    train_view_sampler = hydra.utils.instantiate(
        cfg.view_sampler,
        stage="train",
        step_tracker=step_tracker
    )
    
    if cfg.view_sampler.cfg.num_context_views == 2:
        index_path = "assets/evaluation_index_re10k.json"
    elif cfg.view_sampler.cfg.num_context_views == 4:
        index_path = "assets/evaluation_index_re10k_context4.json"
    else:
        raise ValueError(f"No evaluation json for {cfg.view_sampler.cfg.num_context_views} context views")

    val_view_sampler = ViewSamplerEvaluation(
        ViewSamplerEvaluationCfg(
            name="evaluation",
            index_path=index_path,
            num_context_views=cfg.view_sampler.cfg.num_context_views,
        ),
        stage="test",
        step_tracker=step_tracker
    )
    
    train_dataset = Re10kDataset(cfg, split='train', view_sampler=train_view_sampler)
    val_dataset = Re10kDataset(cfg, split='test', view_sampler=val_view_sampler)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.data.batch_size, 
        shuffle=True, 
        num_workers=cfg.data.num_workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.data.val_batch_size, 
        shuffle=False, 
        num_workers=cfg.data.num_workers,
        pin_memory=False
    )
    
    return train_loader, val_loader


def build_re10k_val_dataloader(cfg, step_tracker):
    if cfg.view_sampler.cfg.num_context_views == 2:
        index_path = "assets/evaluation_index_re10k.json"
    elif cfg.view_sampler.cfg.num_context_views == 4:
        index_path = "assets/evaluation_index_re10k_video_context4_data_100.json"
    else:
        raise ValueError(f"No evaluation json for {cfg.view_sampler.cfg.num_context_views} context views")
    
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

    return val_loader


def build_re10k_test_dataloader(cfg):
    step_tracker = StepTracker()
    test_view_sampler = hydra.utils.instantiate(
        cfg.view_sampler,
        stage="test",
        step_tracker=step_tracker
    )
    
    test_dataset = Re10kDataset(cfg, split='test', view_sampler=test_view_sampler)

    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.data.val_batch_size, 
        shuffle=False, 
        num_workers=cfg.data.num_workers,
        pin_memory=False
    )
    return test_loader