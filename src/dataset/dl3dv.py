import json
import os
import random

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from einops import rearrange, repeat
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.dataset.view_sampler.view_sampler_evaluation import (
    ViewSamplerEvaluation,
    ViewSamplerEvaluationCfg,
)
from src.utils.camera import get_fov
from src.utils.shims import apply_crop_shim
from src.utils.step_tracker import StepTracker

# DL3DV-10K images_8 frames are stored at 270x480; scenes at other resolutions
# are corrupt / mis-processed and are skipped.
DL3DV_IMAGE_SHAPE = (270, 480)


def pad_context_views(views, extrinsics, intrinsics, timestamps, max_context_views):
    """Right-pad a variable number of context views to ``max_context_views``.

    Only used at train time, where a batch mixes scenes with different numbers
    of context views. The real count is tracked separately (``num_context_views``)
    so the model can mask the padding.
    """
    if views.shape[0] >= max_context_views:
        return (
            views[:max_context_views],
            extrinsics[:max_context_views],
            intrinsics[:max_context_views],
            timestamps[:max_context_views],
        )

    padded_views = torch.zeros(max_context_views, *views.shape[1:], device=views.device)
    padded_views[: views.shape[0]] = views

    padded_extrinsics = torch.zeros(max_context_views, 4, 4, device=extrinsics.device)
    padded_extrinsics[: extrinsics.shape[0]] = extrinsics

    padded_intrinsics = torch.zeros(max_context_views, 4, device=intrinsics.device)
    padded_intrinsics[: intrinsics.shape[0]] = intrinsics

    padded_timestamps = torch.zeros(max_context_views, dtype=timestamps.dtype, device=timestamps.device)
    padded_timestamps[: timestamps.shape[0]] = timestamps

    return padded_views, padded_extrinsics, padded_intrinsics, padded_timestamps


class DL3DVDataset(Dataset):
    """DL3DV-10K dataset for CLiFT novel-view synthesis.

    Each scene is a directory with a ``transforms.json`` (Blender-convention
    camera poses + shared intrinsics) and an ``images_8/`` folder of frames.
    Context / target views are chosen by the view sampler; at test time they come
    from a precomputed evaluation index (``view_sampler.cfg.index_path``).
    """

    def __init__(self, cfg, split="train", view_sampler=None, pad_context=True, return_view_indices=False):
        self.cfg = cfg
        self.data_dir = cfg.data.data_dir
        self.split = split
        self.overfit = cfg.data.overfit
        self.image_size = cfg.data.image_size  # (height, width) = (256, 448)
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        # pad_context=False + return_view_indices=True is used by the K-means
        # annotation pass (save_kmeans.py), which needs unpadded views and the
        # sampled context indices so they can be stored alongside the clusters.
        self.pad_context = pad_context
        self.return_view_indices = return_view_indices

        all_scenes = sorted(
            f for f in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, f))
        )
        if self.overfit != -1:
            all_scenes = all_scenes[: self.overfit] * 200

        if split == "train":
            if self.overfit == -1:
                # Exclude the held-out evaluation scenes from training (skipped
                # in overfit debug mode, which may deliberately target them).
                eval_data_path = "assets/dl3dv_start_0_distance_50_ctx_4v_video_0_50.json"
                eval_scene_ids = set()
                if os.path.exists(eval_data_path):
                    with open(eval_data_path, "r") as f:
                        eval_scene_ids = set(json.load(f).keys())
                all_scenes = [scene for scene in all_scenes if scene not in eval_scene_ids]
        else:
            with open(self.view_sampler.cfg.index_path, "r") as f:
                all_scenes = sorted(json.load(f).keys())

        self.data_lists = []
        for scene in tqdm(all_scenes, desc=f"Indexing DL3DV ({split})"):
            scene_dir = os.path.join(self.data_dir, scene)
            meta_path = os.path.join(scene_dir, "transforms.json")
            if not os.path.exists(meta_path):
                print(f"scene {scene} have no metadata")
                continue

            metadata = self.load_metadata(meta_path)
            image_path_list = self.get_images(scene_dir)

            if len(image_path_list) > 0:
                first_shape = np.array(Image.open(image_path_list[0]).convert("RGB")).shape[:2]
                if first_shape != DL3DV_IMAGE_SHAPE:
                    print(f"first image {scene} have invalid shape {first_shape}")
                    continue

            if len(image_path_list) == 0:
                print(f"scene {scene} have no images")
                continue

            if len(image_path_list) != metadata["timestamps"].shape[0]:
                print(f"scene {scene} have different number of images and timestamps")
                continue

            self.data_lists.append({
                "images_path": image_path_list,
                "camera_params": metadata["cameras"],
                "timestamps": metadata["timestamps"],
                "url": metadata["url"],
            })

    def load_metadata(self, example_path):
        """Parse a DL3DV ``transforms.json`` into per-frame cameras and timestamps.

        Poses are converted from Blender c2w to an OpenCV w2c matrix; intrinsics
        are stored normalized by the image resolution (fx/w, fy/h, cx/w, cy/h).
        """
        blender2opencv = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        url = str(example_path).split("/")[-2]
        with open(example_path, "r") as f:
            meta_data = json.load(f)

        store_h, store_w = meta_data["h"], meta_data["w"]
        fx, fy, cx, cy = meta_data["fl_x"], meta_data["fl_y"], meta_data["cx"], meta_data["cy"]
        saved_fx = float(fx) / float(store_w)
        saved_fy = float(fy) / float(store_h)
        saved_cx = float(cx) / float(store_w)
        saved_cy = float(cy) / float(store_h)

        timestamps = []
        cameras = []
        for frame in meta_data["frames"]:
            timestamps.append(
                int(os.path.basename(frame["file_path"]).split(".")[0].split("_")[-1])
            )
            camera = [saved_fx, saved_fy, saved_cx, saved_cy, 0.0, 0.0]
            opencv_c2w = np.array(frame["transform_matrix"]) @ blender2opencv
            camera.extend(np.linalg.inv(opencv_c2w)[:3].flatten().tolist())
            cameras.append(np.array(camera))

        timestamps = torch.tensor(timestamps, dtype=torch.int64)
        cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)
        return {"url": url, "timestamps": timestamps, "cameras": cameras}

    def preprocess_poses(self, in_c2ws, target_c2ws, scene_scale_factor=1):
        """Normalize a scene's cameras into a canonical, unit-scale frame.

        Aligns the world to the average context camera (mean center + mean
        forward/down/right axes) and rescales so the largest context camera
        offset is 1. This makes the model invariant to the arbitrary DL3DV world
        frame and metric scale.
        """
        center = in_c2ws[:, :3, 3].mean(0)
        avg_forward = F.normalize(in_c2ws[:, :3, 2].mean(0), dim=-1)
        avg_down = in_c2ws[:, :3, 1].mean(0)
        avg_right = F.normalize(torch.cross(avg_down, avg_forward, dim=-1), dim=-1)
        avg_down = F.normalize(torch.cross(avg_forward, avg_right, dim=-1), dim=-1)

        avg_pose = torch.eye(4, device=in_c2ws.device)
        avg_pose[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
        avg_pose[:3, 3] = center
        avg_pose = torch.linalg.inv(avg_pose)

        in_c2ws = avg_pose @ in_c2ws
        target_c2ws = avg_pose @ target_c2ws

        scene_scale = scene_scale_factor * torch.max(torch.abs(in_c2ws[:, :3, 3]))
        in_c2ws[:, :3, 3] /= scene_scale
        target_c2ws[:, :3, 3] /= scene_scale
        return in_c2ws, target_c2ws

    def get_images(self, scene_dir):
        images_path = os.path.join(scene_dir, "images_8")
        image_path_list = [
            os.path.join(images_path, p) for p in os.listdir(images_path) if p.endswith(".png")
        ]
        image_path_list.sort(key=lambda x: int(os.path.basename(x).split("frame_")[1].split(".")[0]))
        return image_path_list

    def __len__(self):
        return len(self.data_lists)

    def load_images(self, path):
        return self.to_tensor(Image.open(path).convert("RGB"))

    def convert_poses(self, poses):
        """Split the flat camera vector into (c2w extrinsics, normalized K)."""
        b, _ = poses.shape

        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        c2w = w2c.inverse()
        return c2w, intrinsics

    def scale_intrinsics_to_pixel_coords(self, intrinsics):
        h, w = self.image_size
        fx = intrinsics[:, 0, 0] * w
        fy = intrinsics[:, 1, 1] * h
        cx = intrinsics[:, 0, 2] * w
        cy = intrinsics[:, 1, 2] * h
        return torch.stack([fx, fy, cx, cy], dim=1)

    def __getitem__(self, idx):
        while True:
            data = self.data_lists[idx]
            poses = data["camera_params"]
            extrinsics, intrinsics = self.convert_poses(poses)  # c2w, normalized K
            images_paths = data["images_path"]
            timestamps = data["timestamps"]
            url = data["url"]

            # Skip fisheye-like scenes the model was not trained on.
            if (get_fov(intrinsics).rad2deg() > 100.0).any():
                idx = random.randint(0, len(self.data_lists) - 1)
                print(f"Skipping scene {url} due to large FOV")
                continue

            out_data = self.view_sampler.sample(
                scene=url,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                min_context_views=self.cfg.data.min_context_views,
                max_context_views=self.cfg.data.max_context_views,
            )
            context_indices, target_indices = out_data[:2]

            # Training scenes may contain degenerate poses or unreadable frames;
            # resample the views instead of crashing the epoch.
            if self.split == "train" and not self._valid_cameras(extrinsics, context_indices, target_indices):
                continue

            try:
                context_images = torch.stack(
                    [self.load_images(images_paths[i.item()]) for i in context_indices]
                )
                target_images = torch.stack(
                    [self.load_images(images_paths[i.item()]) for i in target_indices]
                )
            except Exception as e:
                if self.split != "train":
                    raise
                print(f"Error processing images and poses: {e}")
                continue

            context_extrinsics, target_extrinsics = self.preprocess_poses(
                extrinsics[context_indices], extrinsics[target_indices]
            )
            context_timestamps = timestamps[context_indices]
            target_timestamps = timestamps[target_indices]
            break

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
            "scene": url,
        }
        example = apply_crop_shim(example, tuple(self.image_size))

        condition_views_intrinsics = self.scale_intrinsics_to_pixel_coords(example["context"]["intrinsics"])
        target_views_intrinsics = self.scale_intrinsics_to_pixel_coords(example["target"]["intrinsics"])
        num_context_views = example["context"]["image"].shape[0]

        if self.split == "train" and self.pad_context:
            condition_views, condition_extrinsics, condition_views_intrinsics, context_timestamps = (
                pad_context_views(
                    example["context"]["image"],
                    example["context"]["extrinsics"],
                    condition_views_intrinsics,
                    context_timestamps,
                    self.cfg.data.max_context_views,
                )
            )
        else:
            condition_views = example["context"]["image"]
            condition_extrinsics = example["context"]["extrinsics"]

        item = {
            "condition_views": condition_views.permute(0, 2, 3, 1),
            "condition_views_extrinsics": condition_extrinsics,
            "condition_views_intrinsics": condition_views_intrinsics,
            "num_context_views": num_context_views,
            "sampled_views": example["target"]["image"].permute(0, 2, 3, 1),
            "sampled_views_extrinsics": example["target"]["extrinsics"],
            "sampled_views_intrinsics": target_views_intrinsics,
            "data_id": url,
            "condition_timestamps": context_timestamps,
            "sampled_timestamps": target_timestamps,
        }
        if self.return_view_indices:
            item["condition_view_idx"] = example["context"]["index"]
        return item

    def _valid_cameras(self, extrinsics, context_indices, target_indices):
        """Reject degenerate poses: NaN / non-rigid rotations or runaway translations."""
        for indices in (context_indices, target_indices):
            rotations = extrinsics[indices][:, :3, :3]
            det = torch.det(rotations)
            if torch.isnan(det).any() or not torch.allclose(det, det.new_tensor(1.0)):
                print("invalid extrinsics")
                return False
            if (extrinsics[indices][:, :3, 3] > 1e3).any():
                print("extremely large camera translation")
                return False
        return True


def _build_eval_view_sampler(index_path, num_context_views, step_tracker):
    return ViewSamplerEvaluation(
        ViewSamplerEvaluationCfg(
            name="evaluation",
            index_path=index_path,
            num_context_views=num_context_views,
        ),
        stage="test",
        step_tracker=step_tracker,
    )


def build_dl3dv_dataloader(cfg, step_tracker):
    train_view_sampler = hydra.utils.instantiate(cfg.view_sampler, stage="train", step_tracker=step_tracker)
    val_view_sampler = _build_eval_view_sampler(
        "assets/dl3dv_start_0_distance_50_ctx_4v_video_0_50.json",
        cfg.view_sampler.cfg.num_context_views,
        step_tracker,
    )

    train_dataset = DL3DVDataset(cfg, split="train", view_sampler=train_view_sampler)
    val_dataset = DL3DVDataset(cfg, split="val", view_sampler=val_view_sampler)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.val_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def build_dl3dv_val_dataloader(cfg, step_tracker):
    val_view_sampler = _build_eval_view_sampler(
        "assets/dl3dv_start_0_distance_50_ctx_4v_video_0_50.json",
        cfg.view_sampler.cfg.num_context_views,
        step_tracker,
    )
    val_dataset = DL3DVDataset(cfg, split="val", view_sampler=val_view_sampler)
    return DataLoader(
        val_dataset,
        batch_size=cfg.data.val_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )


def build_dl3dv_test_dataloader(cfg):
    step_tracker = StepTracker()
    test_view_sampler = hydra.utils.instantiate(cfg.view_sampler, stage="test", step_tracker=step_tracker)
    test_dataset = DL3DVDataset(cfg, split="test", view_sampler=test_view_sampler)
    return DataLoader(
        test_dataset,
        batch_size=cfg.data.val_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )


def build_dl3dv_annotation_dataloader(cfg, step_tracker):
    """One training-style pass over the train split for K-means annotation
    (``save_kmeans.py``): unpadded context views plus their frame indices.

    Batch size is fixed to 1 because unpadded scenes mix 4-6 context views and
    cannot be collated.
    """
    view_sampler = hydra.utils.instantiate(cfg.view_sampler, stage="train", step_tracker=step_tracker)
    dataset = DL3DVDataset(
        cfg, split="train", view_sampler=view_sampler, pad_context=False, return_view_indices=True
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
