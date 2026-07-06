"""DL3DV training dataset with precomputed K-means assignments.

Second-stage (condenser) training does not run K-means online. Instead,
``save_kmeans.py`` precomputes, per scene, one or more entries of
``{condition_view_idx, num_keep, num_views, anchor_idx, labels}`` stored as
``<kmeans_dir>/<scene>/metadata.json``. Each __getitem__ picks one entry at
random, loads exactly those context views, and samples target views from the
frame range they span (the context views already cover the scene via the view
sampler's farthest-point sampling).
"""
import json
import os
import random

import torch
from torch.utils.data import DataLoader

from src.dataset.dl3dv import DL3DVDataset, build_dl3dv_val_dataloader  # noqa: F401 (re-exported)
from src.utils.shims import apply_crop_shim


def pad_context_views_with_kmeans(
    views, extrinsics, intrinsics, timestamps, max_context_views, anchor_idx, labels, patch_size
):
    """Right-pad context views like ``dl3dv.pad_context_views`` and additionally
    pad the K-means anchor indices / labels to the fixed maximum token count so
    scenes with different numbers of views and clusters can be batched."""
    padded_views = torch.zeros(max_context_views, *views.shape[1:], device=views.device)
    padded_views[: views.shape[0]] = views

    padded_extrinsics = torch.zeros(max_context_views, 4, 4, device=extrinsics.device)
    padded_extrinsics[: extrinsics.shape[0]] = extrinsics

    padded_intrinsics = torch.zeros(max_context_views, 4, device=intrinsics.device)
    padded_intrinsics[: intrinsics.shape[0]] = intrinsics

    padded_timestamps = torch.zeros(max_context_views, dtype=timestamps.dtype, device=timestamps.device)
    padded_timestamps[: timestamps.shape[0]] = timestamps

    h, w = padded_views.shape[2:4]
    max_length = (h // patch_size) * (w // patch_size) * max_context_views

    anchor_idx_padded = torch.zeros(max_length, dtype=torch.long)
    anchor_idx = torch.tensor(anchor_idx, dtype=torch.long)
    anchor_idx_padded[: min(len(anchor_idx), max_length)] = anchor_idx[:max_length]

    labels_padded = torch.zeros(max_length, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    labels_padded[: min(len(labels), max_length)] = labels[:max_length]

    return padded_views, padded_extrinsics, padded_intrinsics, padded_timestamps, anchor_idx_padded, labels_padded


class DL3DVKmeansDataset(DL3DVDataset):
    """Train split of :class:`DL3DVDataset` restricted to scenes that have a
    precomputed K-means annotation, sampling context views from it."""

    def __init__(self, cfg, view_sampler=None):
        super().__init__(cfg, split="train", view_sampler=view_sampler)
        self.kmeans_dir = cfg.data.kmeans_dir
        assert self.kmeans_dir is not None, "data.kmeans_dir must point to the precomputed K-means assignments"

        num_scenes = len(self.data_lists)
        self.data_lists = [
            data for data in self.data_lists
            if os.path.exists(os.path.join(self.kmeans_dir, data["url"], "metadata.json"))
        ]
        skipped = num_scenes - len(self.data_lists)
        if skipped:
            print(f"Skipped {skipped} scenes without K-means metadata; {len(self.data_lists)} remain")

    def load_kmeans_metadata(self, scene_id):
        """Pick one precomputed entry (context set + clustering) at random."""
        with open(os.path.join(self.kmeans_dir, scene_id, "metadata.json"), "r") as f:
            metadata_dict = json.load(f)
        numeric_keys = [k for k in metadata_dict.keys() if k.isdigit()]
        if numeric_keys:
            return metadata_dict[random.choice(numeric_keys)]
        return metadata_dict

    def sample_target_idx(self, condition_view_idx):
        """Sample target frames from the range spanned by the context views.

        Always returns exactly ``num_target_views`` indices (sampling with
        replacement when the span holds fewer candidates) so batches collate.
        """
        num_targets = self.cfg.view_sampler.cfg.num_target_views

        min_idx = int(condition_view_idx.min())
        max_idx = int(condition_view_idx.max())
        candidates = sorted(set(range(min_idx, max_idx + 1)) - set(condition_view_idx.tolist()))
        if not candidates:
            candidates = condition_view_idx.tolist()

        if len(candidates) >= num_targets:
            return torch.tensor(random.sample(candidates, num_targets))
        return torch.tensor(random.choices(candidates, k=num_targets))

    def __getitem__(self, idx):
        while True:
            data = self.data_lists[idx]
            poses = data["camera_params"]
            extrinsics, intrinsics = self.convert_poses(poses)  # c2w, normalized K
            images_paths = data["images_path"]
            timestamps = data["timestamps"]
            url = data["url"]

            kmeans_metadata = self.load_kmeans_metadata(url)
            anchor_idx = kmeans_metadata["anchor_idx"]
            labels = kmeans_metadata["labels"]
            num_keep = kmeans_metadata["num_keep"]
            num_context_views = kmeans_metadata["num_views"]

            context_indices = torch.tensor(kmeans_metadata["condition_view_idx"])
            context_images = torch.stack(
                [self.load_images(images_paths[i.item()]) for i in context_indices]
            )

            # The context set is fixed by the K-means entry, so only target
            # views can be resampled; if the scene keeps failing (degenerate
            # context poses, corrupt frames), fall back to another scene.
            sampled = False
            for _ in range(8):
                target_indices = self.sample_target_idx(context_indices)

                if not self._valid_cameras(extrinsics, context_indices, target_indices):
                    continue
                try:
                    target_images = torch.stack(
                        [self.load_images(images_paths[i.item()]) for i in target_indices]
                    )
                except Exception as e:
                    print(f"Error processing images and poses: {e}")
                    continue
                sampled = True
                break

            if not sampled:
                print(f"Scene {url} keeps failing; resampling another scene")
                idx = random.randint(0, len(self.data_lists) - 1)
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

        (
            condition_views,
            condition_extrinsics,
            condition_views_intrinsics,
            context_timestamps,
            anchor_idx_padded,
            labels_padded,
        ) = pad_context_views_with_kmeans(
            example["context"]["image"],
            example["context"]["extrinsics"],
            condition_views_intrinsics,
            context_timestamps,
            self.cfg.data.max_context_views,
            anchor_idx,
            labels,
            self.cfg.model.patch_size,
        )

        return {
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
            "anchor_idx": anchor_idx_padded,
            "labels": labels_padded,
            "num_tokens": num_keep,
        }


def build_dl3dv_kmeans_dataloader(cfg, step_tracker):
    """Training loader over K-means-annotated scenes (no val loader; use
    ``build_dl3dv_val_dataloader`` from ``src.dataset.dl3dv``)."""
    train_dataset = DL3DVKmeansDataset(cfg)
    return DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
