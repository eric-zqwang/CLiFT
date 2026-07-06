"""K-means annotation pass for second-stage (condenser) training.

Loads a first-stage (encoder-decoder) checkpoint, encodes each training scene's
context views, clusters the encoder tokens with FAISS K-means at a random token
ratio in [0.05, 1.0], and appends one entry per pass to
``<data.kmeans_dir>/<scene>/metadata.json``:

    {condition_view_idx, num_keep, num_views, anchor_idx, labels}

Works for both datasets (the model class set in the config decides the encoder);
the K-means-loading training datasets (``src/dataset/re10k_load_kmeans.py`` /
``src/dataset/dl3dv_load_kmeans.py``) consume these entries during condenser
training. Note that, following the original training pipeline, the clustering
runs on the raw encoder output (before any normalization).

Run several passes with different ``test_seed`` values to give each scene
multiple context sets / token budgets (the save_*_kmeans.sh scripts loop).
"""
import json
import os

import faiss
import numpy as np
import torch
from einops import rearrange

from src.lightning_clift import LightningCLiFTWrapper


class KmeansAnnotator(LightningCLiFTWrapper):
    def test_step(self, data_dict, idx):
        self._get_plucker_coords(data_dict)

        # Encode WITHOUT input_norm: the annotation clusters raw encoder tokens.
        # Batches come unpadded (batch_size=1, all views real), so no mask.
        views = data_dict['condition_views']
        num_views = views.shape[1]
        input_tokens = self.transformer.patchify(data_dict['condition_views_plucker_coords'], views)
        input_tokens = rearrange(input_tokens, '(b v) c h w -> b (v h w) c', v=num_views)
        features = self.transformer.encoder(input_tokens)

        token_ratio = np.random.uniform(0.05, 1.0)
        num_keep = int(features.shape[1] * token_ratio)

        for b in range(features.shape[0]):
            anchor_idx, labels = self.transformer.get_kmeans_centroids_faiss(
                features[b].unsqueeze(0), num_keep, gpu=faiss.get_num_gpus() > 0
            )
            self._save_kmeans_entry(
                scene_id=data_dict['data_id'][b],
                condition_view_idx=data_dict['condition_view_idx'][b],
                anchor_idx=anchor_idx[0],
                labels=labels[0],
            )

    def _save_kmeans_entry(self, scene_id, condition_view_idx, anchor_idx, labels):
        save_dir = os.path.join(self.cfg.data.kmeans_dir, scene_id)
        os.makedirs(save_dir, exist_ok=True)
        json_path = os.path.join(save_dir, "metadata.json")

        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                metadata_dict = json.load(f)
        else:
            metadata_dict = {}

        metadata_dict[str(len(metadata_dict) + 1)] = {
            "condition_view_idx": condition_view_idx.cpu().tolist(),
            "num_keep": len(anchor_idx),
            "num_views": len(condition_view_idx),
            "anchor_idx": anchor_idx.cpu().tolist(),
            "labels": labels.cpu().tolist(),
        }

        with open(json_path, 'w') as f:
            json.dump(metadata_dict, f)
