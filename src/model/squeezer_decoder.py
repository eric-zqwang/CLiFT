from sklearn.cluster import KMeans
from src.model.encoder_decoder import LiFTnvs
from einops import rearrange
import torch
from src.utils.model_utils import batch_index_select
import numpy as np
from scipy.spatial.distance import cdist
import faiss
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from einops.layers.torch import Rearrange
from src.model.modules.condenser import NeuralCondenser

class CLiFTnvs(LiFTnvs):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.squeezer = NeuralCondenser(
            hidden_dim=self.hidden_dim, 
            num_attention_heads=8, 
            num_layers=cfg.squeezer.num_layers
        )


    def get_kmeans_centroids(self, features, num_keep):
        bs, N, _ = features.shape
        features = features[0]  # assuming batch size = 1, shape [4096, D]
        features_np = features.cpu().numpy()

        # run KMeans
        kmeans = KMeans(n_clusters=num_keep, random_state=0).fit(features_np)
        centroids = kmeans.cluster_centers_  # [2048, D]
        assignments = kmeans.labels_         # [4096], each token -> cluster ID

        # now: for each cluster, find the index of the token that's closest to the centroid
        # this gives you the representative/original token ID

        closest_token_ids = []

        for i in range(num_keep):
            idxs = np.where(assignments == i)[0]             # token indices in this cluster
            cluster_tokens = features_np[idxs]               # their feature vectors
            dists = cdist([centroids[i]], cluster_tokens)[0] # compute distances to centroid
            min_idx = idxs[np.argmin(dists)]                 # find the token index with min dist
            closest_token_ids.append(min_idx)

        closest_token_ids = torch.tensor(closest_token_ids).reshape(1, -1)
        assignments = torch.tensor(assignments).reshape(1, -1)

        return closest_token_ids, assignments


    def get_kmeans_centroids_faiss(self, features, num_keep):
        bs, N, D = features.shape
        assert bs == 1, "Only batch size 1 is supported."
        features = features[0]  # shape [N, D]

        # convert to numpy float32 and ensure contiguous
        features_np = features.detach().cpu().contiguous().numpy().astype(np.float32)

        # FAISS kmeans setup
        kmeans = faiss.Kmeans(d=D, k=num_keep, gpu=False)
        kmeans.train(features_np)

        centroids = kmeans.centroids                       # [num_keep, D]
        _, assignments = kmeans.index.search(features_np, 1)  # [N, 1]
        assignments = assignments.reshape(-1)              # [N]

        # for each cluster, find token closest to centroid
        closest_token_ids = []
        for i in range(num_keep):
            idxs = np.where(assignments == i)[0]
            if len(idxs) == 0:
                continue  # safety check: skip empty cluster
            cluster_tokens = features_np[idxs]  # [M, D]
            dists = cdist([centroids[i]], cluster_tokens)[0]
            min_idx = idxs[np.argmin(dists)]
            closest_token_ids.append(min_idx)

        # convert to torch tensors
        closest_token_ids = torch.tensor(closest_token_ids, dtype=torch.long).reshape(1, -1)
        assignments = torch.tensor(assignments, dtype=torch.long).reshape(1, -1)

        return closest_token_ids, assignments


    def encode_and_kmeans(self, input_image, input_view_plucker_coords, token_ratio=0.2):
        """
        Inference time we assume use 2048 token to track performance
        """
        bs = input_image.shape[0]

        num_condition_view = input_image.shape[1]

        input_tokens = self.patchify(input_view_plucker_coords, input_image)
        h, w = input_tokens.shape[2], input_tokens.shape[3]

        input_tokens = rearrange(input_tokens, '(b v) c h w -> b (v h w) c', v=num_condition_view)

        features = self.encoder(input_tokens)

        max_num_token = features.shape[1]
        num_keep = int(max_num_token * token_ratio)

        keep_indices = []
        assignments = []
        for b in range(bs):
            keep_indices_b, assignments_b = self.get_kmeans_centroids(features[b].unsqueeze(0), num_keep)

            keep_indices_b = keep_indices_b.to(features.device)
            assignments_b = assignments_b.to(features.device)
            
            # Padding keep_indices to max_num_token
            padded_indices = torch.full((1, max_num_token), 0, 
                                      dtype=keep_indices_b.dtype, 
                                      device=features.device)
            padded_indices[0, :keep_indices_b.shape[1]] = keep_indices_b
            
            keep_indices.append(padded_indices)
            assignments.append(assignments_b)
        
        keep_indices = torch.cat(keep_indices, dim=0)
        assignments = torch.cat(assignments, dim=0)

        num_keep = torch.full((bs,), num_keep, device=features.device)

        return features, keep_indices, assignments, num_keep
    

    def forward(self, features, anchor_idx, labels, num_tokens, target_view_plucker_coords):
        # Squeeze features
        squeezed_features = self.squeezer(features, anchor_idx, labels, num_tokens)
        
        # Get padding mask
        bs, seq_len, _ = squeezed_features.shape
        idx = torch.arange(seq_len, device=squeezed_features.device)
        padding_mask = (idx[None, :] >= num_tokens[:, None]).to(torch.bool)
        
        # Repeat features and padding mask for each target view
        num_target_views = target_view_plucker_coords.shape[1]
        squeezed_features = squeezed_features.repeat_interleave(num_target_views, dim=0)
        padding_mask = padding_mask.repeat_interleave(num_target_views, dim=0)

        # Render
        out = self.render(squeezed_features, target_view_plucker_coords, bs, padding_mask)

        # Output dict
        output_dict = {
            'pred_sampled_views': out,
        }

        output_dict['hard_keep_decision'] = anchor_idx

        return output_dict