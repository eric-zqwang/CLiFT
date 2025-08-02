from sklearn.cluster import KMeans
from src.model.encoder_decoder import LiFTnvs
from einops import rearrange
import torch
from src.utils.model_utils import batch_index_select
import numpy as np
from scipy.spatial.distance import cdist
import faiss


class LiFTnvsKmeans(LiFTnvs):
    def __init__(self, cfg):
        super().__init__(cfg)

    def get_kmeans_centroids(self, features):
        bs, N, _ = features.shape
        features = features[0]  # assuming batch size = 1, shape [4096, D]
        features_np = features.cpu().numpy()

        token_ratio = self.token_ratio
        num_keep = int(N * token_ratio)

        # run KMeans
        kmeans = KMeans(n_clusters=num_keep, random_state=0).fit(features_np)
        centroids = kmeans.cluster_centers_  # [num_tokens, D]
        assignments = kmeans.labels_         # [4096], each token -> cluster ID

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


    def get_kmeans_centroids_faiss(self, features):
        bs, N, D = features.shape
        assert bs == 1, "Only batch size 1 is supported."
        features = features[0]  # shape [N, D]

        token_ratio = self.token_ratio
        num_keep = int(N * token_ratio)

        # convert to numpy float32 and ensure contiguous
        features_np = features.detach().cpu().contiguous().numpy().astype(np.float32)

        # FAISS kmeans setup
        kmeans = faiss.Kmeans(d=D, k=num_keep, gpu=True)
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


    def forward(self, input_image, input_view_plucker_coords, target_view_plucker_coords):
        bs = input_image.shape[0]

        num_target_views = target_view_plucker_coords.shape[1]
        num_condition_view = input_image.shape[1]

        input_tokens = self.patchify(input_view_plucker_coords, input_image)
        query_patch_emb = self.patchify(target_view_plucker_coords, None)
        h, w = query_patch_emb.shape[2], query_patch_emb.shape[3]

        input_tokens = rearrange(input_tokens, '(b v) c h w -> b (v h w) c', v=num_condition_view)
        query_patch_emb = rearrange(query_patch_emb, 'b c h w -> b (h w) c')

        features = self.encoder(input_tokens)

        token_ratio = self.token_ratio

        if self.cfg.kmeans == 'sklearn':
            keep_indices, assignments = self.get_kmeans_centroids(features)
        elif self.cfg.kmeans == 'faiss':
            keep_indices, assignments = self.get_kmeans_centroids_faiss(features)
        else:
            raise ValueError(f"Invalid kmeans method: {self.cfg.kmeans}")

        keep_indices = keep_indices.to(features.device)
        assignments = assignments.to(features.device)

        features = features.repeat_interleave(num_target_views, dim=0)

        if token_ratio < 1.0:
            B, N, C = features.shape

            features = batch_index_select(features, keep_indices)
            hard_keep_decision = keep_indices
        else:
            hard_keep_decision = None

        out = self.decoder(
            query_patch_emb, 
            features,
        )

        out = self.output_norm(out)

        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w).contiguous()

        out = self.patch_to_image(out)

        out = rearrange(out, '(b v) c h w -> b v h w c', b=bs).contiguous()

        output_dict = {
            'pred_sampled_views': out,
        }

        output_dict['hard_keep_decision'] = hard_keep_decision
        output_dict['kmeans_assignments'] = assignments

        return output_dict