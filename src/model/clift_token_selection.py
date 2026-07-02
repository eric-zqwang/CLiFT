import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from src.model.squeezer_decoder import CLiFTnvs


def ray_angular_distance(dir_a: torch.Tensor, dir_b: torch.Tensor) -> torch.Tensor:
    """Angular distance (radians) between two sets of ray directions.

    Args:
        dir_a: [Na, 3] ray directions.
        dir_b: [Nb, 3] ray directions.

    Returns:
        [Na, Nb] tensor ``acos(cos_sim(dir_a, dir_b))``.
    """
    cos_sim = F.cosine_similarity(dir_a[:, None, :], dir_b[None, :, :], dim=-1)
    cos_sim = cos_sim.clamp(-1.0, 1.0)
    return torch.acos(cos_sim)


def select_render_tokens(
    center_rays: torch.Tensor,
    target_rays: torch.Tensor,
    center_cam_center_dist: torch.Tensor,
    num_render_tokens: int,
    angular_weight: float = 1.0,
    cam_center_weight: float = 0.02,
    momentum_weight: float = 0.0,
    previous_select_weight: float = 0.0,
    last_combined_distances: Optional[torch.Tensor] = None,
    previous_selected: Optional[torch.Tensor] = None,
    exist_same_ray: bool = False,
    patch_topk: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the Nr storage tokens closest to a single target view (Algorithm 1).

    Distance between a storage token and the target view combines the ray-angle
    distance to the closest target sample ray and the camera-center distance of
    the token's source view. Temporal terms make the selection stable across a
    video: a bonus (``previous_select_weight``) for tokens picked in the previous
    frame, and an exponential moving average (``momentum_weight``) of the distance
    field from the previous frame.

    Args:
        center_rays: [Ns, 6] stacked (origin, direction) of the Ns storage tokens.
        target_rays: [Nq, 6] stacked (origin, direction) of the target sample grid.
        center_cam_center_dist: [Ns] camera-center distance of each token's source
            view to the target view.
        num_render_tokens: Nr, the compute budget of tokens to keep.
        angular_weight, cam_center_weight: weights of the two distance terms.
        momentum_weight: EMA weight on the previous frame's distance field.
        previous_select_weight: distance bonus for previously selected tokens.
        last_combined_distances: [Ns, Nq] distance field from the previous frame.
        previous_selected: [Ns] bool mask of tokens selected in the previous frame.
        exist_same_ray: if True, force the angular distance of coincident rays to 0.
        patch_topk: patch-wise coverage budget -- for every target patch keep its
            ``patch_topk`` nearest storage tokens first (the paper's Algorithm 1
            per-patch step), then fill the remaining budget globally. NOTE: the
            paper's pseudocode uses n = Nr // (24*24), but the code that produced
            Table 1 hardcodes ``patch_topk=0`` (pure global fallback).

    Returns:
        selected_idx: [Nr] long indices (into the Ns tokens) of the kept tokens.
        selected_mask: [Ns] bool mask of the kept tokens (for the next frame).
        combined_distances: [Ns, Nq] distance field after momentum/bonus (EMA state).
    """
    o_center, d_center = center_rays[:, :3], center_rays[:, 3:]
    o_target, d_target = target_rays[:, :3], target_rays[:, 3:]

    angular = ray_angular_distance(d_center, d_target)  # [Ns, Nq]
    if exist_same_ray:
        ray_eq = (torch.norm(d_center[:, None] - d_target[None], dim=-1) < 1e-6) & (
            torch.norm(o_center[:, None] - o_target[None], dim=-1) < 1e-3
        )
        angular = torch.where(ray_eq, torch.zeros_like(angular), angular)

    combined = angular_weight * angular + cam_center_weight * center_cam_center_dist[:, None]

    if previous_selected is not None:
        combined = combined - previous_selected.float()[:, None] * previous_select_weight
    if last_combined_distances is not None:
        combined = combined * (1.0 - momentum_weight) + last_combined_distances * momentum_weight

    num_tokens = combined.shape[0]
    num_keep = min(int(num_render_tokens), num_tokens)

    # Patch-wise coverage: keep the `patch_topk` nearest storage tokens for every
    # target patch so the whole view stays covered (Algorithm 1, patch-wise step).
    patch_k = min(int(patch_topk), num_tokens)
    _, patch_idx = torch.topk(combined, k=patch_k, dim=0, largest=False)  # [patch_k, Nq]
    patch_selected = patch_idx.reshape(-1).unique()

    if patch_selected.numel() >= num_keep:
        # More coverage tokens than the budget: keep the closest `num_keep` of them.
        patch_dist = combined[patch_selected].min(dim=-1).values
        _, keep = torch.topk(patch_dist, k=num_keep, largest=False)
        selected_idx = patch_selected[keep]
    else:
        # Global fallback: fill the remaining budget from the not-yet-selected
        # tokens, ranked by their closest distance to any target patch.
        remaining_mask = torch.ones(num_tokens, dtype=torch.bool, device=combined.device)
        remaining_mask[patch_selected] = False
        remaining_rows = torch.nonzero(remaining_mask, as_tuple=False).squeeze(-1)
        remaining_choose = num_keep - patch_selected.numel()
        remaining_dist = combined[remaining_rows].min(dim=-1).values
        _, rem_sel = torch.topk(remaining_dist, k=remaining_choose, largest=False)
        selected_idx = torch.cat([patch_selected, remaining_rows[rem_sel]])

    selected_mask = torch.zeros(num_tokens, dtype=torch.bool, device=combined.device)
    selected_mask[selected_idx] = True

    return selected_idx, selected_mask, combined


class CLiFTnvsTokenSelection(CLiFTnvs):
    """CLiFT renderer with render-time token selection (paper Algorithm 1).

    Extends the official :class:`CLiFTnvs` (encode -> latent K-means -> condense
    into Ns storage CLiFTs) and adds, for every target view, the selection of the
    Nr storage tokens whose source rays are geometrically closest to that view;
    only those tokens are fed to the decoder. Lowering Nr trades quality for fewer
    FLOPs / higher FPS with a single trained network.

    All render-selection hyper-parameters are read from ``cfg.render_selection``.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        rs = cfg.render_selection
        self.num_render_tokens = rs.num_render_tokens
        self.angular_weight = rs.angular_weight
        self.cam_center_weight = rs.cam_center_weight
        self.momentum_weight = rs.momentum_weight
        self.previous_select_weight = rs.previous_select_weight
        self.storage_token_ratio = rs.storage_token_ratio
        self.patch_topk = rs.get("patch_topk", 0)
        self.kmeans_pca_dim = rs.get("kmeans_pca_dim", 16)

    def get_kmeans_centroids(self, features, num_keep):
        """Latent K-means storage-token selection in a PCA subspace.

        Features are projected to ``render_selection.kmeans_pca_dim`` dims (Table 1
        uses 16) before sklearn K-means; the storage token for each cluster is the
        original token closest to the centroid. This matches how the storage CLiFTs
        were selected at training time.
        """
        from sklearn.decomposition import PCA

        features = features[0]
        features_np = features.detach().cpu().numpy()
        features_pca = PCA(n_components=min(self.kmeans_pca_dim, features_np.shape[1])).fit_transform(
            features_np
        )

        kmeans = KMeans(n_clusters=num_keep, random_state=0).fit(features_pca)
        centroids = kmeans.cluster_centers_
        assignments = kmeans.labels_

        closest_token_ids = []
        for i in range(num_keep):
            idxs = np.where(assignments == i)[0]
            cluster_tokens = features_pca[idxs]
            dists = cdist([centroids[i]], cluster_tokens)[0]
            closest_token_ids.append(idxs[np.argmin(dists)])

        closest_token_ids = torch.tensor(closest_token_ids).reshape(1, -1)
        assignments = torch.tensor(assignments).reshape(1, -1)
        return closest_token_ids, assignments

    @staticmethod
    def camera_center_distance(context_extrinsics, target_extrinsic):
        """Per-context-view camera-center distance to a target view.

        Args:
            context_extrinsics: [V, 4, 4] context camera-to-world matrices.
            target_extrinsic: [4, 4] target camera-to-world matrix.

        Returns:
            [V] RMS distance between context and target camera centers.
        """
        context_center = context_extrinsics[..., :3, 3]  # [V, 3]
        target_center = target_extrinsic[..., :3, 3]      # [3]
        return torch.sqrt(torch.mean((context_center - target_center) ** 2, dim=-1))

    def select_tokens(
        self,
        center_rays,
        center_cam_center_dist,
        target_ray,
        exist_same_ray,
        last_combined_distances,
        previous_selected,
    ):
        """Run Algorithm 1 for one target view. See :func:`select_render_tokens`."""
        target_rays = target_ray.reshape(-1, 6)
        return select_render_tokens(
            center_rays,
            target_rays,
            center_cam_center_dist,
            num_render_tokens=self.num_render_tokens,
            angular_weight=self.angular_weight,
            cam_center_weight=self.cam_center_weight,
            momentum_weight=self.momentum_weight,
            previous_select_weight=self.previous_select_weight,
            last_combined_distances=last_combined_distances,
            previous_selected=previous_selected,
            exist_same_ray=exist_same_ray,
            patch_topk=self.patch_topk,
        )

    def render_selected(self, squeezed_features, selected_idx, target_view_plucker_coords, bs):
        """Render a single target view using only the selected Nr storage tokens.

        Returns the rendered image and the wall-clock render time (GPU-synced) so
        the caller can report rendering FPS.
        """
        selected_features = squeezed_features[:, selected_idx]  # [bs, Nr, C]

        torch.cuda.synchronize()
        start = time.time()
        out = self.render(selected_features, target_view_plucker_coords, bs)
        torch.cuda.synchronize()
        render_time = time.time() - start

        return out, render_time
