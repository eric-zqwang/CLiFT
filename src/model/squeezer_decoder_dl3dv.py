"""DL3DV CLiFT model: encoder -> K-means condensation -> condenser -> decoder.

Combines the two parents:
  * ``DL3DVTransformer`` provides the architecture (QK-norm attention,
    ``input_norm``, context-view padding) and the padding-aware
    ``encode_scene``;
  * ``CLiFTnvs`` provides the condenser (``squeezer``), the FAISS K-means
    anchor selection, and the cluster-render ``forward``.

Only the pieces that genuinely differ from RE10K are overridden here:
``encode_and_kmeans`` threads ``num_context_views`` through the padded encoder
and always clusters with FAISS in the full feature space (no PCA).
"""
import torch

from src.model.encoder_decoder_dl3dv import DL3DVTransformer
from src.model.squeezer_decoder import CLiFTnvs


class SqueezerDecoder(DL3DVTransformer, CLiFTnvs):
    # MRO: DL3DVTransformer -> CLiFTnvs -> LiFTnvs. The cooperative __init__
    # chain builds the RE10K model, adds the condenser (CLiFTnvs), then swaps
    # in the QK-norm layers + input_norm (DL3DVTransformer).

    def encode_and_kmeans(self, input_image, input_view_plucker_coords, token_ratio=0.125, num_context_views=None):
        bs = input_image.shape[0]
        features = self.encode_scene(input_image, input_view_plucker_coords, num_context_views)

        max_num_token = features.shape[1]
        num_keep = int(max_num_token * token_ratio)

        keep_indices = []
        assignments = []
        for b in range(bs):
            keep_indices_b, assignments_b = self.get_kmeans_centroids_faiss(features[b].unsqueeze(0), num_keep)
            keep_indices_b = keep_indices_b.to(features.device)
            assignments_b = assignments_b.to(features.device)

            padded_indices = torch.full((1, max_num_token), 0, dtype=keep_indices_b.dtype, device=features.device)
            padded_indices[0, :keep_indices_b.shape[1]] = keep_indices_b

            keep_indices.append(padded_indices)
            assignments.append(assignments_b)

        keep_indices = torch.cat(keep_indices, dim=0)
        assignments = torch.cat(assignments, dim=0)
        num_keep = torch.full((bs,), num_keep, device=features.device)
        return features, keep_indices, assignments, num_keep

    def forward(self, features, anchor_idx, labels, num_tokens, target_view_plucker_coords, num_context_views=None):
        # Render from the condensed clusters (CLiFTnvs.forward), not the
        # first-stage token-drop forward inherited from DL3DVTransformer.
        return CLiFTnvs.forward(
            self, features, anchor_idx, labels, num_tokens, target_view_plucker_coords, num_context_views
        )
