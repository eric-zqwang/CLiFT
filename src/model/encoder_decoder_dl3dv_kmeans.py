"""DL3DV K-means selection baseline ("Ours w/o Condenser").

Combines the two parents the same way ``squeezer_decoder_dl3dv.SqueezerDecoder``
does, minus the condenser: ``DL3DVTransformer`` provides the architecture
(QK-norm attention, ``input_norm``, context-view padding) and
``LiFTnvsKmeans`` provides the K-means anchor selection. The anchor tokens are
rendered directly, without condensing their clusters into them.

Evaluated with the first-stage (encoder-decoder) checkpoint.
"""
from src.model.encoder_decoder_dl3dv import DL3DVTransformer
from src.model.encoder_decoder_kmeans import LiFTnvsKmeans
from src.utils.model_utils import batch_index_select


class DL3DVTransformerKmeans(DL3DVTransformer, LiFTnvsKmeans):
    def forward(self, input_image, input_view_plucker_coords, target_view_plucker_coords, num_context_views=None):
        bs = input_image.shape[0]
        num_target_views = target_view_plucker_coords.shape[1]

        features = self.encode_scene(input_image, input_view_plucker_coords, num_context_views)

        hard_keep_decision = None
        if self.token_ratio < 1.0:
            if self.cfg.kmeans == 'sklearn':
                keep_indices, _ = self.get_kmeans_centroids(features)
            elif self.cfg.kmeans == 'faiss':
                keep_indices, _ = self.get_kmeans_centroids_faiss(features)
            else:
                raise ValueError(f"Invalid kmeans method: {self.cfg.kmeans}")
            keep_indices = keep_indices.to(features.device)
            features = batch_index_select(features, keep_indices)
            hard_keep_decision = keep_indices

        features = features.repeat_interleave(num_target_views, dim=0)
        out = self.render(features, target_view_plucker_coords, bs)

        return {
            'pred_sampled_views': out,
            'hard_keep_decision': hard_keep_decision,
        }
