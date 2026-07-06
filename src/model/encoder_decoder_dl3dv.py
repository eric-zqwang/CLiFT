"""DL3DV encoder/decoder transformer.

Extends the RealEstate10K ``LiFTnvs`` with the two DL3DV-specific changes:
  * QK-normalized attention layers (see ``modules.qk_norm_transformer``), which
    stabilize training on DL3DV's larger, less constrained scenes, plus an
    ``input_norm`` after the encoder;
  * a context-view padding mask so a batch can mix scenes with 4-6 context
    views (padded views are masked in the encoder, the token sampling, and the
    decoder).
"""
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import TransformerDecoder, TransformerEncoder

from src.model.encoder_decoder import LiFTnvs
from src.model.modules.qk_norm_transformer import (
    QKNormTransformerDecoderLayer,
    QKNormTransformerEncoderLayer,
)
from src.utils.model_utils import batch_index_select


class DL3DVTransformer(LiFTnvs):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Replace the standard attention layers with QK-normalized ones and
        # normalize the encoder output before condensation / rendering.
        self.encoder = TransformerEncoder(
            encoder_layer=QKNormTransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=cfg.encoder.num_attention_heads,
                dropout=0.1,
                activation="gelu",
                norm_first=True,
                batch_first=True,
            ),
            num_layers=cfg.encoder.num_layers,
        )
        self.decoder = TransformerDecoder(
            decoder_layer=QKNormTransformerDecoderLayer(
                d_model=self.hidden_dim,
                nhead=cfg.decoder.num_attention_heads,
                dropout=0.1,
                activation="gelu",
                norm_first=True,
                batch_first=True,
            ),
            num_layers=cfg.decoder.num_layers,
        )
        self.input_norm = nn.LayerNorm(self.hidden_dim)

    def create_padding_mask(self, batch_size, num_context_views, max_context_views, patches_per_view, device):
        """Boolean mask, [B, max_context_views * patches_per_view]; True = valid token."""
        padding_mask = torch.ones(batch_size, max_context_views, patches_per_view, dtype=torch.bool, device=device)
        for i in range(batch_size):
            if num_context_views[i] < max_context_views:
                padding_mask[i, num_context_views[i]:, :] = False
        return padding_mask.reshape(batch_size, max_context_views * patches_per_view)

    def sample_tokens_respecting_padding(self, features, padding_mask, token_ratio):
        """Randomly keep ``token_ratio`` of each sample's *valid* tokens (train only).

        Returns the gathered features, a validity mask for the kept slots, and the
        kept token indices; samples with fewer valid tokens than the batch max are
        right-padded with zeros and masked out.
        """
        B, N, C = features.shape

        valid_token_counts = padding_mask.sum(dim=1)  # [B]
        num_tokens_to_keep = (valid_token_counts * token_ratio).int()  # [B]
        max_keep = max(int(num_tokens_to_keep.max().item()), 1)

        sampled_features = torch.zeros(B, max_keep, C, device=features.device)
        sampled_mask = torch.zeros(B, max_keep, dtype=torch.bool, device=padding_mask.device)
        keep_indices = torch.zeros(B, max_keep, dtype=torch.long, device=features.device)

        for i in range(B):
            valid_indices = torch.where(padding_mask[i])[0]
            if len(valid_indices) == 0:
                continue
            num_keep = min(num_tokens_to_keep[i].item(), len(valid_indices))
            perm = torch.randperm(len(valid_indices), device=features.device)
            selected_indices = valid_indices[perm[:num_keep]]

            sampled_features[i, :num_keep] = features[i, selected_indices]
            sampled_mask[i, :num_keep] = True
            keep_indices[i, :num_keep] = selected_indices

        return sampled_features, sampled_mask, keep_indices

    def encode_scene(self, input_image, input_view_plucker_coords, num_context_views=None):
        assert num_context_views is not None, "DL3DV batches must carry num_context_views"
        bs = input_image.shape[0]
        max_context_views = input_image.shape[1]

        # Zero the plucker rays of padded context views (no-op when every view
        # is real, e.g. at evaluation).
        for i in range(bs):
            if num_context_views[i] < max_context_views:
                input_view_plucker_coords[i][num_context_views[i]:] = 0

        input_tokens = self.patchify(input_view_plucker_coords, input_image)
        input_tokens = rearrange(input_tokens, '(b v) c h w -> b (v h w) c', v=max_context_views)

        patches_per_view = int(input_tokens.shape[1] // max_context_views)
        padding_mask = self.create_padding_mask(
            bs, num_context_views, max_context_views, patches_per_view, device=input_tokens.device
        )
        features = self.encoder(input_tokens, src_key_padding_mask=~padding_mask)
        features = self.input_norm(features)
        return features

    def forward(self, input_image, input_view_plucker_coords, target_view_plucker_coords, num_context_views=None):
        """First-stage (LVSM-style) forward: encode context tokens, randomly drop
        tokens to ``token_ratio``, then decode the target views. Padded context
        views are masked in both the encoder and the decoder."""
        bs = input_image.shape[0]
        num_target_views = target_view_plucker_coords.shape[1]
        max_context_views = input_image.shape[1]

        input_tokens = self.patchify(input_view_plucker_coords, input_image)
        patches_per_view = input_tokens.shape[2] * input_tokens.shape[3]
        input_tokens = rearrange(input_tokens, '(b v) c h w -> b (v h w) c', v=max_context_views)

        if self.training:
            padding_mask = self.create_padding_mask(
                bs, num_context_views, max_context_views, patches_per_view, device=input_tokens.device
            )
            features = self.encoder(input_tokens, src_key_padding_mask=~padding_mask)
            padding_mask = padding_mask.repeat_interleave(num_target_views, dim=0)
        else:
            # Validation batches always carry the full number of real views.
            features = self.encoder(input_tokens)

        features = self.input_norm(features)
        features = features.repeat_interleave(num_target_views, dim=0)

        token_ratio = self.token_ratio
        if token_ratio == "random":
            token_ratio = torch.rand(1).item() * (1.0 - 0.25) + 0.25

        if self.training:
            features, keep_mask, hard_keep_decision = self.sample_tokens_respecting_padding(
                features, padding_mask, token_ratio
            )
            out = self.render(features, target_view_plucker_coords, bs, padding_mask=~keep_mask)
        else:
            # Random token keep at the configured ratio (the random-selection
            # baseline; all views are real at evaluation).
            hard_keep_decision = None
            if token_ratio < 1.0:
                B, N, _ = features.shape
                num_keep = int(N * token_ratio)
                rand_indices = torch.rand(N, device=features.device).argsort()
                keep_indices = rand_indices[:num_keep].unsqueeze(0).repeat(B, 1)
                features = batch_index_select(features, keep_indices)
                hard_keep_decision = keep_indices
            out = self.render(features, target_view_plucker_coords, bs)

        return {
            'pred_sampled_views': out,
            'hard_keep_decision': hard_keep_decision,
        }
