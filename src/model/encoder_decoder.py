import torch.nn as nn
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import (
    TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
)
import torch.nn.functional as F
from src.utils.model_utils import batch_index_select


class LiFTnvs(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size

        self.hidden_dim = cfg.encoder.hidden_dim
        self.encoder = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
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
            decoder_layer=TransformerDecoderLayer(
                d_model=self.hidden_dim,
                nhead=cfg.decoder.num_attention_heads,
                dropout=0.1,
                activation="gelu",
                norm_first=True,
                batch_first=True,
            ),
            num_layers=cfg.decoder.num_layers,
        )

        self.plucker_coord_dim = 6

        self.linear_input = nn.Conv2d(
            self.patch_size * self.patch_size * 9, 
            self.hidden_dim,
            kernel_size=1,
        )

        self.linear_target = nn.Conv2d(
            self.patch_size * self.patch_size * 6, 
            self.hidden_dim,
            kernel_size=1,
        )

        self.patch_to_image = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.patch_size * self.patch_size * 3, kernel_size=1),
            Rearrange('b (p1 p2 c) h w -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size),
            nn.Sigmoid()
        )

        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.token_ratio = cfg.token_ratio



    def patchify(self, plucker_coords, image=None):
        num_target_views = plucker_coords.shape[1]

        if image is not None:
            # c = 9
            x = torch.cat([image, plucker_coords], dim=-1)
        else:
            x = plucker_coords
        
        # (B, V, H, W, 9) -> (B*V, 9, H, W)
        x = rearrange(x, 'b v h w c -> (b v) c h w', v=num_target_views)
        
        # image_plucker = rearrange(x, '(b v) c h w -> (b v) c h w', p=self.patch_size)
        x = rearrange(x, 'b c (h p1) (w p2) -> b (p1 p2 c) h w', p1=self.patch_size, p2=self.patch_size)
        if image is not None:
            x = self.linear_input(x)
        else:
            x = self.linear_target(x)
        return x

    def encode_scene(self, input_image, input_view_plucker_coords):
        num_condition_view = input_image.shape[1]

        input_tokens = self.patchify(input_view_plucker_coords, input_image)
        h, w = input_tokens.shape[2], input_tokens.shape[3]

        input_tokens = rearrange(input_tokens, '(b v) c h w -> b (v h w) c', v=num_condition_view)
        features = self.encoder(input_tokens)

        return features

    def render(self, features, target_view_plucker_coords, bs, padding_mask=None):
        query_patch_emb = self.patchify(target_view_plucker_coords, None)
        h, w = query_patch_emb.shape[2], query_patch_emb.shape[3]
        query_patch_emb = rearrange(query_patch_emb, 'b c h w -> b (h w) c')
        
        out = self.decoder(
            query_patch_emb, 
            features,
            memory_key_padding_mask=padding_mask,
        )

        out = self.output_norm(out)

        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w).contiguous()

        out = self.patch_to_image(out)

        out = rearrange(out, '(b v) c h w -> b v h w c', b=bs).contiguous()

        return out
    
    def forward(self, input_image, input_view_plucker_coords, target_view_plucker_coords, num_context_views=None):
        bs = input_image.shape[0]
        features = self.encode_scene(input_image, input_view_plucker_coords)

        num_target_views = target_view_plucker_coords.shape[1]
        features = features.repeat_interleave(num_target_views, dim=0)

        token_ratio = self.token_ratio
        
        if token_ratio == "random":
            token_ratio = torch.rand(1).item() * (1.0 - 0.25) + 0.25

        if token_ratio < 1.0:
            B, N, C = features.shape
            num_keep = int(N * token_ratio)

            if self.training:
                rand_indices = torch.rand(B, N, device=features.device).argsort(dim=1)
                keep_indices = rand_indices[:, :num_keep]
            else:
                rand_indices = torch.rand(N, device=features.device).argsort()
                keep_indices = rand_indices[:num_keep].unsqueeze(0).repeat(B, 1)
            features = batch_index_select(features, keep_indices)
            hard_keep_decision = keep_indices
        else:
            hard_keep_decision = None

        out = self.render(features, target_view_plucker_coords, bs)

        output_dict = {
            'pred_sampled_views': out,
            'hard_keep_decision': hard_keep_decision,
        }

        return output_dict
