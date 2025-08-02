import torch.nn as nn
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import (
    TransformerEncoder, TransformerEncoderLayer
)


class Transformer(nn.Module):
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

        self.scene_embedding = nn.Parameter(
            torch.randn(1, 3072, self.hidden_dim)
        )
        nn.init.trunc_normal_(self.scene_embedding, std=0.02)
        

        self.decoder = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
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

        self.input_norm = nn.LayerNorm(self.hidden_dim)
        self.output_norm = nn.LayerNorm(self.hidden_dim)


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

    

    def forward(self, input_image, input_view_plucker_coords, target_view_plucker_coords):
        bs = input_image.shape[0]

        num_target_views = target_view_plucker_coords.shape[1]
        num_condition_view = input_image.shape[1]

        input_tokens = self.patchify(input_view_plucker_coords, input_image)
        query_patch_emb = self.patchify(target_view_plucker_coords, None)
        h, w = query_patch_emb.shape[2], query_patch_emb.shape[3]

        input_tokens = rearrange(input_tokens, '(b v) c h w -> b (v h w) c', v=num_condition_view)
        query_patch_emb = rearrange(query_patch_emb, 'b c h w -> b (h w) c')

        scene_embedding = self.scene_embedding.repeat(bs, 1, 1)
        num_scene_embedding = scene_embedding.shape[1]

        input_tokens = torch.cat([scene_embedding, input_tokens], dim=1)

        features = self.encoder(input_tokens)

        features = features[:, :num_scene_embedding, :]

        features = self.input_norm(features)

        features = features.repeat_interleave(num_target_views, dim=0)

        all_token_decoder = torch.cat([query_patch_emb, features], dim=1)


        num_query_token = query_patch_emb.shape[1]
       
        out = self.decoder(all_token_decoder)

        out = out[:, :num_query_token, :].contiguous()

        out = self.output_norm(out)

        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w).contiguous()

        out = self.patch_to_image(out)

        out = rearrange(out, '(b v) c h w -> b v h w c', b=bs).contiguous()

        output_dict = {
            'pred_sampled_views': out,
        }

        return output_dict