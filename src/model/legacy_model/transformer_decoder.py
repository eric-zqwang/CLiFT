from diffusers import Transformer2DModel
import torch.nn as nn
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F


class TransformerDecoder(nn.Module):
    def __init__(self, cfg):
        super(TransformerDecoder, self).__init__()
        self.cfg = cfg

        self.patch_size = 8
        self.hidden_dim = cfg.decoder.hidden_dim

        h, w = 256, 256
        self.cond_view_num = 2

        self.transformer = Transformer2DModel(
            in_channels=self.cfg.decoder.hidden_dim,
            out_channels=self.cfg.decoder.hidden_dim,
            num_layers=self.cfg.decoder.num_layers,
            num_attention_heads=self.cfg.decoder.num_attention_heads,
            attention_head_dim=self.cfg.decoder.hidden_dim // self.cfg.decoder.num_attention_heads,
        )


        self.plucker_coord_dim = 6

        self.linear_input = nn.Conv2d(
            self.patch_size * self.patch_size * 9, 
            self.cfg.decoder.hidden_dim,
            kernel_size=1,
        )

        self.linear_target = nn.Conv2d(
            self.patch_size * self.patch_size * 6, 
            self.hidden_dim,
            kernel_size=1,
        )

        self.input_norm = nn.LayerNorm([self.hidden_dim, h//self.patch_size * (self.cond_view_num+1), w//self.patch_size])
        
        self.patch_to_image = nn.Sequential(
            nn.LayerNorm([self.hidden_dim, h//self.patch_size, w//self.patch_size]),
            nn.Conv2d(self.hidden_dim, self.patch_size * self.patch_size * 3, kernel_size=1),
            Rearrange('b (p1 p2 c) h w -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size)
        )


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

        query_patch_emb = self.patchify(target_view_plucker_coords)
        input_tokens = self.patchify(input_view_plucker_coords, input_image)

        input_tokens = rearrange(input_tokens, '(b m) c h w -> b c (m h) w', m=num_condition_view)

        input_tokens = input_tokens.repeat_interleave(num_target_views, dim=0)

        input_token_all = torch.cat([query_patch_emb, input_tokens], dim=2)

        # layernorm
        input_token_all = self.input_norm(input_token_all)
        

        # input_token_all = torch.cat([query_patch_emb, features], dim=2)

        # output is (B, L, D)
        out = self.transformer(
            hidden_states=input_token_all,
        ).sample

        out = out[:, :, :query_patch_emb.shape[2], :]

        out = self.patch_to_image(out)

        out = rearrange(out, '(b v) c h w -> b v h w c', b=bs)
        
        return out


        