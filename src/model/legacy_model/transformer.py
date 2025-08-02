from diffusers import Transformer2DModel
import torch.nn as nn
import torch
from einops import rearrange
from einops.layers.torch import Rearrange


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_size = 8

        self.hidden_dim = cfg.encoder.hidden_dim
        self.encoder = Transformer2DModel(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            num_attention_heads=cfg.encoder.num_attention_heads,
            num_layers=cfg.encoder.num_layers,
            attention_head_dim=self.hidden_dim // cfg.encoder.num_attention_heads,
        )

        self.decoder = Transformer2DModel(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            num_attention_heads=cfg.decoder.num_attention_heads,
            num_layers=cfg.decoder.num_layers,
            attention_head_dim=self.hidden_dim // cfg.encoder.num_attention_heads,
            cross_attention_dim=self.hidden_dim,
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
            # nn.Sigmoid(),
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

        input_tokens = self.patchify(input_view_plucker_coords, input_image)
        query_patch_emb = self.patchify(target_view_plucker_coords, None)


        input_tokens = rearrange(input_tokens, '(b v) c h w -> b c (v h) w', v=num_condition_view)

        features = self.encoder(input_tokens).sample

        features = features.repeat_interleave(num_target_views, dim=0)
        features = rearrange(features, 'b c (v h) w -> b (v h w) c', v=num_condition_view)

        out = self.decoder(
            query_patch_emb, 
            encoder_hidden_states=features
        ).sample


        out = self.patch_to_image(out)

        out = rearrange(out, '(b v) c h w -> b v h w c', b=bs)
        # out = torch.sigmoid(out)

        return out
