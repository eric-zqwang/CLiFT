"""QK-normalized transformer layers used by the DL3DV CLiFT model.

Standard ``nn.TransformerEncoderLayer`` / ``nn.TransformerDecoderLayer`` whose
attention modules additionally RMS-normalize the per-head query and key before
the dot product (QK-norm), which stabilizes training on DL3DV's larger, less
constrained scenes. ``custom_multihead_attn`` mirrors ``nn.MultiheadAttention``
but always routes through ``model_utils.multi_head_attention_forward`` so the
``q_norm`` / ``k_norm`` modules are applied (the native fast path would skip
them).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer

from src.utils.model_utils import multi_head_attention_forward


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight.type_as(x)


class custom_multihead_attn(nn.MultiheadAttention):
    """``nn.MultiheadAttention`` with RMS QK-norm on the per-head query/key."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        # NEVER take nn.MultiheadAttention's native fast path: it calls
        # torch._native_multi_head_attention, which would silently skip
        # q_norm/k_norm (it is only reachable in eval mode without autocast,
        # e.g. fp32 inference).
        why_not_fast_path = "QK-norm requires the slow path (q_norm/k_norm are not applied by the native kernel)"

        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )
        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. "
                                f"The fast path was not hit because {why_not_fast_path}")

        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
                q_norm=self.q_norm, k_norm=self.k_norm)
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
                q_norm=self.q_norm, k_norm=self.k_norm)

        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        return attn_output, attn_output_weights


class QKNormTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.self_attn = custom_multihead_attn(
            embed_dim=kwargs['d_model'],
            num_heads=kwargs['nhead'],
            dropout=kwargs['dropout'],
            batch_first=kwargs['batch_first'],
        )


class QKNormTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.self_attn = custom_multihead_attn(
            embed_dim=kwargs['d_model'],
            num_heads=kwargs['nhead'],
            dropout=kwargs['dropout'],
            batch_first=kwargs['batch_first'],
        )
        self.multihead_attn = custom_multihead_attn(
            embed_dim=kwargs['d_model'],
            num_heads=kwargs['nhead'],
            dropout=kwargs['dropout'],
            batch_first=kwargs['batch_first'],
        )
