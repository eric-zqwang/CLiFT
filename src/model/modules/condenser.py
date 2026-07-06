"""Cross-attention condenser (the CLiFT "squeezer").

Given the full set of encoded tokens plus a K-means clustering (anchor token per
cluster + per-token cluster labels), each anchor cross-attends to the other
tokens of its own cluster to absorb their information, producing the compact set
of storage CLiFTs. ``linear1`` is zero-initialized so the module starts as an
identity over the anchor features and gradually learns the aggregation.

Used by both datasets: for DL3DV, ``num_views`` additionally masks out encoder
tokens that belong to padded (non-existent) context views; RE10K batches have a
fixed number of views and leave it at None.
"""
import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer

from src.utils.model_utils import batch_index_select


class NeuralCondenser(nn.Module):
    def __init__(self, hidden_dim, num_attention_heads, num_layers):
        super().__init__()
        self.num_heads = num_attention_heads
        decoder_layer = TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            batch_first=True,
            norm_first=True,
        )
        if num_layers == 1:
            # Kept unwrapped so parameter names stay compatible with existing checkpoints.
            self.cross_attn = decoder_layer
        else:
            self.cross_attn = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)

    def forward(self, x, anchor_idx, labels, num_tokens, num_views=None, tokens_per_view=1792):
        """Aggregate each cluster's tokens into its anchor via masked cross-attention.

        Args:
            x: encoded tokens, [batch_size, seq_len, hidden_dim]
            anchor_idx: anchor (representative) token indices, [batch_size, max_anchors]
            labels: per-token cluster id, [batch_size, seq_len]
            num_tokens: number of valid anchors per batch element, [batch_size]
            num_views: number of real (non-padded) context views per batch
                element, or None when no views are padded (RE10K)
            tokens_per_view: encoder tokens per context view (32*56 for 256x448,
                patch 8); only used when num_views is given
        Returns:
            Storage tokens (one per anchor), [batch_size, max_anchors, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = x.shape
        device = x.device

        # Queries are the anchor tokens themselves.
        query_features = batch_index_select(x, anchor_idx)  # [B, num_anchors, D]

        idx_seq = torch.arange(seq_len, device=device)
        src_kp_mask = idx_seq.unsqueeze(0) >= num_tokens.unsqueeze(1)

        # An anchor may only attend to memory tokens sharing its cluster label.
        anchor_clusters = torch.gather(labels, 1, anchor_idx)
        attn_mask = anchor_clusters.unsqueeze(2) != labels.unsqueeze(1)

        for b in range(batch_size):
            # Disable padded anchor rows (keeps attention numerically stable).
            attn_mask[b, num_tokens[b]:, :] = False
            if num_views is not None:
                # Mask out memory tokens belonging to padded (non-existent) views.
                valid_labels = tokens_per_view * num_views[b]
                attn_mask[b, :, valid_labels:] = True

        squeezed_features = self.cross_attn(
            query_features,
            x,
            tgt_key_padding_mask=src_kp_mask,
            memory_mask=attn_mask.repeat_interleave(self.num_heads, dim=0),
        )

        query_features = query_features + self.linear1(squeezed_features)
        return query_features
