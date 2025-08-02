import torch
import torch.nn as nn
from torch.nn import TransformerDecoderLayer, TransformerDecoder
from src.utils.model_utils import batch_index_select

class NeuralCondenser(nn.Module):
    def __init__(self, hidden_dim, num_attention_heads, num_layers):
        super().__init__()
        self.num_heads = num_attention_heads
        self.cross_attn = TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            batch_first=True,
            norm_first=True,
        )
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)


    def forward(self, x, anchor_idx, labels, num_tokens):
        """
        Perform cross-attention between anchor tokens and other tokens in the same cluster.
        
        Args:
            x: Input features of shape [batch_size, seq_len, hidden_dim]
            anchor_idx: Indices of anchor tokens, padded to fixed length [batch_size, max_num_anchors]
            labels: Cluster assignment for each token [batch_size, seq_len]
            num_tokens: Number of valid tokens in each batch element [batch_size]
        
        Returns:
            Aggregated features of shape [batch_size, num_anchors, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = x.shape
        device = x.device
        
        # 1. Get the query features directly from anchor indices
        query_features = batch_index_select(x, anchor_idx)  # [batch_size, num_anchors, hidden_dim]


        idx_seq = torch.arange(seq_len, device=device)
        src_kp_mask  = idx_seq.unsqueeze(0) >= num_tokens.unsqueeze(1)

        anchor_clusters = torch.gather(labels, 1, anchor_idx)
        attn_mask = anchor_clusters.unsqueeze(2) != labels.unsqueeze(1)

        # for stable training?
        for b in range(batch_size):
            attn_mask[b, num_tokens[b]:, :] = False

        squeezed_features = self.cross_attn(
            query_features,
            x,
            tgt_key_padding_mask=src_kp_mask,
            memory_mask=attn_mask.repeat_interleave(self.num_heads, dim=0)
        )

        query_features = query_features + self.linear1(squeezed_features)

        return query_features
