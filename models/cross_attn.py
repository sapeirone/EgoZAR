import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttnLayer(nn.Module):
    """EgoZAR Cross Attention layer"""

    def __init__(self, num_heads: int = 8, embed_dimension: int = 1024,
                 bias: bool = False, dropout: float = 0.1):
        super().__init__()
        assert embed_dimension % num_heads == 0
        
        self.n_heads = num_heads
        self.embed_dimension = embed_dimension
        
        # Query, Keys and Values projection
        self.q_attn = nn.Linear(embed_dimension, embed_dimension, bias=bias)
        self.kv_attn = nn.Linear(embed_dimension, 2 * embed_dimension, bias=bias)
        
        # Output projection
        self.c_proj = nn.Sequential(
            nn.LayerNorm(embed_dimension),
            nn.Linear(embed_dimension, embed_dimension, bias=bias)
        )
        
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        """Cross Attention implementation.

        Parameters
        ----------
        x_q : torch.Tensor
            input queries
        x_kv : torch.Tensor
            input keys and values

        Returns
        -------
        torch.Tensor
            _description_
        """
        query = self.q_attn(x_q)
        kv_projected = self.kv_attn(x_kv)

        bs = query.size(0)
        embed_dim = query.size(2)
        head_dim = embed_dim // (self.n_heads)

        key, value = kv_projected.chunk(2, -1)
        query = query.view(bs, -1, self.n_heads, head_dim).transpose(1, 2)
        key = key.view(bs, -1, self.n_heads, head_dim).transpose(1, 2)
        value = value.view(bs, -1, self.n_heads, head_dim).transpose(1, 2)

        if self.training:
            dropout = self.dropout
        else:
            dropout = 0.0

        y = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=dropout)
        y = y.transpose(1, 2).reshape(bs, -1, self.n_heads * head_dim)
        y = x_kv + y

        y = y + self.c_proj(y)
        return self.resid_dropout(y)
