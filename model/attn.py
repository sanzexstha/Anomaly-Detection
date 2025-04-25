import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
import math
from model.Attentions.dozer_attention import DozerAttention

class TriangularCausalMask():
    def __init__(self, B, L, device="cuda"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class AnomalyAttention(nn.Module):
    def __init__(self, configs, win_size, d_model=None, n_heads=None, block_size=None,
                 mask_flag=True, scale=None, attention_dropout=0.0,
                 output_attention=False, use_sparse_attention=False, sparse_attention=None, local_window=None,
                 stride=None,
                 rand_rate=None,
                 vary_len=None, pred_len=None):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.output_attention = output_attention
        self.use_sparse_attention = use_sparse_attention
        self.sparse_attention = sparse_attention

        self.distances = torch.zeros((win_size, win_size)).cuda()
        for i in range(win_size):
            for j in range(win_size):
                self.distances[i][j] = abs(i - j)

        if use_sparse_attention:
            if sparse_attention == "dozer":
              print("================ Using sparse attention: Dozer ====================")
              self.sparse_attention = DozerAttention(local_window=configs.local_window,
                          stride=configs.stride,
                          rand_rate=configs.rand_rate,
                          mask_flag=False,
                          attention_dropout=attention_dropout,
                          vary_len=configs.vary_len,
                          pred_len=pred_len,
                          output_attention=output_attention)
              self.forward_method = self._forward_sparse
        else:
            self.forward_method = self._forward_standard

    def _forward_standard(self, queries, keys, values, sigma, attn_mask):
        B, L, H, E = queries.shape
        S, _, H_val, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = scale * scores
        sigma = sigma.transpose(1, 2)
        window_size = attn.shape[-1]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)

        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda()
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))

        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            return (V.contiguous(), series, prior, sigma)
        else:
            return (V.contiguous(), None)

    def _forward_sparse(self, queries, keys, values, sigma, attn_mask):
        B, L, H, E = queries.shape
        S, _, H_val, D = values.shape

        attn_output, attn = self.sparse_attention(queries, keys, values, attn_mask)

        sigma = sigma.transpose(1, 2)
        window_size = attn.shape[-1]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)

        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda()
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))

        # series = self.dropout(attn)
        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            return (V.contiguous(), series, prior, sigma)
        else:
            return (V.contiguous(), None)

    def forward(self, queries, keys, values, sigma, attn_mask):
        return self.forward_method(queries, keys, values, sigma, attn_mask)

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model, n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)

        out, series, prior, sigma = self.inner_attention(
            queries,
            keys,
            values,
            sigma,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), series, prior, sigma
