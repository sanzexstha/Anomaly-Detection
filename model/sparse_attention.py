import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os

# TriangularCausalMask
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

# --- Final DozerAnomalyAttention ---
class DozerAnomalyAttention(nn.Module):
    """
    AnomalyAttention integrated with Dozer sparse patterns (local, stride)
    for Encoder Self-Attention. Includes fix for full attention fallback.
    """
    def __init__(self, win_size, local_window=None, stride=None, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(DozerAnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag # Controls causal masking
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        # --- AnomalyAttention specific ---
        window_size = win_size
        # Use register_buffer for distance matrix for proper device handling
        self.register_buffer('distances', torch.zeros((window_size, window_size)))
        # Precompute distances
        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j] = abs(i - j)

        # --- DozerAttention specific (Encoder/Self-Attention only) ---
        self.local_window = local_window
        self.stride = stride

        # In class DozerAnomalyAttention(nn.Module):

    def forward(self, queries, keys, values, sigma, attn_mask):
            # B: Batch size, L: Sequence length, H: Heads, E: Query/Key Dim, D: Value Dim
            B, L_Q, H, E = queries.shape
            _, L_K, _, D = values.shape
            if L_Q != L_K:
                raise ValueError(
                    f"Expected L_Q == L_K for DozerAnomalyAttention (Self-Attention), but got L_Q={L_Q}, L_K={L_K}")
            L = L_Q  # Sequence Length

            scale = self.scale or 1. / sqrt(E)

            # 1. Calculate raw scores
            scores = torch.einsum("blhe,bshe->bhls", queries, keys)  # Shape: (B, H, L, L)

            # --- Apply Sparsity (with Full Attention Fallback & VECTORIZED MASK) ---
            sparsity_enabled = (self.local_window is not None and self.local_window > 0) or \
                               (self.stride is not None and self.stride > 0)

            if sparsity_enabled:
                # 2. Generate Dozer sparse mask using VECTORIZED operations
                # Create row and column indices
                rows = torch.arange(L, device=scores.device)
                cols = torch.arange(L, device=scores.device)

                # Initialize mask as False (nothing allowed yet)
                sparse_mask = torch.zeros((L, L), dtype=torch.bool, device=scores.device)

                # Apply Local window attention condition (vectorized)
                if self.local_window is not None and self.local_window > 0:
                    window_half = self.local_window // 2
                    # Condition: abs(row_idx - col_idx) <= window_half
                    local_mask = torch.abs(rows.unsqueeze(1) - cols.unsqueeze(0)) <= window_half
                    sparse_mask = sparse_mask | local_mask  # Combine with OR

                # Apply Strided attention condition (vectorized)
                if self.stride is not None and self.stride > 0:
                    actual_stride = self.stride + 1  # Stride of N means step N+1
                    # Condition: abs(row_idx - col_idx) % actual_stride == 0
                    stride_mask = torch.abs(rows.unsqueeze(1) - cols.unsqueeze(0)) % actual_stride == 0
                    sparse_mask = sparse_mask | stride_mask  # Combine with OR

                # --- Mask generation finished ---

                # Expand sparse mask and apply to scores
                # Mask locations where sparse_mask is False
                scores.masked_fill_(~sparse_mask.unsqueeze(0).unsqueeze(0), -float('inf'))

            # --- Sparsity application finished ---

            # 3. Apply Causal Mask (if enabled) - AFTER sparsity mask
            if self.mask_flag:
                if attn_mask is None:
                    attn_mask = TriangularCausalMask(B, L, device=queries.device)
                scores.masked_fill_(attn_mask.mask, -float('inf'))

            # 4. Calculate Anomaly Attention components (prior and series)
            attn = scale * scores  # Scaled scores, potentially sparse and/or causal

            # --- Calculate Prior (Identical to original AnomalyAttention) ---
            # (Prior calculation code remains the same as your previous version)
            sigma = sigma.transpose(1, 2)  # B L H ->  B H L
            current_L = attn.shape[-1]
            if current_L > self.distances.shape[0]:
                raise ValueError(
                    f"Sequence length L={current_L} exceeds initialized win_size={self.distances.shape[0]} for distance matrix.")
            prior_distances = self.distances[:current_L, :current_L].unsqueeze(0).unsqueeze(0).repeat(B, H, 1, 1)
            sigma = torch.sigmoid(sigma * 5) + 1e-5
            sigma = torch.pow(3, sigma) - 1
            sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, current_L)  # B H L L
            clamped_sigma = torch.clamp(sigma, min=1e-8)
            prior = 1.0 / (math.sqrt(2 * math.pi) * clamped_sigma) * torch.exp(
                -prior_distances.float() ** 2 / (2 * clamped_sigma ** 2))
            prior = torch.nan_to_num(prior, nan=1e-8)

            # 5. --- Calculate Series (Softmax over potentially sparse scores) ---
            series = self.dropout(torch.softmax(attn, dim=-1))
            series = torch.nan_to_num(series, nan=1e-8)

            # 6. Compute output values (V) using the 'series' attention
            V = torch.einsum("bhls,bshd->blhd", series, values)  # (B, L, H, D)

            # 7. Return results
            if self.output_attention:
                return (V.contiguous(), series, prior, sigma)
            else:
                return (V.contiguous(), None)

# --- Your DozerAnomalyAttentionLayer (looks correct, no changes needed) ---
class DozerAnomalyAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(DozerAnomalyAttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
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
        if L != S:
             raise ValueError(f"Encoder self-attention layer expects queries, keys, values to have same sequence length, but got L_Q={L}, L_K={S}")
        H = self.n_heads
        x = queries
        queries_proj = self.query_projection(queries).view(B, L, H, -1)
        keys_proj = self.key_projection(keys).view(B, S, H, -1)
        values_proj = self.value_projection(values).view(B, S, H, -1)
        sigma_proj = self.sigma_projection(x).view(B, L, H)

        outputs = self.inner_attention(
            queries_proj,
            keys_proj,
            values_proj,
            sigma_proj,
            attn_mask
        )

        if self.inner_attention.output_attention:
            out, series, prior, sigma = outputs
            out = out.view(B, L, -1)
            out = self.out_projection(out)
            return out, series, prior, sigma
        else:
            out, _ = outputs
            out = out.view(B, L, -1)
            out = self.out_projection(out)
            return out, None, None, None