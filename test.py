import torch
import torch.nn as nn
from thop import profile

# ---- 1) Define a simple full self‑attention module ----
class FullSelfAttention(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.H = H
        self.d = D // H
        self.qkv_proj = nn.Linear(D, 3*D)
        self.out_proj = nn.Linear(D, D)
    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        qkv = self.qkv_proj(x)                           # [B,L,3D]
        q, k, v = qkv.chunk(3, dim=-1)                   # each [B,L,D]
        q = q.view(B, L, self.H, self.d)
        k = k.view(B, L, self.H, self.d)
        v = v.view(B, L, self.H, self.d)
        # score: [B, H, L, L]
        scores = torch.einsum("blhe,bshe->bhls", q, k)
        attn = torch.softmax(scores / self.d**0.5, dim=-1)
        # out: [B, L, H, d] -> [B, L, D]
        out = torch.einsum("bhls,bshe->blhe", attn, v).reshape(B, L, D)
        return self.out_proj(out)

# ---- 2) Define a window‑based sparse‑attention module ----
class WindowSparseAttention(FullSelfAttention):
    def __init__(self, D, H, window_size):
        super().__init__(D, H)
        self.window = window_size
    def forward(self, x):
        B, L, D = x.shape
        # same qkv as above...
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, self.H, self.d)
        k = k.view(B, L, self.H, self.d)
        v = v.view(B, L, self.H, self.d)
        # build a block‐sparse mask for local window
        scores = torch.zeros(B, self.H, L, L, device=x.device)
        for i in range(L):
            lo = max(0, i - self.window)
            hi = min(L, i + self.window + 1)
            # compute only local dot‐products
            scores[..., i, lo:hi] = (
                (q[:, i : i+1] * k[:, lo:hi]).sum(-1)
                / self.d**0.5
            )
        attn = torch.softmax(scores, dim=-1)
        out = torch.zeros_like(x)
        for i in range(L):
            lo = max(0, i - self.window)
            hi = min(L, i + self.window + 1)
            out[:, i] = (attn[..., i, lo:hi].unsqueeze(-1) * v[:, lo:hi]).sum(1)
        out = out.view(B, L, D)
        return self.out_proj(out)

# ---- 3) Profile both layers ----
if __name__ == "__main__":
    B, L, D, H = 1, 720, 512, 8
    dummy = torch.randn(B, L, D)

    full_attn = FullSelfAttention(D, H)
    macs_full, params_full = profile(full_attn, inputs=(dummy,))
    flops_full = 2 * macs_full
    print(f"Full Attention → MACs: {macs_full:,}, FLOPs ≈ {flops_full:,}, Params: {params_full:,}")

    sparse_attn = WindowSparseAttention(D, H, window_size=24)
    macs_sparse, params_sparse = profile(sparse_attn, inputs=(dummy,))
    flops_sparse = 2 * macs_sparse
    print(f"Sparse Attention → MACs: {macs_sparse:,}, FLOPs ≈ {flops_sparse:,}, Params: {params_sparse:,}")
