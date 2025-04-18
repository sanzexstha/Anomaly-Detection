#
from math import sqrt
import torch.nn as nn
import torch
from model.Attentions.attention_masking import TriangularCausalMask
import numpy as np
from einops import rearrange

# 用于替代torch.einsum,我想我需要注意计算的过程。
from einops import einsum


class DozerAttention_mask(nn.Module):
    def __init__(self, local_window, stride, rand_rate, vary_len, pred_len, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(DozerAttention_mask, self).__init__()
        self.scale = scale
        # Decoder需要mask.
        # Variable length可以通过mask来实现
        self.local_window = local_window
        self.stride = stride
        self.rand_rate = rand_rate
        self.vary_len = vary_len
        self.mask_flag = mask_flag
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, x, queries, keys, values, attn_mask):
        # Batch size, Seq len, Head, dim/head
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        scale = self.scale or 1. / sqrt(D)

        # 全的A矩阵
        # # Batch size, Head, Seq len, Seq len
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        sparse_mask = torch.zeros(L_Q, L_K, device=scores.device)
        # Self Attention
        if L_Q == L_K:
            # 1. Local Attention. 从local开始。对时序很重要。
            # window_size这个参数还是作为超参数传到这里比较好，便于测试。而且对于不同的数据集，应设置为不同的值。
            if self.local_window:
                for w_idx in range(self.local_window//2+1):
                    sparse_mask = torch.diagonal_scatter(sparse_mask, torch.ones(L_Q - w_idx), w_idx)
                    sparse_mask = torch.diagonal_scatter(sparse_mask, torch.ones(L_Q - w_idx), -w_idx)

            # 2. Stride Attention 同时间的步/片段。强周期性数据中应有高相关性。
            if self.stride:
                stride = self.stride + 1
                for w_idx in range(0, L_Q, stride):
                    sparse_mask = torch.diagonal_scatter(sparse_mask, torch.ones(L_Q - w_idx), w_idx)
                    sparse_mask = torch.diagonal_scatter(sparse_mask, torch.ones(L_Q - w_idx), -w_idx)
            # self_full_QKs = torch.numel(sparse_mask)
            # self_dozer_QKs = sparse_mask.unique(return_counts=True)
            # self_saved_QKs = self_dozer_QKs[1][1]/self_full_QKs
            # print('self Full QKs are', self_full_QKs)
            # print('self Dozer QKs are', self_dozer_QKs)
            # print('selfDozer Only used % QKs', self_saved_QKs)
            # a = sparse_mask.detach().cpu().numpy()

            # # 3. Random
            # rand_mask = torch.where(torch.rand(L_Q, L_Q, device=scores.device) > (1-self.rand_rate), 1, 0)
            # sparse_mask = torch.where((sparse_mask + rand_mask) >= 1, 1, 0)

            # # 4. 聚类
            # # 先随便找个方法试试吧。以后可以换别的。
            # from sklearn.cluster import DBSCAN
            # for batch_idx in range(x.size()[0]):
            #     clustering = DBSCAN(eps=3, min_samples=2).fit(x[batch_idx].detach().cpu().numpy())

        # Cross Attention
        if L_Q != L_K:
            # 0. cross attention中非预测的时间步直接去掉。如果取子矩阵可以省约一半（取决于输入序列反而分布）的计算量。


            # 1. local
            if self.local_window:
                local_window = self.local_window//2 if self.local_window>1 else self.local_window
                sparse_mask[:, -local_window:] = 1

            # 2. Stride
            # (1) 同时间的步/片段。强周期性数据中应有高相关性。(2)降采样以后
            # 对角线能实现吗？ 找到Q和K对应的时间步，然后从第一个对应的时间步开始，用两个循环分别向过去和未来stride
            # Query如果同时包含label和pred，label和pred必须可以分别被patch size整除。
            if self.stride:
                start_index = L_K - L_Q//2
                stride = self.stride + 1
                # 未来和过去
                for w_idx in range(start_index, L_K, stride):
                    sparse_mask = torch.diagonal_scatter(sparse_mask,
                                                         torch.ones(len(torch.diagonal(sparse_mask, w_idx))),
                                                         w_idx)
                for w_idx in range(start_index, -L_K, -stride):
                    sparse_mask = torch.diagonal_scatter(sparse_mask,
                                                         torch.ones(len(torch.diagonal(sparse_mask, w_idx))),
                                                         w_idx)

            # # 3. 随机
            # rand_mask = torch.where(torch.rand(L_Q, L_K, device=scores.device) > (1-self.rand_rate), 1, 0)
            # sparse_mask = torch.where((sparse_mask + rand_mask) >= 1, 1, 0)

            # 4. 对距当前时间不同距离的未来时间步，取不同长度的过去时间步序列。暂时没想好如何实现。
            #    1) 用下三角矩阵实现，对未来的预测逐步加一; 目前的实现是自注意力对未来的每一步，取等长的输入序列。Todo 或许改为两倍？或者等长+1天？
            if self.vary_len or type(self.vary_len) is int:
                start_index = -self.pred_len+self.vary_len-1
                var_len_mask = torch.tril(torch.ones(L_Q, L_K, device=scores.device), diagonal=start_index)
                var_len_mask = torch.flip(var_len_mask, [1])
                sparse_mask = torch.where((sparse_mask + var_len_mask) >= 1, 1, 0)
            # a = sparse_mask.detach().cpu().numpy()
            # # print(1)
            # cross_full_QKs = torch.numel(sparse_mask)
            # cross_dozer_QKs = sparse_mask.unique(return_counts=True)
            # cross_saved_QKs = cross_dozer_QKs[1][1]/cross_full_QKs
            # print('cross Full QKs are', cross_full_QKs)
            # print('cross Dozer QKs are', cross_dozer_QKs)
            # print('cross Dozer Only used % QKs', cross_saved_QKs)

        # 将A与掩码矩阵相乘得到稀疏的A
        scores = scores * sparse_mask

        # Decoder中的masked attention的掩码。
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L_Q, device=queries.device)
            # attn_mask is bool
            scores.masked_fill_(attn_mask.mask, -np.inf)
        b = scores[0, 0, :, :].detach().cpu().numpy()
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        # V和Attention matrix“相乘”
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)



class DozerAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(DozerAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        # shen复制
        x = torch.clone(queries)
        # Batch size, Seq len, embed_dim
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Batch size, Seq len, head, embed_dim/head
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # 真正的注意力计算。输出的是输出序列和注意力矩阵A。
        out, attn = self.inner_attention(
            x,
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)
        out = self.out_projection(out)

        return out, attn


class DozerAttention(nn.Module):
    def __init__(self, local_window, stride, rand_rate, vary_len, pred_len, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(DozerAttention, self).__init__()
        self.scale = scale
        # Decoder需要mask.
        # Variable length可以通过mask来实现
        self.local_window = local_window
        self.stride = stride
        self.rand_rate = rand_rate
        self.vary_len = vary_len
        self.mask_flag = mask_flag
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        # Batch size, Seq len, Head, dim/head
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        scale = self.scale or 1. / sqrt(D)

        sparse_mask = torch.zeros(L_Q, L_K, device=queries.device)
        # Self Attention
        if L_Q == L_K:
            # 1. Local Attention. 从local开始。对时序很重要。
            # window_size这个参数还是作为超参数传到这里比较好，便于测试。而且对于不同的数据集，应设置为不同的值。
            if self.local_window:
                for w_idx in range(self.local_window//2+1):
                    sparse_mask = torch.diagonal_scatter(sparse_mask, torch.ones(L_Q - w_idx), w_idx)
                    sparse_mask = torch.diagonal_scatter(sparse_mask, torch.ones(L_Q - w_idx), -w_idx)

            # 2. Stride Attention 同时间的步/片段。强周期性数据中应有高相关性。
            if self.stride:
                stride = self.stride + 1
                for w_idx in range(0, L_Q, stride):
                    sparse_mask = torch.diagonal_scatter(sparse_mask, torch.ones(L_Q - w_idx), w_idx)
                    sparse_mask = torch.diagonal_scatter(sparse_mask, torch.ones(L_Q - w_idx), -w_idx)

        # Cross Attention
        if L_Q != L_K:
            # 1. local
            if self.local_window:
                local_window = self.local_window//2 if self.local_window>1 else self.local_window
                sparse_mask[:, -local_window:] = 1

            # 2. Stride
            # (1) 同时间的步/片段。强周期性数据中应有高相关性。(2)降采样以后
            # 对角线能实现吗？ 找到Q和K对应的时间步，然后从第一个对应的时间步开始，用两个循环分别向过去和未来stride
            # Query如果同时包含label和pred，label和pred必须可以分别被patch size整除。
            if self.stride:
                start_index = L_K - L_Q//2
                stride = self.stride + 1
                # 未来和过去
                for w_idx in range(start_index, L_K, stride):
                    sparse_mask = torch.diagonal_scatter(sparse_mask,
                                                         torch.ones(len(torch.diagonal(sparse_mask, w_idx))),
                                                         w_idx)
                for w_idx in range(start_index, -L_K, -stride):
                    sparse_mask = torch.diagonal_scatter(sparse_mask,
                                                         torch.ones(len(torch.diagonal(sparse_mask, w_idx))),
                                                         w_idx)

            # 4. 对距当前时间不同距离的未来时间步，取不同长度的过去时间步序列。暂时没想好如何实现。
            #    1) 用下三角矩阵实现，对未来的预测逐步加一; 目前的实现是自注意力对未来的每一步，取等长的输入序列。Todo 或许改为两倍？或者等长+1天？
            if self.vary_len or type(self.vary_len) is int:
                # 2024五月二十四日更改
                start_index = -(L_Q - self.pred_len)+self.vary_len-1
                var_len_mask = torch.tril(torch.ones(L_Q, L_K, device=queries.device), diagonal=start_index)
                var_len_mask = torch.flip(var_len_mask, [1])
                sparse_mask = torch.where((sparse_mask + var_len_mask) >= 1, 1, 0)
                # a = sparse_mask.detach().cpu().numpy()

        # 将A与掩码矩阵相乘得到稀疏的A
        # a = var_len_mask.detach().cpu().numpy()
        # 全的A矩阵
        # #scores Dims: Batch size*ts_d, Head, Seq len, Seq len
        #QKV Dims: Batch size*ts_d, Seq len, Head,  embed_dim_head
        scores = torch.zeros(B, H, L_Q, L_K).to(queries.device)
        # scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        for i in range(L_Q):
            seleted_keys_idxs = rearrange(sparse_mask[i, :].nonzero(), 'dim1 dim2 -> (dim1 dim2)')
            scores[:, :, i:i+1, seleted_keys_idxs] = torch.einsum("blhe,bshe->bhls", queries[:, i:i+1, :, :], keys[:, seleted_keys_idxs, :, :])

        # scores_2 = torch.einsum("blhe,bshe->bhls", queries, keys)
        # scores_2 = scores_2 * sparse_mask
        # a = scores[0, 0, :, :].detach().cpu().numpy()
        # b = scores_2[0, 0, :, :].detach().cpu().numpy()
        # c = a == b
        # Decoder中的masked attention的掩码。
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L_Q, device=queries.device)
            # attn_mask is bool
            scores.masked_fill_(attn_mask.mask, -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        v = scale * scores
        # V和Attention matrix“相乘”
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), v)
        else:
            return (V.contiguous(), None)
