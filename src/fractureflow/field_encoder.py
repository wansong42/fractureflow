# -*- coding: utf-8 -*-
"""M2 场编码器: 多层图消息传递 + 注意力机制, 输出逐点特征 h [B,L,d_model]。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, d_in, d_out, d_hidden=256, n_layers=3, dropout=0.15,
                 act=nn.ReLU, norm="ln"):
        super().__init__()
        layers = []
        d = d_in
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(d, d_hidden))
            if norm == "ln":
                layers.append(nn.LayerNorm(d_hidden))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = d_hidden
        layers.append(nn.Linear(d, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """多头注意力用于图节点聚合"""
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5

    def forward(self, h, idx, mask=None):
        """h: [B,L,d], idx: [B,L,k] -> attended features [B,L,d]"""
        B, L, _ = h.shape
        k = idx.shape[2]
        
        # 拿到邻居特征 [B,L,k,d]
        hj = torch.gather(h.unsqueeze(1).expand(B, L, L, h.shape[-1]), 2,
                          idx.unsqueeze(-1).expand(B, L, k, h.shape[-1]))
        
        # Query 来自中心节点, Key/Value 来自邻居
        q = self.q_proj(h)                    # [B,L,d]
        k_proj = self.k_proj(hj)              # [B,L,k,d]
        v = self.v_proj(hj)                   # [B,L,k,d]
        
        # Reshape for multi-head: [B, L, nh, dh]
        q = q.view(B, L, self.n_heads, self.d_head)
        k_proj = k_proj.view(B, L, k, self.n_heads, self.d_head)
        v = v.view(B, L, k, self.n_heads, self.d_head)
        
        # Transpose to [B, nh, L, dh] and [B, nh, L, k, dh]
        q = q.transpose(1, 2)
        k_proj = k_proj.permute(0, 3, 1, 2, 4)  # [B, nh, L, k, dh]
        v = v.permute(0, 3, 1, 2, 4)            # [B, nh, L, k, dh]
        
        # Attention: [B,nh,L,k]
        attn = torch.matmul(q.unsqueeze(3), k_proj.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = mask.view(B, 1, L, 1).expand_as(attn)
            attn = attn.masked_fill(~mask, -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Aggregate: [B,nh,L,dh]
        out = torch.matmul(attn, v).squeeze(3)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_proj(out)


class FieldEncoder(nn.Module):
    """节点特征 + 边特征 -> 逐点特征 h。

    h_i^{t+1} = h_i^t + 1/sqrt(deg_i) * sum_j MLP_edge([h_i, h_j, e_ij]) (residual)
    可选: 注意力聚合替代简单平均
    全局池化读出(可选, 用于组级全局条件)。
    """

    def __init__(self, d_in, d_edge, k, d_model=256, n_layers=3, dropout=0.15, use_attention=True):
        super().__init__()
        self.d_model = d_model
        self.use_attention = use_attention
        self.node_proj = MLP(d_in, d_model, d_hidden=d_model, n_layers=2, dropout=dropout)
        
        if use_attention:
            self.attentions = nn.ModuleList([
                Attention(d_model, n_heads=4, dropout=dropout)
                for _ in range(n_layers)
            ])
            self.edge_mlps = nn.ModuleList([
                MLP(2 * d_model + d_edge, d_model, d_hidden=d_model, n_layers=2, dropout=dropout)
                for _ in range(n_layers)
            ])
        else:
            self.edge_mlps = nn.ModuleList([
                MLP(2 * d_model + d_edge, d_model, d_hidden=d_model, n_layers=2, dropout=dropout)
                for _ in range(n_layers)
            ])
        self.ln = nn.LayerNorm(d_model)

    def message_passing(self, h, edge_mlp, fedge, idx, attention=None):
        B, L, k, _ = fedge.shape
        hn = h[:, :, None, :].expand(B, L, k, h.shape[-1])                 # h_i
        hj = torch.gather(h.unsqueeze(1).expand(B, L, L, h.shape[-1]), 2,
                          idx.unsqueeze(-1).expand(B, L, k, h.shape[-1]))  # h_j
        m = torch.cat([hn, hj, fedge], dim=-1)
        msg = edge_mlp(m)
        deg = k ** 0.5
        agg = msg.mean(2) / deg
        
        if attention is not None:
            attn_out = attention(h, idx)
            return h + agg + attn_out
        return h + agg

    def forward(self, fnode, fedge, idx):
        h = self.node_proj(fnode)
        for i, layer in enumerate(self.edge_mlps):
            attn = self.attentions[i] if self.use_attention else None
            h = self.message_passing(h, layer, fedge, idx, attn)
        return self.ln(h)