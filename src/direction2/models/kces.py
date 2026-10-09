# -*- coding: utf-8 -*-
"""KCES-style 训练免图净化（简化复现）。

参照 Jia et al. (2026) Kernel-Complexity Edge Sanitization 的核心思想：
training-free 地用图核复杂度给边打分，剪掉复杂度异常高（签名稀有）的边，
从而移除疑似对抗注入的边。本实现为简化版（KCES-lite）：
  - 初始标签 = 特征等频分桶签名（粗粒度保证签名有碰撞）；
  - 1 轮 WL 迭代（粗粒度邻居签名，避免签名过度唯一化）；
  - 边复杂度 = 其无序签名在全图的稀有度（负对数频率），向量化统计；
  - 每个源节点保留复杂度最低（最普通）的 top-k 出边。
"""
import numpy as np
import torch


def _initial_codes(x, n_bins=8, max_dims=8):
    """连续特征 -> 前 max_dims 维等频分桶码，逐维组合成紧凑整数签名。
    限制签名空间（n_bins^max_dims）以保证签名间有充分碰撞。"""
    xf = x.detach().cpu().numpy().astype(np.float64)[:, :max_dims]
    qs = np.linspace(0, 1, n_bins + 1)
    bounds = np.quantile(xf, qs, axis=0)
    codes = np.zeros(xf.shape[0], dtype=np.uint64)
    for f in range(xf.shape[1]):
        c = np.clip(np.searchsorted(bounds[:, f], xf[:, f]) - 1, 0, n_bins - 1).astype(np.uint64)
        codes = codes * np.uint64(n_bins) + c
    return codes.astype(np.int64)


def wl_edge_complexity(x, edge_index, n_bins=4, max_dims=6):
    """返回每条有向边的核签名稀有度（越大越稀有/越可疑）。向量化，内存安全。

    注：签名空间取 n_bins^max_dims（默认 4096）以保证稠密图上签名间有充分碰撞，
    使稀有度具有区分度；不使用 WL 迭代（粗标签经组合后签名会重新唯一化）。
    """
    src = edge_index[0].detach().cpu().numpy()
    dst = edge_index[1].detach().cpu().numpy()
    E = src.shape[0]

    lab = _initial_codes(x, n_bins=n_bins, max_dims=max_dims)

    lo_l, hi_l = lab[np.minimum(src, dst)], lab[np.maximum(src, dst)]
    pairs = np.stack([lo_l, hi_l], axis=1)
    _, inv = np.unique(pairs, axis=0, return_inverse=True)
    cnt = np.bincount(inv)
    scores = -np.log(cnt[inv] / E)
    return scores


def kces_sanitize(edge_index, x, num_nodes, k, n_bins=4, max_dims=6):
    """每源节点保留复杂度最低的 top-k 出边（剪掉稀有/可疑边）。"""
    scores = wl_edge_complexity(x, edge_index, n_bins=n_bins, max_dims=max_dims)
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()

    order = np.argsort(src, kind="stable")
    s_src = src[order]
    s_scores = scores[order]
    s_dst = dst[order]
    bnd = np.searchsorted(s_src, np.arange(num_nodes + 1))

    keep = []
    for u in range(num_nodes):
        b0, b1 = bnd[u], bnd[u + 1]
        n = b1 - b0
        if n > k:
            sel = np.argpartition(s_scores[b0:b1], k)[:k]
            keep.extend((b0 + sel).tolist())
        else:
            keep.extend(range(b0, b1))

    if not keep:
        return edge_index
    keep = order[np.array(keep, dtype=np.int64)]
    new_edge = torch.tensor(np.stack([src[keep], dst[keep]]), dtype=torch.long)
    return new_edge.contiguous()


def graphconsis_sanitize(edge_index, x, num_nodes, k):
    """GraphConsis-style 边选择（简化复现）：consistency score = 特征欧氏距离平方，
    每源节点保留距离最小的 top-k 出边（GraphConsis 用温度 softmax 采样，
    这里用确定性 top-k 以与 ICS/Random 的 top-k 协议严格可比）。"""
    src = edge_index[0]
    dst = edge_index[1]
    d = ((x[src] - x[dst]) ** 2).sum(dim=-1).cpu().numpy()

    order = np.argsort(src.numpy(), kind="stable")
    s_src = src.numpy()[order]
    s_d = d[order]
    s_dst = dst.numpy()[order]
    bnd = np.searchsorted(s_src, np.arange(num_nodes + 1))

    keep_idx = []
    for u in range(num_nodes):
        b0, b1 = bnd[u], bnd[u + 1]
        n = b1 - b0
        if n > k:
            sel = np.argsort(s_d[b0:b1], kind="stable")[:k]
            keep_idx.extend((b0 + sel).tolist())
        else:
            keep_idx.extend(range(b0, b1))

    if not keep_idx:
        return edge_index
    keep = order[np.array(keep_idx, dtype=np.int64)]
    new_edge = torch.stack([src[keep], dst[keep]])
    return new_edge.contiguous()
