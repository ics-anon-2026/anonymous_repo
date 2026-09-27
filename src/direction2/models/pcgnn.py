# -*- coding: utf-8 -*-
"""方向二：PC-GNN 简化版基线（Pick and Choose GNN, KDD 2021）。

实现要点：
1. Pick：为每个中心节点选择有信息量的邻居。对于训练节点，按类别平衡策略
   从 1-hop 邻居中挑选 top-k 个最相似的邻居（正/负邻居各一半）；对于无标签节点，
   直接按特征相似度挑选 top-k 邻居。
2. Choose：训练时每个 epoch 从训练集中随机抽取等量的正负样本节点（balanced batch）。

参考：Liu et al., "Pick and Choose: A GNN-based Imbalanced Learning Approach for
Fraud Detection", KDD 2021.
"""

import torch
import torch.nn.functional as F


class PCGNNSampler:
    """PC-GNN 的邻居选择（Pick）模块，输出稀疏化后的 edge_index。

    Args:
        k: 每个节点保留的最大出边数。
        pos_ratio: 正邻居目标比例（默认 0.5，即正负各半）。
        use_labels: 是否利用邻居标签做类别平衡选择。若 False 则退化为
                    特征相似度 top-k（与 RIS no-sim 类似）。
        seed: 随机种子。
    """

    def __init__(self, k=20, pos_ratio=0.5, use_labels=True, seed=None):
        self.k = k
        self.pos_ratio = pos_ratio
        self.use_labels = use_labels
        self.seed = seed

    def fit_transform(self, data, train_mask=None):
        edge_index = data.edge_index
        x = data.x
        y = data.y
        num_edges = edge_index.size(1)

        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()
        num_nodes = int(src.max()) + 1
        if train_mask is not None:
            train_mask = train_mask.cpu().numpy()

        # 构建邻接表
        adj = {i: [] for i in range(num_nodes)}
        for idx, (s, d) in enumerate(zip(src, dst)):
            adj[s].append((idx, d))

        keep = []

        for s in range(num_nodes):
            neighbors = adj[s]
            if len(neighbors) == 0:
                continue
            if len(neighbors) <= self.k:
                keep.extend([idx for idx, _ in neighbors])
                continue

            # 计算特征相似度
            idxs = [idx for idx, _ in neighbors]
            d_nodes = torch.tensor([d for _, d in neighbors], dtype=torch.long)
            sim = F.cosine_similarity(x[s].unsqueeze(0), x[d_nodes], dim=1)

            use_label_for_node = (self.use_labels and train_mask is not None and train_mask[s])
            if use_label_for_node:
                # 按邻居标签分组
                labels = y[d_nodes].long()
                pos_idx = [(i, idxs[i]) for i in range(len(idxs)) if labels[i] == 1]
                neg_idx = [(i, idxs[i]) for i in range(len(idxs)) if labels[i] == 0]

                n_pos = int(self.k * self.pos_ratio)
                n_neg = self.k - n_pos

                selected = []
                # 正邻居：按相似度选 top-n_pos
                if len(pos_idx) > 0:
                    pos_sims = torch.tensor([sim[i].item() for i, _ in pos_idx])
                    top_p = min(n_pos, len(pos_idx))
                    _, p_order = torch.topk(pos_sims, top_p, largest=True)
                    selected.extend([pos_idx[i][1] for i in p_order.tolist()])

                # 负邻居：按相似度选 top-n_neg
                if len(neg_idx) > 0:
                    neg_sims = torch.tensor([sim[i].item() for i, _ in neg_idx])
                    top_n = min(n_neg, len(neg_idx))
                    # 若正/负不足，用另一类补足
                    if len(selected) < self.k:
                        top_n = min(self.k - len(selected), len(neg_idx))
                    _, n_order = torch.topk(neg_sims, top_n, largest=True)
                    selected.extend([neg_idx[i][1] for i in n_order.tolist()])

                # 如果还是不够 k，从剩余邻居中按相似度补足
                if len(selected) < self.k:
                    remaining = [i for i in range(len(idxs)) if idxs[i] not in selected]
                    if len(remaining) > 0:
                        rem_sims = torch.tensor([sim[i].item() for i in remaining])
                        need = self.k - len(selected)
                        _, rem_order = torch.topk(rem_sims, min(need, len(remaining)), largest=True)
                        selected.extend([idxs[remaining[i]] for i in rem_order.tolist()])

                keep.extend(selected)
            else:
                # 不使用标签：直接特征相似度 top-k
                _, top = torch.topk(sim, self.k, largest=True)
                keep.extend([idxs[i] for i in top.tolist()])

        keep = list(set(keep))
        return edge_index[:, torch.tensor(keep, dtype=torch.long)]
