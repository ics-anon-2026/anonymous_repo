# -*- coding: utf-8 -*-
"""方向二：快速因果协同图稀疏化（Fast Causal Collaborative Sparsification, CCS）。

核心思想：
1. 按节点特征把图划分为若干环境（environment）。
2. 对每条边计算"因果稳定性"分数：
   - 节点特征一致性（两端节点特征相似）。
   - 跨环境稳定性：边两端节点在不同环境下的特征关系保持一致（方差小）。
   - 异构关系协同 bonus：边若同时出现在多个关系网络中，更可能是因果相关。
3. 对每个节点保留 top-k 条高稳定性边，替代随机稀疏化。

相比随机 top-k，CCS 在不增加下游训练时间的前提下，
更有可能保留与欺诈标签相关的因果边。
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch_geometric.utils import to_undirected


class CausalCollaborativeSparsification:
    """快速因果协同图稀疏化器。

    Args:
        n_envs: 环境划分数。
        topk: 每个节点保留的最大出边数。
        env_stability_weight: 跨环境稳定性项权重。
        collab_bonus: 多关系协同 bonus 系数。
        device: 计算设备。
    """

    def __init__(self, n_envs=4, topk=100, env_stability_weight=1.0,
                 collab_bonus=0.1, use_stab=True, use_collab=True, device="cpu",
                 seed=None, normalize_scores=True, env_mode="kmeans",
                 feat_dropout=0.2):
        self.n_envs = n_envs
        self.topk = topk
        self.env_stability_weight = env_stability_weight
        self.collab_bonus = collab_bonus
        self.use_stab = use_stab
        self.use_collab = use_collab
        self.device = device
        self.seed = seed
        self.normalize_scores = normalize_scores
        self.env_mode = env_mode  # "kmeans" or "dropout"
        self.feat_dropout = feat_dropout

    def fit_transform(self, data, edge_index_dict=None):
        """对 data 进行稀疏化并返回新 Data 与环境标签。"""
        data = data.to(self.device)
        env_labels = self._partition_envs(data)
        edge_scores = self._score_edges(data, env_labels, edge_index_dict)
        new_edge_index = self._select_topk_per_node(data.edge_index, edge_scores)

        new_data = data.clone()
        new_data.edge_index = new_edge_index
        return new_data, env_labels

    def _partition_envs(self, data):
        """基于标准化节点特征做 KMeans 环境划分（保留用于兼容）。

        环境划分是数据固有属性，固定 random_state=42，不随训练 seed 变化，
        以保证消融实验在相同环境下公平比较。
        """
        x = data.x.detach().cpu().numpy()
        # Standardize features for stable clustering
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True) + 1e-8
        x = (x - mean) / std
        kmeans = KMeans(n_clusters=self.n_envs, init="k-means++", n_init=10,
                        max_iter=300, random_state=42)
        labels = kmeans.fit_predict(x)
        return torch.from_numpy(labels).long().to(self.device)

    def _get_env_features(self, x):
        """Generate environment-specific features.

        Mode 'dropout': create environments by randomly dropping features.
        Mode 'kmeans': return same features (environments handled by labels).
        """
        if self.env_mode == "dropout":
            envs = []
            rng = torch.Generator(device=x.device)
            rng.manual_seed(42)  # 环境构造固定，不随训练 seed 变化
            for _ in range(self.n_envs):
                mask = torch.rand(x.size(1), generator=rng, device=x.device) > self.feat_dropout
                # Keep at least one feature
                if mask.sum() == 0:
                    mask[0] = True
                envs.append(x * mask.float().unsqueeze(0))
            return envs
        else:
            return [x] * self.n_envs

    def _score_edges(self, data, env_labels, edge_index_dict):
        """计算每条有向边的保留优先级分数（越高越好）。"""
        x = data.x  # [N, F]
        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        device = self.device

        src = edge_index[0].to(device)
        dst = edge_index[1].to(device)
        x = x.to(device)

        def chunk_cos_sim_by_idx(x, idx_a, idx_b, chunk=250_000):
            """按边索引分块计算 cosine similarity，避免大图 OOM。"""
            n = idx_a.size(0)
            out = torch.empty(n, device=device)
            for i in range(0, n, chunk):
                j = min(i + chunk, n)
                a = F.normalize(x[idx_a[i:j]], dim=1)
                b = F.normalize(x[idx_b[i:j]], dim=1)
                out[i:j] = (a * b).sum(dim=1)
            return out

        # 1) 节点特征一致性：cosine similarity
        cos_sim = chunk_cos_sim_by_idx(x, src, dst)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

        # 2) 跨环境稳定性：边在不同特征环境下的余弦相似度保持稳定
        env_sims = []
        if self.env_mode == "kmeans":
            # K-means 环境下：用每个环境内节点特征均值中心化后计算相似度
            for m in range(self.n_envs):
                mask = (env_labels == m)
                if mask.sum() == 0:
                    env_sims.append(torch.zeros(num_edges, device=device))
                    continue
                mu = x[mask].mean(dim=0)
                xe = x - mu
                env_sims.append(torch.clamp(chunk_cos_sim_by_idx(xe, src, dst), -1.0, 1.0))
        else:
            # dropout 环境下：对特征做随机 dropout 扰动
            env_features = self._get_env_features(x)
            for xe in env_features:
                env_sims.append(torch.clamp(chunk_cos_sim_by_idx(xe, src, dst), -1.0, 1.0))

        env_sims = torch.stack(env_sims, dim=0)  # [n_envs, E]
        env_mean = env_sims.mean(dim=0)
        env_var = env_sims.var(dim=0)
        # Higher mean similarity and lower variance -> more stable edge.
        # ReLU keeps the term non-negative so it never actively hurts edges
        # that are already well-separated by raw feature similarity.
        stability = torch.clamp(env_mean, min=0.0) - self.env_stability_weight * env_var
        stability = torch.clamp(stability, min=0.0)

        # 综合分数：特征一致性 + 跨环境稳定性
        scores = cos_sim
        if self.use_stab:
            scores = scores + stability

        # 3) 协同 bonus：边出现在越多种异构关系中，越可能是标签相关边
        if self.use_collab and edge_index_dict:
            # 用 64 位编码 (u,v) 并统计每个有向边出现的关系数（向量化，避免 Python dict 内存爆炸）
            num_nodes = data.num_nodes
            enc_list = []
            for rel_ei in edge_index_dict.values():
                rel_ei = rel_ei.to(self.device)
                enc_list.append(rel_ei[0] * num_nodes + rel_ei[1])
                enc_list.append(rel_ei[1] * num_nodes + rel_ei[0])  # 反向边同样计入
            enc = torch.cat(enc_list)
            uniq, counts = torch.unique(enc, return_counts=True)

            e_full = edge_index[0] * num_nodes + edge_index[1]
            idx = torch.searchsorted(uniq, e_full)
            found = (idx < uniq.size(0)) & (uniq[idx] == e_full)
            bonus = torch.zeros(num_edges, device=self.device)
            bonus[found] = self.collab_bonus * counts[idx[found]].float()
            scores = scores + bonus

        if self.normalize_scores:
            # Min-max normalize to keep components on comparable scales
            s_min, s_max = scores.min(), scores.max()
            if s_max > s_min:
                scores = (scores - s_min) / (s_max - s_min)

        return scores

    def _select_topk_per_node(self, edge_index, edge_scores):
        """对每个源节点保留 topk 条得分最高的出边。"""
        src = edge_index[0].cpu().numpy()
        num_nodes = int(src.max()) + 1
        adj = {i: [] for i in range(num_nodes)}
        for idx, s in enumerate(src):
            adj[s].append(idx)

        keep = []
        for s, idxs in adj.items():
            if len(idxs) == 0:
                continue
            if len(idxs) <= self.topk:
                keep.extend(idxs)
            else:
                scores_s = edge_scores[idxs]
                _, top_idx = torch.topk(scores_s, self.topk, largest=True)
                keep.extend([idxs[i] for i in top_idx.tolist()])

        keep = list(set(keep))
        return edge_index[:, torch.tensor(keep, dtype=torch.long)]


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
    from fraud_dataset import FraudDataset

    root = Path(__file__).resolve().parents[3] / "data" / "raw" / "Fraud"
    ds = FraudDataset(root=root, name="YelpChi", use_hetero=True)
    data = ds[0]
    print("Original:", data)

    ccs = CausalCollaborativeSparsification(n_envs=3, topk=50, device="cpu")
    new_data, env_labels = ccs.fit_transform(data,
                                             edge_index_dict=data.edge_index_dict)
    print("After CCS:", new_data)
    print("Env distribution:", torch.bincount(env_labels).tolist())
