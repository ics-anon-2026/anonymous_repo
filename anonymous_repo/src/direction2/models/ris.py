# -*- coding: utf-8 -*-
"""方向二：关系级不变稀疏化（Relation-level Invariant Sparsification, RIS）。

核心思想：
1. 把异构图的每种关系类型视为一个“环境”。
2. 在每个关系子图上计算边的重要性（默认使用节点特征相似度，可选预训练 GNN）。
3. 对每条边计算跨关系稳定性：在多个关系下都重要且重要性波动小 -> 保留。
4. 结合完整图上的特征一致性做 top-k 稀疏化。

相比 ICS 的随机 feature-dropout 环境，RIS 的环境是真实存在的异构关系，
因此 cross-relation stability 具有更明确的语义，也更容易在消融中证明独立价值。
"""

import torch
import torch.nn.functional as F


def relation_edge_importance_gat(x, edge_index, model, device="cpu"):
    """用 GAT 的注意力权重估计边重要性（保留备用）。"""
    model.eval()
    x = x.to(device)
    edge_index = edge_index.to(device)

    with torch.no_grad():
        out, (att_edge_index, att_weights) = model.conv1(
            x, edge_index, return_attention_weights="auto"
        )
    if att_weights.dim() == 2:
        att = att_weights.mean(dim=1)
    else:
        att = att_weights.squeeze(-1)
    return att.cpu()


def relation_edge_importance_gradient(x, edge_index, model, data, device="cpu"):
    """基于边权重梯度估计边重要性（保留备用）。"""
    model.eval()
    x = x.to(device)
    edge_index = edge_index.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)

    edge_weight = torch.ones(edge_index.size(1), device=device, requires_grad=True)
    out = model(x, edge_index, edge_weight=edge_weight)
    loss = F.binary_cross_entropy_with_logits(out[train_mask], y[train_mask].float())
    loss.backward()
    return edge_weight.grad.abs().detach().cpu()


class RelationInvariantSparsification:
    """关系级不变稀疏化器。

    Args:
        topk: 每个节点保留的最大出边数。
        lambda_stab: 跨关系稳定性中的方差惩罚系数。
        alpha: 稳定性项与特征一致性项的相对权重。
        importance_mode: "relation_sim"（默认，基于关系子图特征相似度）、
                        "attention" 或 "gradient"（需要预训练 GNN）。
        pretrain_epochs: 预训练 GNN 的轮数（仅 attention/gradient 模式使用）。
        use_sim: 是否使用完整图特征一致性项。
        use_stab: 是否使用跨关系稳定性项。
        device: 计算设备。
        seed: 随机种子。
    """

    def __init__(self, topk=100, lambda_stab=1.0, alpha=1.0,
                 importance_mode="relation_sim", pretrain_epochs=20,
                 use_sim=True, use_stab=True,
                 device="cpu", seed=None):
        self.topk = topk
        self.lambda_stab = lambda_stab
        self.alpha = alpha
        self.importance_mode = importance_mode
        self.pretrain_epochs = pretrain_epochs
        self.use_sim = use_sim
        self.use_stab = use_stab
        self.device = device
        self.seed = seed

    def fit_transform(self, data, edge_index_dict, pretrain_model=None):
        """对 data 进行稀疏化并返回新 Data。"""
        data = data.to(self.device)

        # 1) 计算完整图特征一致性
        sim_scores = (self._feature_similarity(data.edge_index, data.x)
                      if self.use_sim else
                      torch.zeros(data.edge_index.size(1), device=self.device))

        # 2) 计算跨关系稳定性
        stab_scores = torch.zeros_like(sim_scores)
        if self.use_stab and edge_index_dict is not None and len(edge_index_dict) > 0:
            if self.importance_mode == "relation_sim":
                edge_importance = self._compute_relation_similarity(
                    data, edge_index_dict)
            else:
                if pretrain_model is None:
                    pretrain_model = self._pretrain(data)
                else:
                    pretrain_model = pretrain_model.to(self.device)
                    pretrain_model.eval()
                edge_importance = self._compute_relation_importance(
                    data, edge_index_dict, pretrain_model)
            stab_scores = self._compute_stability(edge_importance)

        # 3) 合并分数并 top-k 选择
        scores = sim_scores + self.alpha * stab_scores
        new_edge_index = self._select_topk_per_node(data.edge_index, scores)

        new_data = data.clone()
        new_data.edge_index = new_edge_index
        return new_data

    def _feature_similarity(self, edge_index, x):
        """计算原始特征余弦相似度。"""
        src, dst = edge_index[0], edge_index[1]
        x_src = x[src]
        x_dst = x[dst]
        sim = F.cosine_similarity(x_src, x_dst, dim=1)
        return torch.clamp(sim, -1.0, 1.0)

    def _compute_relation_similarity(self, data, edge_index_dict):
        """对每个关系子图计算特征相似度，并映射回完整 homogeneous 边集。"""
        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        x = data.x
        relation_scores = {}

        # 为完整图边建立 (u,v) -> index 字典（只处理一次）
        full_edges = edge_index.t().tolist()
        edge_to_idx = {}
        for i, (u, v) in enumerate(full_edges):
            if (u, v) not in edge_to_idx:
                edge_to_idx[(u, v)] = i

        for rel_name, rel_ei in edge_index_dict.items():
            rel_ei = rel_ei.to(self.device)
            src, dst = rel_ei[0], rel_ei[1]
            sim = F.cosine_similarity(x[src], x[dst], dim=1)
            sim = torch.clamp(sim, -1.0, 1.0)

            full_scores = torch.zeros(num_edges, device=self.device)
            rel_list = rel_ei.t().tolist()
            for i, (u, v) in enumerate(rel_list):
                s = sim[i].item()
                if (u, v) in edge_to_idx:
                    idx = edge_to_idx[(u, v)]
                    if s > full_scores[idx]:
                        full_scores[idx] = s
                if (v, u) in edge_to_idx:
                    idx = edge_to_idx[(v, u)]
                    if s > full_scores[idx]:
                        full_scores[idx] = s
            relation_scores[rel_name] = full_scores
        return relation_scores

    def _compute_relation_importance(self, data, edge_index_dict, model):
        """返回 dict[relation_name -> Tensor of importance scores for data.edge_index]。"""
        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        relation_scores = {}

        full_edges = edge_index.t().tolist()
        edge_to_idx = {tuple(e): i for i, e in enumerate(full_edges)}

        for rel_name, rel_ei in edge_index_dict.items():
            rel_ei = rel_ei.to(self.device)
            if self.importance_mode == "attention" and hasattr(model, "conv1"):
                try:
                    scores = relation_edge_importance_gat(
                        data.x, rel_ei, model, device=self.device)
                except Exception:
                    scores = relation_edge_importance_gradient(
                        data.x, rel_ei, model, data, device=self.device)
            else:
                scores = relation_edge_importance_gradient(
                    data.x, rel_ei, model, data, device=self.device)

            full_scores = torch.zeros(num_edges, device=self.device)
            rel_ei_list = rel_ei.t().tolist()
            for i, (u, v) in enumerate(rel_ei_list):
                s = scores[i].item()
                if (u, v) in edge_to_idx:
                    idx = edge_to_idx[(u, v)]
                    if s > full_scores[idx]:
                        full_scores[idx] = s
                if (v, u) in edge_to_idx:
                    idx = edge_to_idx[(v, u)]
                    if s > full_scores[idx]:
                        full_scores[idx] = s
            relation_scores[rel_name] = full_scores
        return relation_scores

    def _compute_stability(self, relation_scores):
        """输入 dict[str, Tensor]，输出每条边的跨关系稳定性分数。"""
        if len(relation_scores) == 0:
            return None

        scores = torch.stack(list(relation_scores.values()), dim=0)  # [R, E]
        mean_score = scores.mean(dim=0)
        var_score = scores.var(dim=0)
        stability = mean_score - self.lambda_stab * var_score
        return stability

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

    def _random_subsample(self, edge_index, num_nodes, max_edges):
        """按源节点随机保留出边，使总有向边数不超过 max_edges。"""
        if edge_index.size(1) <= max_edges:
            return edge_index
        rng = torch.Generator(device="cpu").manual_seed(
            self.seed if self.seed is not None else 42)
        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()
        adj = {i: [] for i in range(num_nodes)}
        for s, d in zip(src, dst):
            adj[s].append(d)

        k_per_node = max(1, int(max_edges / num_nodes))
        new_edges = []
        for s in range(num_nodes):
            neighbors = adj[s]
            if len(neighbors) > k_per_node:
                idx = torch.randperm(len(neighbors), generator=rng)[:k_per_node]
                neighbors = [neighbors[i] for i in idx.tolist()]
            for d in neighbors:
                new_edges.append([s, d])

        if len(new_edges) == 0:
            return edge_index
        return torch.tensor(new_edges, dtype=torch.long).t().contiguous()

    def _pretrain(self, data):
        """训练一个轻量 GCN 用于收集边重要性（仅 attention/gradient 模式）。"""
        from gnn import BinaryGCN

        torch.manual_seed(self.seed if self.seed is not None else 42)
        in_ch = data.num_features
        hidden = max(16, in_ch // 8)

        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        max_pretrain_edges = 1_000_000
        if num_edges > max_pretrain_edges:
            edge_index = self._random_subsample(edge_index, data.num_nodes, max_pretrain_edges)
        edge_index = edge_index.to(self.device)

        model = BinaryGCN(in_ch, hidden, num_layers=2, dropout=0.1).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        y = data.y.to(self.device)
        train_mask = data.train_mask.to(self.device)
        pos_weight = ((y == 0).sum() / (y == 1).sum()).clamp_min(1.0)

        model.train()
        for epoch in range(1, self.pretrain_epochs + 1):
            optimizer.zero_grad()
            out = model(data.x, edge_index)
            loss = F.binary_cross_entropy_with_logits(
                out[train_mask], y[train_mask].float(), pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
        return model
