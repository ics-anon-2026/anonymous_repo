# -*- coding: utf-8 -*-
"""结构攻击实现（Phase 0 起）。

包含：
- DiceAttack：面向欺诈语义的 DICE 变体（删欺诈-欺诈边 + 注入欺诈-良性边）。
  算法从 src/direction2/experiments/run_attack.py 的 dice_attack 迁移，
  改为继承 BaseAttack 并经由 _finalize 强制不变量。
- CamoAttack：关系伪装攻击（relation camouflage），实例化 CARE-GNN (CIKM'20)
  的 relation-camouflage 威胁模型——删欺诈-欺诈边 + 加 欺诈→良性(热门) 边。

两者均为纯结构、免训练攻击（不训练 victim，不训练 surrogate），
符合 Phase 0 "零训练设计验证" 的边界；camo 的 popular_bias 选项贴合
CARE-GNN 描述的 "popular benign nodes" 伪装行为。
"""
from __future__ import annotations

import torch

from base import BaseAttack


class DiceAttack(BaseAttack):
    name = "dice"

    def attack(self, data):
        g = torch.Generator().manual_seed(self.seed)
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        y = data.y
        E = edge_index.size(1)
        n_del = int(E * self.budget / 2)
        n_add = int(E * self.budget / 2)

        a, b = edge_index[0], edge_index[1]
        lo = torch.minimum(a, b)
        hi = torch.maximum(a, b)
        seen = set((lo * num_nodes + hi).tolist())

        # 1) 删欺诈-欺诈无向边
        ff_mask = (y[a] == 1) & (y[b] == 1) & (a < b)
        ff_idx = ff_mask.nonzero(as_tuple=True)[0]
        if ff_idx.size(0) > n_del:
            sel = torch.randperm(ff_idx.size(0), generator=g)[:n_del]
            del_core = ff_idx[sel]
        else:
            del_core = ff_idx
        del_set = set()
        for i in del_core.tolist():
            u, v = a[i].item(), b[i].item()
            del_set.add((u, v))
            del_set.add((v, u))
        keep = [i for i in range(E) if (a[i].item(), b[i].item()) not in del_set]
        new_ei = edge_index[:, torch.tensor(keep, dtype=torch.long)]

        # 2) 注入欺诈-良性边
        fraud_nodes = (y == 1).nonzero(as_tuple=True)[0]
        benign_nodes = (y == 0).nonzero(as_tuple=True)[0]
        add_pairs = []
        attempts = 0
        seen_set = set(seen)
        # 每条待加边以 (u,v)+(v,u) 两个有向 entry 计入 add_pairs，
        # 故循环目标应为 n_add*2 个 entry（即 n_add 对欺诈-良性边）
        while len(add_pairs) < n_add * 2 and attempts < n_add * 40:
            attempts += 1
            u = fraud_nodes[torch.randint(fraud_nodes.size(0), (1,), generator=g)].item()
            v = benign_nodes[torch.randint(benign_nodes.size(0), (1,), generator=g)].item()
            if u == v:
                continue
            lo_, hi_ = (u, v) if u < v else (v, u)
            enc = lo_ * num_nodes + hi_
            if enc in seen_set:
                continue
            seen_set.add(enc)
            add_pairs.append((u, v))
            add_pairs.append((v, u))
        if add_pairs:
            add_t = torch.tensor(add_pairs, dtype=torch.long).t()
            new_ei = torch.cat([new_ei, add_t], dim=1)
        return self._finalize(data, new_ei, n_add=len(add_pairs) // 2, n_del=len(del_core))


class CamoAttack(BaseAttack):
    name = "camo"

    def __init__(self, budget=0.05, seed=0, device="cpu", popular_bias=True, **kwargs):
        super().__init__(budget, seed, device, **kwargs)
        self.popular_bias = bool(popular_bias)

    def attack(self, data):
        g = torch.Generator().manual_seed(self.seed)
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        y = data.y
        E = edge_index.size(1)
        n_del = int(E * self.budget / 2)
        n_add = int(E * self.budget / 2)

        a, b = edge_index[0], edge_index[1]
        lo = torch.minimum(a, b)
        hi = torch.maximum(a, b)
        seen = set((lo * num_nodes + hi).tolist())

        # 1) 删欺诈-欺诈无向边（弱化欺诈聚类信号）
        ff_mask = (y[a] == 1) & (y[b] == 1) & (a < b)
        ff_idx = ff_mask.nonzero(as_tuple=True)[0]
        if ff_idx.size(0) > n_del:
            sel = torch.randperm(ff_idx.size(0), generator=g)[:n_del]
            del_core = ff_idx[sel]
        else:
            del_core = ff_idx
        del_set = set()
        for i in del_core.tolist():
            u, v = a[i].item(), b[i].item()
            del_set.add((u, v))
            del_set.add((v, u))
        keep = [i for i in range(E) if (a[i].item(), b[i].item()) not in del_set]
        new_ei = edge_index[:, torch.tensor(keep, dtype=torch.long)]

        # 2) 加 欺诈→良性 边；popular_bias=True 时按 degree 加权采样良性目标
        fraud_nodes = (y == 1).nonzero(as_tuple=True)[0]
        benign_nodes = (y == 0).nonzero(as_tuple=True)[0]
        if self.popular_bias and benign_nodes.size(0) > 0:
            # 良性节点度（基于原始边）
            deg = torch.zeros(num_nodes, dtype=torch.long)
            deg = deg.scatter_add(0, a, torch.ones(E, dtype=torch.long))
            deg = deg.scatter_add(0, b, torch.ones(E, dtype=torch.long))
            ben_deg = deg[benign_nodes].float().clamp_min(1.0)
            ben_prob = ben_deg / ben_deg.sum()
        else:
            ben_prob = None

        add_pairs = []
        attempts = 0
        seen_set = set(seen)
        while len(add_pairs) < n_add * 2 and attempts < n_add * 100:
            attempts += 1
            u = fraud_nodes[torch.randint(fraud_nodes.size(0), (1,), generator=g)].item()
            if ben_prob is not None:
                v_idx = torch.multinomial(ben_prob, 1, generator=g).item()
                v = benign_nodes[v_idx].item()
            else:
                v = benign_nodes[torch.randint(benign_nodes.size(0), (1,), generator=g)].item()
            if u == v:
                continue
            lo_, hi_ = (u, v) if u < v else (v, u)
            enc = lo_ * num_nodes + hi_
            if enc in seen_set:
                continue
            seen_set.add(enc)
            add_pairs.append((u, v))
            add_pairs.append((v, u))
        if add_pairs:
            add_t = torch.tensor(add_pairs, dtype=torch.long).t()
            new_ei = torch.cat([new_ei, add_t], dim=1)
        return self._finalize(
            data, new_ei, n_add=len(add_pairs) // 2, n_del=len(del_core),
            extra_info={"popular_bias": self.popular_bias},
        )
