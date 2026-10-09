# -*- coding: utf-8 -*-
"""ICS 鲁棒性评估 —— 统一攻击 harness（冻结契约，Phase 0, 2026-10-07）。

设计目标：
- direction2（ICS 论文）与方向三综述复现脚本共用同一套攻击接口。
- 接口契约（frozen）：
    * 所有攻击继承 BaseAttack。
    * 构造：Attack(budget: float = 0.05, seed: int = 0, device: str = "cpu", **kwargs)
    * 主 API：attack(data: torch_geometric.data.Data) -> Data
        返回 *克隆* 的 data，含被污染的 .edge_index（必要时 .x）。
        额外挂载 .attack_info (dict) 与 .edge_weight（若攻击产生权重）。
    * attack_info 至少包含：name, budget, seed, n_add, n_del, self_loops, dup_edges。
    * 不变量由 BaseAttack._finalize 强制：无自环、无重复无向边；
      攻击器只需产出"候选边集"，去重/去自环与统计交给 _finalize，
      保证所有攻击产出满足 ICS 论文 Table 的一致性要求。

注意：本文件仅定义契约与工具，Phase 0 不要求实现任何新攻击算法。
"""
from __future__ import annotations

import torch
from torch_geometric.data import Data


class BaseAttack:
    """所有攻击的基类，定义统一接口与不变量工具。"""

    # 子类覆盖：攻击在 results/论文中的短名（dice / camo / metattack ...）
    name = "base"

    def __init__(self, budget: float = 0.05, seed: int = 0, device: str = "cpu", **kwargs):
        self.budget = float(budget)
        self.seed = int(seed)
        self.device = device
        self.extra = kwargs

    def attack(self, data: Data) -> Data:
        """对 data 施加攻击，返回污染后的克隆对象。子类必须实现。"""
        raise NotImplementedError(f"{self.__class__.__name__} 未实现 attack()")

    # ---------- 不变量工具 ----------

    @staticmethod
    def _undirected_key(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """将每条有向边编码为无向唯一整数键（lo*N + hi）。"""
        a, b = edge_index[0].long(), edge_index[1].long()
        lo, hi = torch.minimum(a, b), torch.maximum(a, b)
        return lo * num_nodes + hi

    def _finalize(self, data: Data, edge_index: torch.Tensor,
                  x: torch.Tensor | None = None,
                  n_add: int = 0, n_del: int = 0,
                  extra_info: dict | None = None) -> Data:
        """强制不变量（去自环、去重），记录 attack_info，返回克隆数据。

        edge_index: 候选有向边 [2, E']，可含自环/重复，本函数统一清洗。
        """
        num_nodes = data.num_nodes
        edge_index = edge_index.long().contiguous()

        # 1) 去自环
        sl = edge_index[0] == edge_index[1]
        n_self = int(sl.sum().item())
        edge_index = edge_index[:, ~sl]

        # 2) 去"完全相同的有向边"（(u,v) 出现两次）。
        #    注意：本代码库无向图以对称双方向存储（与 run_attack.py 一致），
        #    反向边 (v,u) 是合法的"同一条无向边"的另一方向，不能删，
        #    因此用有向键 a*N+b 去重，而非无向键 min*N+max。
        key = edge_index[0] * num_nodes + edge_index[1]
        uniq, inv = torch.unique(key, return_inverse=True)
        order = torch.arange(edge_index.size(1))
        first_pos = torch.full((uniq.size(0),), edge_index.size(1), dtype=torch.long)
        first_pos = first_pos.scatter_(0, inv, order)  # 每个有向键首次出现的位置
        keep = first_pos[inv] == order
        n_dup = int((~keep).sum().item())
        edge_index = edge_index[:, keep].contiguous()

        new_data = data.clone()
        new_data.edge_index = edge_index
        if x is not None:
            new_data.x = x
        new_data.attack_info = {
            "name": self.name,
            "budget": self.budget,
            "seed": self.seed,
            "n_add": int(n_add),
            "n_del": int(n_del),
            "self_loops": n_self,
            "dup_edges": n_dup,
        }
        if extra_info:
            new_data.attack_info.update(extra_info)
        return new_data


def build_synthetic_graph(num_nodes: int = 200, fraud_ratio: float = 0.2,
                          p_edge: float = 0.05, fraud_cluster: float = 0.5,
                          seed: int = 0) -> Data:
    """构造带类内聚类的小图，供冒烟测试（无需加载真实数据集）。

    欺诈节点之间以更高概率连边，模拟真实欺诈聚类（便于检验 camo 删欺诈-欺诈边、
    dice 删欺诈-欺诈边等行为是否符合语义）。
    """
    g = torch.Generator().manual_seed(seed)
    y = (torch.rand(num_nodes, generator=g) < fraud_ratio).long()
    fraud = y == 1
    benign = y == 0
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            same = (y[i] == y[j]) and (y[i] == 1)
            p = fraud_cluster if same else p_edge
            if torch.rand(1, generator=g).item() < p:
                edges.append((i, j))
                edges.append((j, i))
    if len(edges) == 0:
        # 保底：至少连一条
        edges = [(0, 1), (1, 0)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.randn(num_nodes, 16, generator=g)
    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_nodes = num_nodes
    return data
