# -*- coding: utf-8 -*-
"""统一攻击分发：把各攻击器接到 run_attack.py 的 PyG 数据流。

支持的攻击器：
- camo    : CARE-GNN 关系伪装攻击（本仓库实现，免训练，src/common/attacks/structural.py）
- prbcd   : PRBCD（Geisler et al., NeurIPS 2021）强投毒攻击，deeprobust 实现 + PyG surrogate 适配器
- metattack: Metattack（Zügner & Günnemann, ICDM 2019）强投毒攻击，deeprobust 实现（稠密 A，
            仅 Amazon / 服务器可用；YelpChi 内存过大，CPU 计划不跑）
- binarized: BinarizedAttack 风格的定向结构投毒（Zhu et al., ICDE 2022，GAD 专用）。
            当前为简化实现（梯度定向选边 + 目标欺诈节点分数翻转），
            完整 bi-level + 二值投影见 TODO（实现核对卡已标注）。

所有攻击返回 (edge_index_new, info)，edge_index 为无向对称存储（与 run_attack.py 一致）。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "common" / "attacks"))
sys.path.insert(0, str(ROOT / "src" / "direction1" / "attacks"))

from structural import CamoAttack
from deeprobust_wrapper import prbcd_attack, metattack_attack

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def attack_to_edge_index(data, attack_name, budget=0.05, seed=0, device="cpu", **kw):
    if attack_name == "camo":
        attacked = CamoAttack(budget=budget, seed=seed, device=device,
                              popular_bias=kw.get("popular_bias", True)).attack(data)
        return attacked.edge_index, attacked.attack_info
    if attack_name == "prbcd":
        d = prbcd_attack(model=None, data=data, perturbations=0, budget=budget,
                         device=device, **kw)
        return d.edge_index, {"method": "prbcd"}
    if attack_name == "metattack":
        d = metattack_attack(model=None, data=data,
                             perturbations=int(budget * data.edge_index.size(1) / 2),
                             device=device, **kw)
        return d.edge_index, {"method": "metattack"}
    if attack_name == "binarized":
        return binarized_style_attack(data, budget=budget, seed=seed, device=device, **kw)
    raise ValueError(f"unknown attack {attack_name}")


class _SurrogateBinaryGCN(nn.Module):
    """单输出二分类 GCN surrogate（用于 BinarizedAttack 梯度定向打分）。"""

    def __init__(self, nfeat, nhid):
        super().__init__()
        self.c1 = GCNConv(nfeat, nhid)
        self.c2 = GCNConv(nhid, 1)

    def forward(self, x, edge_index, edge_weight=None):
        h = F.relu(self.c1(x, edge_index, edge_weight))
        return self.c2(h, edge_index, edge_weight).squeeze(-1)


def _train_binary_surrogate(data, device="cpu", epochs=60, hidden=None, seed=0):
    if hidden is None:
        E = data.edge_index.size(1)
        hidden = 16 if E >= 1_000_000 else 32 if E >= 200_000 else 64
    torch.manual_seed(seed)
    sur = _SurrogateBinaryGCN(data.num_features, hidden).to(device)
    data_d = data.to(device)
    opt = torch.optim.Adam(sur.parameters(), lr=0.01, weight_decay=5e-4)
    tr = data_d.train_mask.nonzero(as_tuple=True)[0]
    if tr.size(0) == 0:
        tr = torch.arange(data_d.num_nodes, device=device)
    yb = data_d.y[tr].float()
    for _ in range(epochs):
        sur.train(); opt.zero_grad()
        out = sur(data_d.x, data_d.edge_index)
        loss = F.binary_cross_entropy_with_logits(out[tr], yb)
        loss.backward(); opt.step()
    sur.eval()
    return sur


def binarized_style_attack(data, budget=0.05, seed=0, device="cpu",
                           n_targets=20, epochs=60, **kw):
    """BinarizedAttack 风格的**目标定向**结构投毒（梯度定向 + 二值投影，单级）。

    语义贴合 Zhu et al., ICDE 2022（GAD 专用投毒）：选定少量目标欺诈节点，
    用 surrogate 梯度把"暴露欺诈身份"的边删除、注入 欺诈→良性 边使其伪装，
    目标是把这些目标节点的欺诈分数压低/翻转（hide anomalies）。

    实现要点（与完整 bi-level 的差异已诚实标注）：
    - 删除：在欺诈关联边上按 surrogate 对"hide 目标"目标的梯度排序，删梯度最负者
      （保留该边反而妨碍伪装），等价于二值投影（保留/删除 = 1/0）。
    - 注入：在 目标欺诈→良性 候选非边上按梯度排序，注入梯度最正者（注入助伪装）。
    - 完整 BinarizedAttack 的 bi-level 外层重训 + 二值投影迭代在 Phase 2 后按需补。

    论文引用须标注为 "BinarizedAttack-style (gradient-targeted, single-level)"。
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    y = data.y
    edge_index = data.edge_index
    E = edge_index.size(1)
    num_nodes = data.num_nodes
    n_del = max(1, int(E * budget / 2))
    n_add = max(1, int(E * budget / 2))

    fraud = (y == 1).nonzero(as_tuple=True)[0]
    benign = (y == 0).nonzero(as_tuple=True)[0]
    if fraud.size(0) == 0 or benign.size(0) == 0:
        return edge_index, {"method": "binarized", "skipped": True}

    sur = _train_binary_surrogate(data, device=device, epochs=epochs, seed=seed)
    targets = fraud[torch.randperm(fraud.size(0), generator=g)[:min(n_targets, fraud.size(0))]].to(device)

    a, b = edge_index[0], edge_index[1]
    x_dev = data.x.to(device)
    ei_dev = edge_index.to(device)

    # ---------- 删除候选：现有欺诈关联边，按 hide 梯度排序 ----------
    # 用张量运算替代 4.4M 次 Python 循环（Amazon 规模下纯 Python 循环过慢）
    fraud_mask_edges = (y[a] == 1) | (y[b] == 1)
    del_cand_idx = fraud_mask_edges.nonzero(as_tuple=True)[0]
    sel_del = []
    if del_cand_idx.size(0) > 0:
        w = torch.ones(E, device=device, requires_grad=True)
        out = sur(x_dev, ei_dev, w)
        loss = F.binary_cross_entropy_with_logits(
            out[targets], torch.ones(targets.size(0), device=device))
        loss.backward()
        grad_all = w.grad.detach().cpu()
        del_grad = grad_all[del_cand_idx]
        n_take = min(n_del, del_cand_idx.size(0))
        take = torch.topk(-del_grad, n_take).indices  # 最负优先（删之助伪装）
        sel_del = del_cand_idx[take].tolist()
    if sel_del:
        del_keys = (torch.minimum(a[sel_del], b[sel_del]) * num_nodes
                    + torch.maximum(a[sel_del], b[sel_del]))
        del_keys_uniq = torch.unique(del_keys)
        edge_keys = torch.minimum(a, b) * num_nodes + torch.maximum(a, b)
        keep_mask = ~torch.isin(edge_keys, del_keys_uniq)
        new_ei = edge_index[:, keep_mask]
    else:
        new_ei = edge_index

    # ---------- 注入候选：目标欺诈 → 良性 非边，按梯度排序 ----------
    # 仅对「与目标欺诈节点关联的边」建去重集合（候选 u 来自 targets），
    # 避免对全图 4.4M 边建 Python 集合导致 OOM（Amazon 规模）。
    tgt_inc = torch.isin(a, targets) | torch.isin(b, targets)
    inc_edges = edge_index[:, tgt_inc]
    seen = set((torch.minimum(inc_edges[0], inc_edges[1]) * num_nodes
               + torch.maximum(inc_edges[0], inc_edges[1])).tolist())
    cand = []
    attempts = 0
    max_attempts = max(1000, n_add * 200)
    while len(cand) < n_add * 2 and attempts < max_attempts:
        attempts += 1
        u = targets[torch.randint(targets.size(0), (1,), generator=g)].item()
        v = benign[torch.randint(benign.size(0), (1,), generator=g)].item()
        if u == v:
            continue
        lo_, hi_ = (u, v) if u < v else (v, u)
        enc = lo_ * num_nodes + hi_
        if enc in seen:
            continue
        seen.add(enc)
        cand.append((u, v)); cand.append((v, u))
    if cand:
        cand_t = torch.tensor(cand, dtype=torch.long).t().to(device)
        new_ei_dev = new_ei.to(device)
        E2 = new_ei_dev.size(1)
        aug_edge = torch.cat([new_ei_dev, cand_t], dim=1)
        w2 = torch.ones(E2 + len(cand), device=device, requires_grad=True)
        out2 = sur(x_dev, aug_edge, w2)
        loss2 = F.binary_cross_entropy_with_logits(
            out2[targets], torch.ones(targets.size(0), device=device))
        loss2.backward()
        grad_add = w2.grad[E2:].detach().cpu()
        gp = grad_add.view(-1, 2).max(dim=1).values  # 每条候选(无向对)取最大梯度
        n_take = min(n_add, gp.size(0))
        take_add = torch.topk(gp, n_take).indices
        chosen = [cand[i * 2] for i in take_add.tolist()]
        add_t = torch.tensor(chosen, dtype=torch.long).t()
        new_ei = torch.cat([new_ei, add_t], dim=1)
    return new_ei.to(device), {"method": "binarized", "n_targets": int(targets.size(0))}
