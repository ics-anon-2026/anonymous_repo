# -*- coding: utf-8 -*-
"""统一封装 DeepRobust 中的图对抗攻击，输出 PyG 格式。

注意：DeepRobust 的 PGD / Metattack / Nettack 都需要一个内部 surrogate GCN
（即 DeepRobust 自带的 GCN，具有 gc1/gc2 结构）。本文件在攻击前会先用训练集
训练一个 surrogate GCN，再用它生成对抗图，最后让外部受害者模型在该图上评估。
"""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix

from deeprobust.graph.global_attack import PGDAttack, Metattack, MetaApprox, PRBCD, Random, DICE
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.defense import GCN as DeepRobustGCN


class PyGSurrogateGCN(nn.Module):
    """PyG 原生 GCN，满足 deeprobust PRBCD 对 surrogate 的契约：
    - forward(x, edge_index, edge_weight=None) -> logits
    - predict(x, edge_index, edge_weight=None) -> log_softmax
    PRBCD 内部以 edge_weight 做带权传播，故两方法都支持可选 edge_weight。
    """
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super().__init__()
        self.c1 = GCNConv(nfeat, nhid)
        self.c2 = GCNConv(nhid, nclass)
        self.dropout = dropout

    def _propagate(self, x, edge_index, edge_weight):
        if edge_weight is not None and edge_weight.dim() == 1:
            out = self.c1(x, edge_index, edge_weight)
        else:
            out = self.c1(x, edge_index)
        return out

    def forward(self, x, edge_index, edge_weight=None):
        h = self._propagate(x, edge_index, edge_weight)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        if edge_weight is not None and edge_weight.dim() == 1:
            h = self.c2(h, edge_index, edge_weight)
        else:
            h = self.c2(h, edge_index)
        return h

    def predict(self, x, edge_index, edge_weight=None):
        return F.log_softmax(self.forward(x, edge_index, edge_weight), dim=1)

    def fit(self, data, idx_train, idx_val, epochs=200, lr=0.01, weight_decay=5e-4, device="cpu"):
        self.to(device)
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        x, y = data.x.to(device), data.y.to(device)
        ei = data.edge_index.to(device)
        idx_train = torch.as_tensor(idx_train).to(device)
        for _ in range(epochs):
            self.train()
            opt.zero_grad()
            out = self.forward(x, ei)
            loss = F.cross_entropy(out[idx_train], y[idx_train].long())
            loss.backward()
            opt.step()
        self.eval()
        return self


def pyg_data_to_deeprobust(data):
    """PyG Data -> (scipy csr adj, feature numpy, label numpy)"""
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocsr()
    features = data.x.cpu().numpy()
    labels = data.y.cpu().numpy()
    return adj, features, labels


def deeprobust_adj_to_pyg(adj_csr):
    """scipy csr / torch tensor / ndarray -> PyG edge_index"""
    if isinstance(adj_csr, torch.Tensor):
        adj_csr = sp.csr_matrix(adj_csr.detach().cpu().numpy())
    if isinstance(adj_csr, np.ndarray):
        adj_csr = sp.csr_matrix(adj_csr)
    edge_index, edge_weight = from_scipy_sparse_matrix(adj_csr)
    return edge_index, edge_weight


def _train_deeprobust_surrogate(data, device, hidden=64, epochs=200):
    """训练 DeepRobust GCN surrogate（攻击器需要其 gc1/gc2 结构）。"""
    adj, features, labels = pyg_data_to_deeprobust(data)
    nclass = int(labels.max()) + 1
    surrogate = DeepRobustGCN(nfeat=features.shape[1], nhid=hidden, nclass=nclass,
                              dropout=0.5, device=device).to(device)
    surrogate.fit(features, adj, labels,
                  idx_train=data.train_mask.cpu().numpy(),
                  idx_val=data.val_mask.cpu().numpy() if hasattr(data, "val_mask") else None,
                  train_iters=epochs, verbose=False)
    return surrogate, features, labels, adj


def _to_dense_torch(adj, device):
    if isinstance(adj, torch.Tensor):
        return adj.float().to(device)
    return torch.from_numpy(adj.toarray()).float().to(device)


def _build_attacked_data(data, adj_attack, device):
    edge_index, edge_weight = deeprobust_adj_to_pyg(adj_attack)
    data_attack = data.clone()
    data_attack.edge_index = edge_index.to(device)
    data_attack.edge_weight = edge_weight.to(device) if edge_weight is not None else None
    return data_attack


def pgd_attack(model, data, perturbations, device="cpu", attack_structure=True,
               attack_features=False, **kwargs):
    """PGD 逃逸攻击。model 参数仅用于接口统一，实际攻击使用 surrogate GCN。"""
    surrogate, features, labels, adj = _train_deeprobust_surrogate(data, device)
    attack = PGDAttack(surrogate, nnodes=adj.shape[0], attack_structure=attack_structure,
                       attack_features=attack_features, device=device).to(device)
    adj_torch = _to_dense_torch(adj, device)
    attack.attack(features, adj_torch, labels,
                  idx_train=data.train_mask.cpu().numpy(),
                  n_perturbations=int(perturbations), **kwargs)
    adj_attack = attack.modified_adj
    return _build_attacked_data(data, adj_attack, device)


def nettack_attack(model, data, perturbations, device="cpu", **kwargs):
    """Nettack 目标攻击。攻击测试集前 n_target 个目标节点。"""
    surrogate, features, labels, adj = _train_deeprobust_surrogate(data, device)
    attack = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True,
                     attack_features=False, device=device).to(device)

    test_idx = data.test_mask.nonzero(as_tuple=True)[0].cpu().numpy()
    n_target = kwargs.pop("n_target", min(20, len(test_idx)))
    targets = test_idx[:n_target]

    # Nettack 的 perturbations 是 per-target 预算；均分全局预算
    per_target_budget = max(1, int(perturbations) // max(1, n_target))
    features_lil = sp.lil_matrix(features)
    adj_attack = adj.copy().tolil()
    for target in targets:
        attack.attack(features_lil, adj_attack, labels, target, per_target_budget, **kwargs)
        adj_attack = attack.modified_adj.copy().tolil()
    return _build_attacked_data(data, adj_attack.tocsr(), device)


def metattack_attack(model, data, perturbations, device="cpu", lambda_=0.0, **kwargs):
    """Metattack 投毒攻击。"""
    surrogate, features, labels, adj = _train_deeprobust_surrogate(data, device)
    attack = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
                       attack_structure=True, attack_features=False,
                       device=device, lambda_=lambda_).to(device)
    adj_torch = _to_dense_torch(adj, device)
    attack.attack(features, adj_torch, labels,
                  idx_train=data.train_mask.cpu().numpy(),
                  idx_unlabeled=data.test_mask.cpu().numpy(),
                  n_perturbations=int(perturbations), ll_constraint=False, **kwargs)
    adj_attack = attack.modified_adj
    return _build_attacked_data(data, adj_attack, device)


def metapprox_attack(model, data, perturbations, device="cpu", lambda_=0.5, **kwargs):
    """MetaApprox：Metattack 的快速近似版。"""
    surrogate, features, labels, adj = _train_deeprobust_surrogate(data, device)
    attack = MetaApprox(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
                        attack_structure=True, attack_features=False,
                        device=device, lambda_=lambda_).to(device)
    adj_torch = _to_dense_torch(adj, device)
    attack.attack(features, adj_torch, labels,
                  idx_train=data.train_mask.cpu().numpy(),
                  idx_unlabeled=data.test_mask.cpu().numpy(),
                  n_perturbations=int(perturbations), ll_constraint=False, **kwargs)
    adj_attack = attack.modified_adj
    return _build_attacked_data(data, adj_attack, device)


def random_attack(model, data, perturbations, device="cpu", attack_type="flip", **kwargs):
    """随机添加/删除边攻击（弱基线，极快）。"""
    adj, features, labels = pyg_data_to_deeprobust(data)
    attack = Random(device=device)
    attack.attack(adj, n_perturbations=int(perturbations), type=attack_type, **kwargs)
    return _build_attacked_data(data, attack.modified_adj, device)


def dice_attack(model, data, perturbations, device="cpu", **kwargs):
    """DICE 攻击：删除类内边、添加类间边（需要标签）。"""
    adj, features, labels = pyg_data_to_deeprobust(data)
    attack = DICE(device=device)
    attack.attack(adj, labels, n_perturbations=int(perturbations), **kwargs)
    return _build_attacked_data(data, attack.modified_adj, device)


def _surrogate_hidden(num_edges):
    """按边数自适应 surrogate hidden：大图降维避免消息传播 OOM（Amazon 4.4M 边）。"""
    if num_edges >= 1_000_000:
        return 16
    if num_edges >= 200_000:
        return 32
    return 64


def prbcd_attack(model, data, perturbations, device="cpu", **kwargs):
    """PRBCD 攻击（Geisler et al., NeurIPS 2021）的可扩展投毒攻击，适合大图。

    修正点（相对原始 deeprobust_wrapper）：
    - 改用 PyG 原生 surrogate（PyGSurrogateGCN），满足 PRBCD 的
      forward(x, edge_index, edge_weight) + predict(...) 契约；
    - PRBCD.attack 用 ptb_rate（叛变边占比），n_perturbations = int(ptb_rate*E//2)，
      故传入 ptb_rate = 2*budget 使总叛变边数 ≈ budget*E。
    返回：扰动后的 PyG Data（edge_index 为被翻转边的对称存储）。
    """
    E = data.edge_index.size(1)
    surrogate = PyGSurrogateGCN(data.num_features, _surrogate_hidden(E),
                                int(data.y.max().item()) + 1).to(device)
    surrogate.fit(data, data.train_mask.cpu().numpy(),
                  data.val_mask.cpu().numpy() if hasattr(data, "val_mask") else None,
                  epochs=200, device=device)
    prbcd_data = Data(x=data.x, edge_index=data.edge_index, y=data.y)
    for attr in ("train_mask", "val_mask", "test_mask"):
        if hasattr(data, attr):
            setattr(prbcd_data, attr, getattr(data, attr))
    budget = kwargs.pop("budget", None)
    ptb_rate = 2.0 * (budget if budget is not None else (perturbations / E))
    n_pert = int(ptb_rate * E // 2)
    # 轻量默认参数：CPU 友好。search_space_size 限制在 ~n_pert 的 10 倍且不超过全图，
    # 避免大图（Amazon 4.4M 边）上 PRBCD 采样块过慢/内存爆炸；迭代轮数同步下调。
    prbcd_kwargs = dict(
        search_space_size=min(E, max(50000, n_pert * 10)),
        epochs=50, fine_tune_epochs=25, max_final_samples=10,
        with_early_stopping=False, do_synchronize=False,
    )
    prbcd_kwargs.update(kwargs.pop("prbcd_kwargs", {}))
    attack = PRBCD(model=surrogate, data=prbcd_data, device=device, **prbcd_kwargs)
    edge_index, edge_weight = attack.attack(ptb_rate=ptb_rate)
    # 保留 weight≈1 的边（PRBCD 在采样块内翻转权重，非块内边保持 1）
    keep = (edge_weight > 0.5)
    new_edge_index = edge_index[:, keep]
    data_attack = data.clone()
    data_attack.edge_index = new_edge_index.to(device)
    return data_attack


ATTACK_REGISTRY = {
    "fga": None,  # 本地实现，不通过本 wrapper
    "pgd": pgd_attack,
    "nettack": nettack_attack,
    "metattack": metattack_attack,
    "metapprox": metapprox_attack,
    "prbcd": prbcd_attack,
    "random": random_attack,
    "dice": dice_attack,
}


def run_attack(attack_name, model, data, perturbations, device="cpu", **kwargs):
    """统一入口。"""
    if attack_name not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack {attack_name}. Choose from {list(ATTACK_REGISTRY.keys())}")
    return ATTACK_REGISTRY[attack_name](model, data, perturbations, device=device, **kwargs)
