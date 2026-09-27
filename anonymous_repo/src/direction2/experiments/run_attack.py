# -*- coding: utf-8 -*-
"""方向二：结构攻击鲁棒性实验。

威胁模型：攻击者在部署前污染原始图（删除/注入边），防御方
（ICS / Random top-k）在污染后的图上做稀疏化，再训练 victim BinaryGAT。

攻击方式：
- dice：面向欺诈语义的 DICE 变体——删除欺诈-欺诈边 + 注入欺诈-良性边。
- nettack：surrogate 梯度攻击——训练 2 层 GCN surrogate，
  用损失梯度选删除候选、用 fraud logit 梯度选注入候选。

协议：攻击作用于原始密集图（budget = 5% 总边数，删一半加一半），
然后 ICS/Random 稀疏化（k=20）→ 训练 → 测试 AUC。
clean（无攻击）基线取自主实验表，不重跑。

用法：
  python run_attack.py --dataset Amazon --attack dice --defense ics --seed 0
  python run_attack.py --all   # 跑全部矩阵（2数据集×2攻击×2防御×3seeds）
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))

from ccs import CausalCollaborativeSparsification
from fraud_dataset import FraudDataset
from elliptic_dataset import EllipticDataset
from gnn import BinaryGAT, GCN
from train_eval import train_epoch, evaluate_with_val_threshold

ROOT = Path(__file__).resolve().parents[3]


def random_sparsify(edge_index, num_nodes, k, seed=42):
    """与 run_ccs.py 相同的随机 top-k 出边稀疏化。"""
    rng = torch.Generator(device="cpu").manual_seed(seed)
    edge_list = edge_index.t().tolist()
    adj = {i: [] for i in range(num_nodes)}
    for src, dst in edge_list:
        adj[src].append(dst)
    new_edges = []
    for src in range(num_nodes):
        neighbors = adj[src]
        if len(neighbors) > k:
            idx = torch.randperm(len(neighbors), generator=rng)[:k]
            neighbors = [neighbors[i] for i in idx.tolist()]
        for dst in neighbors:
            new_edges.append([src, dst])
    if len(new_edges) == 0:
        return edge_index
    return torch.tensor(new_edges, dtype=torch.long).t().contiguous()


def to_undirected_edge_set(edge_index, num_nodes):
    """返回 64 位编码的无向边集合（用于快速查重）。"""
    a, b = edge_index[0].long(), edge_index[1].long()
    lo = torch.minimum(a, b)
    hi = torch.maximum(a, b)
    enc = lo * num_nodes + hi
    return torch.unique(enc)


def dice_attack(data, budget=0.05, seed=0):
    """欺诈语义 DICE：删 fraud-fraud 边 + 加 fraud-benign 边。

    删除集与注入集各占 budget/2（相对总边数）。
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    y = data.y
    E = edge_index.size(1)
    n_del = int(E * budget / 2)
    n_add = int(E * budget / 2)

    a, b = edge_index[0], edge_index[1]
    # 无向化编码查重
    lo = torch.minimum(a, b)
    hi = torch.maximum(a, b)
    seen = set((lo * num_nodes + hi).tolist())

    # 1) 删除 fraud-fraud 边（无向意义下）
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
    new_edge_index = edge_index[:, torch.tensor(keep, dtype=torch.long)]

    # 2) 注入 fraud-benign 边
    fraud_nodes = (y == 1).nonzero(as_tuple=True)[0]
    benign_nodes = (y == 0).nonzero(as_tuple=True)[0]
    add_pairs = []
    attempts = 0
    seen_set = set(seen)
    while len(add_pairs) < n_add and attempts < n_add * 20:
        attempts += 1
        u = fraud_nodes[torch.randint(fraud_nodes.size(0), (1,), generator=g)].item()
        v = benign_nodes[torch.randint(benign_nodes.size(0), (1,), generator=g)].item()
        if u == v:
            continue
        lo_, hi_ = (u, v) if u < v else (v, u)
        enc = lo_ * num_nodes + hi_
        if enc in seen_set:
            continue  # 已存在
        seen_set.add(enc)
        add_pairs.append((u, v))
        add_pairs.append((v, u))

    if add_pairs:
        add_t = torch.tensor(add_pairs, dtype=torch.long).t()
        new_edge_index = torch.cat([new_edge_index, add_t], dim=1)
    return new_edge_index.contiguous()


def train_surrogate(data, device="cpu", epochs=30, hidden=32, seed=0):
    """训练 2 层 GCN surrogate（用于 nettack 边打分）。

    hidden 取 32 以控制 8M+ 边上的内存峰值（边级消息 [E, hidden]）。
    """
    torch.manual_seed(seed)
    in_ch = data.num_features
    sur = GCN(in_ch, hidden, 1, num_layers=2, dropout=0.5)
    sur = sur.to(device)
    data = data.to(device)
    fraud_ratio = (data.y == 1).float().mean().item()
    pos_weight = torch.tensor(1.0 / fraud_ratio, device=device)
    opt = torch.optim.Adam(sur.parameters(), lr=0.01, weight_decay=5e-4)
    train_idx = data.train_mask.nonzero(as_tuple=True)[0]

    class _BinData:
        pass

    for _ in range(epochs):
        sur.train()
        opt.zero_grad()
        out = sur(data.x, data.edge_index).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(out[train_idx], data.y[train_idx].float(),
                                                  pos_weight=pos_weight)
        loss.backward()
        opt.step()
    return sur


def nettack_attack(data, budget=0.05, seed=0, device="cpu", n_candidate_nodes=400,
                   n_nonneighbors=100):
    """surrogate 梯度结构攻击。

    - 删除：现有边中 |∂L/∂A| 最大的欺诈节点关联边（top E*budget/2）。
    - 注入：抽样欺诈节点的非邻居候选对中，-∂logit_fraud/∂A 最大者
      （即使得欺诈 logit 下降最多的边），top E*budget/2。
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    edge_index = data.edge_index.clone()
    num_nodes = data.num_nodes
    E = edge_index.size(1)
    n_del = int(E * budget / 2)
    n_add = int(E * budget / 2)
    y = data.y

    sur = train_surrogate(data, device=device, seed=seed)

    # ---------- 删除候选：一次 backward 拿全边损失梯度 ----------
    sur.train()
    data_d = data.clone()
    edge_weight = torch.ones(E, device=device, requires_grad=True)
    out = sur(data_d.x, edge_index, edge_weight).squeeze(-1)
    train_idx = data.train_mask.nonzero(as_tuple=True)[0]
    fraud_ratio = (data.y == 1).float().mean().item()
    pos_weight = torch.tensor(1.0 / fraud_ratio, device=device)
    loss = F.binary_cross_entropy_with_logits(out[train_idx], data.y[train_idx].float(),
                                              pos_weight=pos_weight)
    loss.backward()
    grad_del = edge_weight.grad.abs().detach().cpu()

    a, b = edge_index[0], edge_index[1]
    fraud_edge_mask = (y[a] == 1) | (y[b] == 1)
    cand_idx = fraud_edge_mask.nonzero(as_tuple=True)[0]
    if cand_idx.size(0) > n_del:
        scores = grad_del[cand_idx]
        top = torch.topk(scores, n_del).indices
        del_directed = cand_idx[top].tolist()
    else:
        del_directed = cand_idx.tolist()
    del_set = set()
    for i in del_directed:
        u, v = a[i].item(), b[i].item()
        del_set.add((u, v))
        del_set.add((v, u))
    keep = [i for i in range(E) if (a[i].item(), b[i].item()) not in del_set]
    edge_index_kept = edge_index[:, torch.tensor(keep, dtype=torch.long)]

    # ---------- 注入候选：候选边 weight=0 加入图，一次 backward 拿梯度 ----------
    fraud_nodes_all = (y == 1).nonzero(as_tuple=True)[0]
    benign_nodes = (y == 0).nonzero(as_tuple=True)[0]
    n_cand_nodes = min(n_candidate_nodes, fraud_nodes_all.size(0))
    sel_nodes = fraud_nodes_all[torch.randperm(fraud_nodes_all.size(0), generator=g)[:n_cand_nodes]]

    # 现有邻接集合（无向编码）
    ka, kb = edge_index_kept[0], edge_index_kept[1]
    klo = torch.minimum(ka, kb)
    khi = torch.maximum(ka, kb)
    adj_set = set((klo * num_nodes + khi).tolist())

    cand_pairs = []
    for u in sel_nodes.tolist():
        cnt = 0
        tries = 0
        while cnt < n_nonneighbors and tries < n_nonneighbors * 10:
            tries += 1
            v = benign_nodes[torch.randint(benign_nodes.size(0), (1,), generator=g)].item()
            if u == v:
                continue
            lo_, hi_ = (u, v) if u < v else (v, u)
            if lo_ * num_nodes + hi_ in adj_set:
                continue
            adj_set.add(lo_ * num_nodes + hi_)
            cand_pairs.append((u, v))
            cnt += 1

    if cand_pairs:
        cand_t = torch.tensor(cand_pairs, dtype=torch.long).t().to(device)
        E2 = edge_index_kept.size(1)
        aug_edge = torch.cat([edge_index_kept.to(device), cand_t], dim=1)
        aug_weight = torch.ones(E2 + cand_t.size(1), device=device)
        aug_weight.requires_grad_(True)
        out2 = sur(data.x, aug_edge, aug_weight).squeeze(-1)
        # 攻击者目标：降低欺诈节点的 fraud logit
        fraud_idx_all = (y == 1).nonzero(as_tuple=True)[0].to(device)
        inject_loss = out2[fraud_idx_all].mean()
        sur.zero_grad()
        inject_loss.backward()
        grad_add = aug_weight.grad[E2:].detach().cpu()  # 越小（负）越好
        add_score = -grad_add
        n_take = min(n_add, add_score.size(0))
        top_add = torch.topk(add_score, n_take).indices
        add_chosen = [cand_pairs[i] for i in top_add.tolist()]
        add_directed = []
        for (u, v) in add_chosen:
            add_directed.append((u, v))
            add_directed.append((v, u))
        add_t = torch.tensor(add_directed, dtype=torch.long).t()
        edge_index_kept = torch.cat([edge_index_kept, add_t], dim=1)

    return edge_index_kept.contiguous()


def train_victim(model, data, epochs=100, lr=1e-3, device="cpu", patience=20):
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    fraud_ratio = (data.y == 1).float().mean().item()
    pos_weight = torch.tensor(1.0 / fraud_ratio, device=device)

    best_val = -1.0
    best_state = None
    patience_counter = 0
    for _ in range(epochs):
        train_epoch(model, data, optimizer, device, pos_weight,
                    use_focal=False, balanced_batch=True)
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_idx = data.val_mask.nonzero(as_tuple=True)[0]
            val_prob = torch.sigmoid(out[val_idx])
            val_auc = roc_auc_safe(data.y[val_idx].cpu(), val_prob.cpu())
        if val_auc > best_val:
            best_val = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def roc_auc_safe(y_true, y_prob):
    from sklearn.metrics import roc_auc_score
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.5
    return roc_auc_score(y_true, y_prob)


def run_one(dataset_name, attack_name, defense, seed, budget=0.05, topk=20,
            n_envs=3, device="cpu"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    use_hetero = defense == "ics"
    if dataset_name == "Elliptic":
        ds = EllipticDataset(root=ROOT / "data" / "elliptic", use_hetero=use_hetero)
        topk_ds = 10
    else:
        ds = FraudDataset(root=ROOT / "data" / "raw" / "Fraud", name=dataset_name,
                          use_hetero=use_hetero)
        topk_ds = topk
    data = ds[0]
    print(f"\n=== {dataset_name} | attack={attack_name} | defense={defense} | seed={seed} ===")
    print(f"Original edges={data.num_edges//2}")

    t0 = time.time()
    if attack_name == "dice":
        data.edge_index = dice_attack(data, budget=budget, seed=seed)
    elif attack_name == "nettack":
        data.edge_index = nettack_attack(data, budget=budget, seed=seed, device=device)
    else:
        raise ValueError(attack_name)
    t_attack = time.time() - t0
    print(f"Attacked edges={data.num_edges//2} (attack prep {t_attack:.1f}s)")

    t0 = time.time()
    if defense == "random":
        data.edge_index = random_sparsify(data.edge_index, data.num_nodes, topk_ds, seed=42)
        data.edge_index = data.edge_index.contiguous()
    elif defense == "ics":
        ccs = CausalCollaborativeSparsification(
            n_envs=n_envs, topk=topk_ds, env_stability_weight=1.0,
            device=device, collab_bonus=0.1, use_stab=True, use_collab=True,
            seed=seed, env_mode="kmeans")
        edge_index_dict = data.edge_index_dict if hasattr(data, "edge_index_dict") else None
        data, _ = ccs.fit_transform(data, edge_index_dict=edge_index_dict)
        data = data.to("cpu")
    t_def = time.time() - t0
    print(f"After defense edges={data.num_edges//2} (defense {t_def:.1f}s)")

    model = BinaryGAT(data.num_features, 64, heads=8, dropout=0.1)
    t0 = time.time()
    model = train_victim(model, data, device=device)
    t_train = time.time() - t0

    (test_acc, test_f1, test_auc), threshold = evaluate_with_val_threshold(
        model, data, data.test_mask, device)
    print(f"Test acc={test_acc:.4f} f1={test_f1:.4f} auc={test_auc:.4f} "
          f"(train {t_train:.1f}s)")

    out_dir = ROOT / "results" / "direction2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{dataset_name}_BinaryGAT_{defense}_attack-{attack_name}_s{seed}.csv"
    with open(out_file, "w") as f:
        f.write("dataset,model,defense,attack,topk,n_envs,seed,budget,"
                "test_acc,test_f1,test_auc,attack_time,defense_time,train_time\n")
        f.write(f"{dataset_name},BinaryGAT,{defense},{attack_name},{topk_ds},{n_envs},"
                f"{seed},{budget},{test_acc:.4f},{test_f1:.4f},{test_auc:.4f},"
                f"{t_attack:.2f},{t_def:.2f},{t_train:.2f}\n")
    print(f"Saved to {out_file}")
    return test_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Amazon")
    parser.add_argument("--attack", default="dice", choices=["dice", "nettack"])
    parser.add_argument("--defense", default="ics", choices=["random", "ics"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=float, default=0.05)
    parser.add_argument("--all", action="store_true",
                        help="Run full matrix: 2 datasets x 2 attacks x 2 defenses x 3 seeds")
    args = parser.parse_args()

    if args.all:
        for ds_name in ["Amazon", "YelpChi"]:
            for attack_name in ["dice", "nettack"]:
                for defense in ["random", "ics"]:
                    for seed in [0, 1, 2]:
                        run_one(ds_name, attack_name, defense, seed,
                                budget=args.budget)
    else:
        run_one(args.dataset, args.attack, args.defense, args.seed,
                budget=args.budget)


if __name__ == "__main__":
    main()
