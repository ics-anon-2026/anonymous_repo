# -*- coding: utf-8 -*-
"""
批量实验脚本：多种子、多参数、多baseline。
基于现有 run_ccs.py 的 train/evaluate 逻辑，扩展了 Elliptic 数据集和 DropEdge。
"""

import sys, os, time, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))

from ccs import CausalCollaborativeSparsification
from fraud_dataset import FraudDataset
from gnn import GCN, GAT, HAN
from train_eval import train_epoch, evaluate, evaluate_with_val_threshold


def random_sparsify(edge_index, num_nodes, k, seed=42):
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


def load_elliptic():
    """Load Elliptic Bitcoin dataset as PyG Data + masks."""
    from torch_geometric.datasets import EllipticBitcoinDataset
    ell_dir = str(ROOT / "data" / "elliptic")
    ds = EllipticBitcoinDataset(root=ell_dir)
    data = ds[0]
    # Convert labels: 1=illicit(fraud), 2=licit(normal), nan=unknown
    y = data.y.clone()
    y_new = torch.full_like(y, -1, dtype=torch.long)
    y_new[y == 1] = 1
    y_new[y == 2] = 0
    known = y_new >= 0
    # Random split of known nodes
    known_idx = torch.where(known)[0].numpy()
    np.random.seed(42)
    np.random.shuffle(known_idx)
    n = len(known_idx)
    train_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
    val_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
    test_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
    train_mask[known_idx[:int(n*0.5)]] = True
    val_mask[known_idx[int(n*0.5):int(n*0.75)]] = True
    test_mask[known_idx[int(n*0.75):]] = True
    data.y = y_new
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data


def train_model(model, data, epochs=40, lr=5e-4, device="cpu", pos_weight=None,
                dropedge_rate=0.0, patience=15):
    """Train GNN model. If dropedge_rate > 0, randomly drop edges each epoch."""
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0
    orig_edge_index = data.edge_index.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        if dropedge_rate > 0:
            n_edges = orig_edge_index.shape[1]
            keep = torch.rand(n_edges, device=device) > dropedge_rate
            data.edge_index = orig_edge_index[:, keep]
        else:
            data.edge_index = orig_edge_index

        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        train_y = data.y[data.train_mask].long()
        loss = F.cross_entropy(out[data.train_mask], train_y, weight=torch.tensor([1.0, pos_weight], device=device))
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            data.edge_index = orig_edge_index  # restore full edges for eval
            out = model(data.x, data.edge_index)
            probs = torch.softmax(out, dim=-1)[:, 1].cpu().numpy()

        val_mask = data.val_mask.cpu().numpy()
        val_labels = data.y.cpu().numpy()
        valid = val_mask & (val_labels >= 0)
        if valid.sum() == 0:
            continue
        # Find best threshold on val
        best_t, best_f1 = 0.5, 0.0
        for t in np.linspace(0.05, 0.95, 19):
            preds = (probs[valid] >= t).astype(int)
            from sklearn.metrics import f1_score
            f1 = f1_score(val_labels[valid], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        if best_f1 > best_val_f1:
            best_val_f1 = best_f1
            best_state = (model.state_dict(), best_t)
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    if best_state is not None:
        state, best_t = best_state
        model.load_state_dict(state)
    else:
        best_t = 0.5
    return model, best_t


def run_exp(args, seed):
    """Run one experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    if args.dataset == "Elliptic":
        data = load_elliptic()
    else:
        ds = FraudDataset(root=str(ROOT / "data" / "raw" / "Fraud"), name=args.dataset, use_hetero=True)
        data = ds[0]

    fraud_ratio = (data.y == 1).float().mean().item()
    pos_weight = 1.0 / max(fraud_ratio, 0.01)

    # Sparsification
    t_prep = time.time()
    if args.method == "ccs":
        ccs = CausalCollaborativeSparsification(
            n_envs=args.n_envs, topk=args.topk, env_stability_weight=args.lam,
            device=args.device, collab_bonus=args.beta,
            use_stab=not args.no_stab, use_collab=not args.no_collab)
        edge_index_dict = data.edge_index_dict if hasattr(data, "edge_index_dict") else None
        data, _ = ccs.fit_transform(data, edge_index_dict=edge_index_dict)
        data = data.to("cpu")
    elif args.method == "random":
        data.edge_index = random_sparsify(data.edge_index, data.num_nodes, args.topk, seed=seed)
    elif args.method == "dropedge":
        pass  # handled in training
    prep_time = time.time() - t_prep

    # Build model
    hidden = 64
    if args.model == "GCN":
        model = GCN(data.num_features, hidden, 2, num_layers=2, dropout=0.5)
    elif args.model == "GAT":
        model = GAT(data.num_features, hidden, 2, heads=4, dropout=0.5)
    else:
        raise ValueError(args.model)

    dropedge_rate = 0.5 if args.method == "dropedge" else 0.0

    t_train = time.time()
    model, best_t = train_model(model, data, epochs=args.epochs, lr=args.lr,
                                device=args.device, pos_weight=pos_weight,
                                dropedge_rate=dropedge_rate, patience=15)
    train_time = time.time() - t_train

    # Evaluate
    data = data.to(args.device)
    (test_acc, test_f1, test_auc), threshold = evaluate_with_val_threshold(
        model, data, data.test_mask, args.device)
    return {'acc': test_acc, 'f1': test_f1, 'auc': test_auc,
            'prep_time': prep_time, 'train_time': train_time}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="YelpChi")
    parser.add_argument("--model", type=str, default="GAT")
    parser.add_argument("--method", type=str, default="ccs")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--n_envs", type=int, default=3)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--no_stab", action="store_true")
    parser.add_argument("--no_collab", action="store_true")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    results = []
    for s in range(args.n_seeds):
        r = run_exp(args, s)
        results.append(r)
        print(f"  seed={s}: acc={r['acc']:.4f} f1={r['f1']:.4f} auc={r['auc']:.4f}")

    accs = [r['acc'] for r in results]
    f1s = [r['f1'] for r in results]
    aucs = [r['auc'] for r in results]
    print(f"\n{args.dataset} {args.model} {args.method}: "
          f"Acc={np.mean(accs):.4f}+-{np.std(accs):.4f} "
          f"F1={np.mean(f1s):.4f}+-{np.std(f1s):.4f} "
          f"AUC={np.mean(aucs):.4f}+-{np.std(aucs):.4f}")


if __name__ == "__main__":
    main()
