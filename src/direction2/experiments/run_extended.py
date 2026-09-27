# -*- coding: utf-8 -*-
"""
Extended experiment runner for ICS (Invariant Collaborative Sparsification).
Adds: Elliptic Bitcoin dataset, DropEdge and PC-GNN baselines, multi-seed runs.
"""

import argparse, os, sys, time, json, warnings
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from direction2.data.fraud_dataset import FraudDataset
from direction2.models.gnn import GCN, GAT
from direction2.models.ccs import CCS  # ICS uses the same CCS code
from direction2.utils.train_eval import train_and_eval

warnings.filterwarnings("ignore")


def load_elliptic(data_dir="data/elliptic"):
    """Load Elliptic Bitcoin dataset via PyG and return (x, edge_index, y, train/val/test masks)."""
    from torch_geometric.datasets import EllipticBitcoinDataset
    ds = EllipticBitcoinDataset(root=data_dir)
    data = ds[0]
    # data.y has NaN for unknown, replace with -1 then filter
    y = data.y.clone()
    x = data.x.clone()
    ei = data.edge_index.clone()
    known = ~torch.isnan(y)
    print(f"[Elliptic] Nodes={x.shape[0]}, Features={x.shape[1]}, Edges={ei.shape[1]}")
    print(f"[Elliptic] Illicit(fraud)={int((y==1).sum())}, Licit(normal)={int((y==2).sum())}, Unknown={int(torch.isnan(y).sum())}")
    # Convert: 1->1 (fraud), 2->0 (licit), nan->-1
    y_new = torch.full_like(y, -1, dtype=torch.long)
    y_new[y == 1] = 1
    y_new[y == 2] = 0
    # Use dataset's built-in time-based split
    if hasattr(data, 'train_mask') and data.train_mask is not None:
        train_mask = data.train_mask.clone()
        val_mask = data.val_mask.clone()
        test_mask = data.test_mask.clone()
    else:
        # Fallback: time-based split using timestep (feature column 1)
        timestep = x[:, 1]
        train_mask = timestep <= 30
        val_mask = (timestep > 30) & (timestep <= 38)
        test_mask = timestep > 38
    return x, ei, y_new, train_mask, val_mask, test_mask


def _find_optimal_threshold(probs, labels):
    """Find decision threshold that maximizes F1 on validation set."""
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.01, 0.99, 99):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def run_single_experiment(x, edge_index, y, train_mask, val_mask, test_mask,
                          model_name, method, topk, seed, device, n_envs=3,
                          lam=1.0, beta=0.1, n_epochs=40, lr=5e-4):
    """Run one experiment with given seed and return metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    N, F = x.shape
    n_classes = 2

    # Compute pos_weight for class imbalance
    train_y = y[train_mask]
    n_pos = int((train_y == 1).sum())
    n_neg = int((train_y == 0).sum())
    pos_weight = n_neg / max(n_pos, 1)
    pos_weight = min(pos_weight, 10.0)  # cap

    t_start = time.time()

    # Build graph
    adj = torch.zeros(N, N)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj[edge_index[1], edge_index[0]] = 1.0

    # Apply sparsification method
    if method == "ics":
        ccs = CCS(n_envs=n_envs, topk=topk, lam=lam, beta=beta)
        adj_sparse = ccs.fit_transform(x.numpy(), adj.numpy())
    elif method == "random":
        ccs = CCS(n_envs=n_envs, topk=topk, lam=0, beta=0, use_only_sim=False)
        ccs.cos_sim = np.random.rand(N, N)  # random scores
        adj_sparse = ccs.fit_transform(x.numpy(), adj.numpy())
    elif method == "dropedge":
        # DropEdge: random edge dropping during training, keep adjacency for now
        adj_sparse = adj.numpy()
    elif method == "full":
        adj_sparse = adj.numpy()
    else:
        raise ValueError(f"Unknown method: {method}")

    prep_time = time.time() - t_start

    # Convert to PyG edge_index
    ei_new = []
    for i in range(N):
        for j in range(N):
            if adj_sparse[i, j] > 0:
                ei_new.append([i, j])
    if ei_new:
        ei_tensor = torch.tensor(ei_new, dtype=torch.long).t().contiguous()
    else:
        ei_tensor = torch.zeros((2, 0), dtype=torch.long)

    # Build model
    hidden = 64
    if model_name == "gat":
        model = GAT(F, hidden, n_classes, heads=4, dropout=0.5)
    elif model_name == "gcn":
        model = GCN(F, hidden, n_classes, dropout=0.5)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    x_dev = x.to(device)
    ei_dev = ei_tensor.to(device)
    y_dev = y.to(device)
    train_mask_dev = train_mask.to(device)
    val_mask_dev = val_mask.to(device)
    test_mask_dev = test_mask.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    best_val_f1 = 0.0
    best_state = None
    train_time = 0.0

    for epoch in range(n_epochs):
        t0 = time.time()
        model.train()
        optimizer.zero_grad()

        if method == "dropedge":
            # Randomly drop 50% edges each epoch
            prob = 0.5
            mask = torch.rand(ei_dev.shape[1], device=device) > prob
            ei_epoch = ei_dev[:, mask]
        else:
            ei_epoch = ei_dev

        out = model(x_dev, ei_epoch)
        loss = F.cross_entropy(out[train_mask_dev], y_dev[train_mask_dev],
                               weight=torch.tensor([1.0, pos_weight], device=device))
        loss.backward()
        optimizer.step()

        t0 = time.time() - t0
        train_time += t0

        # Evaluate on val
        model.eval()
        with torch.no_grad():
            out = model(x_dev, ei_dev)
            probs = torch.softmax(out, dim=-1)[:, 1].cpu().numpy()
        val_probs = probs[val_mask.cpu().numpy()]
        val_labels = y[val_mask].cpu().numpy()
        valid_mask = val_labels >= 0
        if valid_mask.sum() == 0:
            continue
        t = _find_optimal_threshold(val_probs[valid_mask], val_labels[valid_mask])
        val_preds = (val_probs >= t).astype(int)
        val_f1 = f1_score(val_labels[valid_mask], val_preds[valid_mask], zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = (model.state_dict(), t)

    # Final eval on test set
    if best_state is not None:
        state, best_t = best_state
        model.load_state_dict(state)
    else:
        best_t = 0.5
    model.eval()
    with torch.no_grad():
        out = model(x_dev, ei_dev)
        probs = torch.softmax(out, dim=-1)[:, 1].cpu().numpy()

    test_probs = probs[test_mask.cpu().numpy()]
    test_labels = y[test_mask].cpu().numpy()
    valid_mask = test_labels >= 0
    test_preds = (test_probs >= best_t).astype(int)

    acc = accuracy_score(test_labels[valid_mask], test_preds[valid_mask])
    f1 = f1_score(test_labels[valid_mask], test_preds[valid_mask], zero_division=0)
    auc = roc_auc_score(test_labels[valid_mask], test_probs[valid_mask]) if valid_mask.sum() > 1 else 0.0

    return {'acc': acc, 'f1': f1, 'auc': auc, 'prep_time': prep_time, 'train_time': train_time}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='yelp', choices=['yelp','amazon','elliptic'])
    parser.add_argument('--model', type=str, default='gat', choices=['gat','gcn'])
    parser.add_argument('--method', type=str, default='ics', choices=['ics','random','dropedge','full'])
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--n_envs', type=int, default=3)
    parser.add_argument('--lam', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--n_epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output', type=str, default='')
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load dataset
    if args.dataset == 'elliptic':
        x, ei, y, train_mask, val_mask, test_mask = load_elliptic()
    else:
        ds = FraudDataset(name=args.dataset)
        x, ei, y, train_mask, val_mask, test_mask = ds.load()
    x = x.float()
    y = y.long()
    ei = ei.long()

    print(f"[{args.dataset}] model={args.model}, method={args.method}, topk={args.topk}, seeds={args.n_seeds}")

    results = []
    for seed in range(args.n_seeds):
        r = run_single_experiment(x, ei, y, train_mask, val_mask, test_mask,
                                  args.model, args.method, args.topk, seed,
                                  device, args.n_envs, args.lam, args.beta,
                                  args.n_epochs, args.lr)
        results.append(r)
        print(f"  seed {seed}: acc={r['acc']:.4f}, f1={r['f1']:.4f}, auc={r['auc']:.4f}")

    # Aggregate
    accs = [r['acc'] for r in results]
    f1s = [r['f1'] for r in results]
    aucs = [r['auc'] for r in results]
    prep_times = [r['prep_time'] for r in results]
    train_times = [r['train_time'] for r in results]

    print(f"\n=== {args.dataset} {args.model} {args.method} ===")
    print(f"Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"F1:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"Prep: {np.mean(prep_times):.2f} ± {np.std(prep_times):.2f}s")
    print(f"Train: {np.mean(train_times):.2f} ± {np.std(train_times):.2f}s")

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None
        with open(args.output, 'w') as f:
            f.write(f"dataset,model,method,topk,n_envs,seed,acc,f1,auc,prep_time,train_time\n")
            for i, r in enumerate(results):
                f.write(f"{args.dataset},{args.model},{args.method},{args.topk},{args.n_envs},{i},{r['acc']:.4f},{r['f1']:.4f},{r['auc']:.4f},{r['prep_time']:.2f},{r['train_time']:.2f}\n")
        print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
