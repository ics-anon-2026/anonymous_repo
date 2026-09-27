# -*- coding: utf-8 -*-
"""方向二：欺诈检测基线实验。"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))

from fraud_dataset import FraudDataset
from gnn import GCN, GAT, HAN
from train_eval import train_epoch, evaluate


ROOT = Path(__file__).resolve().parents[3]


def sparsify_edge_index(edge_index, num_nodes, k=100, seed=42):
    """For each node, randomly keep at most k outgoing edges (for dense homo graphs)."""
    rng = torch.Generator(device='cpu').manual_seed(seed)
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


def train(model, data, epochs=100, lr=0.01, weight_decay=5e-4, device="cpu",
          pos_weight=None, patience=20):
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_auc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, data, optimizer, device, pos_weight)
        train_acc, train_f1, train_auc = evaluate(model, data, data.train_mask, device)
        val_acc, val_f1, val_auc = evaluate(model, data, data.val_mask, device)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} loss={loss:.4f} "
                  f"train acc={train_acc:.4f} f1={train_f1:.4f} auc={train_auc:.4f} | "
                  f"val acc={val_acc:.4f} f1={val_f1:.4f} auc={val_auc:.4f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="YelpChi", choices=["YelpChi", "Amazon"])
    parser.add_argument("--model", type=str, default="GCN", choices=["GCN", "GAT", "HAN"])
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--pos_weight", type=float, default=None,
                        help="weight for positive class; auto-compute if None")
    parser.add_argument("--sparsify_k", type=int, default=None,
                        help="randomly keep at most k outgoing edges per node for dense homo graphs")
    args = parser.parse_args()

    root = ROOT / "data" / "raw" / "Fraud"
    ds = FraudDataset(root=root, name=args.dataset, use_hetero=(args.model == "HAN"))
    data = ds[0]

    if args.sparsify_k is not None and args.sparsify_k > 0:
        orig_edges = data.num_edges
        data.edge_index = sparsify_edge_index(data.edge_index, data.num_nodes, k=args.sparsify_k)
        print(f"[SPARSIFY] {orig_edges//2} -> {data.num_edges//2} undirected edges (k={args.sparsify_k})")

    fraud_ratio = (data.y == 1).float().mean().item()
    pos_weight = args.pos_weight if args.pos_weight is not None else (1.0 / fraud_ratio)
    print(f"\n=== {args.dataset} | {args.model} | fraud_ratio={fraud_ratio:.4f} | pos_weight={pos_weight:.2f} ===")

    in_ch = data.num_features
    out_ch = int(data.y.max()) + 1

    if args.model == "GCN":
        model = GCN(in_ch, args.hidden, out_ch, num_layers=2, dropout=args.dropout)
    elif args.model == "GAT":
        model = GAT(in_ch, args.hidden, out_ch, heads=8, dropout=args.dropout)
    elif args.model == "HAN":
        metadata = (["user"], [("user", rel, "user") for rel in data.edge_index_dict.keys()])
        model = HAN(in_ch, args.hidden, out_ch, metadata=metadata, heads=8, dropout=args.dropout)
        # Replace forward to use edge_index_dict
        def han_forward(x, edge_index, edge_weight=None):
            return model.han(x, data.edge_index_dict)
        model.forward = han_forward
    else:
        raise ValueError(args.model)

    t0 = time.time()
    model = train(model, data, epochs=args.epochs, lr=args.lr, device=args.device,
                  pos_weight=pos_weight)
    train_time = time.time() - t0

    test_acc, test_f1, test_auc = evaluate(model, data, data.test_mask, args.device)
    print(f"\nTest  acc={test_acc:.4f} f1={test_f1:.4f} auc={test_auc:.4f} (train {train_time:.1f}s)")

    out_dir = ROOT / "results" / "direction2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.dataset}_{args.model}_baseline.csv"
    with open(out_file, "w") as f:
        f.write("dataset,model,test_acc,test_f1,test_auc,train_time\n")
        f.write(f"{args.dataset},{args.model},{test_acc:.4f},{test_f1:.4f},{test_auc:.4f},{train_time:.2f}\n")
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
