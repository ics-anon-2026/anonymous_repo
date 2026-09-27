# -*- coding: utf-8 -*-
"""方向二：因果协同稀疏化（CCS）实验。

对比方法：
- full: 使用完整 homo 图（Amazon 上可能 OOM）。
- random: 随机 top-k 出边稀疏化。
- ccs:  因果协同稀疏化。
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))

from ccs import CausalCollaborativeSparsification
from ris import RelationInvariantSparsification
from pcgnn import PCGNNSampler
from fraud_dataset import FraudDataset
from elliptic_dataset import EllipticDataset
from gnn import GCN, GAT, HAN, BinaryGCN, BinaryGAT
from train_eval import train_epoch, evaluate, evaluate_with_val_threshold
from ood_split import domain_split


ROOT = Path(__file__).resolve().parents[3]


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


def train(model, data, epochs=100, lr=0.01, weight_decay=5e-4, device="cpu",
          pos_weight=None, use_focal=False, balanced_batch=False, patience=20,
          dropedge_p=0.0, metric="f1"):
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_score = -1.0
    best_state = None
    patience_counter = 0
    orig_edge_index = data.edge_index  # save original for eval

    for epoch in range(1, epochs + 1):
        if dropedge_p > 0:
            n_edges = orig_edge_index.shape[1]
            keep = torch.rand(n_edges, device=device) > dropedge_p
            data.edge_index = orig_edge_index[:, keep]
        else:
            data.edge_index = orig_edge_index

        loss = train_epoch(model, data, optimizer, device, pos_weight,
                           use_focal, balanced_batch)
        data.edge_index = orig_edge_index  # restore for eval
        train_acc, train_f1, train_auc = evaluate(model, data, data.train_mask, device)
        val_acc, val_f1, val_auc = evaluate(model, data, data.val_mask, device)

        val_score = val_f1 if metric == "f1" else val_auc
        if val_score > best_val_score:
            best_val_score = val_score
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


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="YelpChi",
                        choices=["YelpChi", "Amazon", "Elliptic"])
    parser.add_argument("--model", type=str, default="GCN",
                        choices=["GCN", "GAT", "HAN", "BinaryGCN", "BinaryGAT"])
    parser.add_argument("--method", type=str, default="ccs",
                        choices=["full", "random", "ccs", "dropedge", "ris", "pcgnn"])
    parser.add_argument("--dropedge_p", type=float, default=0.5,
                        help="drop edge probability for dropedge method (per epoch)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed")
    parser.add_argument("--topk", type=int, default=100,
                        help="max outgoing edges per node after sparsification")
    parser.add_argument("--n_envs", type=int, default=3,
                        help="number of environments for CCS")
    parser.add_argument("--env_mode", type=str, default="kmeans",
                        choices=["kmeans", "dropout"],
                        help="environment construction for CCS: kmeans or feature dropout")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--pos_weight", type=float, default=None)
    parser.add_argument("--collab_bonus", type=float, default=0.1)
    parser.add_argument("--env_stability_weight", type=float, default=1.0,
                        help="variance penalty weight for CCS cross-environment stability")
    parser.add_argument("--lambda_stab", type=float, default=1.0,
                        help="variance penalty for RIS cross-relation stability")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="weight of RIS stability term relative to feature similarity")
    parser.add_argument("--importance_mode", type=str, default="relation_sim",
                        choices=["relation_sim", "attention", "gradient"],
                        help="how to estimate edge importance in RIS")
    parser.add_argument("--pretrain_epochs", type=int, default=20,
                        help="pretraining epochs for RIS edge-importance model")
    parser.add_argument("--no_stab", action="store_true",
                        help="ablation: disable cross-environment stability term")
    parser.add_argument("--no_collab", action="store_true",
                        help="ablation: disable collaborative relation bonus")
    parser.add_argument("--no_sim", action="store_true",
                        help="RIS ablation: disable feature similarity term")
    parser.add_argument("--ris_no_stab", action="store_true",
                        help="RIS ablation: disable cross-relation stability term")
    parser.add_argument("--pcgnn_pos_ratio", type=float, default=0.5,
                        help="PC-GNN target ratio of positive neighbors")
    parser.add_argument("--pcgnn_use_labels", action="store_true",
                        help="PC-GNN use neighbor labels for balanced picking")
    parser.add_argument("--use_focal", action="store_true",
                        help="use focal loss instead of weighted cross-entropy")
    parser.add_argument("--balanced_batch", action="store_true",
                        help="use balanced node sampling per epoch")
    parser.add_argument("--use_val_threshold", action="store_true",
                        help="select classification threshold on validation set (default: 0.5)")
    parser.add_argument("--ood_split", action="store_true",
                        help="use domain-based OOD split instead of random split")
    parser.add_argument("--ood_domains", type=int, default=3,
                        help="number of domains for OOD split")
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    use_hetero = (args.model == "HAN") or (args.method in ("ccs", "ris") and not args.no_collab)
    if args.dataset == "Elliptic":
        root = ROOT / "data" / "elliptic"
        ds = EllipticDataset(root=root, use_hetero=use_hetero)
    else:
        root = ROOT / "data" / "raw" / "Fraud"
        ds = FraudDataset(root=root, name=args.dataset, use_hetero=use_hetero)
    data = ds[0]

    if args.ood_split:
        print(f"Using OOD domain split (n_domains={args.ood_domains})")
        data, _ = domain_split(data, n_domains=args.ood_domains, seed=args.seed)

    print(f"\n=== {args.dataset} | {args.model} | method={args.method} | topk={args.topk} ===")
    fraud_ratio = (data.y == 1).float().mean().item()
    pos_weight = args.pos_weight if args.pos_weight is not None else (1.0 / fraud_ratio)
    print(f"Original nodes={data.num_nodes} edges={data.num_edges//2} fraud_ratio={fraud_ratio:.4f}")

    t_prep = time.time()
    if args.method == "full":
        pass
    elif args.method == "random":
        data.edge_index = random_sparsify(data.edge_index, data.num_nodes, args.topk)
        data.edge_index = data.edge_index.contiguous()
    elif args.method == "dropedge":
        # DropEdge: keep full graph, randomly drop edges during training
        # Store drop probability for training
        pass
    elif args.method == "ccs":
        ccs = CausalCollaborativeSparsification(
            n_envs=args.n_envs, topk=args.topk,
            env_stability_weight=args.env_stability_weight, device=args.device,
            collab_bonus=args.collab_bonus,
            use_stab=not args.no_stab,
            use_collab=not args.no_collab,
            seed=args.seed, env_mode=args.env_mode)
        edge_index_dict = data.edge_index_dict if hasattr(data, "edge_index_dict") else None
        data, env_labels = ccs.fit_transform(data, edge_index_dict=edge_index_dict)
        data = data.to("cpu")
        print("Env distribution:", torch.bincount(env_labels).tolist())
    elif args.method == "ris":
        ris = RelationInvariantSparsification(
            topk=args.topk, lambda_stab=args.lambda_stab, alpha=args.alpha,
            importance_mode=args.importance_mode,
            pretrain_epochs=args.pretrain_epochs,
            use_sim=not args.no_sim,
            use_stab=not args.ris_no_stab,
            device=args.device, seed=args.seed)
        edge_index_dict = data.edge_index_dict if hasattr(data, "edge_index_dict") else None
        data = ris.fit_transform(data, edge_index_dict=edge_index_dict)
        data = data.to("cpu")
    elif args.method == "pcgnn":
        sampler = PCGNNSampler(
            k=args.topk, pos_ratio=args.pcgnn_pos_ratio,
            use_labels=args.pcgnn_use_labels, seed=args.seed)
        data.edge_index = sampler.fit_transform(data, train_mask=data.train_mask)
        data.edge_index = data.edge_index.contiguous()
    prep_time = time.time() - t_prep
    print(f"After prep edges={data.num_edges//2} (prep {prep_time:.1f}s)")

    in_ch = data.num_features
    out_ch = int(data.y.max()) + 1

    if args.model == "GCN":
        model = GCN(in_ch, args.hidden, out_ch, num_layers=2, dropout=args.dropout)
    elif args.model == "GAT":
        model = GAT(in_ch, args.hidden, out_ch, heads=8, dropout=args.dropout)
    elif args.model == "HAN":
        metadata = (["user"], list(data.edge_index_dict.keys()))
        model = HAN(in_ch, args.hidden, out_ch, metadata=metadata, heads=8, dropout=args.dropout)
    elif args.model == "BinaryGCN":
        model = BinaryGCN(in_ch, args.hidden, num_layers=2, dropout=args.dropout)
    elif args.model == "BinaryGAT":
        model = BinaryGAT(in_ch, args.hidden, heads=8, dropout=args.dropout)
    else:
        raise ValueError(args.model)

    t0 = time.time()
    dropedge_p = args.dropedge_p if args.method == "dropedge" else 0.0
    model = train(model, data, epochs=args.epochs, lr=args.lr, device=args.device,
                  pos_weight=pos_weight, use_focal=args.use_focal,
                  balanced_batch=args.balanced_batch, dropedge_p=dropedge_p,
                  metric="auc")
    train_time = time.time() - t0

    if args.use_val_threshold:
        (test_acc, test_f1, test_auc), threshold = evaluate_with_val_threshold(
            model, data, data.test_mask, args.device)
    else:
        test_acc, test_f1, test_auc = evaluate(model, data, data.test_mask, args.device, threshold=0.5)
        threshold = 0.5
    print(f"\nTest  acc={test_acc:.4f} f1={test_f1:.4f} auc={test_auc:.4f} "
          f"threshold={threshold:.3f} (prep {prep_time:.1f}s, train {train_time:.1f}s)")

    out_dir = ROOT / "results" / "direction2"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if args.method == "ccs":
        if args.no_stab:
            suffix += "_noStab"
        if args.no_collab:
            suffix += "_noCollab"
    elif args.method == "ris":
        suffix += f"_lam{args.lambda_stab}_a{args.alpha}"
        if args.no_sim:
            suffix += "_noSim"
        if args.ris_no_stab:
            suffix += "_noStab"
    elif args.method == "pcgnn":
        suffix += f"_pr{args.pcgnn_pos_ratio}"
        if args.pcgnn_use_labels:
            suffix += "_label"
    if args.ood_split:
        suffix += f"_ood{args.ood_domains}"
    out_file = out_dir / f"{args.dataset}_{args.model}_{args.method}_k{args.topk}_s{args.seed}{suffix}.csv"
    with open(out_file, "w") as f:
        f.write("dataset,model,method,topk,n_envs,seed,test_acc,test_f1,test_auc,prep_time,train_time\n")
        f.write(f"{args.dataset},{args.model},{args.method},{args.topk},{args.n_envs},{args.seed},"
                f"{test_acc:.4f},{test_f1:.4f},{test_auc:.4f},{prep_time:.2f},{train_time:.2f}\n")
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
