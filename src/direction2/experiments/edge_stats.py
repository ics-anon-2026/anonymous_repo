# -*- coding: utf-8 -*-
"""Compute edge retention statistics for ICS vs random sparsification."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))

from ccs import CausalCollaborativeSparsification
from elliptic_dataset import EllipticDataset
from fraud_dataset import FraudDataset


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


def load_dataset(name, use_hetero=True):
    root = Path(__file__).resolve().parents[3] / "data"
    if name == "Elliptic":
        return EllipticDataset(root=root / "elliptic", use_hetero=use_hetero)[0]
    return FraudDataset(root=root / "raw" / "Fraud", name=name, use_hetero=use_hetero)[0]


def analyze(name, topk):
    data = load_dataset(name)
    n = data.num_nodes
    orig_edges = data.num_edges

    # ICS
    ccs = CausalCollaborativeSparsification(
        n_envs=3, topk=topk, device="cpu", collab_bonus=0.1, seed=42)
    edge_index_dict = data.edge_index_dict if hasattr(data, "edge_index_dict") else None
    new_data, _ = ccs.fit_transform(data, edge_index_dict=edge_index_dict)
    ics_edges = new_data.num_edges

    # Random
    rand_ei = random_sparsify(data.edge_index, n, topk, seed=42)
    rand_edges = rand_ei.shape[1]

    print(f"{name:<10} nodes={n:<7} orig_edges={orig_edges:<8} "
          f"topk={topk:<4} ICS_edges={ics_edges:<8} random_edges={rand_edges:<8} "
          f"ICS_ret={ics_edges/orig_edges:.3f} random_ret={rand_edges/orig_edges:.3f}")


if __name__ == "__main__":
    print(f"{'Dataset':<10} {'Nodes':<7} {'OrigEdges':<8} {'TopK':<5} "
          f"{'ICS_Edges':<8} {'Rand_Edges':<8} {'ICS_Ret':<8} {'Rand_Ret':<8}")
    analyze("YelpChi", 20)
    analyze("Amazon", 20)
    analyze("Elliptic", 10)
