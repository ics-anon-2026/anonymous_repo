# -*- coding: utf-8 -*-
"""Elliptic Bitcoin transaction fraud dataset loader.

The dataset contains:
- elliptic_txs_features.csv: first column is tx id, second is time step,
  columns 3-95 are features.
- elliptic_txs_edgelist.csv: directed edges between transactions.
- elliptic_txs_classes.csv: tx id -> class (1=licit, 2=illicite, unknown='unknown').

We treat it as a homogeneous fraud graph and expose a PyG Data object compatible
with the existing direction2 pipeline.
"""

import os
import zipfile

import pandas as pd
import torch
from torch_geometric.data import Data


class EllipticDataset:
    def __init__(self, root, use_hetero=False):
        self.root = root
        self.use_hetero = use_hetero
        self.raw_dir = os.path.join(root, "raw")
        os.makedirs(self.raw_dir, exist_ok=True)
        self._extract()
        self.data = self._process()

    def _extract(self):
        for fname in ["elliptic_txs_features.csv.zip",
                      "elliptic_txs_edgelist.csv.zip",
                      "elliptic_txs_classes.csv.zip"]:
            path = os.path.join(self.raw_dir, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            csv_name = fname.replace(".zip", "")
            csv_path = os.path.join(self.raw_dir, csv_name)
            if not os.path.exists(csv_path):
                with zipfile.ZipFile(path, "r") as z:
                    z.extractall(self.raw_dir)

    def _process(self):
        feat_path = os.path.join(self.raw_dir, "elliptic_txs_features.csv")
        edge_path = os.path.join(self.raw_dir, "elliptic_txs_edgelist.csv")
        cls_path = os.path.join(self.raw_dir, "elliptic_txs_classes.csv")

        features = pd.read_csv(feat_path, header=None)
        edges = pd.read_csv(edge_path)
        classes = pd.read_csv(cls_path)

        # Map tx id to consecutive index
        tx_ids = features.iloc[:, 0].values
        id_map = {tx_id: i for i, tx_id in enumerate(tx_ids)}
        n = len(tx_ids)

        # Features: drop tx id and time step (column 1)
        x = torch.tensor(features.iloc[:, 2:].values, dtype=torch.float)

        # Labels: 0=licit, 1=illicit, -1=unknown
        # Per Elliptic docs: class='1' -> illicit, class='2' -> licit
        cls_map = classes.set_index("txId")["class"].to_dict()
        y = torch.full((n,), -1, dtype=torch.long)
        for tx_id, label in cls_map.items():
            if tx_id in id_map:
                idx = id_map[tx_id]
                if label == "2":
                    y[idx] = 0
                elif label == "1":
                    y[idx] = 1

        # Edges
        src = [id_map.get(u, -1) for u in edges["txId1"].values]
        dst = [id_map.get(v, -1) for v in edges["txId2"].values]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        # Remove edges with unknown ids and self-loops
        valid = (edge_index[0] >= 0) & (edge_index[1] >= 0) & (edge_index[0] != edge_index[1])
        edge_index = edge_index[:, valid]

        data = Data(x=x, edge_index=edge_index, y=y)
        data.num_nodes = n

        # Split by time step (column 1 of features).
        # Use first 34 time steps for train, 35-39 for val, 40-49 for test.
        # Only known-labeled nodes participate in splits.
        time_step = features.iloc[:, 1].values
        known = (y >= 0).numpy()
        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)

        is_train = known & (time_step <= 34)
        is_val = known & ((time_step >= 35) & (time_step <= 39))
        is_test = known & (time_step >= 40)
        train_mask[torch.from_numpy(is_train)] = True
        val_mask[torch.from_numpy(is_val)] = True
        test_mask[torch.from_numpy(is_test)] = True

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        if self.use_hetero:
            # Expose as single-relation heterogeneous graph for compatibility
            data.edge_index_dict = {("user", "transacts", "user"): edge_index}

        return data

    def __getitem__(self, idx):
        return self.data

    def __len__(self):
        return 1


if __name__ == "__main__":
    root = (Path(__file__).resolve().parents[3] / "data" / "elliptic")
    ds = EllipticDataset(root, use_hetero=True)
    data = ds[0]
    print(data)
    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    print(f"Train: {data.train_mask.sum().item()}, Val: {data.val_mask.sum().item()}, Test: {data.test_mask.sum().item()}")
    print(f"Labels: {torch.bincount(data.y[data.y >= 0])}")
