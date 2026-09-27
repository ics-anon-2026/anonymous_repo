# -*- coding: utf-8 -*-
"""方向二：异构图金融欺诈检测数据集加载器。

支持 YelpChi 与 Amazon（来自 CARE-GNN 公开数据）。
由于官方 GitHub 在国内访问不稳定，本文件优先从本地 raw 目录读取；
若不存在，则给出手动下载说明。

数据格式：
  1. CARE-GNN release 的 .mat 文件（推荐）：
       {YelpChi,Amazon}.mat
     内含 homo（同质图）以及异构关系矩阵（YelpChi: net_rur/rtr/rsr；
     Amazon: net_upu/usu/uvu）、features、label。
  2. DGL 风格的 pickle/npy 文件：
       {dataset}_adj_lists.pickle, {dataset}_feat.npy, {dataset}_label.pickle
"""
import functools
import pickle
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import from_scipy_sparse_matrix

# PyTorch 2.6 defaults torch.load to weights_only=True, which breaks PyG pickles.
torch.load = functools.partial(torch.load, weights_only=False)


SUPPORTED = {"YelpChi", "Amazon"}

# CARE-GNN .mat 中异构关系名
HETERO_KEYS = {
    "YelpChi": ["net_rur", "net_rtr", "net_rsr"],
    "Amazon": ["net_upu", "net_usu", "net_uvu"],
}


class FraudDataset(InMemoryDataset):
    """本地 CARE-GNN / DGL 风格欺诈检测数据集。"""
    def __init__(self, root, name="YelpChi", transform=None, use_hetero=False):
        assert name in SUPPORTED, f"name must be one of {SUPPORTED}"
        self.name = name
        self.use_hetero = use_hetero
        super().__init__(root=Path(root) / name, transform=transform)
        out = torch.load(self.processed_paths[0])
        if isinstance(out, tuple) and len(out) == 2:
            self.data, self.slices = out
        else:
            self.data = out

    @property
    def raw_dir(self):
        return Path(self.root) / "raw"

    @property
    def processed_dir(self):
        return Path(self.root) / "processed"

    @property
    def raw_file_names(self):
        # Prefer .mat (CARE-GNN release)
        if (self.raw_dir / f"{self.name}.mat").exists():
            return [f"{self.name}.mat"]
        return [f"{self.name}_adj_lists.pickle",
                f"{self.name}_feat.npy",
                f"{self.name}_label.pickle"]

    @property
    def processed_file_names(self):
        return [f"data_hetero{self.use_hetero}.pt"]

    def download(self):
        raise RuntimeError(
            f"\n请手动下载 {self.name} 数据集到 {self.raw_dir}。\n"
            f"来源 1（GitHub）：https://github.com/YingtongDou/CARE-GNN/tree/master/data\n"
            f"来源 2（DGL）：https://data.dgl.ai/dataset/"
            + ("YelpChi.zip" if self.name == "YelpChi" else "Amazon.zip") + "\n"
            "需要以下任一格式：\n"
            f"  - {self.name}.mat  （CARE-GNN release，推荐）\n"
            "或 DGL 风格三个文件：\n"
            f"  - {self.name}_adj_lists.pickle\n"
            f"  - {self.name}_feat.npy\n"
            f"  - {self.name}_label.pickle\n"
        )

    def process(self):
        raw_files = list(self.raw_dir.iterdir())
        mat_files = [f for f in raw_files if f.suffix == ".mat"]

        if mat_files:
            data = self._process_mat(mat_files[0])
        else:
            data = self._process_pickle()

        # Random train/val/test split (imbalanced fraud detection)
        num_nodes = data.num_nodes
        torch.manual_seed(42)
        perm = torch.randperm(num_nodes)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        n_train = int(0.5 * num_nodes)
        n_val = int(0.25 * num_nodes)
        train_mask[perm[:n_train]] = True
        val_mask[perm[n_train:n_train + n_val]] = True
        test_mask[perm[n_train + n_val:]] = True
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        torch.save(self.collate([data]), self.processed_paths[0])

    def _process_mat(self, mat_path):
        mat = scipy.io.loadmat(str(mat_path))
        features = mat["features"]
        if sp.isspmatrix(features):
            features = features.todense().A
        elif isinstance(features, np.matrix):
            features = np.asarray(features)
        x = torch.from_numpy(features).float()

        labels = mat["label"]
        if labels.ndim > 1:
            labels = labels.flatten()
        y = torch.from_numpy(labels).long()
        num_nodes = x.size(0)

        if self.use_hetero:
            edge_index_dict = {}
            for key in HETERO_KEYS[self.name]:
                if key in mat:
                    adj = mat[key]
                    if sp.isspmatrix(adj):
                        adj = adj.tocoo()
                    edge_index, _ = from_scipy_sparse_matrix(adj)
                    # HANConv 期望三元组 edge_type
                    edge_index_dict[("user", key, "user")] = edge_index
            # Default to homo graph as primary edge_index
            homo = mat.get("homo")
            if homo is None:
                # Fallback: union of hetero edges
                all_edges = []
                for ei in edge_index_dict.values():
                    all_edges.append(ei)
                edge_index = torch.cat(all_edges, dim=1)
            else:
                if sp.isspmatrix(homo):
                    homo = homo.tocoo()
                edge_index, _ = from_scipy_sparse_matrix(homo)
            data = Data(x=x, edge_index=edge_index, y=y,
                        edge_index_dict=edge_index_dict)
        else:
            homo = mat["homo"]
            if sp.isspmatrix(homo):
                homo = homo.tocoo()
            edge_index, _ = from_scipy_sparse_matrix(homo)
            data = Data(x=x, edge_index=edge_index, y=y)

        data.num_nodes = num_nodes
        return data

    def _process_pickle(self):
        adj_path = self.raw_dir / f"{self.name}_adj_lists.pickle"
        feat_path = self.raw_dir / f"{self.name}_feat.npy"
        label_path = self.raw_dir / f"{self.name}_label.pickle"

        with open(adj_path, "rb") as f:
            adj_lists = pickle.load(f)

        rows, cols = [], []
        for src, neighbors in adj_lists.items():
            for dst in neighbors:
                rows.append(src)
                cols.append(dst)
        num_nodes = len(adj_lists)
        adj = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(num_nodes, num_nodes))

        features = np.load(feat_path)
        with open(label_path, "rb") as f:
            labels = pickle.load(f)
        labels = np.array(labels)

        edge_index, _ = from_scipy_sparse_matrix(adj)
        x = torch.from_numpy(features).float()
        y = torch.from_numpy(labels).long()

        data = Data(x=x, edge_index=edge_index, y=y)
        data.num_nodes = num_nodes
        return data


if __name__ == "__main__":
    import sys
    root = Path(__file__).resolve().parents[3] / "data" / "raw" / "Fraud"
    try:
        ds = FraudDataset(root=root, name="Amazon")
        data = ds[0]
        print("Loaded Amazon:", data)
        print("nodes", data.num_nodes, "edges", data.num_edges,
              "features", data.num_features, "fraud ratio",
              (data.y == 1).float().mean().item())
    except RuntimeError as e:
        print(e)
        sys.exit(1)
