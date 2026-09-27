# -*- coding: utf-8 -*-
"""方向二：OOD（分布外）数据划分工具。

用于验证图稀疏化方法在不同特征分布下的泛化能力。
实现：对节点特征做 K-means 聚类得到若干"域"（domain），
      训练集使用部分域，验证集/测试集使用其他域。
"""

import numpy as np
import torch
from sklearn.cluster import KMeans


def domain_split(data, n_domains=3, train_domains=None, val_domain=None, test_domain=None, seed=42):
    """按特征分布将节点划分为多个域，并生成 train/val/test mask。

    Args:
        data: PyG Data，包含 x。
        n_domains: 域数量。
        train_domains: list[int]，训练域索引。默认 [0,1]。
        val_domain: int，验证域索引。默认 2%n_domains。
        test_domain: int，测试域索引。默认 (n_domains-1)。
        seed: K-means 随机种子。

    Returns:
        data: 更新后的 Data（train_mask, val_mask, test_mask）。
        domain_labels: 每个节点的域标签 Tensor。
    """
    x = data.x.detach().cpu().numpy()
    # 标准化特征以稳定聚类
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-8
    x_norm = (x - mean) / std

    kmeans = KMeans(n_clusters=n_domains, random_state=seed, n_init=10)
    domain_labels = kmeans.fit_predict(x_norm)
    domain_labels = torch.from_numpy(domain_labels).long()

    if train_domains is None:
        train_domains = list(range(n_domains - 1))
    if val_domain is None:
        val_domain = (n_domains - 1) % n_domains
    if test_domain is None:
        test_domain = n_domains - 1

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    for d in train_domains:
        train_mask[domain_labels == d] = True
    val_mask[domain_labels == val_domain] = True
    test_mask[domain_labels == test_domain] = True

    # 确保每个集合都有正负样本
    y = data.y.cpu()
    for mask, name in [(train_mask, "train"), (val_mask, "val"), (test_mask, "test")]:
        pos = (y[mask] == 1).sum().item()
        neg = (y[mask] == 0).sum().item()
        print(f"OOD {name}: {mask.sum().item()} nodes, pos={pos}, neg={neg}")

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data, domain_labels


def temporal_split(data, time_attr="time", train_ratio=0.5, val_ratio=0.25):
    """按时间属性划分 train/val/test（若数据集有时间特征）。"""
    if not hasattr(data, time_attr):
        raise ValueError(f"Data does not have time attribute {time_attr}")
    t = getattr(data, time_attr).cpu().numpy()
    order = np.argsort(t)
    n = len(order)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    train_mask[order[:n_train]] = True
    val_mask[order[n_train:n_train + n_val]] = True
    test_mask[order[n_train + n_val:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data
