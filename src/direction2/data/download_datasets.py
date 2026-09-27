# -*- coding: utf-8 -*-
"""下载方向二所需欺诈检测数据集"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
os.environ["PYG_DATASET_DIR"] = str(RAW_DIR)

from torch_geometric.datasets import FraudDataset


def download_fraud_yelp():
    dataset = FraudDataset(root=RAW_DIR / "Fraud", name="Yelp")
    data = dataset[0]
    print(f"YelpChi: {data.num_nodes} nodes, {data.num_edges} edges, {data.num_features} features")
    return dataset


def download_fraud_amazon():
    dataset = FraudDataset(root=RAW_DIR / "Fraud", name="Amazon")
    data = dataset[0]
    print(f"Amazon: {data.num_nodes} nodes, {data.num_edges} edges, {data.num_features} features")
    return dataset


if __name__ == "__main__":
    print("Downloading Direction-2 fraud datasets ...")
    download_fraud_yelp()
    download_fraud_amazon()
    print("All direction-2 datasets ready.")
