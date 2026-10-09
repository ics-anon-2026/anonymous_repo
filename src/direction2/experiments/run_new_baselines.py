# -*- coding: utf-8 -*-
"""新基线（回应审稿 C3/M2）：
1. IID 主表：GraphConsis-lite / KCES-lite x {Amazon, YelpChi} x seeds 0-4 = 20 runs
2. attack 对比：复用 run_attack.py（--defense kces / graphconsis），2 攻击 x 2 数据集 x seeds 0-4
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))

from fraud_dataset import FraudDataset
from gnn import BinaryGAT
from train_eval import train_epoch, evaluate_with_val_threshold
from kces import kces_sanitize, graphconsis_sanitize

ROOT = Path(__file__).resolve().parents[3]


def roc_auc_safe(y_true, y_prob):
    from sklearn.metrics import roc_auc_score
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.5
    return roc_auc_score(y_true, y_prob)


def train_victim(model, data, epochs=100, lr=1e-3, device="cpu", patience=20):
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    fraud_ratio = (data.y == 1).float().mean().item()
    pos_weight = torch.tensor(1.0 / fraud_ratio, device=device)
    best_val, best_state, patience_counter = -1.0, None, 0
    for _ in range(epochs):
        train_epoch(model, data, optimizer, device, pos_weight,
                    use_focal=False, balanced_batch=True)
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_idx = data.val_mask.nonzero(as_tuple=True)[0]
            val_auc = roc_auc_safe(data.y[val_idx].cpu(), torch.sigmoid(out[val_idx]).cpu())
        if val_auc > best_val:
            best_val, best_state, patience_counter = val_auc, \
                {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run_iid(dataset_name, defense, seed, topk=20, device="cpu"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = FraudDataset(root=ROOT / "data" / "raw" / "Fraud", name=dataset_name,
                      use_hetero=False)
    data = ds[0]
    t0 = time.time()
    if defense == "kces":
        data.edge_index = kces_sanitize(data.edge_index, data.x, data.num_nodes, topk)
    else:
        data.edge_index = graphconsis_sanitize(data.edge_index, data.x, data.num_nodes, topk)
    t_prep = time.time() - t0
    model = BinaryGAT(data.num_features, 64, heads=8, dropout=0.1)
    t0 = time.time()
    model = train_victim(model, data, device=device)
    t_train = time.time() - t0
    (acc, f1, auc), thr = evaluate_with_val_threshold(model, data, data.test_mask, device)
    print(f">>> IID {dataset_name} {defense} seed={seed}: "
          f"acc={acc:.4f} f1={f1:.4f} auc={auc:.4f} (prep {t_prep:.1f}s)", flush=True)
    out = ROOT / "results" / "direction2" / \
        f"{dataset_name}_BinaryGAT_{defense}_k{topk}_s{seed}.csv"
    with open(out, "w") as fh:
        fh.write("dataset,model,method,topk,seed,test_acc,test_f1,test_auc,prep_time,train_time\n")
        fh.write(f"{dataset_name},BinaryGAT,{defense},{topk},{seed},"
                 f"{acc:.4f},{f1:.4f},{auc:.4f},{t_prep:.2f},{t_train:.2f}\n")


def run_attack_matrix(seeds):
    atk = Path(__file__).resolve().parent / "run_attack.py"
    for ds_name in ["Amazon", "YelpChi"]:
        for attack in ["dice", "nettack"]:
            for defense in ["kces", "graphconsis"]:
                for seed in seeds:
                    cmd = [sys.executable, str(atk), "--dataset", ds_name,
                           "--attack", attack, "--defense", defense,
                           "--seed", str(seed)]
                    print(f">>> ATTACK {ds_name} {attack} {defense} s{seed}", flush=True)
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                    for line in r.stdout.splitlines():
                        if "Test acc=" in line:
                            print("   ", line.strip(), flush=True)
                    if r.returncode != 0:
                        print("ERROR:", r.stderr[-300:], file=sys.stderr, flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "iid"
    if which == "iid":
        for dataset in ["Amazon", "YelpChi"]:
            for defense in ["graphconsis", "kces"]:
                for seed in range(5):
                    run_iid(dataset, defense, seed)
    elif which == "attack":
        run_attack_matrix(range(5))
    print("DONE")
