# -*- coding: utf-8 -*-
"""Collect multi-seed experiment results and compute mean ± std."""
import os
import glob
import numpy as np
from collections import defaultdict

results_dir = (Path(__file__).resolve().parents[3] / "results" / "direction2")

rows = []
for f in glob.glob(os.path.join(results_dir, "*.csv")):
    with open(f) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("dataset"):
                continue
            rows.append(line.split(","))

groups = defaultdict(list)
for r in rows:
    if len(r) < 10:
        continue
    key = (r[0], r[1], r[2], r[3], r[4])  # dataset, model, method, topk, n_envs
    try:
        groups[key].append({
            'acc': float(r[6]),
            'f1': float(r[7]),
            'auc': float(r[8]),
            'prep': float(r[9]),
            'train': float(r[10]),
        })
    except (ValueError, IndexError):
        continue


def fmt(mean, std):
    return f"{mean:.4f}±{std:.4f}"


print(f"{'Dataset':<12} {'Model':<10} {'Method':<12} {'TopK':<6} {'N':>2} "
      f"{'Acc':>14} {'F1':>14} {'AUC':>14} {'Train(s)':>10}")
print("-" * 95)
for key in sorted(groups.keys()):
    vals = groups[key]
    if len(vals) < 2:
        continue
    accs = [v['acc'] for v in vals]
    f1s = [v['f1'] for v in vals]
    aucs = [v['auc'] for v in vals]
    trains = [v['train'] for v in vals]
    print(f"{key[0]:<12} {key[1]:<10} {key[2]:<12} {key[3]:<6} {len(vals):>2} "
          f"{fmt(np.mean(accs), np.std(accs)):>14} "
          f"{fmt(np.mean(f1s), np.std(f1s)):>14} "
          f"{fmt(np.mean(aucs), np.std(aucs)):>14} "
          f"{np.mean(trains):>10.1f}")

print("\n--- Single-seed results ---")
for key in sorted(groups.keys()):
    vals = groups[key]
    if len(vals) == 1:
        v = vals[0]
        print(f"{key[0]:<12} {key[1]:<10} {key[2]:<12} {key[3]:<6} "
              f"{v['acc']:7.4f} {v['f1']:7.4f} {v['auc']:7.4f}")
