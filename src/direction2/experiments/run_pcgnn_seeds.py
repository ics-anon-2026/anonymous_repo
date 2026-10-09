# -*- coding: utf-8 -*-
"""补跑 Amazon BinaryGAT PC-GNN seeds 0-4（与 YelpChi 的 5-seed 对齐）。"""
import subprocess
import sys
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[3]
RUN_CCS = Path(__file__).resolve().parent / "run_ccs.py"
PYTHON = Path(r"D:\Miniconda3\python.exe")

out_file = ROOT / "results" / "direction2" / "pcgnn_amazon_seeds0_4.csv"
with open(out_file, "w", newline="") as fh:
    csv.writer(fh).writerow(["dataset", "model", "method", "seed", "test_acc", "test_f1", "test_auc"])

for seed in range(5):
    cmd = [
        str(PYTHON), str(RUN_CCS),
        "--dataset", "Amazon",
        "--model", "BinaryGAT",
        "--method", "pcgnn",
        "--topk", "20",
        "--seed", str(seed),
        "--pcgnn_pos_ratio", "0.5",
        "--balanced_batch",
        "--use_val_threshold",
        "--epochs", "100",
        "--lr", "0.001",
        "--dropout", "0.1",
    ]
    print(f"\n>>> PC-GNN Amazon seed={seed}")
    row = [seed, None, None, None]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        tail = result.stdout[-600:] if len(result.stdout) > 600 else result.stdout
        print(tail)
        if result.returncode != 0:
            print("ERROR:", result.stderr[-500:], file=sys.stderr)
        else:
            for line in result.stdout.splitlines():
                if "Test  acc=" in line:
                    parts = line.strip().split()
                    acc = float(parts[1].split("=")[1])
                    f1 = float(parts[2].split("=")[1])
                    auc = float(parts[3].split("=")[1])
                    row = [seed, acc, f1, auc]
                    print(f"    AUC={auc:.4f}")
                    break
    except Exception as e:
        print(f"Exception seed={seed}: {e}", file=sys.stderr)
    with open(out_file, "a", newline="") as fh:
        csv.writer(fh).writerow(["Amazon", "BinaryGAT", "pcgnn"] + row)
print("\nDONE")
