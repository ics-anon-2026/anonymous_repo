# -*- coding: utf-8 -*-
"""用当前代码重跑 Elliptic BinaryGAT ICS k=10 seeds 0-2（数据一致性修复）。"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN_CCS = Path(__file__).resolve().parent / "run_ccs.py"
PYTHON = Path(r"D:\Miniconda3\python.exe")

for seed in [0, 1, 2]:
    cmd = [
        str(PYTHON), str(RUN_CCS),
        "--dataset", "Elliptic",
        "--model", "BinaryGAT",
        "--method", "ccs",
        "--topk", "10",
        "--n_envs", "3",
        "--seed", str(seed),
        "--balanced_batch",
        "--use_val_threshold",
        "--epochs", "100",
        "--lr", "0.001",
        "--dropout", "0.1",
    ]
    print(f"\n>>> Elliptic seed={seed}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    for line in result.stdout.splitlines():
        if "Test  acc=" in line:
            print("   ", line.strip(), flush=True)
            break
    if result.returncode != 0:
        print("ERROR:", result.stderr[-300:], file=sys.stderr, flush=True)
print("\nDONE")
