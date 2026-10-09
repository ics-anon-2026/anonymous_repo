# -*- coding: utf-8 -*-
"""用当前代码重跑 Amazon BinaryGAT ICS k=20 seeds 2-4（主表数据一致性修复）。"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN_CCS = Path(__file__).resolve().parent / "run_ccs.py"
PYTHON = Path(r"D:\Miniconda3\python.exe")

for seed in [2, 3, 4]:
    cmd = [
        str(PYTHON), str(RUN_CCS),
        "--dataset", "Amazon",
        "--model", "BinaryGAT",
        "--method", "ccs",
        "--topk", "20",
        "--n_envs", "3",
        "--seed", str(seed),
        "--balanced_batch",
        "--use_val_threshold",
        "--epochs", "100",
        "--lr", "0.001",
        "--dropout", "0.1",
    ]
    print(f"\n>>> seed={seed}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    for line in result.stdout.splitlines():
        if "Test  acc=" in line:
            print(line.strip())
    if result.returncode != 0:
        print("ERROR:", result.stderr[-400:], file=sys.stderr)
print("\nDONE")
