# -*- coding: utf-8 -*-
"""数据一致性修复（最终轮）：重跑全部 Random 基线与 Elliptic full（均跑在 8/18 train_eval 加 gradient clipping 之前）。

清单：
1. Amazon Random k=20 seeds 0-4
2. YelpChi Random k=20 seeds 0-4
3. Elliptic Random k=10 seeds 0-2
4. Elliptic full k=100 seed 0
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN_CCS = Path(__file__).resolve().parent / "run_ccs.py"
PYTHON = Path(sys.executable)

COMMON = [
    "--model", "BinaryGAT",
    "--balanced_batch",
    "--use_val_threshold",
    "--epochs", "100",
    "--lr", "0.001",
    "--dropout", "0.1",
]

JOBS = (
    [("Amazon", "random", 20, s) for s in range(5)]
    + [("YelpChi", "random", 20, s) for s in range(5)]
    + [("Elliptic", "random", 10, s) for s in range(3)]
    + [("Elliptic", "full", 100, 0)]
)

for dataset, method, topk, seed in JOBS:
    cmd = [str(PYTHON), str(RUN_CCS),
           "--dataset", dataset, "--method", method,
           "--topk", str(topk), "--seed", str(seed)] + COMMON
    print(f"\n>>> {dataset} {method} k={topk} seed={seed}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    for line in result.stdout.splitlines():
        if "Test  acc=" in line:
            print("   ", line.strip(), flush=True)
            break
    if result.returncode != 0:
        print("ERROR:", result.stderr[-300:], file=sys.stderr, flush=True)
print("\nALL DONE")
