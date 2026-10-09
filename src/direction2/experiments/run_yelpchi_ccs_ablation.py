# -*- coding: utf-8 -*-
"""Sequential YelpChi CCS ablation to avoid background task issues."""
import subprocess
import sys
from pathlib import Path

PY = "d:/Miniconda3/python.exe"
RUN = "src/direction2/experiments/run_ccs.py"
COMMON = [
    "--dataset", "YelpChi", "--model", "BinaryGAT", "--topk", "20",
    "--epochs", "100", "--device", "cpu", "--use_val_threshold",
    "--lr", "0.001", "--dropout", "0.1", "--balanced_batch",
    "--pos_weight", "1.0", "--n_envs", "3", "--collab_bonus", "0.1",
    "--env_mode", "kmeans"
]

variants = [
    ("full", []),
    ("no_stab", ["--no_stab"]),
    ("no_collab", ["--no_collab"]),
    ("only_sim", ["--no_stab", "--no_collab"]),
]

for seed in [0, 1, 2, 3, 4]:
    for name, flags in variants:
        print(f"\n=== YelpChi {name} seed={seed} ===")
        cmd = [PY, RUN] + COMMON + ["--seed", str(seed)] + flags
        result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[3])
        if result.returncode != 0:
            print(f"FAILED: YelpChi {name} seed={seed}", file=sys.stderr)
            sys.exit(1)

print("\n=== YelpChi CCS ablation done ===")
