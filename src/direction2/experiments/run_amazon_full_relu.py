# -*- coding: utf-8 -*-
"""Amazon full ICS (ReLU stability) 5 seeds."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "direction2" / "experiments"))
sys.path.insert(0, str(ROOT / "src" / "direction2" / "data"))
sys.path.insert(0, str(ROOT / "src" / "direction2" / "models"))
sys.path.insert(0, str(ROOT / "src" / "direction2" / "utils"))

import run_ccs

COMMON = [
    "--dataset", "Amazon", "--model", "BinaryGAT", "--topk", "20",
    "--epochs", "100", "--device", "cpu", "--use_val_threshold",
    "--lr", "0.001", "--dropout", "0.1", "--balanced_batch",
    "--pos_weight", "1.0", "--n_envs", "3", "--collab_bonus", "0.1",
    "--env_mode", "kmeans",
]

for seed in [1, 2, 3, 4]:
    print(f"\n{'='*60}\n=== Amazon full ReLU seed={seed} ===\n{'='*60}", flush=True)
    run_ccs.main(COMMON + ["--method", "ccs", "--seed", str(seed)])

print("\n=== Amazon full ReLU done ===", flush=True)
