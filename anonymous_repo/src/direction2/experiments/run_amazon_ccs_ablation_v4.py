# -*- coding: utf-8 -*-
"""Amazon CCS ablation driver: in-process sequential execution."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "direction2" / "experiments"))
sys.path.insert(0, str(ROOT / "src" / "direction2" / "data"))
sys.path.insert(0, str(ROOT / "src" / "direction2" / "models"))
sys.path.insert(0, str(ROOT / "src" / "direction2" / "utils"))

import run_ccs

BASE = [
    "--dataset", "Amazon", "--model", "BinaryGAT", "--topk", "20",
    "--epochs", "100", "--device", "cpu", "--use_val_threshold",
    "--lr", "0.001", "--dropout", "0.1", "--balanced_batch",
    "--pos_weight", "1.0", "--n_envs", "3", "--collab_bonus", "0.1",
    "--env_mode", "kmeans",
]

variants = [
    ("full", []),
    ("no_stab", ["--no_stab"]),
    ("no_collab", ["--no_collab"]),
    ("only_sim", ["--no_stab", "--no_collab"]),
]

for seed in [0, 1, 2, 3, 4]:
    for name, flags in variants:
        print(f"\n{'='*60}\n=== Amazon {name} seed={seed} ===\n{'='*60}", flush=True)
        run_ccs.main(BASE + ["--seed", str(seed)] + flags)

print("\n=== Amazon CCS ablation done ===", flush=True)
