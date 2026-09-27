# -*- coding: utf-8 -*-
"""OOD 泛化实验：Random vs ICS under domain-based OOD split."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "direction2" / "experiments"))
sys.path.insert(0, str(ROOT / "src" / "direction2" / "data"))
sys.path.insert(0, str(ROOT / "src" / "direction2" / "models"))
sys.path.insert(0, str(ROOT / "src" / "direction2" / "utils"))

import run_ccs

COMMON = [
    "--epochs", "100", "--device", "cpu", "--use_val_threshold",
    "--lr", "0.001", "--dropout", "0.1", "--balanced_batch",
    "--pos_weight", "1.0", "--n_envs", "3", "--collab_bonus", "0.1",
    "--env_mode", "kmeans", "--ood_split", "--ood_domains", "3",
]

# Amazon OOD
for seed in [0, 1, 2]:
    print(f"\n{'='*60}\n=== Amazon OOD random seed={seed} ===\n{'='*60}", flush=True)
    run_ccs.main(COMMON + ["--dataset", "Amazon", "--model", "BinaryGAT",
                           "--topk", "20", "--method", "random", "--seed", str(seed)])
    print(f"\n{'='*60}\n=== Amazon OOD ICS seed={seed} ===\n{'='*60}", flush=True)
    run_ccs.main(COMMON + ["--dataset", "Amazon", "--model", "BinaryGAT",
                           "--topk", "20", "--method", "ccs", "--seed", str(seed)])

# YelpChi OOD
for seed in [0, 1, 2]:
    print(f"\n{'='*60}\n=== YelpChi OOD random seed={seed} ===\n{'='*60}", flush=True)
    run_ccs.main(COMMON + ["--dataset", "YelpChi", "--model", "BinaryGAT",
                           "--topk", "20", "--method", "random", "--seed", str(seed)])
    print(f"\n{'='*60}\n=== YelpChi OOD ICS seed={seed} ===\n{'='*60}", flush=True)
    run_ccs.main(COMMON + ["--dataset", "YelpChi", "--model", "BinaryGAT",
                           "--topk", "20", "--method", "ccs", "--seed", str(seed)])

print("\n=== OOD experiments done ===", flush=True)
