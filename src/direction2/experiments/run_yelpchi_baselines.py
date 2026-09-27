# -*- coding: utf-8 -*-
"""YelpChi random baseline (stable config) + PC-GNN baseline."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "direction2" / "experiments"))
sys.path.insert(0, str(ROOT / "src" / "direction2" / "data"))
sys.path.insert(0, str(ROOT / "src" / "direction2" / "models"))
sys.path.insert(0, str(ROOT / "src" / "direction2" / "utils"))

import run_ccs

COMMON = [
    "--dataset", "YelpChi", "--model", "BinaryGAT", "--topk", "20",
    "--epochs", "100", "--device", "cpu", "--use_val_threshold",
    "--lr", "0.001", "--dropout", "0.1", "--balanced_batch",
    "--pos_weight", "1.0",
]

for seed in [0, 1, 2, 3, 4]:
    print(f"\n{'='*60}\n=== YelpChi random seed={seed} ===\n{'='*60}", flush=True)
    run_ccs.main(COMMON + ["--method", "random", "--seed", str(seed)])

for seed in [0, 1, 2, 3, 4]:
    print(f"\n{'='*60}\n=== YelpChi pcgnn seed={seed} ===\n{'='*60}", flush=True)
    run_ccs.main(COMMON + ["--method", "pcgnn", "--seed", str(seed)])

print("\n=== YelpChi baselines done ===", flush=True)
