# -*- coding: utf-8 -*-
"""补跑攻击实验 seeds 3-4（与主表 5-seed 对齐）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_attack import run_one

for ds_name in ["Amazon", "YelpChi"]:
    for attack_name in ["dice", "nettack"]:
        for defense in ["random", "ics"]:
            for seed in [3, 4]:
                run_one(ds_name, attack_name, defense, seed, budget=0.05)
print("\nALL SEEDS 3-4 DONE")
