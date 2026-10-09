# -*- coding: utf-8 -*-
"""数据一致性修复：用当前代码（8/19 后 ccs.py）重跑全部受版本混合影响的主实验与消融。

清单：
1. YelpChi ICS 主实验 seeds 0,2,3,4（s1 已重跑=0.6858）
2. Amazon 消融 3 配置 × seeds 0-4（full=主实验已重跑）
3. YelpChi 消融 3 配置 × seeds 0-4
注意：随机稀疏化不经过 ccs.py，Random 基线不受影响；OOD/attack/PC-GNN/λβ 均为当前代码结果。
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN_CCS = Path(__file__).resolve().parent / "run_ccs.py"
PYTHON = Path(r"D:\Miniconda3\python.exe")

COMMON = [
    "--model", "BinaryGAT",
    "--method", "ccs",
    "--topk", "20",
    "--n_envs", "3",
    "--balanced_batch",
    "--use_val_threshold",
    "--epochs", "100",
    "--lr", "0.001",
    "--dropout", "0.1",
]

ABLATIONS = {
    "noStab_noCollab": ["--no_stab", "--no_collab"],
    "noStab": ["--no_stab"],
    "noCollab": ["--no_collab"],
}


def run(dataset, seed, extra, tag):
    cmd = [str(PYTHON), str(RUN_CCS), "--dataset", dataset, "--seed", str(seed)] + COMMON + extra
    print(f"\n>>> {dataset} seed={seed} {tag}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    for line in result.stdout.splitlines():
        if "Test  acc=" in line:
            print("   ", line.strip(), flush=True)
            break
    if result.returncode != 0:
        print("ERROR:", result.stderr[-300:], file=sys.stderr, flush=True)


# 1. YelpChi 主实验缺失 seeds（s1 已重跑）
for seed in [0, 2, 3, 4]:
    run("YelpChi", seed, [], "MAIN")

# 2+3. 两数据集消融
for dataset in ["Amazon", "YelpChi"]:
    for tag, extra in ABLATIONS.items():
        for seed in range(5):
            run(dataset, seed, extra, tag)

print("\nALL RERUN DONE")
