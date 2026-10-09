# -*- coding: utf-8 -*-
"""OOD-v2 实验：与 ICS 环境划分解耦的域外泛化评估。

背景（审稿意见 C2）：旧 OOD 协议 domain_split(M=3, seed=训练seed) 与 ICS 内部环境
划分（KMeans M=3, random_state=42）同源于同一聚类机制，测试域可能是 ICS 自己的
环境簇，构成循环论证。

解耦设计：OOD 切分改为 KMeans M=5, seed=2026（固定，与训练 seed 无关，且参数与
ICS 的 M=3/seed=42 完全不同源）。ICS 内部仍用 M=3/seed=42 做环境划分。

实验矩阵（输出文件名带 _ood5 后缀，与旧 _ood3 区分，不互相覆盖）：
1. 主实验：{ICS, Random} x {Amazon, YelpChi} x seeds 0-2 = 12 runs
2. OOD 消融：{full, sim-only, sim+stab, sim+collab} x 2 数据集 x seeds 0-2 = 24 runs
   ——检验 stab/collab 组件在 OOD 下是否有显著贡献（回应审稿意见 C1）。
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
    "--ood_split",
    "--ood_domains", "5",
    "--ood_seed", "2026",
]

ABLATIONS = {
    "sim_only": ["--no_stab", "--no_collab"],
    "sim_stab": ["--no_collab"],
    "sim_collab": ["--no_stab"],
}


def run(dataset, seed, method, extra, tag):
    cmd = [str(PYTHON), str(RUN_CCS),
           "--dataset", dataset, "--method", method,
           "--topk", "20", "--n_envs", "3",
           "--seed", str(seed)] + COMMON + extra
    print(f"\n>>> {dataset} seed={seed} method={method} [{tag}]", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    for line in result.stdout.splitlines():
        if "Test  acc=" in line:
            print("   ", line.strip(), flush=True)
            break
        if "OOD" in line:
            print("   ", line.strip(), flush=True)
    if result.returncode != 0:
        print("ERROR:", result.stderr[-300:], file=sys.stderr, flush=True)


# 1. 主实验
for dataset in ["Amazon", "YelpChi"]:
    for seed in [0, 1, 2]:
        run(dataset, seed, "ccs", [], "MAIN-ICS")
        run(dataset, seed, "random", [], "MAIN-Random")

# 2. OOD 消融
for dataset in ["Amazon", "YelpChi"]:
    for tag, extra in ABLATIONS.items():
        for seed in [0, 1, 2]:
            run(dataset, seed, "ccs", extra, f"ABL-{tag}")

print("\nALL OOD-V2 DONE")
