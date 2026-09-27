# -*- coding: utf-8 -*-
"""Batch runner for direction2 experiments.

Runs a grid of (dataset, model, method, topk, seed) experiments and saves
individual CSV files. Use collect_stats.py to aggregate results.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
RUNNER = ROOT / "src" / "direction2" / "experiments" / "run_ccs.py"


def run_one(dataset, model, method, topk, seed, extra_args=None):
    cmd = [
        str(PYTHON), str(RUNNER),
        "--dataset", dataset,
        "--model", model,
        "--method", method,
        "--topk", str(topk),
        "--seed", str(seed),
        "--epochs", "40",
        "--lr", "5e-4",
        "--device", "cpu",
    ]
    if extra_args:
        cmd.extend(extra_args)
    print(f"[RUN] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, timeout=1800)
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {dataset} {model} {method} k={topk} s={seed}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {dataset} {model} {method} k={topk} s={seed}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["YelpChi"])
    parser.add_argument("--models", nargs="+", default=["GAT"])
    parser.add_argument("--methods", nargs="+", default=["full", "random", "ccs", "dropedge"])
    parser.add_argument("--topks", nargs="+", type=int, default=[20])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--ccs_extra", nargs="*", default=["--collab_bonus", "0.1"])
    args = parser.parse_args()

    for dataset in args.datasets:
        for model in args.models:
            for method in args.methods:
                for topk in args.topks:
                    for seed in args.seeds:
                        extra = []
                        if method == "ccs":
                            extra = args.ccs_extra
                        run_one(dataset, model, method, topk, seed, extra)


if __name__ == "__main__":
    main()
