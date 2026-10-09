# -*- coding: utf-8 -*-
"""补跑 λ/β 敏感性 seeds 1-4（seed 0 已有，凑齐 5 seeds 与主表对齐）。

每个 seed 的结果写入独立 CSV：lambda_beta_sensitivity_Amazon_s<seed>.csv
跑完后用 combine_lambda_beta.py 合并 5 个 seed 计算 mean±std。
"""
import subprocess
import sys
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[3]
RUN_CCS = Path(__file__).resolve().parent / "run_ccs.py"
PYTHON = Path(r"D:\Miniconda3\python.exe")

lambdas = [0.0, 0.5, 1.0, 2.0]
betas = [0.0, 0.05, 0.1, 0.2]
SEEDS = [1, 2, 3, 4]

for seed in SEEDS:
    out_file = ROOT / "results" / "direction2" / f"lambda_beta_sensitivity_Amazon_s{seed}.csv"
    with open(out_file, "w", newline="") as fh:
        csv.writer(fh).writerow(["lambda", "beta", "test_acc", "test_f1", "test_auc"])

    for lam in lambdas:
        for beta in betas:
            cmd = [
                str(PYTHON), str(RUN_CCS),
                "--dataset", "Amazon",
                "--model", "BinaryGAT",
                "--method", "ccs",
                "--topk", "20",
                "--n_envs", "3",
                "--seed", str(seed),
                "--env_stability_weight", str(lam),
                "--collab_bonus", str(beta),
                "--balanced_batch",
                "--use_val_threshold",
                "--epochs", "100",
                "--lr", "0.001",
                "--dropout", "0.1",
            ]
            print(f"\n>>> seed={seed} lambda={lam} beta={beta}")
            row = [lam, beta, None, None, None]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    print("ERROR:", result.stderr[-400:], file=sys.stderr)
                else:
                    for line in result.stdout.splitlines():
                        if "Test  acc=" in line:
                            parts = line.strip().split()
                            acc = float(parts[1].split("=")[1])
                            f1 = float(parts[2].split("=")[1])
                            auc = float(parts[3].split("=")[1])
                            row = [lam, beta, acc, f1, auc]
                            print(f"    AUC={auc:.4f}")
                            break
            except Exception as e:
                print(f"Exception seed={seed} lam={lam} beta={beta}: {e}", file=sys.stderr)
            with open(out_file, "a", newline="") as fh:
                csv.writer(fh).writerow(row)
print("\nALL SEEDS DONE")
