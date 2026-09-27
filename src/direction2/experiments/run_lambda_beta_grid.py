# -*- coding: utf-8 -*-
"""Run lambda/beta sensitivity grid for ICS on Amazon."""
import subprocess
import sys
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[3]
RUN_CCS = Path(__file__).resolve().parent / "run_ccs.py"
# Use the Python interpreter that runs this script (requires torch + PyG installed)
PYTHON = Path(sys.executable)

lambdas = [0.0, 0.5, 1.0, 2.0]
betas = [0.0, 0.05, 0.1, 0.2]

out_file = ROOT / "results" / "direction2" / "lambda_beta_sensitivity_Amazon.csv"
# Write header
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
            "--seed", "0",
            "--env_stability_weight", str(lam),
            "--collab_bonus", str(beta),
            "--balanced_batch",
            "--use_val_threshold",
            "--epochs", "100",
            "--lr", "0.001",
            "--dropout", "0.1",
        ]
        print(f"\n>>> Running lambda={lam}, beta={beta}")
        row = [lam, beta, None, None, None]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            print(result.stdout[-800:] if len(result.stdout) > 800 else result.stdout)
            if result.returncode != 0:
                print("ERROR:", result.stderr[-500:], file=sys.stderr)
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
            print(f"Exception for lambda={lam}, beta={beta}: {e}", file=sys.stderr)
        with open(out_file, "a", newline="") as fh:
            csv.writer(fh).writerow(row)
