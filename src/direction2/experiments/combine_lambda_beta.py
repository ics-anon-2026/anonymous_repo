# -*- coding: utf-8 -*-
"""合并 λ/β 敏感性 5 seeds（seed 0 + seeds 1-4），输出 mean±std 并更新热力图。"""
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np

RES = Path(r"D:\workbuddy工作区\SCI论文发表\results\direction2")
OUT_DIR = Path(r"D:\workbuddy工作区\SCI论文发表\papers\direction2")

SEED_FILES = [RES / "lambda_beta_sensitivity_Amazon.csv"] + \
             [RES / f"lambda_beta_sensitivity_Amazon_s{s}.csv" for s in [1, 2, 3, 4]]

groups = defaultdict(list)
for f in SEED_FILES:
    if not f.exists():
        print(f"MISSING: {f}")
        continue
    with open(f) as fh:
        for row in csv.DictReader(fh):
            if row.get("test_auc"):
                key = (float(row["lambda"]), float(row["beta"]))
                groups[key].append(float(row["test_auc"]))

# 合并 CSV
out_csv = RES / "lambda_beta_sensitivity_Amazon_5seeds.csv"
with open(out_csv, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["lambda", "beta", "n_seeds", "auc_mean", "auc_std"])
    for (lam, beta) in sorted(groups.keys()):
        vals = groups[(lam, beta)]
        w.writerow([lam, beta, len(vals), f"{np.mean(vals):.4f}", f"{np.std(vals):.4f}"])
print(f"Saved {out_csv}")

# 打印汇总
print("\nlambda\\beta | " + " | ".join(f"b={b}" for b in [0.0, 0.05, 0.1, 0.2]))
for lam in [0.0, 0.5, 1.0, 2.0]:
    row_cells = []
    for beta in [0.0, 0.05, 0.1, 0.2]:
        vals = groups.get((lam, beta), [])
        if vals:
            row_cells.append(f"{np.mean(vals):.3f}±{np.std(vals):.3f}")
        else:
            row_cells.append("N/A")
    print(f"lam={lam} | " + " | ".join(row_cells))

# 更新热力图（5-seed mean）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

lambda_vals = sorted(set(k[0] for k in groups))
beta_vals = sorted(set(k[1] for k in groups))
Z = np.full((len(beta_vals), len(lambda_vals)), np.nan)
for (lam, beta), vals in groups.items():
    Z[beta_vals.index(beta), lambda_vals.index(lam)] = np.mean(vals)

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(Z, cmap="YlGnBu", aspect="auto", origin="lower")
ax.set_xticks(np.arange(len(lambda_vals)))
ax.set_yticks(np.arange(len(beta_vals)))
ax.set_xticklabels([f"{x:.1f}" for x in lambda_vals])
ax.set_yticklabels([f"{x:.2f}" for x in beta_vals])
ax.set_xlabel("Stability variance weight $\\lambda$", fontsize=11)
ax.set_ylabel("Collaborative bonus $\\beta$", fontsize=11)
ax.set_title("Amazon AUC sensitivity to $\\lambda$ and $\\beta$\n(BinaryGAT, $k=20$, 5 seeds)",
             fontsize=12, fontweight="bold")
for i in range(len(beta_vals)):
    for j in range(len(lambda_vals)):
        if not np.isnan(Z[i, j]):
            ax.text(j, i, f"{Z[i, j]:.3f}", ha="center", va="center", color="black", fontsize=9)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Test AUC (5-seed mean)", rotation=270, labelpad=15)
plt.tight_layout()
plt.savefig(OUT_DIR / "lambda_beta_heatmap.pdf", format="pdf", bbox_inches="tight", dpi=300)
print("Saved lambda_beta_heatmap.pdf (5-seed mean)")
