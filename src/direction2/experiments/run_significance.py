# -*- coding: utf-8 -*-
"""主表显著性检验：ICS vs Random / ICS vs PC-GNN 的 5-seed AUC paired 检验。"""
import csv
from pathlib import Path
from scipy import stats
import numpy as np

RES = Path(r"D:\workbuddy工作区\SCI论文发表\results\direction2")


def read_aucs(pattern):
    vals = []
    for f in sorted(RES.glob(pattern)):
        with open(f) as fh:
            for row in csv.DictReader(fh):
                vals.append(float(row["test_auc"]))
    return np.array(vals)


def compare(a, b, name_a, name_b, tag):
    t, p_t = stats.ttest_rel(a, b)
    try:
        w, p_w = stats.wilcoxon(a, b)
    except ValueError:
        w, p_w = np.nan, np.nan
    diff = a.mean() - b.mean()
    print(f"{tag}: {name_a} {a.mean():.4f} vs {name_b} {b.mean():.4f} "
          f"(diff {diff:+.4f}) | paired t-test p={p_t:.5f} | wilcoxon p={p_w:.5f}")
    return p_t, p_w


print("=== 主表显著性（5 seeds）===")
for ds in ["YelpChi", "Amazon"]:
    ics = read_aucs(f"{ds}_BinaryGAT_ccs_k20_s[0-4].csv")
    rnd = read_aucs(f"{ds}_BinaryGAT_random_k20_s[0-4].csv")
    pcg = read_aucs(f"{ds}_BinaryGAT_pcgnn_k20_s[0-4]_pr0.5.csv")
    compare(ics, rnd, "ICS", "Random", f"{ds} ICS>Random")
    compare(ics, pcg, "ICS", "PC-GNN", f"{ds} ICS vs PC-GNN")

print("\n=== 消融显著性（YelpChi，5 seeds）===")
yc_ics = read_aucs("YelpChi_BinaryGAT_ccs_k20_s[0-4].csv")
yc_sim = read_aucs("YelpChi_BinaryGAT_ccs_k20_s[0-4]_noStab_noCollab.csv")
yc_sc = read_aucs("YelpChi_BinaryGAT_ccs_k20_s[0-4]_noStab.csv")
if len(yc_sim) == 5 and len(yc_sc) == 5:
    compare(yc_ics, yc_sim, "Full", "sim-only", "YelpChi full vs sim")
    compare(yc_sc, yc_sim, "sim+collab", "sim-only", "YelpChi sim+collab vs sim")
else:
    print(f"missing ablation files: sim={len(yc_sim)}, sim+collab={len(yc_sc)}")

print("\n=== attack 显著性（5 seeds）===")
for ds in ["YelpChi", "Amazon"]:
    for atk in ["dice", "nettack"]:
        ics = read_aucs(f"{ds}_BinaryGAT_ics_attack-{atk}_s[0-4].csv")
        rnd = read_aucs(f"{ds}_BinaryGAT_random_attack-{atk}_s[0-4].csv")
        if len(ics) == 5 and len(rnd) == 5:
            compare(ics, rnd, "ICS", "Random", f"{ds} {atk}")
        else:
            print(f"{ds} {atk}: missing files ics={len(ics)} rnd={len(rnd)}")
