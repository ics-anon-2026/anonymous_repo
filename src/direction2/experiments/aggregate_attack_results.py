"""聚合 Phase 2 攻击矩阵结果 -> 论文用 AUC 均值表。

读取 results/direction2/*_BinaryGAT_*_attack-*.csv，按 (dataset, attack, defense)
跨 seed 取 test_auc / test_acc / test_f1 的均值±标准差，输出 Markdown 表格。
用法: python aggregate_attack_results.py [--out summary.md]
"""
import argparse
import csv
import glob
import os
import re
import statistics
from pathlib import Path

RES = Path(__file__).resolve().parents[3] / "results" / "direction2"

FNAME = re.compile(r"^(?P<ds>\w+)_BinaryGAT_(?P<def>\w+)_attack-(?P<atk>\w+)_s(?P<seed>\d+)\.csv$")


def load():
    rows = {}  # (ds, atk, defense) -> {metric: [vals]}
    for fp in sorted(glob.glob(str(RES / "*_BinaryGAT_*_attack-*.csv"))):
        m = FNAME.match(os.path.basename(fp))
        if not m:
            continue
        ds, defense, atk = m.group("ds"), m.group("def"), m.group("atk")
        with open(fp, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for line in r:
                if not line.get("test_auc"):
                    continue
                key = (ds, atk, defense)
                rows.setdefault(key, {"auc": [], "acc": [], "f1": []})
                rows[key]["auc"].append(float(line["test_auc"]))
                rows[key]["acc"].append(float(line["test_acc"]))
                rows[key]["f1"].append(float(line["test_f1"]))
    return rows


def fmt(vals):
    if not vals:
        return "  -  "
    mean = statistics.mean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return f"{mean:.4f}±{std:.4f}"


def render(rows):
    out = []
    # 每个数据集一张表
    datasets = sorted({k[0] for k in rows})
    attacks = ["dice", "nettack", "camo", "prbcd", "binarized"]
    defenses = ["ics", "random", "kces", "graphconsis"]
    for ds in datasets:
        out.append(f"\n## {ds}\n")
        out.append("| 攻击 \\ 防御 | " + " | ".join(defenses) + " |")
        out.append("|" + "---|" * (len(defenses) + 1))
        # 该数据集实际出现的攻击
        ds_atks = [a for a in attacks if any(k[0] == ds and k[1] == a for k in rows)]
        for atk in ds_atks:
            cells = []
            for d in defenses:
                key = (ds, atk, d)
                cells.append(fmt(rows.get(key, {}).get("auc", [])))
            out.append(f"| {atk} | " + " | ".join(cells) + " |")
        out.append("")
        out.append(f"_表：{ds} 各攻击×防御的 test_auc 均值±总体标准差（跨 seed；"
                   f"数字越高=检测越好=攻击越失效，ICS 为本文方法应最高）。_\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES / "attack_matrix_summary.md"))
    args = ap.parse_args()
    rows = load()
    text = ("# Phase 2 攻击矩阵 AUC 汇总\n\n"
            f"_共 {len(rows)} 个 (dataset, attack, defense) 组合；"
            "AUC 越高=欺诈检测越好=攻击越失效。_\n") + render(rows)
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"已写出: {args.out}")
    print(f"组合总数: {len(rows)}")
    # 控制台也打印关键对照：ICS vs 各防御 在 prbcd/camo 下的均值
    print("\n=== ICS 方法对照（跨数据集×攻击，各防御 mean AUC）===")
    by_def = {}
    for (ds, atk, d), v in rows.items():
        by_def.setdefault(d, []).extend(v["auc"])
    for d in ["ics", "random", "kces", "graphconsis"]:
        vals = by_def.get(d, [])
        if vals:
            print(f"  {d:12s}: {statistics.mean(vals):.4f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
