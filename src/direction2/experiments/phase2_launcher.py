# -*- coding: utf-8 -*-
"""Phase 2 全矩阵训练 launcher（A1+A2+A4，约 240 runs）。

设计要点：
- 启动即应用 70% CPU/内存硬上限（resource_limiter：线程+亲和性+Win Job Object）。
- 断点续跑：结果 CSV 已存在则跳过；崩溃/中断后重跑会从上次进度继续。
- 单组合 try/except 容错：某一个 (数据集×攻击×防御×seed) 失败不影响整体，
  失败信息写入日志，最后汇总。
- 进度 / ETA 实时打印到控制台与 results/direction2/phase2_run.log。

攻击矩阵（A1+A2+A4）：
  主矩阵：{Amazon, YelpChi} × {dice, nettack, camo, prbcd, binarized}
          × {ics, random, kces, graphconsis} × {0..4}
  A4 Elliptic：{dice, nettack, camo} × {ics, random} × {0,1,2}

用法（在用户真实机器运行，沙箱只做 --smoke 验证）：
  python phase2_launcher.py                 # 跑完整 A1+A2+A4 矩阵
  python phase2_launcher.py --dry-run       # 只打印待跑组合数，不执行
  python phase2_launcher.py --smoke         # 小规模冒烟（YelpChi subset=2000）
  python phase2_launcher.py --budget 0.10   # 改攻击预算（默认 0.05）
  python phase2_launcher.py --max-hours 3    # 跑满 3h 后优雅停（下次重跑自动续）
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "common" / "attacks"))
from resource_limiter import apply_limits

# 每个组合用独立子进程跑 run_attack.py：子进程退出后 OS 干净回收内存，
# 避免单进程内多组合累积导致内存泄漏/段错误（沙箱 Job Object 内存上限失效时尤其必要）。
# 单个组合若段错误只挂自己，launcher 记 fail 继续往下跑，断点续跑不受影响。
RUN_ATTACK = Path(__file__).resolve().parent / "run_attack.py"
PY = sys.executable


def run_combo_subprocess(ds, atk, defense, seed, budget, subset, logf):
    """用子进程跑单个 (数据集×攻击×防御×seed) 组合。返回 ('ok'|'fail', detail)。"""
    cmd = [PY, str(RUN_ATTACK), "--dataset", ds, "--attack", atk,
           "--defense", defense, "--seed", str(seed), "--budget", str(budget)]
    if subset and subset > 0:
        cmd += ["--subset", str(subset)]
    try:
        r = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent),
                           capture_output=True, text=True, timeout=7200)
    except subprocess.TimeoutExpired:
        return "fail", "subprocess timeout >7200s"
    if r.returncode == 0:
        for line in r.stdout.splitlines():
            logf.write("    | " + line + "\n")
        return "ok", ""
    detail = (r.stderr or r.stdout or "")[:2000]
    return "fail", f"rc={r.returncode}; {detail}"

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "results" / "direction2"
RES.mkdir(parents=True, exist_ok=True)
LOG = RES / "phase2_run.log"

# Amazon（440 万边）规模下 binarized 当前实现 CPU 内存/时耗不可行，从 Amazon 矩阵剔除
# （与完整 Metattack 同处理）；GAD 专用攻击覆盖由 Amazon camo + YelpChi/Elliptic binarized 提供。
ATTACKS_AMAZON = ["dice", "nettack", "camo", "prbcd"]
ATTACKS_YELPCHI = ["dice", "nettack", "camo", "prbcd", "binarized"]
DEFENSES = ["ics", "random", "kces", "graphconsis"]
SEEDS = [0, 1, 2, 3, 4]
DATASETS = ["Amazon", "YelpChi"]
ELL_ATTACKS = ["dice", "nettack", "camo"]
ELL_DEFENSES = ["ics", "random"]
ELL_SEEDS = [0, 1, 2]


def out_path(ds, atk, defense, seed, subset=0):
    sub_tag = f"_sub{subset}" if subset and subset > 0 else ""
    return RES / f"{ds}_BinaryGAT_{defense}_attack-{atk}{sub_tag}_s{seed}.csv"


def build_matrix():
    combos = []
    per_ds_attacks = {"Amazon": ATTACKS_AMAZON, "YelpChi": ATTACKS_YELPCHI}
    for ds in DATASETS:
        for atk in per_ds_attacks[ds]:
            for defense in DEFENSES:
                for seed in SEEDS:
                    combos.append((ds, atk, defense, seed))
    for atk in ELL_ATTACKS:
        for defense in ELL_DEFENSES:
            for seed in ELL_SEEDS:
                combos.append(("Elliptic", atk, defense, seed))
    return combos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="小规模冒烟（YelpChi subset=2000，4攻击×random×s0）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印待跑组合数，不执行")
    ap.add_argument("--budget", type=float, default=0.05)
    ap.add_argument("--subset", type=int, default=0,
                    help="抽取 N 节点子集（调试用，0=全量）")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="墙钟上限（小时）；到点先跑完当前组合再停，下次重跑自动续。默认不限制")
    args = ap.parse_args()

    apply_limits(0.70, 0.70)

    combos = build_matrix()
    if args.smoke:
        combos = [
            ("YelpChi", "dice", "random", 0),
            ("YelpChi", "camo", "random", 0),
            ("YelpChi", "prbcd", "random", 0),
            ("YelpChi", "binarized", "random", 0),
        ]
        args.subset = 2000

    total = len(combos)
    print(f"[phase2] 计划组合 {total} 个；budget={args.budget}; subset={args.subset}")
    print(f"[phase2] 结果落点: {RES}")
    print(f"[phase2] 日志: {LOG}")

    remain = [c for c in combos if not out_path(*c).exists()]
    print(f"[phase2] 待跑 {len(remain)} / {total}（已存在则跳过）")
    if args.dry_run:
        return

    done = skipped = failed = 0
    t0 = time.time()
    with open(LOG, "a", encoding="utf-8") as logf:
        logf.write(f"\n=== Phase2 启动 {time.strftime('%Y-%m-%d %H:%M:%S')} "
                   f"(combos={total}, budget={args.budget}, subset={args.subset}, "
                   f"max_hours={args.max_hours}) ===\n")
        for i, (ds, atk, defense, seed) in enumerate(combos):
            p = out_path(ds, atk, defense, seed, args.subset)
            if p.exists():
                skipped += 1
                continue
            # 墙钟上限：到点先不启动新组合，优雅收尾（当前组合已在跑的不打断）
            if args.max_hours is not None and (time.time() - t0) / 3600.0 >= args.max_hours:
                print(f"\n[phase2] 已达 --max-hours {args.max_hours}h 上限，"
                      f"已跳过尚未启动的组合，本次收尾。下次重跑自动续跑。")
                logf.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} STOP --max-hours "
                           f"{args.max_hours}h reached; remaining deferred.\n")
                break
            msg = f"[{i + 1}/{total}] {ds} {atk} {defense} s{seed}"
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            print(msg)
            logf.write(f"{ts} START {msg}\n"); logf.flush()
            try:
                status, detail = run_combo_subprocess(
                    ds, atk, defense, seed, args.budget, args.subset, logf)
                logf.flush()
            except Exception as e:
                status, detail = "fail", f"{type(e).__name__}: {e}"
            if status == "ok":
                done += 1
                logf.write(f"{ts} OK    {msg}\n")
            else:
                failed += 1
                logf.write(f"{ts} FAIL  {msg} :: {detail}\n")
                print(f"  !! FAIL {msg}: {detail}")
            logf.flush()
            elapsed = time.time() - t0
            finished = done + failed
            per = elapsed / max(1, finished)
            eta = per * (total - i - 1)
            print(f"  progress done={done} skip={skipped} fail={failed} | "
                  f"已用 {elapsed / 3600:.1f}h ETA {eta / 3600:.1f}h")
    print(f"[phase2] 完成 done={done} skipped={skipped} failed={failed}")
    print(f"[phase2] 详见日志: {LOG}")


if __name__ == "__main__":
    main()
