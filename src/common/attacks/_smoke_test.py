# -*- coding: utf-8 -*-
"""Phase 0 冒烟测试：仅校验攻击 harness 的扰动不变量与语义（不训练 victim）。

用合成小图（build_synthetic_graph）跑 dice / camo，检查：
  1) 不变量：self_loops == 0, dup_edges == 0, 边数变化 ≈ budget
  2) 语义：camo 应使欺诈节点的"良性邻居占比"显著上升（relation camouflage 生效）
结果写入 results/direction2/attack_audit/smoke_stats.json。
"""
import json
import sys
from pathlib import Path

import torch

# 让脚本可直接运行：把 src/ 加入 path
SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

from common.attacks import (  # noqa: E402
    DiceAttack, CamoAttack, build_synthetic_graph,
)


def fraud_benign_ratio(data):
    """每个欺诈节点的良性邻居数 / 邻居总数 的均值。"""
    y = data.y
    ei = data.edge_index
    fraud = (y == 1).nonzero(as_tuple=True)[0]
    if fraud.numel() == 0:
        return 0.0
    ratios = []
    for u in fraud.tolist():
        nbr = ei[1][ei[0] == u].tolist() + ei[0][ei[1] == u].tolist()
        if not nbr:
            ratios.append(0.0)
            continue
        ben = sum(1 for v in set(nbr) if y[v] == 0)
        ratios.append(ben / len(set(nbr)))
    return sum(ratios) / len(ratios)


def run():
    torch.manual_seed(0)
    # 用稀疏图：保证存在大量"欺诈-良性"非边，使加边能达预算（稠密图会过早耗尽候选）
    data = build_synthetic_graph(num_nodes=300, fraud_ratio=0.2,
                                p_edge=0.015, fraud_cluster=0.15, seed=0)
    E0 = data.edge_index.size(1)
    base_ratio = fraud_benign_ratio(data)
    print(f"[smoke] synthetic graph: nodes={data.num_nodes} edges={E0} "
          f"fraud_nodes={(data.y==1).sum().item()} base_fraud_benign_ratio={base_ratio:.3f}")

    results = {"num_nodes": data.num_nodes, "edges": E0,
               "base_fraud_benign_ratio": base_ratio, "attacks": {}}
    ok = True

    for name, atk in [("dice", DiceAttack(budget=0.1, seed=0)),
                      ("camo_pop", CamoAttack(budget=0.1, seed=0, popular_bias=True)),
                      ("camo_nopop", CamoAttack(budget=0.1, seed=0, popular_bias=False))]:
        out = atk.attack(data)
        info = out.attack_info
        E1 = out.edge_index.size(1)
        ratio = fraud_benign_ratio(out)
        expected = int(E0 * 0.1 / 2)
        # 不变量断言
        inv_ok = (info["self_loops"] == 0) and (info["dup_edges"] == 0)
        n_chg_ok = (info["n_add"] == expected) and (info["n_del"] == expected)
        ok = ok and inv_ok and n_chg_ok
        print(f"  [{name}] edges {E0}->{E1} n_add={info['n_add']}(exp {expected}) "
              f"n_del={info['n_del']}(exp {expected}) self_loops={info['self_loops']} "
              f"dup={info['dup_edges']} fraud_benign_ratio={ratio:.3f} invariants_ok={inv_ok}")
        results["attacks"][name] = {
            "edge_index_change": E1 - E0, "n_add": info["n_add"], "n_del": info["n_del"],
            "self_loops": info["self_loops"], "dup_edges": info["dup_edges"],
            "fraud_benign_ratio_after": ratio, "invariants_ok": inv_ok, "n_change_ok": n_chg_ok,
        }

    # camo 语义断言：良性邻居占比应高于 base（伪装生效）
    camo_ratio = results["attacks"]["camo_pop"]["fraud_benign_ratio_after"]
    sem_ok = camo_ratio > base_ratio
    ok = ok and sem_ok
    print(f"[smoke] camo semantic (fraud_benign_ratio up): {base_ratio:.3f} -> {camo_ratio:.3f} ok={sem_ok}")

    results["all_ok"] = ok
    out_dir = SRC.parent / "results" / "direction2" / "attack_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "smoke_stats.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[smoke] wrote {out_file}")
    print(f"[smoke] RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
