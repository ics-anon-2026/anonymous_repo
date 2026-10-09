# -*- coding: utf-8 -*-
"""统一攻击 harness 包（Phase 0 冻结）。

提供：
- BaseAttack：统一接口基类与不变量强制（base.py）
- structural：dice / camo（关系伪装）等结构攻击
- build_synthetic_graph：冒烟测试用的小图构造

后续 Metattack / PRBCD / FGA / BinarizedAttack / 特征攻击 在 Phase 1 接入。
"""
from .base import BaseAttack, build_synthetic_graph
from .structural import DiceAttack, CamoAttack

ATTACK_REGISTRY = {
    "dice": DiceAttack,
    "camo": CamoAttack,
}

__all__ = [
    "BaseAttack",
    "build_synthetic_graph",
    "DiceAttack",
    "CamoAttack",
    "ATTACK_REGISTRY",
]
