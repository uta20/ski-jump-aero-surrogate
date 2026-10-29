"""データ生成に使う空力シナリオ(仮の値).

**ここにある数値はすべてプレースホルダであり, 実在のCFD結果でも文献値でもない.**
実データ(非定常CFDの3ケース)が届くまで, データ生成とその後の機械学習・感度分析の
パイプラインを先に動かすための, 形だけを模した仮の値である. 実データが届いたら
``cfd.cfd_scenario`` で作った表に差し替える(コードの他の部分は変更不要).

シナリオの形は, 実際の3ケースの違い(迎角の増加の速さ)を模している:

- ``lift_oriented``: 迎角の増加が比較的急速 -> 有効揚力面積が早く大きくなる
- ``balanced``: 中間
- ``drag_suppressed``: 迎角の増加が比較的緩やか -> 有効抗力面積が小さく抑えられる

各シナリオは t=0 から 0.5 s までの有効面積の表で, 範囲外は端の値を保持する.
"""

from __future__ import annotations

import numpy as np

from .effective_areas import ConstantEffectiveAreas, TimeTableEffectiveAreas

SCENARIO_NAMES = ("lift_oriented", "balanced", "drag_suppressed")

# (A_L の初期値, A_L の最終値, A_D の初期値, A_D の最終値, 立ち上がり時間 [s])
_PLACEHOLDER_SHAPES = {
    "lift_oriented": (0.15, 0.50, 0.30, 0.42, 0.20),
    "balanced": (0.15, 0.45, 0.30, 0.38, 0.30),
    "drag_suppressed": (0.15, 0.40, 0.30, 0.34, 0.45),
}

# 0.5 s より後を文献値で置き換える想定の, 仮の一定値 (A_L, A_D) [m^2]
PLACEHOLDER_LATE_AREAS = (0.34, 0.38)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def placeholder_scenarios(
    duration: float = 0.5, step: float = 0.001
) -> dict[str, TimeTableEffectiveAreas]:
    """仮の3シナリオ(有効面積の表)を返す."""
    t = np.arange(0.0, duration + 1e-12, step)
    scenarios = {}
    for name, (l0, l1, d0, d1, tau) in _PLACEHOLDER_SHAPES.items():
        ramp = _smoothstep(t / tau)
        scenarios[name] = TimeTableEffectiveAreas(
            t, l0 + (l1 - l0) * ramp, d0 + (d1 - d0) * ramp
        )
    return scenarios


def placeholder_late_model() -> ConstantEffectiveAreas:
    """0.5 s より後に使う, 仮の一定の有効面積."""
    lift_area, drag_area = PLACEHOLDER_LATE_AREAS
    return ConstantEffectiveAreas(lift_area, drag_area)
