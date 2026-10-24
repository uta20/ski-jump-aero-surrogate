"""有効揚力面積 A_L と有効抗力面積 A_D のモデル.

空気力は, 揚力係数 Cl・抗力係数 Cd と基準面積 S を使って
F_L = (1/2) rho Cl S v^2, F_D = (1/2) rho Cd S v^2 と書ける. 本プロジェクトでは
積をまとめた有効揚力面積 A_L = Cl S と有効抗力面積 A_D = Cd S [m^2] を直接扱う
(F_L = (1/2) rho A_L v^2, F_D = (1/2) rho A_D v^2). 基準面積 S の定義を別に管理する
必要がなく, CFDで得た力を (1/2) rho v^2 で割れば有効面積になる.

物理コア(``physics.py``)は有効面積の出所を知らず, ``EffectiveAreaModel`` という
呼び出し規約だけに依存する. これにより, 同じ軌道計算に対して

- 一定値(動作確認・解析解との比較用)
- 時間の関数としての表(非定常CFDの出力 A_L(t), A_D(t) など)
- (今後)CFD区間と文献値区間をなめらかにつないだ値

を差し替えて使える.

**注意**: 文献値を取り込むときは, 有効面積として与えられている値を使うこと.
Cl, Cd だけが示されている文献は, その基準面積 S が分からないと有効面積に換算できない.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class EffectiveAreaModel(Protocol):
    """時刻 t と気流に対する相対速度 v_rel から (A_L, A_D) [m^2] を返す."""

    def __call__(self, t: float, v_rel: np.ndarray) -> tuple[float, float]: ...


@dataclass(frozen=True)
class ConstantEffectiveAreas:
    """時間にも速度にも依存しない一定の有効面積."""

    lift_area: float  # 有効揚力面積 A_L [m^2]
    drag_area: float  # 有効抗力面積 A_D [m^2]

    def __call__(self, t: float, v_rel: np.ndarray) -> tuple[float, float]:
        return self.lift_area, self.drag_area


class TimeTableEffectiveAreas:
    """時刻の表 (t, A_L, A_D) を線形補間した有効面積.

    表の範囲外では端の値を保持する(外挿しない). 非定常CFDのように
    「時刻ごとの空力値」が得られているときに使う. 表がカバーしない時間帯に
    何を仮定するかは, 呼び出し側が明示的に決めること.
    """

    def __init__(
        self,
        times: np.ndarray,
        lift_area: np.ndarray,
        drag_area: np.ndarray,
    ) -> None:
        times = np.asarray(times, dtype=float)
        lift_area = np.asarray(lift_area, dtype=float)
        drag_area = np.asarray(drag_area, dtype=float)
        if times.ndim != 1 or len(times) < 2:
            raise ValueError("times は2点以上の1次元配列である必要があります")
        if not (times.shape == lift_area.shape == drag_area.shape):
            raise ValueError("times, lift_area, drag_area の長さが一致していません")
        if np.any(np.diff(times) <= 0):
            raise ValueError("times は厳密に増加している必要があります")

        self._times = times
        self._lift_area = lift_area
        self._drag_area = drag_area

    def __call__(self, t: float, v_rel: np.ndarray) -> tuple[float, float]:
        lift_area = float(np.interp(t, self._times, self._lift_area))
        drag_area = float(np.interp(t, self._times, self._drag_area))
        return lift_area, drag_area


class BlendedEffectiveAreas:
    """2つのモデルを, 区間 [blend_start, blend_end] でなめらかにつなぐ.

    t <= blend_start では ``early``, t >= blend_end では ``late`` をそのまま使い,
    間は smoothstep の重み w = 3x^2 - 2x^3 (x は区間内の位置 0..1)で
    線形結合する. smoothstep は両端で傾きが0なので, 境目で値も傾きも連続になる.

    非定常CFDがカバーする区間(``early``)と, 文献値で置き換える後半(``late``)を
    つなぐために使う. 境目に段差があると, 軌道が人工的に乱れる.
    """

    def __init__(
        self,
        early: EffectiveAreaModel,
        late: EffectiveAreaModel,
        blend_start: float,
        blend_end: float,
    ) -> None:
        if not (0.0 <= blend_start < blend_end):
            raise ValueError("0 <= blend_start < blend_end である必要があります")
        self._early = early
        self._late = late
        self._start = blend_start
        self._end = blend_end

    def weight(self, t: float) -> float:
        """late 側の重み(0..1)."""
        x = (t - self._start) / (self._end - self._start)
        x = min(1.0, max(0.0, x))
        return x * x * (3.0 - 2.0 * x)

    def __call__(self, t: float, v_rel: np.ndarray) -> tuple[float, float]:
        w = self.weight(t)
        if w <= 0.0:
            return self._early(t, v_rel)
        if w >= 1.0:
            return self._late(t, v_rel)
        e_lift, e_drag = self._early(t, v_rel)
        l_lift, l_drag = self._late(t, v_rel)
        return (
            (1.0 - w) * e_lift + w * l_lift,
            (1.0 - w) * e_drag + w * l_drag,
        )
