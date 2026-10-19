"""揚力係数 Cl・抗力係数 Cd のモデル。

物理コア(``physics.py``)は係数の出所を知らず、``CoefficientModel`` という
呼び出し規約だけに依存する。これにより、同じ軌道計算に対して

- 一定値(動作確認・解析解との比較用)
- 時間の関数としての係数表(非定常CFDの出力 Cl(t), Cd(t) など)
- (今後)CFD区間と文献値区間をなめらかにつないだ係数

を差し替えて使える。

**注意**: Cl・Cd は基準面積 S の定義に依存する。複数の出所(CFDと文献値など)の
係数を混ぜる場合は、S の定義が一致していることを必ず確認すること。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class CoefficientModel(Protocol):
    """時刻 t と気流に対する相対速度 v_rel から (Cl, Cd) を返す。"""

    def __call__(self, t: float, v_rel: np.ndarray) -> tuple[float, float]: ...


@dataclass(frozen=True)
class ConstantCoefficients:
    """時間にも速度にも依存しない一定の係数。"""

    cl: float
    cd: float

    def __call__(self, t: float, v_rel: np.ndarray) -> tuple[float, float]:
        return self.cl, self.cd


class TimeTableCoefficients:
    """時刻の表 (t, Cl, Cd) を線形補間した係数。

    表の範囲外では端の値を保持する(外挿しない)。非定常CFDの出力のように
    「時刻ごとの係数」が得られているときに使う。表がカバーしない時間帯に
    何を仮定するかは、呼び出し側が明示的に決めること。
    """

    def __init__(self, times: np.ndarray, cl: np.ndarray, cd: np.ndarray) -> None:
        times = np.asarray(times, dtype=float)
        cl = np.asarray(cl, dtype=float)
        cd = np.asarray(cd, dtype=float)
        if times.ndim != 1 or len(times) < 2:
            raise ValueError("times は2点以上の1次元配列である必要があります")
        if not (times.shape == cl.shape == cd.shape):
            raise ValueError("times, cl, cd の長さが一致していません")
        if np.any(np.diff(times) <= 0):
            raise ValueError("times は厳密に増加している必要があります")

        self._times = times
        self._cl = cl
        self._cd = cd

    def __call__(self, t: float, v_rel: np.ndarray) -> tuple[float, float]:
        cl = float(np.interp(t, self._times, self._cl))
        cd = float(np.interp(t, self._times, self._cd))
        return cl, cd
