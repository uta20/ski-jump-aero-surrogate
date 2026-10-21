"""ヒル(着地斜面)の形状を表すモジュール.

座標系: 踏切点を原点とし, x はジャンプ方向(水平, 前方が正),
y は鉛直上向きが正. 斜面は y = height(x) で与える(下り斜面なので通常は負).

飛距離は「踏切点から着地点まで, 斜面に沿って測った長さ」で定義する
(実際の競技の飛距離計測も斜面沿いである).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator


class Hill:
    """点列 (x, y) を PCHIP 補間した斜面.

    PCHIP は単調性を保つ補間なので, 点列の間で斜面がうねる(オーバーシュート
    する)ことがない. 点列の範囲外は, 端点の傾きで直線的に延長する.
    """

    def __init__(self, xs: np.ndarray, ys: np.ndarray) -> None:
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        if xs.ndim != 1 or xs.shape != ys.shape or len(xs) < 2:
            raise ValueError("xs, ys は同じ長さ(2以上)の1次元配列である必要があります")
        if np.any(np.diff(xs) <= 0):
            raise ValueError("xs は厳密に増加している必要があります")

        self._xs = xs
        self._ys = ys
        self._interp = PchipInterpolator(xs, ys, extrapolate=False)
        deriv = self._interp.derivative()
        self._slope_start = float(deriv(xs[0]))
        self._slope_end = float(deriv(xs[-1]))

    @classmethod
    def from_points(cls, xs: np.ndarray, ys: np.ndarray) -> Hill:
        return cls(xs, ys)

    @classmethod
    def constant_slope(cls, slope_deg: float, length: float = 1000.0) -> Hill:
        """一定勾配の下り斜面(水平面から slope_deg 度下る).

        解析解と比較しやすいため, テストや動作確認に使う.
        """
        slope = np.tan(np.radians(slope_deg))
        return cls(np.array([0.0, length]), np.array([0.0, -slope * length]))

    def height(self, x: float) -> float:
        """位置 x における斜面の高さ y を返す. """
        x = float(x)
        if x < self._xs[0]:
            return float(self._ys[0] + self._slope_start * (x - self._xs[0]))
        if x > self._xs[-1]:
            return float(self._ys[-1] + self._slope_end * (x - self._xs[-1]))
        return float(self._interp(x))

    def surface_distance(self, x_land: float, n: int = 2000) -> float:
        """x=0 から x_land までの, 斜面に沿った長さ(折れ線近似)を返す. """
        if x_land <= 0:
            return 0.0
        xs = np.linspace(0.0, x_land, n)
        ys = np.array([self.height(x) for x in xs])
        return float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
