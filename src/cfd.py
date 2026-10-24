"""非定常CFDの力の時系列(CSV)を, 軌道計算用の有効面積の表に変換する.

処理の流れ:

1. ``load_force_csv``: 時刻と (Fx, Fy, Fz) [N] を読み込む
2. ``forces_to_areas``: 自由流の速さ U と空気密度 rho から ``A = F / (1/2 rho U^2)`` で
   有効抗力面積 A_D, 有効揚力面積 A_L, 横力の有効面積 A_S [m^2] に変換する
3. ``hold_initial``: 計算開始直後の過渡的な立ち上がりを, その直後の平均値で置き換える
4. ``smooth_and_resample``: 移動平均で平滑化し, 粗い時間刻みに間引く
5. ``to_time_table``: ``TimeTableEffectiveAreas`` に変換して物理コアに渡す

## 軸の向きと符号について

軸の向きや力の符号を取り違えると, 揚力と抗力が静かに入れ替わったり符号が反転したり
する. そのため ``FlowConvention`` の全項目は**必須の明示指定**にしてあり, 既定値は無い.

- ``flow_direction``: 空気が物体に対して流れていく向き(下流方向)の単位ベクトル.
  選手が +x 方向に進む場合, 空気は物体に対して -x 向きに流れるので (-1, 0, 0) となる
- ``up_direction``: 鉛直上向きに対応するベクトル
- ``force_on_body``: CSVの力が「流体から物体が受ける力」なら True, 逆(物体が流体に
  及ぼす力)なら False

抗力は自由流方向の成分, 揚力は鉛直上向きのうち自由流に垂直な成分として定義する.
横力(Fz に相当する成分)は2次元モデルでは使わないが, 大きさの確認用に A_S を出力する.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .effective_areas import TimeTableEffectiveAreas


@dataclass(frozen=True)
class ForceSeries:
    """力の時系列. ``force`` は shape (n, 3) で単位は N."""

    t: np.ndarray
    force: np.ndarray


@dataclass(frozen=True)
class EffectiveAreaSeries:
    """有効面積の時系列 [m^2]."""

    t: np.ndarray
    lift_area: np.ndarray
    drag_area: np.ndarray
    side_area: np.ndarray


@dataclass(frozen=True, kw_only=True)
class FlowConvention:
    """CFDの座標系・符号・流れの条件. 既定値は意図的に持たせていない."""

    flow_direction: tuple[float, float, float]
    up_direction: tuple[float, float, float]
    force_on_body: bool
    rho: float  # CFDで使った空気密度 [kg/m^3]
    speed: float  # 自由流の速さ [m/s]

    def __post_init__(self) -> None:
        if self.rho <= 0 or self.speed <= 0:
            raise ValueError("rho と speed は正の値である必要があります")
        flow = np.asarray(self.flow_direction, dtype=float)
        up = np.asarray(self.up_direction, dtype=float)
        if np.linalg.norm(flow) == 0 or np.linalg.norm(up) == 0:
            raise ValueError("flow_direction と up_direction は零ベクトルにできません")
        d = flow / np.linalg.norm(flow)
        if np.linalg.norm(np.cross(d, up)) < 1e-9:
            raise ValueError(
                "up_direction が自由流と平行で, 揚力の向きを決められません"
            )

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(抗力方向 d, 揚力方向 l, 横力方向 s) の正規直交基底を返す."""
        d = np.asarray(self.flow_direction, dtype=float)
        d = d / np.linalg.norm(d)
        up = np.asarray(self.up_direction, dtype=float)
        lift = up - np.dot(up, d) * d  # 鉛直上向きのうち自由流に垂直な成分
        lift = lift / np.linalg.norm(lift)
        side = np.cross(lift, d)
        return d, lift, side


def load_force_csv(
    path: str | Path,
    *,
    has_header: bool,
    t_col: int = 0,
    fx_col: int = 1,
    fy_col: int = 2,
    fz_col: int = 3,
    expected_dt: float | None = 1.0e-4,
    zero_time: bool = True,
) -> ForceSeries:
    """時刻と力 (Fx, Fy, Fz) [N] を含むCSVを読み込む.

    ``has_header`` は必須の明示指定. 列の取り違え(例: 時刻列と力の列の入れ替わり)を
    検知するため, ``expected_dt`` を与えると時間刻みの中央値が一致するかを確認する.
    ``zero_time=True`` なら, 最初のサンプルの時刻が0になるように時刻を平行移動する.
    """
    usecols = [t_col, fx_col, fy_col, fz_col]
    try:
        df = pd.read_csv(
            path, header=0 if has_header else None, usecols=usecols, dtype=float
        )
    except ValueError as e:
        raise ValueError(
            f"{path} を数値として読み込めませんでした. has_header の指定や列番号を"
            f"確認してください: {e}"
        ) from e

    # pandas は usecols の指定順ではなく, ファイル内の列順で返す. 指定した順
    # (時刻, Fx, Fy, Fz)に並べ替える. ヘッダーの有無に関わらず列の位置で引く.
    file_order = sorted(usecols)
    data = df.to_numpy()[:, [file_order.index(c) for c in usecols]]
    t = data[:, 0]
    force = data[:, 1:4]

    if len(t) < 2:
        raise ValueError("データが2行未満です")
    dts = np.diff(t)
    if np.any(dts <= 0):
        raise ValueError("時刻が厳密に増加していません. 時刻列の指定を確認してください")
    if expected_dt is not None:
        median_dt = float(np.median(dts))
        if abs(median_dt - expected_dt) > 0.01 * expected_dt:
            raise ValueError(
                f"時間刻みの中央値 {median_dt:.3e} s が想定 {expected_dt:.3e} s と"
                f"一致しません. 時刻列の指定を確認してください"
            )
    if zero_time:
        t = t - t[0]
    return ForceSeries(t=t, force=force)


def forces_to_areas(series: ForceSeries, conv: FlowConvention) -> EffectiveAreaSeries:
    """力 [N] を有効面積 [m^2] に変換する: A = F / (1/2 rho U^2)."""
    d, lift, side = conv.basis()
    sign = 1.0 if conv.force_on_body else -1.0
    q = 0.5 * conv.rho * conv.speed**2
    f = sign * series.force
    return EffectiveAreaSeries(
        t=series.t,
        lift_area=(f @ lift) / q,
        drag_area=(f @ d) / q,
        side_area=(f @ side) / q,
    )


def hold_initial(
    areas: EffectiveAreaSeries, t_skip: float, ref_window: float = 5.0e-3
) -> EffectiveAreaSeries:
    """t < t_skip の区間を, t_skip 以降の ref_window [s] の平均値で置き換える.

    計算開始直後の過渡的な立ち上がりを取り除きつつ, 時間軸(t=0 が踏切)を保つ.
    1サンプルではなく窓の平均を使うのは, 乱流由来の細かい変動に置き換え値が
    偏らないようにするため. 平滑化の前に行うこと(先に平滑化すると過渡が周囲に広がる).
    """
    if t_skip <= 0:
        return areas
    start = int(np.searchsorted(areas.t, t_skip, side="left"))
    if start >= len(areas.t):
        raise ValueError("t_skip がデータの長さ以上です")
    end = int(np.searchsorted(areas.t, t_skip + ref_window, side="left"))
    end = max(end, start + 1)

    def hold(y: np.ndarray) -> np.ndarray:
        out = y.copy()
        out[:start] = y[start:end].mean()
        return out

    return replace(
        areas,
        lift_area=hold(areas.lift_area),
        drag_area=hold(areas.drag_area),
        side_area=hold(areas.side_area),
    )


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    """中心化した移動平均. 端では窓に入るサンプルだけで平均する(端の偏りを避ける)."""
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1  # 中心化のため奇数にそろえる
    kernel = np.ones(window)
    total = np.convolve(y, kernel, mode="same")
    count = np.convolve(np.ones_like(y), kernel, mode="same")
    return total / count


def impulse(t: np.ndarray, y: np.ndarray) -> float:
    """台形則による時間積分 ∫ y dt."""
    return float(np.sum((y[1:] + y[:-1]) * np.diff(t)) / 2.0)


def smooth_and_resample(
    areas: EffectiveAreaSeries, window: float, step: float
) -> EffectiveAreaSeries:
    """幅 window [s] の移動平均で平滑化し, step [s] 刻みに間引く."""
    dt = float(np.median(np.diff(areas.t)))
    n = int(round(window / dt))
    new_t = np.arange(areas.t[0], areas.t[-1] + 1e-12, step)

    def process(y: np.ndarray) -> np.ndarray:
        return np.interp(new_t, areas.t, moving_average(y, n))

    return EffectiveAreaSeries(
        t=new_t,
        lift_area=process(areas.lift_area),
        drag_area=process(areas.drag_area),
        side_area=process(areas.side_area),
    )


def to_time_table(areas: EffectiveAreaSeries) -> TimeTableEffectiveAreas:
    """揚力・抗力の有効面積の時系列を, 物理コア用の表に変換する(横力は使わない)."""
    return TimeTableEffectiveAreas(areas.t, areas.lift_area, areas.drag_area)
