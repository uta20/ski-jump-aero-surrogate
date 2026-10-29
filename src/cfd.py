"""非定常CFDの抗力・揚力の時系列(CSV)を, 軌道計算用の有効面積の表に変換する.

コードに与えられるデータは, 空気密度 rho [kg/m^3], 自由流の速さ U [m/s], および
時刻・抗力 D [N]・揚力 L [N] の時系列であると仮定する.

処理の流れ:

1. ``load_drag_lift_csv``: 時刻と抗力・揚力 [N] を読み込む
2. ``to_effective_areas``: ``A = F / (1/2 rho U^2)`` で有効抗力面積 A_D と
   有効揚力面積 A_L [m^2] に変換する
3. ``hold_initial``: 計算開始直後の過渡的な立ち上がりを, その直後の平均値で置き換える
4. ``smooth_and_resample``: 移動平均で平滑化し, 粗い時間刻みに間引く
5. ``to_time_table``: ``TimeTableEffectiveAreas`` に変換して物理コアに渡す

## 符号の規約

抗力は自由流方向を正, 揚力は鉛直上向き側を正とした値を与えること. 抗力の平均が
負なら符号の規約が逆である可能性が高いため, 読み込み時にエラーにする. 揚力は
符号が変わりうる(開始直後に負になる等)ため, 同様のチェックはしない.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .effective_areas import TimeTableEffectiveAreas


@dataclass(frozen=True)
class DragLiftSeries:
    """抗力・揚力の時系列 [N]."""

    t: np.ndarray
    drag: np.ndarray
    lift: np.ndarray


@dataclass(frozen=True)
class EffectiveAreaSeries:
    """有効面積の時系列 [m^2]."""

    t: np.ndarray
    lift_area: np.ndarray
    drag_area: np.ndarray


def load_drag_lift_csv(
    path: str | Path,
    *,
    has_header: bool,
    t_col: int = 0,
    drag_col: int = 1,
    lift_col: int = 2,
    expected_dt: float | None = 1.0e-4,
    zero_time: bool = True,
) -> DragLiftSeries:
    """時刻・抗力 [N]・揚力 [N] を含むCSVを読み込む.

    ``has_header`` は必須の明示指定. 列の取り違え(時刻列と力の列の入れ替わり)を
    検知するため, ``expected_dt`` を与えると時間刻みの中央値が一致するかを確認する.
    ``zero_time=True`` なら, 最初のサンプルの時刻が0になるように時刻を平行移動する.
    """
    usecols = [t_col, drag_col, lift_col]
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
    # (時刻, 抗力, 揚力)に並べ替える. ヘッダーの有無に関わらず列の位置で引く.
    file_order = sorted(usecols)
    data = df.to_numpy()[:, [file_order.index(c) for c in usecols]]
    t, drag, lift = data[:, 0], data[:, 1], data[:, 2]

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
    if drag.mean() <= 0:
        raise ValueError(
            "抗力の平均が正ではありません. 抗力は自由流方向を正とした値を与えるか, "
            "列の指定を確認してください"
        )
    if zero_time:
        t = t - t[0]
    return DragLiftSeries(t=t, drag=drag, lift=lift)


def to_effective_areas(
    series: DragLiftSeries, *, rho: float, speed: float
) -> EffectiveAreaSeries:
    """抗力・揚力 [N] を有効面積 [m^2] に変換する: A = F / (1/2 rho U^2)."""
    if rho <= 0 or speed <= 0:
        raise ValueError("rho と speed は正の値である必要があります")
    q = 0.5 * rho * speed**2
    return EffectiveAreaSeries(
        t=series.t, lift_area=series.lift / q, drag_area=series.drag / q
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
        areas, lift_area=hold(areas.lift_area), drag_area=hold(areas.drag_area)
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
    )


def to_time_table(areas: EffectiveAreaSeries) -> TimeTableEffectiveAreas:
    """有効面積の時系列を, 物理コア用の表に変換する."""
    return TimeTableEffectiveAreas(areas.t, areas.lift_area, areas.drag_area)


def cfd_scenario(
    path: str | Path,
    *,
    has_header: bool,
    rho: float,
    speed: float,
    skip: float,
    window: float,
    step: float,
    t_col: int = 0,
    drag_col: int = 1,
    lift_col: int = 2,
    expected_dt: float | None = 1.0e-4,
) -> TimeTableEffectiveAreas:
    """CSVから有効面積の表までの一連の処理をまとめた関数.

    実データが届いたとき, 仮のシナリオの代わりに使う入口になる.
    """
    series = load_drag_lift_csv(
        path, has_header=has_header, t_col=t_col, drag_col=drag_col,
        lift_col=lift_col, expected_dt=expected_dt,
    )
    areas = to_effective_areas(series, rho=rho, speed=speed)
    areas = hold_initial(areas, skip)
    return to_time_table(smooth_and_resample(areas, window=window, step=step))
