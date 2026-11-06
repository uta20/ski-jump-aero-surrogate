"""SkiTB の ``JP_data.csv`` を使った, シミュレータの妥当性検証.

``JP_data.csv`` には各ジャンプの踏切速度 (``AthleteSpeed`` [km/h]), 飛距離
(``AthleteDistance`` [m]), ヒル規模 (``HillK``, ``HillHS``) などがある.
**風の列は無い**ため, 風の効果の向きは, このファイルでは検証できない.

## 実データで検証できること / できないこと

- できる: 同じヒル内の飛距離のばらつき(ノイズの大きさの妥当性), 踏切速度の
  現実的な範囲(サンプリング範囲の根拠)
- 難しい: 踏切速度に対する飛距離の感度. 同じヒル内の速度の幅が狭く, ゲート・風・
  選手の力量が交絡し, データ数も少ないため, 推定は不安定になる
- できない: 風の効果の向き(風の列が無い)

## データの偏りについて

SkiTB は競技の放送映像から作られたデータで, 飛距離は K 点を平均して 10 m 以上
超えるなど, **良いジャンプに偏っている**. 競技全体の分布を代表するものではなく,
ここで得られる実データのばらつきは, 実際より小さめに出ている可能性が高い.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .datagen import DataGenConfig, Range, generate_dataset, simulate_jump
from .effective_areas import EffectiveAreaModel

REQUIRED_COLUMNS = [
    "ID", "AthleteGender", "AthleteSpeed", "AthleteDistance",
    "HillLocation", "HillK", "HillHS",
]


def hill_class(k: float) -> str:
    """K 点によるヒルの区分(FISの区分に準拠. 要確認)."""
    if k < 85:
        return "small"
    if k < 110:
        return "normal"
    if k < 185:
        return "large"
    return "flying"


def load_jp_data(path: str | Path) -> pd.DataFrame:
    """``JP_data.csv`` から検証に使う列だけを取り出し, 単位を SI に直す.

    出力の列: id, gender, speed_ms, distance, hill_k, hill_hs, hill_class,
    hill_key (会場+K+HSで同じヒルを識別), d_minus_k, d_over_hs
    """
    raw = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"必要な列がありません: {missing}")

    df = pd.DataFrame(
        {
            "id": raw["ID"],
            "gender": raw["AthleteGender"],
            "speed_ms": raw["AthleteSpeed"].astype(float) / 3.6,  # km/h -> m/s
            "distance": raw["AthleteDistance"].astype(float),
            "hill_k": raw["HillK"].astype(int),
            "hill_hs": raw["HillHS"].astype(int),
        }
    )
    df["hill_class"] = df["hill_k"].map(hill_class)
    df["hill_key"] = (
        raw["HillLocation"].astype(str)
        + "|K" + df["hill_k"].astype(str)
        + "|HS" + df["hill_hs"].astype(str)
    )
    df["d_minus_k"] = df["distance"] - df["hill_k"]
    df["d_over_hs"] = df["distance"] / df["hill_hs"]
    return df


def speed_ranges_by_class(
    df: pd.DataFrame, lower: float = 0.05, upper: float = 0.95
) -> pd.DataFrame:
    """ヒル区分ごとの踏切速度 [m/s] の範囲(分位点)を返す.

    シミュレータの ``speed`` のサンプリング範囲を, 実データに基づいて決める根拠にする.
    """
    grouped = df.groupby("hill_class")["speed_ms"]
    out = pd.DataFrame(
        {
            "n": grouped.size(),
            "low": grouped.quantile(lower),
            "median": grouped.median(),
            "high": grouped.quantile(upper),
        }
    )
    return out.sort_values("median")


def pooled_within_group_sd(
    df: pd.DataFrame, value: str, group_cols: list[str], min_n: int = 4
) -> tuple[float, int, int]:
    """グループ内の標準偏差を, 自由度で重みづけして1つにまとめる(pooled sd).

    返り値は (pooled sd, 使ったグループ数, 自由度の合計). n が min_n 未満の
    グループは, 分散の推定が不安定なので使わない.
    """
    num = 0.0
    dof = 0
    n_groups = 0
    for _, g in df.groupby(group_cols):
        if len(g) < min_n:
            continue
        num += float(((g[value] - g[value].mean()) ** 2).sum())
        dof += len(g) - 1
        n_groups += 1
    if dof == 0:
        raise ValueError(f"n >= {min_n} のグループが1つもありません")
    return math.sqrt(num / dof), n_groups, dof


def within_group_slope(
    df: pd.DataFrame,
    y: str,
    x: str,
    group_cols: list[str],
    min_n: int = 4,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """グループ内で中心化した x に対する y の回帰の傾きと, ブートストラップ95%区間.

    グループごとに平均を引く(固定効果)ので, ヒルや性別の違いによる差は除かれる.
    ブートストラップは, 各グループの中で行を復元抽出する(層別).
    """
    groups = [g for _, g in df.groupby(group_cols) if len(g) >= min_n]
    if not groups:
        raise ValueError(f"n >= {min_n} のグループが1つもありません")

    def slope_of(parts: list[pd.DataFrame]) -> float:
        xs = np.concatenate([p[x].to_numpy() - p[x].mean() for p in parts])
        ys = np.concatenate([p[y].to_numpy() - p[y].mean() for p in parts])
        denom = float(np.sum(xs**2))
        return float(np.sum(xs * ys) / denom) if denom > 0 else float("nan")

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        parts = [g.iloc[rng.integers(0, len(g), len(g))] for g in groups]
        s = slope_of(parts)
        if not math.isnan(s):
            boots.append(s)
    if not boots:
        raise ValueError("x のばらつきがグループ内にないため, 傾きを推定できません")
    low, high = np.percentile(boots, [2.5, 97.5])
    return {
        "slope": slope_of(groups),
        "ci_low": float(low),
        "ci_high": float(high),
        "n": int(sum(len(g) for g in groups)),
        "n_groups": len(groups),
    }


# --- シミュレータ側 -----------------------------------------------------------


def simulated_within_condition_sd(
    config: DataGenConfig,
    scenarios: dict[str, EffectiveAreaModel],
    late: EffectiveAreaModel,
    speed_mean: float,
    speed_sd: float,
    n: int = 400,
) -> float:
    """踏切速度だけを実データ相当にばらつかせたときの, 飛距離の標準偏差 [m].

    観測条件(角度・風・密度・質量・斜面)は固定し, ノイズと空力の不確かさは
    ``config`` の設定どおり入れる. ``athlete_speed_sd`` は0にする(実データの速度は
    実際に使われた速度なので, 速度のばらつきは ``speed_sd`` だけで表す).
    速度は一様分布で与え, 標準偏差が ``speed_sd`` になる幅にする.
    """
    half = math.sqrt(3.0) * speed_sd
    fixed = {
        "speed": Range(speed_mean - half, speed_mean + half),
        "angle_deg": Range(0.0, 0.0),
        "wind_x": Range(0.0, 0.0),
        "rho": Range(1.2, 1.2),
        "mass": Range(65.0, 65.0),
        "slope_deg": Range(35.0, 35.0),
    }
    cfg = replace(config, n_samples=n, athlete_speed_sd=0.0, **fixed)
    df = generate_dataset(cfg, scenarios, late)
    return float(df.loc[df["landed"], "distance"].std(ddof=1))


def _distance(early, late, **kwargs) -> float:
    base = {
        "speed": 25.0, "angle_deg": 0.0, "wind_x": 0.0, "rho": 1.2, "mass": 65.0,
        "slope_deg": 35.0,
    }
    base.update(kwargs)
    return simulate_jump(early=early, late=late, **base)[0]


def simulated_speed_sensitivity(
    early: EffectiveAreaModel,
    late: EffectiveAreaModel,
    speed: float = 25.0,
    delta: float = 0.5,
) -> float:
    """踏切速度が 1 m/s 増えたときの飛距離の変化 [m per (m/s)] (中心差分)."""
    upper = _distance(early, late, speed=speed + delta)
    lower = _distance(early, late, speed=speed - delta)
    return (upper - lower) / (2.0 * delta)


def simulated_wind_sensitivity(
    early: EffectiveAreaModel, late: EffectiveAreaModel, delta: float = 1.0
) -> float:
    """追い風が 1 m/s 強くなったときの飛距離の変化 [m per (m/s)] (中心差分).

    負なら追い風で飛距離が縮む(向かい風が有利). 実データに風の列が無いため,
    この向きの妥当性は実データでは確認できない.
    """
    upper = _distance(early, late, wind_x=+delta)
    lower = _distance(early, late, wind_x=-delta)
    return (upper - lower) / (2.0 * delta)
