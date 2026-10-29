"""軌道シミュレーションから, 疑似データセットを生成する.

## 生成のしくみ

1ジャンプ = 1行. 次の変数をラテン超方格(LHS)でサンプリングし, 軌道を計算して
飛距離などを目的変数として記録する.

- 観測可能な条件(``OBSERVED_COLUMNS``): 踏切速度, 踏切角, 風, 空気密度, 質量,
  斜面の勾配, 空力シナリオ
- 空力の不確かさ(``UNCERTAINTY_COLUMNS``): CFD区間の有効面積にかかる乗数
  ``cfd_scale`` と, 0.5 s 以降の文献値区間にかかる乗数 ``late_scale``
  (どちらの区間の不確かさが飛距離に効くかを, 感度分析で分けて見るため)
- 潜在変数(``LATENT_COLUMNS``): 疑似選手ごとの空力・踏切速度の個人差と,
  ジャンプごとのノイズ(突風, 踏切角のぶれ). 現実には観測できない量

疑似選手は ``n_athletes`` 人で, 各ジャンプはランダムにいずれかの選手に割り当てる.
同じ選手は同じ個人差を持つため, 評価では選手単位で分割(GroupKFold)する.

**範囲やノイズの大きさの既定値は, すべてプレースホルダ**(出典のある値ではない).
物理的に意味のある結論を出す前に, 文献や実データで置き換える必要がある.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .effective_areas import (
    BlendedEffectiveAreas,
    EffectiveAreaModel,
    ScaledEffectiveAreas,
)
from .hill import Hill
from .physics import Environment, JumperParams, TakeoffState, simulate

OBSERVED_COLUMNS = [
    "speed", "angle_deg", "wind_x", "rho", "mass", "slope_deg", "scenario",
]
UNCERTAINTY_COLUMNS = ["cfd_scale", "late_scale"]
LATENT_COLUMNS = [
    "athlete_id", "athlete_lift_factor", "athlete_drag_factor",
    "athlete_speed_offset", "gust", "angle_noise",
]
TARGET_COLUMNS = ["distance", "flight_time", "landed"]

# LHSでサンプリングする連続変数(この順に次元を割り当てる)
_SAMPLED = [
    "speed", "angle_deg", "wind_x", "rho", "mass", "slope_deg",
    "cfd_scale", "late_scale",
]


@dataclass(frozen=True)
class Range:
    low: float
    high: float


@dataclass(frozen=True)
class DataGenConfig:
    """データ生成の設定. 範囲・ノイズの既定値はすべてプレースホルダ."""

    n_samples: int = 1000
    n_athletes: int = 20
    seed: int = 0

    speed: Range = field(default_factory=lambda: Range(24.0, 26.5))  # [m/s]
    angle_deg: Range = field(default_factory=lambda: Range(-2.0, 3.0))  # [deg]
    wind_x: Range = field(default_factory=lambda: Range(-3.0, 3.0))  # [m/s]
    rho: Range = field(default_factory=lambda: Range(1.10, 1.30))  # [kg/m^3]
    mass: Range = field(default_factory=lambda: Range(55.0, 75.0))  # [kg]
    slope_deg: Range = field(default_factory=lambda: Range(33.0, 37.0))  # [deg]
    cfd_scale: Range = field(default_factory=lambda: Range(0.90, 1.10))
    late_scale: Range = field(default_factory=lambda: Range(0.85, 1.15))

    athlete_aero_sd: float = 0.05  # 選手ごとの有効面積の乗数の標準偏差
    athlete_speed_sd: float = 0.3  # 選手ごとの踏切速度のずれの標準偏差 [m/s]
    gust_sd: float = 0.5  # ジャンプごとの突風の標準偏差 [m/s]
    angle_noise_sd: float = 0.3  # ジャンプごとの踏切角のぶれの標準偏差 [deg]

    blend_start: float = 0.4  # CFD区間から文献値区間へのつなぎ [s]
    blend_end: float = 0.5

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def from_json(cls, path: str | Path) -> DataGenConfig:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        for name in _SAMPLED:
            raw[name] = Range(**raw[name])
        return cls(**raw)


def simulate_jump(
    *,
    speed: float,
    angle_deg: float,
    wind_x: float,
    rho: float,
    mass: float,
    slope_deg: float,
    early: EffectiveAreaModel,
    late: EffectiveAreaModel,
    cfd_scale: float = 1.0,
    late_scale: float = 1.0,
    lift_factor: float = 1.0,
    drag_factor: float = 1.0,
    blend_start: float = 0.4,
    blend_end: float = 0.5,
) -> tuple[float, float, bool]:
    """1ジャンプを計算し, (飛距離 [m], 滞空時間 [s], 着地したか) を返す.

    ``lift_factor``, ``drag_factor`` は選手ごとの空力の個人差(乗数).
    """
    areas = BlendedEffectiveAreas(
        ScaledEffectiveAreas(
            early,
            lift_scale=cfd_scale * lift_factor,
            drag_scale=cfd_scale * drag_factor,
        ),
        ScaledEffectiveAreas(
            late,
            lift_scale=late_scale * lift_factor,
            drag_scale=late_scale * drag_factor,
        ),
        blend_start,
        blend_end,
    )
    result = simulate(
        TakeoffState(speed=speed, angle_deg=angle_deg),
        JumperParams(mass=mass),
        Environment(rho=rho, wind_x=wind_x),
        areas,
        Hill.constant_slope(slope_deg),
    )
    return result.distance, result.flight_time, result.landed


def generate_dataset(
    config: DataGenConfig,
    scenarios: dict[str, EffectiveAreaModel],
    late: EffectiveAreaModel,
) -> pd.DataFrame:
    """設定に従って疑似データセットを生成する.

    ``scenarios`` は CFD区間の空力シナリオ(名前 -> モデル), ``late`` は 0.5 s より
    後に使う有効面積. 同じ ``config.seed`` なら同じ結果になる(再現可能).
    """
    n = config.n_samples
    rng = np.random.default_rng(config.seed)

    # 連続変数: LHS
    unit = qmc.LatinHypercube(d=len(_SAMPLED), seed=config.seed).random(n)
    columns: dict[str, np.ndarray] = {}
    for i, name in enumerate(_SAMPLED):
        r: Range = getattr(config, name)
        columns[name] = r.low + (r.high - r.low) * unit[:, i]

    # 離散変数: 空力シナリオ(ランダムに割り当てる)
    names = list(scenarios)
    scenario = rng.choice(names, size=n)

    # 疑似選手ごとの個人差(同じ選手は同じ値)
    athlete_id = rng.integers(0, config.n_athletes, size=n)
    lift_factor_by = rng.normal(1.0, config.athlete_aero_sd, config.n_athletes)
    drag_factor_by = rng.normal(1.0, config.athlete_aero_sd, config.n_athletes)
    speed_offset_by = rng.normal(0.0, config.athlete_speed_sd, config.n_athletes)

    # ジャンプごとのノイズ
    gust = rng.normal(0.0, config.gust_sd, n)
    angle_noise = rng.normal(0.0, config.angle_noise_sd, n)

    distance = np.empty(n)
    flight_time = np.empty(n)
    landed = np.empty(n, dtype=bool)
    for i in range(n):
        a = athlete_id[i]
        distance[i], flight_time[i], landed[i] = simulate_jump(
            speed=columns["speed"][i] + speed_offset_by[a],
            angle_deg=columns["angle_deg"][i] + angle_noise[i],
            wind_x=columns["wind_x"][i] + gust[i],
            rho=columns["rho"][i],
            mass=columns["mass"][i],
            slope_deg=columns["slope_deg"][i],
            early=scenarios[scenario[i]],
            late=late,
            cfd_scale=columns["cfd_scale"][i],
            late_scale=columns["late_scale"][i],
            lift_factor=lift_factor_by[a],
            drag_factor=drag_factor_by[a],
            blend_start=config.blend_start,
            blend_end=config.blend_end,
        )

    df = pd.DataFrame(
        {
            "speed": columns["speed"],
            "angle_deg": columns["angle_deg"],
            "wind_x": columns["wind_x"],
            "rho": columns["rho"],
            "mass": columns["mass"],
            "slope_deg": columns["slope_deg"],
            "scenario": scenario,
            "cfd_scale": columns["cfd_scale"],
            "late_scale": columns["late_scale"],
            "athlete_id": athlete_id,
            "athlete_lift_factor": lift_factor_by[athlete_id],
            "athlete_drag_factor": drag_factor_by[athlete_id],
            "athlete_speed_offset": speed_offset_by[athlete_id],
            "gust": gust,
            "angle_noise": angle_noise,
            "distance": distance,
            "flight_time": flight_time,
            "landed": landed,
        }
    )
    df.index.name = "jump_id"
    return df
