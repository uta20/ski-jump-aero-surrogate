"""validation.py のテスト.

実データは使わず, 既知の値を持つ合成データで検証する.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.datagen import DataGenConfig, Range
from src.scenarios import placeholder_late_model, placeholder_scenarios
from src.validation import (
    hill_class,
    load_jp_data,
    pooled_within_group_sd,
    simulated_speed_sensitivity,
    simulated_wind_sensitivity,
    simulated_within_condition_sd,
    speed_ranges_by_class,
    within_group_slope,
)

SCENARIOS = placeholder_scenarios()
LATE = placeholder_late_model()


def _write_jp(path, rows):
    """必要な列だけを持つ JP_data.csv 相当のファイルを作る."""
    pd.DataFrame(rows).to_csv(path, index=False)


def _row(i, speed=90.0, dist=130.0, k=120, hs=134, gender="M", loc="X"):
    return {
        "ID": f"JP{i:04d}", "AthleteGender": gender, "AthleteSpeed": speed,
        "AthleteDistance": dist, "HillLocation": loc, "HillK": k, "HillHS": hs,
    }


# --- 読み込み ---------------------------------------------------------------


def test_load_converts_units_and_derives_columns(tmp_path):
    path = tmp_path / "jp.csv"
    _write_jp(path, [_row(1, speed=90.0, dist=130.0, k=120, hs=134)])
    df = load_jp_data(path)
    assert df["speed_ms"][0] == pytest.approx(25.0)  # 90 km/h = 25 m/s
    assert df["d_minus_k"][0] == pytest.approx(10.0)
    assert df["d_over_hs"][0] == pytest.approx(130.0 / 134.0)
    assert df["hill_class"][0] == "large"
    assert df["hill_key"][0] == "X|K120|HS134"


def test_hill_key_distinguishes_venues_with_same_size(tmp_path):
    path = tmp_path / "jp.csv"
    _write_jp(path, [_row(1, loc="A"), _row(2, loc="B")])
    df = load_jp_data(path)
    assert df["hill_key"].nunique() == 2


def test_load_rejects_missing_columns(tmp_path):
    path = tmp_path / "jp.csv"
    pd.DataFrame([{"ID": "JP0001", "AthleteSpeed": 90.0}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="必要な列"):
        load_jp_data(path)


@pytest.mark.parametrize(
    ("k", "expected"),
    [(84, "small"), (85, "normal"), (109, "normal"), (110, "large"),
     (184, "large"), (185, "flying"), (200, "flying")],
)
def test_hill_class_boundaries(k, expected):
    assert hill_class(k) == expected


# --- 速度の範囲 ---------------------------------------------------------------


def test_speed_ranges_by_class_uses_quantiles(tmp_path):
    path = tmp_path / "jp.csv"
    speeds_kmh = [80.0, 82.0, 84.0, 86.0, 88.0]  # 22.2 .. 24.4 m/s
    _write_jp(path, [_row(i, speed=v, k=95, hs=105) for i, v in enumerate(speeds_kmh)])
    ranges = speed_ranges_by_class(load_jp_data(path), lower=0.0, upper=1.0)
    assert ranges.loc["normal", "n"] == 5
    assert ranges.loc["normal", "low"] == pytest.approx(80.0 / 3.6)
    assert ranges.loc["normal", "high"] == pytest.approx(88.0 / 3.6)
    assert ranges.loc["normal", "median"] == pytest.approx(84.0 / 3.6)


# --- pooled sd ----------------------------------------------------------------


def test_pooled_within_group_sd_matches_hand_calculation():
    """グループ1 [1,2,3] (分散1, 自由度2), グループ2 [10,12,14,16] (分散20/3, 自由度3)
    -> pooled 分散 = (2*1 + 3*20/3) / 5 = 4.4.
    """
    df = pd.DataFrame(
        {"g": ["a"] * 3 + ["b"] * 4, "y": [1.0, 2.0, 3.0, 10.0, 12.0, 14.0, 16.0]}
    )
    sd, n_groups, dof = pooled_within_group_sd(df, "y", ["g"], min_n=3)
    assert sd == pytest.approx(math.sqrt(4.4))
    assert (n_groups, dof) == (2, 5)


def test_pooled_within_group_sd_skips_small_groups():
    df = pd.DataFrame(
        {"g": ["a"] * 3 + ["b"] * 2, "y": [1.0, 2.0, 3.0, 100.0, 200.0]}
    )
    sd, n_groups, dof = pooled_within_group_sd(df, "y", ["g"], min_n=3)
    assert n_groups == 1  # n=2 のグループは使わない
    assert sd == pytest.approx(1.0)


def test_pooled_within_group_sd_requires_a_usable_group():
    df = pd.DataFrame({"g": ["a", "b"], "y": [1.0, 2.0]})
    with pytest.raises(ValueError):
        pooled_within_group_sd(df, "y", ["g"], min_n=4)


# --- グループ内回帰 -----------------------------------------------------------


def _grouped_linear(slope, noise_sd=0.0, seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for g, offset in enumerate([100.0, 140.0, 230.0]):  # ヒルごとに大きく異なる水準
        x = rng.normal(25.0, 0.5, 12)
        y = offset + slope * (x - 25.0) + rng.normal(0.0, noise_sd, 12)
        parts.append(pd.DataFrame({"g": g, "x": x, "y": y}))
    return pd.concat(parts, ignore_index=True)


def test_within_group_slope_recovers_exact_slope_without_noise():
    result = within_group_slope(_grouped_linear(3.0), "y", "x", ["g"], n_boot=200)
    assert result["slope"] == pytest.approx(3.0)
    assert result["ci_low"] == pytest.approx(3.0)
    assert result["ci_high"] == pytest.approx(3.0)
    assert result["n_groups"] == 3


def test_within_group_slope_ignores_between_group_offsets():
    """グループ間の水準差が大きくても, グループ内の傾きだけが推定される."""
    df = _grouped_linear(0.0)  # グループ内では x と y は無関係
    result = within_group_slope(df, "y", "x", ["g"], n_boot=200)
    assert result["slope"] == pytest.approx(0.0, abs=1e-9)


def test_within_group_slope_interval_covers_true_value_with_noise():
    df = _grouped_linear(5.0, noise_sd=2.0, seed=1)
    result = within_group_slope(df, "y", "x", ["g"], n_boot=500)
    assert result["ci_low"] < 5.0 < result["ci_high"]
    assert result["ci_low"] < result["slope"] < result["ci_high"]


def test_within_group_slope_rejects_constant_x():
    df = pd.DataFrame({"g": [0] * 5, "x": [25.0] * 5, "y": [1.0, 2.0, 3.0, 4.0, 5.0]})
    with pytest.raises(ValueError):
        within_group_slope(df, "y", "x", ["g"], n_boot=50)


# --- シミュレータ側 -----------------------------------------------------------


def test_simulated_speed_sensitivity_is_positive():
    assert simulated_speed_sensitivity(SCENARIOS["balanced"], LATE) > 0


def test_simulated_wind_sensitivity_matches_direct_difference():
    from src.validation import _distance

    early = SCENARIOS["balanced"]
    expected = (
        _distance(early, LATE, wind_x=+1.0) - _distance(early, LATE, wind_x=-1.0)
    ) / 2.0
    assert simulated_wind_sensitivity(early, LATE) == pytest.approx(expected)


def test_simulated_sd_is_zero_without_any_variation():
    config = DataGenConfig(
        athlete_aero_sd=0.0, gust_sd=0.0, angle_noise_sd=0.0,
        cfd_scale=Range(1.0, 1.0), late_scale=Range(1.0, 1.0),
    )
    sd = simulated_within_condition_sd(
        config, {"balanced": SCENARIOS["balanced"]}, LATE,
        speed_mean=25.0, speed_sd=0.0, n=20,
    )
    assert sd == pytest.approx(0.0, abs=1e-9)


def test_simulated_sd_grows_with_speed_variation():
    config = DataGenConfig(
        athlete_aero_sd=0.0, gust_sd=0.0, angle_noise_sd=0.0,
        cfd_scale=Range(1.0, 1.0), late_scale=Range(1.0, 1.0),
    )
    kwargs = {"config": config, "scenarios": {"balanced": SCENARIOS["balanced"]},
              "late": LATE, "speed_mean": 25.0, "n": 60}
    small = simulated_within_condition_sd(speed_sd=0.1, **kwargs)
    large = simulated_within_condition_sd(speed_sd=0.8, **kwargs)
    # 速度の標準偏差が8倍 -> 飛距離のばらつきもほぼ比例して増える
    assert large > 4 * small
