"""scenarios.py, ScaledEffectiveAreas, datagen.py のテスト.

ここで使う範囲・ノイズ・空力シナリオはすべてプレースホルダであり, 実データではない.
"""

import numpy as np
import pandas as pd
import pytest

from src.datagen import (
    LATENT_COLUMNS,
    OBSERVED_COLUMNS,
    TARGET_COLUMNS,
    UNCERTAINTY_COLUMNS,
    DataGenConfig,
    Range,
    generate_dataset,
    simulate_jump,
)
from src.effective_areas import ConstantEffectiveAreas, ScaledEffectiveAreas
from src.scenarios import (
    SCENARIO_NAMES,
    placeholder_late_model,
    placeholder_scenarios,
)

V = np.array([25.0, 0.0])


def _small_config(**overrides):
    kwargs = {"n_samples": 40, "n_athletes": 5, "seed": 0}
    kwargs.update(overrides)
    return DataGenConfig(**kwargs)


def _generate(config):
    return generate_dataset(config, placeholder_scenarios(), placeholder_late_model())


# --- ScaledEffectiveAreas -----------------------------------------------------


def test_scaled_areas_multiply_lift_and_drag_separately():
    base = ConstantEffectiveAreas(lift_area=0.4, drag_area=0.5)
    scaled = ScaledEffectiveAreas(base, lift_scale=1.5, drag_scale=0.8)
    assert scaled(0.0, V) == pytest.approx((0.6, 0.4))


def test_scaled_areas_default_is_identity_and_rejects_negative():
    base = ConstantEffectiveAreas(0.4, 0.5)
    assert ScaledEffectiveAreas(base)(0.0, V) == (0.4, 0.5)
    with pytest.raises(ValueError):
        ScaledEffectiveAreas(base, lift_scale=-0.1)


# --- 仮のシナリオ -------------------------------------------------------------


def test_placeholder_scenarios_have_expected_names_and_span():
    scenarios = placeholder_scenarios()
    assert tuple(scenarios) == SCENARIO_NAMES
    assert len(scenarios) == 3


def test_placeholder_scenarios_differ_in_the_way_the_real_cases_do():
    """迎角の増加が速いほど, 早い時刻で有効揚力面積が大きい."""
    s = placeholder_scenarios()
    lift_early = {name: model(0.15, V)[0] for name, model in s.items()}
    assert (
        lift_early["lift_oriented"]
        > lift_early["balanced"]
        > lift_early["drag_suppressed"]
    )
    # 抗力は抑制型が最も小さい
    drag_end = {name: model(0.5, V)[1] for name, model in s.items()}
    assert (
        drag_end["drag_suppressed"]
        < drag_end["balanced"]
        < drag_end["lift_oriented"]
    )


# --- simulate_jump ------------------------------------------------------------


def _jump(**overrides):
    kwargs = {
        "speed": 25.0, "angle_deg": 0.0, "wind_x": 0.0, "rho": 1.2, "mass": 65.0,
        "slope_deg": 35.0, "early": placeholder_scenarios()["balanced"],
        "late": placeholder_late_model(),
    }
    kwargs.update(overrides)
    return simulate_jump(**kwargs)


def test_higher_takeoff_speed_gives_longer_distance():
    d = [_jump(speed=v)[0] for v in (23.0, 25.0, 27.0)]
    assert d[0] < d[1] < d[2]


def test_simulate_jump_returns_landed_flag_and_positive_values():
    distance, flight_time, landed = _jump()
    assert landed
    assert distance > 0
    assert flight_time > 0


def test_unit_scales_reproduce_the_unscaled_model():
    """乗数がすべて1なら, 乗数なしと同じ結果になる."""
    a = _jump()
    b = _jump(cfd_scale=1.0, late_scale=1.0, lift_factor=1.0, drag_factor=1.0)
    assert a == b


# --- generate_dataset ---------------------------------------------------------


def test_generation_is_reproducible_for_the_same_seed():
    a = _generate(_small_config())
    b = _generate(_small_config())
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_give_different_data():
    a = _generate(_small_config(seed=0))
    b = _generate(_small_config(seed=1))
    assert not a["distance"].equals(b["distance"])


def test_columns_are_exactly_the_documented_groups():
    df = _generate(_small_config())
    documented = (
        OBSERVED_COLUMNS + UNCERTAINTY_COLUMNS + LATENT_COLUMNS + TARGET_COLUMNS
    )
    assert sorted(df.columns) == sorted(documented)
    assert len(df) == 40


def test_sampled_columns_stay_within_their_ranges():
    config = _small_config(n_samples=60)
    df = _generate(config)
    for name in ["speed", "angle_deg", "wind_x", "rho", "mass", "slope_deg",
                 "cfd_scale", "late_scale"]:
        r = getattr(config, name)
        assert df[name].between(r.low, r.high).all(), name


def test_athletes_share_their_individual_differences():
    df = _generate(_small_config(n_samples=60, n_athletes=5))
    assert df["athlete_id"].between(0, 4).all()
    for col in ["athlete_lift_factor", "athlete_drag_factor", "athlete_speed_offset"]:
        assert (df.groupby("athlete_id")[col].nunique() == 1).all(), col
    assert df["athlete_id"].nunique() > 1  # 複数の選手にジャンプが割り当たっている


def test_all_scenarios_appear_and_jumps_land_with_placeholder_defaults():
    df = _generate(_small_config(n_samples=60))
    assert set(df["scenario"]) == set(SCENARIO_NAMES)
    assert df["landed"].all()


def test_noise_free_degenerate_config_matches_direct_simulation():
    """範囲を1点に潰しノイズを0にすると, 全行が同じ条件の直接計算と一致する."""
    fixed = {
        "speed": Range(25.0, 25.0), "angle_deg": Range(1.0, 1.0),
        "wind_x": Range(-1.0, -1.0), "rho": Range(1.2, 1.2),
        "mass": Range(65.0, 65.0), "slope_deg": Range(35.0, 35.0),
        "cfd_scale": Range(1.0, 1.0), "late_scale": Range(1.0, 1.0),
    }
    config = _small_config(
        n_samples=5, n_athletes=1, athlete_aero_sd=0.0, athlete_speed_sd=0.0,
        gust_sd=0.0, angle_noise_sd=0.0, **fixed,
    )
    scenarios = {"balanced": placeholder_scenarios()["balanced"]}
    df = generate_dataset(config, scenarios, placeholder_late_model())

    expected, _, _ = simulate_jump(
        speed=25.0, angle_deg=1.0, wind_x=-1.0, rho=1.2, mass=65.0, slope_deg=35.0,
        early=scenarios["balanced"], late=placeholder_late_model(),
    )
    assert df["distance"].to_numpy() == pytest.approx(np.full(5, expected))


def test_config_json_round_trip(tmp_path):
    config = _small_config(n_samples=123, speed=Range(20.0, 22.0), gust_sd=0.9)
    path = tmp_path / "config.json"
    config.to_json(path)
    assert DataGenConfig.from_json(path) == config
