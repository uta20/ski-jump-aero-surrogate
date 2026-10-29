"""cfd.py と BlendedEffectiveAreas のテスト.

実データは使わず, 既知の値を持つ合成データで検証する.
"""

import numpy as np
import pandas as pd
import pytest

from src.cfd import (
    DragLiftSeries,
    EffectiveAreaSeries,
    cfd_scenario,
    hold_initial,
    impulse,
    load_drag_lift_csv,
    moving_average,
    smooth_and_resample,
    to_effective_areas,
    to_time_table,
)
from src.effective_areas import BlendedEffectiveAreas, ConstantEffectiveAreas
from src.hill import Hill
from src.physics import Environment, JumperParams, TakeoffState, simulate

RHO = 1.2
SPEED = 26.0
Q = 0.5 * RHO * SPEED**2  # 動圧 [Pa]


# --- CSVの読み込み ---------------------------------------------------------


def _write_csv(path, n=50, header=False, dt=1e-4, t0=0.0, drag=100.0, lift=50.0):
    t = t0 + np.arange(n) * dt
    df = pd.DataFrame({"t": t, "drag": drag, "lift": lift})
    df.to_csv(path, header=header, index=False)


def test_load_without_header(tmp_path):
    path = tmp_path / "case.csv"
    _write_csv(path, header=False)
    series = load_drag_lift_csv(path, has_header=False)
    assert len(series.t) == 50
    assert series.t[0] == 0.0
    assert series.t[1] == pytest.approx(1e-4)
    assert series.drag[0] == pytest.approx(100.0)
    assert series.lift[0] == pytest.approx(50.0)


def test_load_with_header_and_shifted_time(tmp_path):
    path = tmp_path / "case.csv"
    _write_csv(path, header=True, t0=1e-4)
    series = load_drag_lift_csv(path, has_header=True)
    assert series.t[0] == 0.0  # zero_time=True で先頭が0になる
    assert series.drag[0] == pytest.approx(100.0)


def test_load_column_mapping(tmp_path):
    path = tmp_path / "case.csv"
    df = pd.DataFrame({"lift": 50.0, "t": np.arange(10) * 1e-4, "drag": 100.0})
    df.to_csv(path, header=False, index=False)
    series = load_drag_lift_csv(
        path, has_header=False, t_col=1, drag_col=2, lift_col=0
    )
    assert series.drag[0] == pytest.approx(100.0)
    assert series.lift[0] == pytest.approx(50.0)


def test_load_detects_wrong_header_flag(tmp_path):
    path = tmp_path / "case.csv"
    _write_csv(path, header=True)
    with pytest.raises(ValueError):
        load_drag_lift_csv(path, has_header=False)  # ヘッダー行を数値として読めない


def test_load_detects_wrong_dt(tmp_path):
    path = tmp_path / "case.csv"
    _write_csv(path, dt=1e-3)  # 想定(1e-4)と違う
    with pytest.raises(ValueError):
        load_drag_lift_csv(path, has_header=False)


def test_load_rejects_negative_drag_as_sign_error(tmp_path):
    path = tmp_path / "case.csv"
    _write_csv(path, drag=-100.0)  # 抗力の符号の規約が逆
    with pytest.raises(ValueError, match="抗力"):
        load_drag_lift_csv(path, has_header=False)


def test_load_allows_negative_lift(tmp_path):
    path = tmp_path / "case.csv"
    _write_csv(path, lift=-20.0)  # 揚力は符号が変わりうるので許容する
    series = load_drag_lift_csv(path, has_header=False)
    assert series.lift[0] == pytest.approx(-20.0)


# --- 有効面積への変換 --------------------------------------------------------


def test_effective_areas_from_known_forces():
    series = DragLiftSeries(
        t=np.array([0.0, 1e-4]), drag=np.array([100.0, 100.0]),
        lift=np.array([50.0, 50.0]),
    )
    areas = to_effective_areas(series, rho=RHO, speed=SPEED)
    assert areas.drag_area[0] == pytest.approx(100.0 / Q)
    assert areas.lift_area[0] == pytest.approx(50.0 / Q)


def test_effective_area_reproduces_force_in_physics_convention():
    """A = F / q から F = (1/2) rho A v^2 に戻すと元の力になる(物理コアとの整合)."""
    series = DragLiftSeries(
        t=np.array([0.0, 1e-4]), drag=np.array([200.0, 200.0]),
        lift=np.array([80.0, 80.0]),
    )
    areas = to_effective_areas(series, rho=RHO, speed=SPEED)
    assert 0.5 * RHO * areas.drag_area[0] * SPEED**2 == pytest.approx(200.0)
    assert 0.5 * RHO * areas.lift_area[0] * SPEED**2 == pytest.approx(80.0)


def test_effective_areas_reject_invalid_conditions():
    series = DragLiftSeries(np.array([0.0, 1.0]), np.array([1.0, 1.0]),
                            np.array([1.0, 1.0]))
    with pytest.raises(ValueError):
        to_effective_areas(series, rho=0.0, speed=SPEED)
    with pytest.raises(ValueError):
        to_effective_areas(series, rho=RHO, speed=-1.0)


# --- 初期過渡・平滑化 ---------------------------------------------------------


def _areas(n=1000, dt=1e-4, value=1.0):
    t = np.arange(n) * dt
    ones = np.full(n, value)
    return EffectiveAreaSeries(t=t, lift_area=ones.copy(), drag_area=ones.copy())


def test_hold_initial_replaces_only_the_start():
    areas = _areas()
    spike = areas.drag_area.copy()
    spike[:20] = 50.0  # 最初の2 ms に過渡的なスパイク
    areas = EffectiveAreaSeries(areas.t, areas.lift_area, spike)

    held = hold_initial(areas, t_skip=2e-3)

    assert np.all(held.drag_area[:20] == 1.0)  # t_skip 以降の最初の値で置き換わる
    assert np.all(held.drag_area[20:] == 1.0)
    assert held.t is areas.t  # 時間軸は変わらない


def test_hold_initial_uses_window_mean_not_a_single_sample():
    """置き換え値は1サンプルではなく参照窓の平均(ノイズに偏らない)."""
    n = 1000
    t = np.arange(n) * 1e-4
    y = np.zeros(n)
    y[:20] = 50.0  # 過渡スパイク
    y[20:70] = np.linspace(0.0, 4.9, 50)  # t_skip 後の 5 ms は 0..4.9 に変動
    areas = EffectiveAreaSeries(t, y, y.copy())

    held = hold_initial(areas, t_skip=2e-3, ref_window=5e-3)

    expected = y[20:70].mean()
    assert np.all(held.lift_area[:20] == pytest.approx(expected))
    assert expected != y[20]  # 1サンプル目とは異なる値になっている


def test_hold_initial_with_zero_skip_is_noop():
    areas = _areas()
    assert hold_initial(areas, 0.0) is areas


def test_hold_initial_rejects_skip_beyond_data():
    with pytest.raises(ValueError):
        hold_initial(_areas(n=10), t_skip=1.0)


def test_moving_average_keeps_constant_including_edges():
    y = np.full(100, 3.0)
    assert np.allclose(moving_average(y, 11), 3.0)


def test_moving_average_keeps_linear_ramp_in_interior():
    y = np.arange(100, dtype=float)
    smoothed = moving_average(y, 11)
    assert np.allclose(smoothed[10:-10], y[10:-10])


def test_smooth_and_resample_step_and_range():
    areas = _areas(n=5000)  # 0.5 s
    out = smooth_and_resample(areas, window=5e-3, step=1e-3)
    assert np.allclose(np.diff(out.t), 1e-3)
    assert out.t[0] == 0.0
    assert out.t[-1] == pytest.approx(0.4999, abs=1e-3)


def test_smoothing_preserves_impulse_of_smooth_signal():
    t = np.arange(5000) * 1e-4
    rng = np.random.default_rng(0)
    base = 0.4 + 0.1 * np.sin(2 * np.pi * t / 0.5)
    noisy = base + rng.normal(0.0, 0.05, size=t.shape)  # 乱流由来の細かい変動を模擬
    areas = EffectiveAreaSeries(t, noisy, noisy)

    out = smooth_and_resample(areas, window=5e-3, step=1e-3)

    assert impulse(out.t, out.drag_area) == pytest.approx(
        impulse(areas.t, areas.drag_area), rel=0.02
    )
    assert np.std(np.diff(out.drag_area)) < np.std(np.diff(noisy))  # ばらつきが減る


# --- 物理コアとの接続・つなぎ -------------------------------------------------


def test_time_table_from_constant_series_matches_constant_model():
    """一定の有効面積の表は, ConstantEffectiveAreas と同じ飛距離を与える."""
    hill = Hill.constant_slope(35.0)
    params = JumperParams(mass=65.0)
    takeoff = TakeoffState(speed=25.0, angle_deg=0.0)
    areas = _areas(n=60000, dt=1e-3, value=0.4)  # 60 s 分(t_max までカバー)

    from_table = simulate(takeoff, params, Environment(), to_time_table(areas), hill)
    from_const = simulate(takeoff, params, Environment(),
                          ConstantEffectiveAreas(0.4, 0.4), hill)

    assert from_table.distance == pytest.approx(from_const.distance, rel=1e-6)


def test_blend_uses_each_model_outside_the_window():
    early = ConstantEffectiveAreas(1.0, 2.0)
    late = ConstantEffectiveAreas(3.0, 4.0)
    blend = BlendedEffectiveAreas(early, late, blend_start=0.4, blend_end=0.5)
    v = np.array([25.0, 0.0])
    assert blend(0.0, v) == (1.0, 2.0)
    assert blend(0.4, v) == (1.0, 2.0)
    assert blend(0.5, v) == (3.0, 4.0)
    assert blend(2.0, v) == (3.0, 4.0)


def test_blend_is_continuous_and_monotone_in_the_window():
    early = ConstantEffectiveAreas(1.0, 2.0)
    late = ConstantEffectiveAreas(3.0, 4.0)
    blend = BlendedEffectiveAreas(early, late, blend_start=0.4, blend_end=0.5)
    v = np.array([25.0, 0.0])
    ts = np.linspace(0.4, 0.5, 101)
    lifts = np.array([blend(t, v)[0] for t in ts])
    assert np.all(np.diff(lifts) >= -1e-12)  # 単調に early -> late
    assert np.max(np.abs(np.diff(lifts))) < 0.1  # 段差が無い(1ステップの変化が小さい)
    # 中央では両者の平均
    assert blend(0.45, v) == pytest.approx((2.0, 3.0))


def test_blend_rejects_invalid_window():
    c = ConstantEffectiveAreas(1.0, 1.0)
    with pytest.raises(ValueError):
        BlendedEffectiveAreas(c, c, blend_start=0.5, blend_end=0.5)
    with pytest.raises(ValueError):
        BlendedEffectiveAreas(c, c, blend_start=-0.1, blend_end=0.5)


def test_cfd_scenario_end_to_end(tmp_path):
    """CSV -> 有効面積の表 までの一連の処理が, 既知の値を再現する."""
    path = tmp_path / "case.csv"
    n = 5000  # 0.5 s
    t = np.arange(n) * 1e-4
    df = pd.DataFrame({"t": t, "drag": 0.4 * Q, "lift": 0.3 * Q})
    df.to_csv(path, header=False, index=False)

    model = cfd_scenario(
        path, has_header=False, rho=RHO, speed=SPEED,
        skip=0.002, window=0.005, step=0.001,
    )

    lift_area, drag_area = model(0.25, np.array([25.0, 0.0]))
    assert drag_area == pytest.approx(0.4)
    assert lift_area == pytest.approx(0.3)
