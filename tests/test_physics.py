"""物理コアのテスト。

ここで使う質量・面積・係数などの数値は、テストのための代表値であり、
実際の選手や用具の値ではない。
"""

import math

import numpy as np
import pytest

from src.coefficients import ConstantCoefficients
from src.hill import Hill
from src.physics import (
    Environment,
    G,
    JumperParams,
    TakeoffState,
    acceleration,
    simulate,
)

PARAMS = JumperParams(mass=65.0, area=0.8)
HILL = Hill.constant_slope(35.0)
TAKEOFF = TakeoffState(speed=25.0, angle_deg=0.0)


def _distance(coeffs, env=None, takeoff=TAKEOFF):
    env = env or Environment()
    return simulate(takeoff, PARAMS, env, coeffs, HILL).distance


def test_vacuum_matches_analytic_projectile():
    """空気がなければ放物運動になり、飛距離は解析解と一致する。"""
    speed, angle_deg, slope_deg = 25.0, 5.0, 35.0
    env = Environment(rho=0.0)
    result = simulate(
        TakeoffState(speed=speed, angle_deg=angle_deg),
        PARAMS,
        env,
        ConstantCoefficients(cl=0.7, cd=0.5),  # rho=0 なので係数は効かない
        Hill.constant_slope(slope_deg),
    )

    vx = speed * math.cos(math.radians(angle_deg))
    vy = speed * math.sin(math.radians(angle_deg))
    tan_b = math.tan(math.radians(slope_deg))
    t_land = 2 * (vy + vx * tan_b) / G
    expected_distance = vx * t_land / math.cos(math.radians(slope_deg))

    assert result.landed
    assert result.flight_time == pytest.approx(t_land, rel=1e-6)
    assert result.distance == pytest.approx(expected_distance, rel=1e-6)


def test_vacuum_conserves_mechanical_energy():
    result = simulate(
        TakeoffState(speed=25.0, angle_deg=3.0),
        PARAMS,
        Environment(rho=0.0),
        ConstantCoefficients(cl=0.7, cd=0.5),
        HILL,
    )
    energy = 0.5 * (result.vx**2 + result.vy**2) + G * result.y
    assert np.allclose(energy, energy[0], rtol=1e-8)


def test_terminal_velocity_has_zero_acceleration():
    """鉛直落下の終端速度 v_t = sqrt(2 m g / (rho S Cd)) で加速度がゼロになる。"""
    env = Environment()
    cd = 0.9
    v_t = math.sqrt(2 * PARAMS.mass * env.g / (env.rho * PARAMS.area * cd))
    coeffs = ConstantCoefficients(0.0, cd)
    a = acceleration(0.0, np.array([0.0, -v_t]), PARAMS, env, coeffs)
    assert a[1] == pytest.approx(0.0, abs=1e-9)
    assert a[0] == pytest.approx(0.0, abs=1e-12)


def test_lift_points_up_for_forward_flight():
    """前方に水平に飛ぶとき、揚力は鉛直上向きに働く(抗力ゼロで確認)。"""
    env = Environment()
    v = 25.0
    cl = 0.8
    coeffs = ConstantCoefficients(cl, 0.0)
    a = acceleration(0.0, np.array([v, 0.0]), PARAMS, env, coeffs)
    expected_ay = -env.g + 0.5 * env.rho * PARAMS.area * cl * v**2 / PARAMS.mass
    assert a[0] == pytest.approx(0.0, abs=1e-12)
    assert a[1] == pytest.approx(expected_ay)


def test_drag_opposes_relative_velocity():
    env = Environment()
    coeffs = ConstantCoefficients(0.0, 0.6)
    a = acceleration(0.0, np.array([25.0, 0.0]), PARAMS, env, coeffs)
    assert a[0] < 0  # 前進方向と逆向きに減速する


def test_more_lift_increases_distance():
    base = _distance(ConstantCoefficients(cl=0.4, cd=0.5))
    more_lift = _distance(ConstantCoefficients(cl=0.8, cd=0.5))
    assert more_lift > base


def test_more_drag_decreases_distance():
    base = _distance(ConstantCoefficients(cl=0.6, cd=0.4))
    more_drag = _distance(ConstantCoefficients(cl=0.6, cd=0.8))
    assert more_drag < base


def test_wind_is_equivalent_to_shifted_velocity():
    """風の符号・実装の確認: 空気に対する相対速度が同じなら加速度も同じ。

    風速 w のもとで速度 v で飛ぶことは、無風で速度 v - w で飛ぶことと
    (空気力に関しては)等価である。
    """
    coeffs = ConstantCoefficients(cl=0.7, cd=0.5)
    v = np.array([25.0, -2.0])
    w = -3.0  # 向かい風
    with_wind = acceleration(0.0, v, PARAMS, Environment(wind_x=w), coeffs)
    shifted = acceleration(0.0, v - np.array([w, 0.0]), PARAMS, Environment(), coeffs)
    assert np.allclose(with_wind, shifted)


def test_drag_only_headwind_reduces_distance():
    """抗力だけなら、向かい風は相対速度を増やして減速を強めるため飛距離は縮む。

    揚力がある場合に向かい風が有利か不利かは係数の組に依存し、この単純な
    一定係数モデルでは定まらない(README「既知の挙動」を参照)。そのため、
    ここでは揚力ゼロの場合に限った厳密な性質だけを確認する。
    """
    coeffs = ConstantCoefficients(cl=0.0, cd=0.5)
    headwind = _distance(coeffs, Environment(wind_x=-3.0))
    tailwind = _distance(coeffs, Environment(wind_x=+3.0))
    assert headwind < tailwind


def test_takeoff_below_slope_lands_immediately():
    """斜面より急な下向きに踏み切った場合は、ほぼ即座に着地扱いになる。"""
    result = simulate(
        TakeoffState(speed=25.0, angle_deg=-60.0),
        PARAMS,
        Environment(),
        ConstantCoefficients(0.6, 0.5),
        HILL,
    )
    assert result.landed
    assert result.flight_time < 0.1


def test_never_landing_is_reported():
    """t_max までに着地しないときは landed=False で、距離は nan になる。"""
    result = simulate(
        TAKEOFF,
        PARAMS,
        Environment(),
        ConstantCoefficients(0.6, 0.5),
        HILL,
        t_max=0.1,
    )
    assert not result.landed
    assert math.isnan(result.distance)
    assert result.flight_time == pytest.approx(0.1)
