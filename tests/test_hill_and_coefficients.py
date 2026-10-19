import math

import numpy as np
import pytest

from src.coefficients import ConstantCoefficients, TimeTableCoefficients
from src.hill import Hill


def test_constant_slope_height():
    hill = Hill.constant_slope(35.0)
    x = 100.0
    assert hill.height(x) == pytest.approx(-x * math.tan(math.radians(35.0)))
    assert hill.height(0.0) == pytest.approx(0.0)


def test_constant_slope_surface_distance_matches_geometry():
    hill = Hill.constant_slope(35.0)
    x = 90.0
    expected = x / math.cos(math.radians(35.0))
    assert hill.surface_distance(x) == pytest.approx(expected, rel=1e-9)


def test_from_points_reproduces_nodes():
    xs = np.array([0.0, 50.0, 100.0, 150.0])
    ys = np.array([0.0, -20.0, -50.0, -65.0])
    hill = Hill.from_points(xs, ys)
    for x, y in zip(xs, ys, strict=True):
        assert hill.height(x) == pytest.approx(y)


def test_height_extends_linearly_beyond_range():
    hill = Hill.from_points(np.array([0.0, 10.0]), np.array([0.0, -5.0]))
    # 傾き -0.5 の直線が範囲外にもそのまま延びる
    assert hill.height(20.0) == pytest.approx(-10.0)
    assert hill.height(-10.0) == pytest.approx(5.0)


def test_hill_rejects_invalid_points():
    with pytest.raises(ValueError):
        Hill.from_points(np.array([0.0]), np.array([0.0]))
    with pytest.raises(ValueError):
        Hill.from_points(np.array([0.0, 0.0]), np.array([0.0, 1.0]))


def test_constant_coefficients_ignore_time_and_velocity():
    coeffs = ConstantCoefficients(cl=0.6, cd=0.5)
    assert coeffs(0.0, np.array([20.0, 0.0])) == (0.6, 0.5)
    assert coeffs(3.0, np.array([5.0, -3.0])) == (0.6, 0.5)


def test_time_table_interpolates_and_clamps():
    coeffs = TimeTableCoefficients(
        times=np.array([0.0, 1.0]),
        cl=np.array([0.0, 1.0]),
        cd=np.array([1.0, 0.0]),
    )
    v = np.array([20.0, 0.0])
    assert coeffs(0.5, v) == pytest.approx((0.5, 0.5))
    # 表の範囲外は端の値を保持する(外挿しない)
    assert coeffs(-1.0, v) == pytest.approx((0.0, 1.0))
    assert coeffs(5.0, v) == pytest.approx((1.0, 0.0))


def test_time_table_rejects_invalid_input():
    with pytest.raises(ValueError):
        TimeTableCoefficients(
            np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0])
        )
    with pytest.raises(ValueError):
        TimeTableCoefficients(
            np.array([0.0, 1.0]), np.array([0.0]), np.array([0.0, 1.0])
        )
