"""ジャンプ飛行の2次元質点モデル(揚力・抗力つき).

## 運動方程式

選手+用具を質量 m の質点とみなし, 空気に対する相対速度を
``v_rel = v - w`` (w は風速ベクトル)とすると,

    m dv/dt = m g
              - (1/2) rho A_D |v_rel| v_rel                 (抗力)
              + (1/2) rho A_L |v_rel| (-v_rel_y, v_rel_x)   (揚力)

A_D = Cd S は有効抗力面積, A_L = Cl S は有効揚力面積 [m^2] である
(``effective_areas.py``). 基準面積 S は別に持たない.

抗力は相対速度と逆向き, 揚力は相対速度を反時計回りに90度回した向き
(前方に飛んでいるとき上向き)に働く.

## 座標系と符号

- 原点は踏切点, x は前方(ジャンプ方向)が正, y は鉛直上向きが正
- 風は水平成分のみ ``wind_x``. +x 方向に吹く風(追い風)が正,
  向かい風は負

## このモデルの限界

姿勢の変化・選手の回転運動・スキー板と身体の相互作用は扱わない.
空力的な振る舞いはすべて有効面積 A_L, A_D (``effective_areas.py``)に押し込めている.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .effective_areas import EffectiveAreaModel
from .hill import Hill

G = 9.80665  # 標準重力加速度 [m/s^2]


@dataclass(frozen=True)
class JumperParams:
    """選手+用具の物理パラメータ. """

    mass: float  # 質量 [kg]


@dataclass(frozen=True)
class Environment:
    """環境条件. """

    rho: float = 1.225  # 空気密度 [kg/m^3]
    wind_x: float = 0.0  # 水平風 [m/s]. +x向き(追い風)が正, 向かい風は負
    g: float = G


@dataclass(frozen=True)
class TakeoffState:
    """踏切直後の状態. """

    speed: float  # 地面に対する速さ [m/s]
    angle_deg: float  # 水平面からの角度 [deg]. 上向きが正, 下向きが負


@dataclass
class FlightResult:
    """1回の飛行シミュレーションの結果. """

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    landed: bool  # t_max までに着地したか
    flight_time: float  # 滞空時間 [s](未着地なら t_max)
    landing_x: float  # 着地点のx [m](未着地ならnan)
    landing_y: float  # 着地点のy [m](未着地ならnan)
    distance: float  # 斜面に沿った飛距離 [m](未着地ならnan)


def acceleration(
    t: float,
    vel: np.ndarray,
    params: JumperParams,
    env: Environment,
    areas: EffectiveAreaModel,
) -> np.ndarray:
    """速度 vel=(vx, vy) のときの加速度 (ax, ay) を返す. """
    gravity = np.array([0.0, -env.g])
    v_rel = np.asarray(vel, dtype=float) - np.array([env.wind_x, 0.0])
    speed = float(math.hypot(v_rel[0], v_rel[1]))
    if speed < 1e-12:
        return gravity  # 相対速度がゼロなら空気力は働かない

    lift_area, drag_area = areas(t, v_rel)
    q = 0.5 * env.rho * speed  # (1/2) rho |v_rel|
    drag = -q * drag_area * v_rel
    lift = q * lift_area * np.array([-v_rel[1], v_rel[0]])
    return gravity + (drag + lift) / params.mass


def simulate(
    takeoff: TakeoffState,
    params: JumperParams,
    env: Environment,
    areas: EffectiveAreaModel,
    hill: Hill,
    t_max: float = 15.0,
    rtol: float = 1e-9,
    atol: float = 1e-9,
) -> FlightResult:
    """踏切から着地までの軌道を数値積分する.

    着地は「選手の位置が斜面の高さを上から下へ横切る瞬間」として
    ``solve_ivp`` のイベント検出で求める.
    """
    angle = math.radians(takeoff.angle_deg)
    vx0 = takeoff.speed * math.cos(angle)
    vy0 = takeoff.speed * math.sin(angle)

    def rhs(t: float, s: np.ndarray) -> list[float]:
        a = acceleration(t, s[2:4], params, env, areas)
        return [s[2], s[3], a[0], a[1]]

    def ground(t: float, s: np.ndarray) -> float:
        return s[1] - hill.height(s[0])

    ground.terminal = True
    ground.direction = -1  # 斜面の上から下へ抜けるときだけ検出する

    sol = solve_ivp(
        rhs,
        (0.0, t_max),
        [0.0, 0.0, vx0, vy0],
        events=ground,
        rtol=rtol,
        atol=atol,
    )

    landed = sol.status == 1
    if landed:
        landing_x = float(sol.y_events[0][0][0])
        landing_y = float(sol.y_events[0][0][1])
        distance = hill.surface_distance(landing_x)
        flight_time = float(sol.t_events[0][0])
    else:
        landing_x = landing_y = distance = math.nan
        flight_time = float(sol.t[-1])

    return FlightResult(
        t=sol.t,
        x=sol.y[0],
        y=sol.y[1],
        vx=sol.y[2],
        vy=sol.y[3],
        landed=landed,
        flight_time=flight_time,
        landing_x=landing_x,
        landing_y=landing_y,
        distance=distance,
    )
