"""1回のジャンプ飛行をシミュレーションし、結果を表示するデモ。

**ここで使っている質量・面積・係数・踏切条件はすべて動作確認用の
プレースホルダであり、実在の選手・用具・CFD結果の値ではない。**
得られる飛距離も現実の競技の飛距離を再現するものではない。

使い方:
    uv run scripts/demo_flight.py
    uv run scripts/demo_flight.py --wind -2.0 --cl 0.5 --cd 0.5
"""

from __future__ import annotations

import argparse

from src.coefficients import ConstantCoefficients
from src.hill import Hill
from src.physics import Environment, JumperParams, TakeoffState, simulate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", type=float, default=25.0, help="踏切速度 [m/s]")
    parser.add_argument(
        "--angle", type=float, default=0.0, help="踏切角 [deg](上向き正)"
    )
    parser.add_argument(
        "--wind", type=float, default=0.0, help="水平風 [m/s](追い風が正)"
    )
    parser.add_argument(
        "--cl", type=float, default=0.5, help="揚力係数(プレースホルダ)"
    )
    parser.add_argument(
        "--cd", type=float, default=0.5, help="抗力係数(プレースホルダ)"
    )
    parser.add_argument("--slope", type=float, default=35.0, help="斜面の勾配 [deg]")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = simulate(
        takeoff=TakeoffState(speed=args.speed, angle_deg=args.angle),
        params=JumperParams(mass=65.0, area=0.8),  # プレースホルダ
        env=Environment(wind_x=args.wind),
        coeffs=ConstantCoefficients(cl=args.cl, cd=args.cd),
        hill=Hill.constant_slope(args.slope),
    )

    if not result.landed:
        print(f"t_max までに着地しませんでした(滞空 {result.flight_time:.2f} s 以上)")
        return

    print(f"飛距離(斜面沿い): {result.distance:.1f} m")
    print(f"滞空時間        : {result.flight_time:.2f} s")
    print(f"着地点          : x={result.landing_x:.1f} m, y={result.landing_y:.1f} m")


if __name__ == "__main__":
    main()
