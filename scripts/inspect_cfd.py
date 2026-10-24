"""非定常CFDのCSVを読み込み, 有効面積への変換結果を確認するスクリプト.

目的は, 実データを見て次の2つを人間が決めること:

- 初期の過渡区間の長さ ``--skip`` (この区間は最初の信頼できる値で置き換える)
- 平滑化の幅 ``--window`` (力積が保たれる範囲で選ぶ)

軸の向き・符号・空気密度などは, 取り違えると揚力と抗力が静かに入れ替わるため,
すべて必須の引数にしてある(既定値は無い).

使い方の例:
    uv run scripts/inspect_cfd.py data/cfd/case1.csv \\
        --header no --flow-direction -1 0 0 --up 0 1 0 \\
        --force-sign body --rho 1.2 --speed 26 \\
        --skip 0.005 --window 0.005 --step 0.001 \\
        --plot outputs/case1.png

出力: 要約(行数, 時間刻み, 各成分の範囲, 初期過渡の目安, 力積の変化)と, 任意でグラフ.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.cfd import (
    FlowConvention,
    forces_to_areas,
    hold_initial,
    impulse,
    load_force_csv,
    smooth_and_resample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csv", type=Path, help="CFDの力の時系列CSV")
    parser.add_argument(
        "--header", choices=["yes", "no"], required=True, help="CSVにヘッダー行があるか"
    )
    parser.add_argument("--t-col", type=int, default=0, help="時刻の列番号(0始まり)")
    parser.add_argument("--fx-col", type=int, default=1)
    parser.add_argument("--fy-col", type=int, default=2)
    parser.add_argument("--fz-col", type=int, default=3)
    parser.add_argument(
        "--flow-direction", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"),
        help="空気が物体に対して流れていく向き. 選手が+x方向に進むなら -1 0 0",
    )
    parser.add_argument(
        "--up", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"),
        help="鉛直上向きに対応するベクトル(例: 0 1 0)",
    )
    parser.add_argument(
        "--force-sign", choices=["body", "fluid"], required=True,
        help="CSVの力が 'body'=流体から物体が受ける力, 'fluid'=その逆",
    )
    parser.add_argument("--rho", type=float, required=True, help="空気密度 [kg/m^3]")
    parser.add_argument("--speed", type=float, required=True, help="自由流の速さ [m/s]")
    parser.add_argument("--dt", type=float, default=1.0e-4, help="想定する時間刻み [s]")
    parser.add_argument("--skip", type=float, default=0.0, help="初期過渡の長さ [s]")
    parser.add_argument("--window", type=float, default=0.005, help="平滑化の幅 [s]")
    parser.add_argument("--step", type=float, default=0.001, help="間引き後の刻み [s]")
    parser.add_argument("--plot", type=Path, default=None, help="グラフの保存先(PNG)")
    return parser.parse_args()


def startup_ratio(t: np.ndarray, y: np.ndarray, head: float, ref: float) -> float:
    """最初の head [s] での |y| の最大値 / その後 ref [s] での |y| の中央値.

    1より大幅に大きければ, 初期に過渡的なスパイクがある目安になる.
    """
    first = np.abs(y[t < head])
    later = np.abs(y[(t >= head) & (t < head + ref)])
    if len(first) == 0 or len(later) == 0 or np.median(later) == 0:
        return float("nan")
    return float(first.max() / np.median(later))


def main() -> None:
    args = parse_args()
    conv = FlowConvention(
        flow_direction=tuple(args.flow_direction),
        up_direction=tuple(args.up),
        force_on_body=(args.force_sign == "body"),
        rho=args.rho,
        speed=args.speed,
    )
    series = load_force_csv(
        args.csv, has_header=(args.header == "yes"),
        t_col=args.t_col, fx_col=args.fx_col, fy_col=args.fy_col, fz_col=args.fz_col,
        expected_dt=args.dt,
    )
    raw = forces_to_areas(series, conv)

    q = 0.5 * args.rho * args.speed**2
    print(
        f"行数: {len(series.t)}, 長さ: {series.t[-1]:.4f} s, "
        f"動圧 1/2 rho U^2: {q:.1f} Pa"
    )
    print("\n有効面積 [m^2] (生データ):")
    for name, y in [("揚力 A_L", raw.lift_area), ("抗力 A_D", raw.drag_area),
                    ("横力 A_S", raw.side_area)]:
        print(
            f"  {name}: 平均 {y.mean():+.4f}, "
            f"最小 {y.min():+.4f}, 最大 {y.max():+.4f}"
        )
    main_scale = max(np.abs(raw.lift_area).mean(), np.abs(raw.drag_area).mean())
    side_ratio = np.abs(raw.side_area).mean() / main_scale
    print(f"  横力の大きさ / 揚力・抗力の大きさ(平均): {side_ratio:.2f}"
          "  (小さいほど2次元モデルで無視しやすい)")

    print("\n初期過渡の目安 (最初の10 ms の最大 / その後50 ms の中央値):")
    print(f"  揚力: {startup_ratio(raw.t, raw.lift_area, 0.01, 0.05):.2f}")
    print(f"  抗力: {startup_ratio(raw.t, raw.drag_area, 0.01, 0.05):.2f}")

    held = hold_initial(raw, args.skip)
    processed = smooth_and_resample(held, window=args.window, step=args.step)

    print(f"\n処理: skip={args.skip} s, window={args.window} s, step={args.step} s")
    for name, a, b in [("揚力", raw.lift_area, processed.lift_area),
                       ("抗力", raw.drag_area, processed.drag_area)]:
        before = impulse(raw.t, a)
        after = impulse(processed.t, b)
        print(f"  {name}の力積 ∫A dt: 処理前 {before:.5f}, 処理後 {after:.5f}"
              f" (変化 {100 * (after - before) / before:+.2f}%)")
    print("\n注意: skip を大きくした場合の力積の変化は, 過渡区間を捨てた分を含む.")

    if args.plot is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        for ax, label, a, b in [
            (axes[0], "A_L (lift) [m^2]", raw.lift_area, processed.lift_area),
            (axes[1], "A_D (drag) [m^2]", raw.drag_area, processed.drag_area),
            (axes[2], "A_S (side) [m^2]", raw.side_area, processed.side_area),
        ]:
            ax.plot(raw.t, a, color="0.7", lw=0.7, label="raw")
            ax.plot(processed.t, b, color="C0", lw=1.5, label="processed")
            ax.axvline(args.skip, color="C3", ls="--", lw=0.8)
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
        axes[0].legend()
        axes[-1].set_xlabel("t [s]")
        fig.suptitle(f"{args.csv.name} (red dashed: --skip)")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"\n[saved] {args.plot}")


if __name__ == "__main__":
    main()
