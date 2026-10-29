"""非定常CFDの抗力・揚力のCSVを読み込み, 有効面積への変換結果を確認するスクリプト.

目的は, 実データを見て次の2つを人間が決めること:

- 初期の過渡区間の長さ ``--skip`` (この区間は直後の平均値で置き換える)
- 平滑化の幅 ``--window`` (力積が保たれる範囲で選ぶ)

CSVには時刻・抗力 [N]・揚力 [N] が含まれ, 抗力は自由流方向を正, 揚力は鉛直上向き側を
正とした値であると仮定する. 空気密度と自由流の速さは必須の引数(既定値は無い).

使い方の例:
    uv run scripts/inspect_cfd.py data/cfd/case1.csv \\
        --header no --rho 1.2 --speed 26 \\
        --skip 0.005 --window 0.005 --step 0.001 \\
        --plot outputs/case1.png

出力: 要約(行数, 時間刻み, 各成分の範囲, 初期過渡の目安, 力積の変化)と, 任意でグラフ.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.cfd import (
    hold_initial,
    impulse,
    load_drag_lift_csv,
    smooth_and_resample,
    to_effective_areas,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csv", type=Path, help="抗力・揚力の時系列CSV")
    parser.add_argument(
        "--header", choices=["yes", "no"], required=True, help="CSVにヘッダー行があるか"
    )
    parser.add_argument("--t-col", type=int, default=0, help="時刻の列番号(0始まり)")
    parser.add_argument("--drag-col", type=int, default=1, help="抗力の列番号")
    parser.add_argument("--lift-col", type=int, default=2, help="揚力の列番号")
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
    series = load_drag_lift_csv(
        args.csv, has_header=(args.header == "yes"), t_col=args.t_col,
        drag_col=args.drag_col, lift_col=args.lift_col, expected_dt=args.dt,
    )
    raw = to_effective_areas(series, rho=args.rho, speed=args.speed)

    q = 0.5 * args.rho * args.speed**2
    print(
        f"行数: {len(series.t)}, 長さ: {series.t[-1]:.4f} s, "
        f"動圧 1/2 rho U^2: {q:.1f} Pa"
    )
    print("\n有効面積 [m^2] (生データ):")
    for name, y in [("揚力 A_L", raw.lift_area), ("抗力 A_D", raw.drag_area)]:
        print(
            f"  {name}: 平均 {y.mean():+.4f}, "
            f"最小 {y.min():+.4f}, 最大 {y.max():+.4f}"
        )

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
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        for ax, label, a, b in [
            (axes[0], "A_L (lift) [m^2]", raw.lift_area, processed.lift_area),
            (axes[1], "A_D (drag) [m^2]", raw.drag_area, processed.drag_area),
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
