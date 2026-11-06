"""SkiTB の JP_data.csv と, シミュレータ(仮の空力シナリオ)を照合するスクリプト.

何が検証できて, 何ができないかを区別して報告する(詳細は ``src/validation.py``).

- A. 実データの概要(飛距離が K 点を超える方向に偏っていることの確認を含む)
- B. 踏切速度の範囲: 実データに基づく範囲と, 現在の既定値との比較
- C. 同じヒル・同じ性別の中での飛距離のばらつき: 実データ vs シミュレータ
- D. 踏切速度に対する飛距離の感度: 実データの回帰 vs シミュレータ
- E. 風の効果の向き: 実データに風の列が無いため検証不可. シミュレータ側だけ表示

**シミュレータ側の空力値は仮のシナリオ(プレースホルダ)**なので, ここでの比較から
物理的な結論は出せない. 実データ側の統計と, 比較の枠組みの確認が目的である.

使い方:
    uv run scripts/validate_against_real.py --jp-data path/to/JP_data.csv

JP_data.csv は SkiTB 公式サイトから取得する(本リポジトリには含めない).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.datagen import DataGenConfig
from src.scenarios import placeholder_late_model, placeholder_scenarios
from src.validation import (
    load_jp_data,
    pooled_within_group_sd,
    simulated_speed_sensitivity,
    simulated_wind_sensitivity,
    simulated_within_condition_sd,
    speed_ranges_by_class,
    within_group_slope,
)

GROUP_COLS = ["hill_key", "gender"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--jp-data", type=Path, required=True, help="JP_data.csv")
    parser.add_argument("--min-n", type=int, default=4, help="グループの最小件数")
    parser.add_argument(
        "--n-sim", type=int, default=400, help="シミュレータのサンプル数"
    )
    parser.add_argument("--n-boot", type=int, default=2000, help="ブートストラップ回数")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/validation_report.json"),
        help="主な数値の保存先(JSON)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_jp_data(args.jp_data)
    config = DataGenConfig(seed=args.seed)
    scenarios = placeholder_scenarios()
    late = placeholder_late_model()
    report: dict = {}

    # --- A. 概要 ---
    print("== A. 実データの概要 ==")
    counts = df["hill_class"].value_counts().to_dict()
    below_k = int((df["d_minus_k"] < 0).sum())
    print(f"ジャンプ数: {len(df)}, ヒル区分: {counts}")
    print(
        "飛距離 - K点 [m]: "
        f"平均 {df['d_minus_k'].mean():+.1f}, 標準偏差 {df['d_minus_k'].std():.1f}, "
        f"最小 {df['d_minus_k'].min():+.1f}, 最大 {df['d_minus_k'].max():+.1f}"
    )
    print(f"K点を下回ったジャンプ: {below_k} / {len(df)}")
    print(
        "注意: 良いジャンプに偏ったデータで, 競技全体の分布を代表しない. "
        "実データのばらつきは小さめに出る\n"
    )
    report["n_jumps"] = len(df)
    report["d_minus_k_mean"] = float(df["d_minus_k"].mean())

    # --- B. 速度の範囲 ---
    print("== B. 踏切速度の範囲 [m/s] (実データの5-95%) ==")
    ranges = speed_ranges_by_class(df)
    print(ranges.round(2).to_string())
    print(
        f"現在の既定値: {config.speed.low} - {config.speed.high} m/s "
        "(仮の値. ヒル区分ごとの範囲は上の表を参照)\n"
    )
    report["speed_ranges"] = ranges.round(3).to_dict(orient="index")

    # --- C. ヒル内のばらつき ---
    print("== C. 同じヒル・同じ性別の中での飛距離のばらつき ==")
    real_sd, n_groups, dof = pooled_within_group_sd(
        df, "distance", GROUP_COLS, min_n=args.min_n
    )
    speed_sd, _, _ = pooled_within_group_sd(
        df, "speed_ms", GROUP_COLS, min_n=args.min_n
    )
    sim_sd = simulated_within_condition_sd(
        config, scenarios, late, speed_mean=25.0, speed_sd=speed_sd, n=args.n_sim
    )
    print(
        f"実データ: 飛距離の標準偏差 {real_sd:.2f} m "
        f"(グループ {n_groups}, 自由度 {dof}), 速度の標準偏差 {speed_sd:.2f} m/s"
    )
    print(
        f"シミュレータ(速度のばらつきを実データ相当にした場合): {sim_sd:.2f} m "
        f"(比 {sim_sd / real_sd:.2f})"
    )
    print(
        "合計のばらつきの桁が合っていても, 内訳(速度由来かそれ以外か)が合っている"
        "とは限らない. 次の D を参照\n"
    )
    report["within_hill_sd"] = {
        "real": real_sd, "simulated": sim_sd, "real_speed_sd": speed_sd,
        "n_groups": n_groups, "dof": dof,
    }

    # --- D. 速度感度 ---
    print("== D. 踏切速度に対する飛距離の感度 [m per (m/s)] ==")
    real = within_group_slope(
        df, "distance", "speed_ms", GROUP_COLS, min_n=args.min_n,
        n_boot=args.n_boot, seed=args.seed,
    )
    sim = simulated_speed_sensitivity(scenarios["balanced"], late)
    print(
        f"実データ(グループ内回帰): {real['slope']:+.2f} "
        f"[95%区間 {real['ci_low']:+.2f}, {real['ci_high']:+.2f}], "
        f"n={real['n']}, グループ {real['n_groups']}"
    )
    print(f"シミュレータ(balanced): {sim:+.2f}")
    inside = real["ci_low"] <= sim <= real["ci_high"]
    print(
        "判定: シミュレータの値は実データの95%区間"
        + ("の中" if inside else "の外")
        + "にある"
    )
    if not inside:
        print(
            "ただし交絡(ゲート選択・風・選手の力量)と少ないデータのため, "
            "シミュレータが誤っているとは断定できない. 感度分析(v0.6)の結果を"
            "解釈する前に確認が必要な項目として扱う"
        )
    print()
    report["speed_sensitivity"] = {"real": real, "simulated": sim, "inside_ci": inside}

    # --- E. 風 ---
    print("== E. 風の効果の向き ==")
    print("実データ: JP_data.csv に風の列が無いため, 検証できない")
    print(
        "シミュレータ: 追い風 1 m/s あたりの飛距離の変化 [m per (m/s)] "
        "(負なら向かい風が有利)"
    )
    wind = {}
    for name, model in scenarios.items():
        wind[name] = simulated_wind_sensitivity(model, late)
        print(f"  {name}: {wind[name]:+.2f}")
    if any(v > 0 for v in wind.values()):
        print(
            "注意: 実際の競技では向かい風が有利とされる(FISの風補正点もその前提). "
            "シミュレータで追い風の方が遠くへ飛ぶ(正の値)なら, 符号が逆であり, "
            "有効面積を一定とする仮定や質点近似の限界を示している"
        )
    report["wind_sensitivity_simulated"] = wind

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
