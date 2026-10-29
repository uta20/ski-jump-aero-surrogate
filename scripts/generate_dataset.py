"""軌道シミュレーションから疑似データセットを生成するCLI.

2つのモードがある.

1. **仮の値モード**(既定): 仮の3シナリオと仮の後半値で動かす. 数値はすべて
   プレースホルダなので, パイプラインの動作確認用であり, 物理的な結論には使えない.
   実行時に警告を表示する.

       uv run scripts/generate_dataset.py --n 2000 --seed 0

2. **CFDモード**: ``--cfd 名前=CSVのパス`` を指定すると, 実データのシナリオを使う.
   仮の値が実データに混ざらないよう, このモードでは次の引数がすべて必須になる:
   ``--header --rho --speed --skip --window --step --late-lift-area --late-drag-area``
   (後半の有効面積は, 文献値として出典を確認した値を指定すること).

       uv run scripts/generate_dataset.py \\
           --cfd lift_oriented=data/cfd/a.csv --cfd balanced=data/cfd/b.csv \\
           --cfd drag_suppressed=data/cfd/c.csv \\
           --header no --rho 1.2 --speed 26 \\
           --skip 0.005 --window 0.005 --step 0.001 \\
           --late-lift-area 0.34 --late-drag-area 0.38

出力: ``--out`` のCSV, 同じ場所に ``<名前>_config.json``(設定)と
``<名前>_source.json``(空力シナリオの出所).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from src.cfd import cfd_scenario
from src.datagen import DataGenConfig, generate_dataset
from src.effective_areas import ConstantEffectiveAreas
from src.scenarios import (
    PLACEHOLDER_LATE_AREAS,
    placeholder_late_model,
    placeholder_scenarios,
)

CFD_REQUIRED = [
    "header", "rho", "speed", "skip", "window", "step",
    "late_lift_area", "late_drag_area",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=None, help="設定のJSON")
    parser.add_argument("--n", type=int, default=None, help="サンプル数")
    parser.add_argument("--n-athletes", type=int, default=None, help="疑似選手の人数")
    parser.add_argument("--seed", type=int, default=None, help="乱数seed")
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/dataset.csv"), help="出力CSV"
    )
    parser.add_argument(
        "--cfd", action="append", default=[], metavar="NAME=CSV",
        help="CFDシナリオ(複数指定可). 指定するとCFDモードになる",
    )
    parser.add_argument("--header", choices=["yes", "no"], default=None)
    parser.add_argument("--rho", type=float, default=None, help="空気密度 [kg/m^3]")
    parser.add_argument("--speed", type=float, default=None, help="自由流の速さ [m/s]")
    parser.add_argument("--skip", type=float, default=None, help="初期過渡の長さ [s]")
    parser.add_argument("--window", type=float, default=None, help="平滑化の幅 [s]")
    parser.add_argument("--step", type=float, default=None, help="間引き後の刻み [s]")
    parser.add_argument("--late-lift-area", type=float, default=None)
    parser.add_argument("--late-drag-area", type=float, default=None)
    args = parser.parse_args()

    if args.cfd:
        missing = [f"--{k.replace('_', '-')}" for k in CFD_REQUIRED
                   if getattr(args, k) is None]
        if missing:
            parser.error("CFDモードでは次の引数が必須です: " + " ".join(missing))
    return args


def build_config(args: argparse.Namespace) -> DataGenConfig:
    config = (
        DataGenConfig.from_json(args.config) if args.config else DataGenConfig()
    )
    overrides = {}
    if args.n is not None:
        overrides["n_samples"] = args.n
    if args.n_athletes is not None:
        overrides["n_athletes"] = args.n_athletes
    if args.seed is not None:
        overrides["seed"] = args.seed
    return dataclasses.replace(config, **overrides)


def main() -> None:
    args = parse_args()
    config = build_config(args)

    if args.cfd:
        scenarios = {}
        for item in args.cfd:
            name, _, path = item.partition("=")
            if not name or not path or name in scenarios:
                raise SystemExit(f"--cfd の指定が不正または重複しています: {item}")
            scenarios[name] = cfd_scenario(
                path, has_header=(args.header == "yes"), rho=args.rho,
                speed=args.speed, skip=args.skip, window=args.window, step=args.step,
            )
        late = ConstantEffectiveAreas(args.late_lift_area, args.late_drag_area)
        source = {
            "mode": "cfd",
            "files": dict(item.split("=", 1) for item in args.cfd),
            "rho": args.rho, "speed": args.speed, "skip": args.skip,
            "window": args.window, "step": args.step,
            "late_lift_area": args.late_lift_area,
            "late_drag_area": args.late_drag_area,
        }
    else:
        print(
            "警告: 仮の値(プレースホルダ)で生成します. 数値は実在のCFD結果でも\n"
            "文献値でもありません. パイプラインの動作確認用であり, 物理的な結論には\n"
            "使えません.\n"
        )
        scenarios = placeholder_scenarios()
        late = placeholder_late_model()
        source = {
            "mode": "placeholder",
            "late_areas": list(PLACEHOLDER_LATE_AREAS),
            "note": "all aerodynamic values are placeholders",
        }

    df = generate_dataset(config, scenarios, late)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out)
    config.to_json(args.out.with_name(args.out.stem + "_config.json"))
    args.out.with_name(args.out.stem + "_source.json").write_text(
        json.dumps(source, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    landed = df[df["landed"]]
    print(f"{len(df)} ジャンプを生成, 着地した割合 {df['landed'].mean():.1%}")
    print("飛距離 [m]:", landed["distance"].describe().round(1).to_dict())
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
