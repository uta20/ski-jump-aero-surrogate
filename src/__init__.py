"""ski-jump-aero-surrogate: ジャンプ軌道の空力シミュレーションとサロゲートモデル."""

from .cfd import (
    DragLiftSeries,
    EffectiveAreaSeries,
    cfd_scenario,
    hold_initial,
    load_drag_lift_csv,
    smooth_and_resample,
    to_effective_areas,
    to_time_table,
)
from .effective_areas import (
    BlendedEffectiveAreas,
    ConstantEffectiveAreas,
    ScaledEffectiveAreas,
    TimeTableEffectiveAreas,
)
from .hill import Hill
from .physics import (
    Environment,
    FlightResult,
    JumperParams,
    TakeoffState,
    acceleration,
    simulate,
)

__all__ = [
    "BlendedEffectiveAreas",
    "ConstantEffectiveAreas",
    "DragLiftSeries",
    "EffectiveAreaSeries",
    "Environment",
    "FlightResult",
    "Hill",
    "JumperParams",
    "ScaledEffectiveAreas",
    "TakeoffState",
    "TimeTableEffectiveAreas",
    "acceleration",
    "cfd_scenario",
    "hold_initial",
    "load_drag_lift_csv",
    "simulate",
    "smooth_and_resample",
    "to_effective_areas",
    "to_time_table",
]
