"""
SmartCoach AI - Algorithmic Engines Package
============================================
Sports science mathematical modeling for:
- Progressive overload and autoregulation (Epley 1RM, RPE-based load scaling, volume tracking)
- Adaptive TDEE and precision macronutrient partitioning
"""

from .overload import (
    calculate_progressive_overload,
    calculate_epley_1rm,
    calculate_effective_volume,
    calculate_total_volume,
    round_to_nearest,
)
from .metabolic import (
    calculate_adaptive_tdee,
    calculate_weight_delta,
    calculate_daily_caloric_imbalance,
    calculate_true_tdee,
    calculate_macro_split,
)

__all__ = [
    "calculate_progressive_overload",
    "calculate_epley_1rm",
    "calculate_effective_volume",
    "calculate_total_volume",
    "round_to_nearest",
    "calculate_adaptive_tdee",
    "calculate_weight_delta",
    "calculate_daily_caloric_imbalance",
    "calculate_true_tdee",
    "calculate_macro_split",
]
