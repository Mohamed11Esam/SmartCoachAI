"""
SmartCoach AI - Progressive Overload Engine
============================================
Mathematical engine implementing progressive overload, autoregulation, and
tonnage telemetry per BLUEPRINT.md Section 12.

Key Formulas:
- Epley 1RM: 1RM = Weight * (1 + Reps / 30.0)
- Effective Stimulating Volume: sum(Reps * Weight) for working sets where RPE >= 7.0
- Total Volume: sum(Reps * Weight) for all sets
- Autoregulated Load Scaling:
    - avg RPE <= 7.0: +2.5% (upper body) or +5.0% (lower body), rounded to nearest increment (0.5kg / 2.5kg)
    - avg RPE 7.5 - 9.0: linear progression / small increment (+1.0125 or +1.0kg)
    - avg RPE >= 9.5 or failure: hold load, fix volume, extend rest timer (+30 seconds)
"""

from typing import List, Optional, Union
import math
from schemas import (
    LoggedSet,
    ProgressiveOverloadInput,
    ProgressiveOverloadOutput,
)


def calculate_epley_1rm(weight: float, reps: int) -> float:
    """
    Calculate estimated 1-Repetition Maximum (1RM) using the Epley formula:
    Estimated 1RM = Weight * (1 + Reps / 30.0)

    Args:
        weight: Load lifted in kilograms.
        reps: Repetitions completed.

    Returns:
        Estimated 1RM in kilograms rounded to 2 decimal places.
    """
    if weight <= 0.0 or reps <= 0:
        return 0.0
    return round(float(weight) * (1.0 + float(reps) / 30.0), 2)


def round_to_nearest(value: float, increment: float = 0.5) -> float:
    """
    Round a given value to the nearest discrete increment (e.g., 0.5kg or 2.5kg plate steps)
    using standard half-up arithmetic rounding.

    Args:
        value: Raw load value in kg.
        increment: Plate increment step (default: 0.5kg).

    Returns:
        Rounded value in kg.
    """
    if increment <= 0.0:
        return round(value, 2)
    steps = math.floor((value / increment) + 0.5)
    return round(steps * increment, 4)


def calculate_effective_volume(sets: List[LoggedSet]) -> float:
    """
    Calculate effective stimulating hypertrophic volume (tonnage):
    effective_volume = sum(s.reps * s.weight for s in sets if s.rpe >= 7.0)

    Sets performed below RPE 7.0 provide submaximal mechanical tension
    and are classified as warmups or preparatory sets.

    Args:
        sets: List of logged training sets.

    Returns:
        Effective volume in kg.
    """
    return round(
        sum(float(s.reps) * float(s.weight) for s in sets if float(s.rpe) >= 7.0),
        2,
    )


def calculate_total_volume(sets: List[LoggedSet]) -> float:
    """
    Calculate total cumulative volume (tonnage):
    total_volume = sum(s.reps * s.weight for s in sets)

    Args:
        sets: List of logged training sets.

    Returns:
        Total volume in kg.
    """
    return round(
        sum(float(s.reps) * float(s.weight) for s in sets),
        2,
    )


def calculate_progressive_overload(
    input_data: Union[ProgressiveOverloadInput, dict]
) -> ProgressiveOverloadOutput:
    """
    Prescribe progressive overload adjustment based on athlete performance telemetry.

    Adheres strictly to BLUEPRINT.md Section 12:
    - If avg RPE <= 7.0:
        next_weight = weight * (1.025 if is_upper_body else 1.05) (rounded to nearest 0.5kg or 2.5kg)
    - If avg RPE between 7.5 and 9.0 (7.0 < avg_rpe < 9.5):
        Maintain linear progression or small increment (1.0125 or +1.0kg)
    - If avg RPE >= 9.5 or reps failed:
        next_weight = weight, fix volume, extend rest timer by 30 seconds (rest_seconds += 30)

    Args:
        input_data: ProgressiveOverloadInput instance or dictionary.

    Returns:
        ProgressiveOverloadOutput containing prescribed weight, rest period, volumes,
        and scientific rationale.
    """
    if isinstance(input_data, dict):
        payload = ProgressiveOverloadInput(**input_data)
    else:
        payload = input_data

    sets = payload.sets
    is_upper_body = payload.is_upper_body
    rounding_inc = payload.rounding_increment if payload.rounding_increment > 0 else 0.5
    rest_seconds = payload.current_rest_seconds

    # If no sets provided, return neutral output
    if not sets:
        base_weight = payload.current_weight or 0.0
        return ProgressiveOverloadOutput(
            next_weight=base_weight,
            weight_change=0.0,
            rest_seconds=rest_seconds,
            effective_volume=0.0,
            total_volume=0.0,
            estimated_1rm=0.0,
            avg_rpe=0.0,
            reps_failed=False,
            progression_action="maintain",
            scientific_rationale="No completed sets provided. Preserving baseline load and rest interval.",
            is_upper_body=is_upper_body,
        )

    # Determine baseline working weight
    if payload.current_weight is not None and payload.current_weight > 0.0:
        current_weight = float(payload.current_weight)
    else:
        current_weight = max((float(s.weight) for s in sets), default=0.0)

    # Calculate volumes
    effective_vol = calculate_effective_volume(sets)
    total_vol = calculate_total_volume(sets)

    # Calculate peak estimated 1RM across all sets
    peak_1rm = max((calculate_epley_1rm(s.weight, s.reps) for s in sets), default=0.0)

    # Calculate average RPE
    avg_rpe = round(sum(float(s.rpe) for s in sets) / len(sets), 2)

    # Check for muscular failure or missed rep targets
    reps_failed = any(
        bool(s.failed)
        or (s.target_reps is not None and int(s.reps) < int(s.target_reps))
        for s in sets
    )

    # Algorithmic Progression Logic
    if avg_rpe >= 9.5 or reps_failed:
        # High fatigue threshold or missed repetitions: Fix volume, hold weight, extend rest
        next_weight = current_weight
        new_rest_seconds = rest_seconds + 30
        progression_action = "hold_and_extend_rest"
        failure_detail = "target reps failed" if reps_failed else f"avg RPE reached {avg_rpe:.1f} (>= 9.5)"
        rationale = (
            f"Neuromuscular fatigue boundary reached: {failure_detail}. "
            f"Load is locked at {next_weight:.1f}kg and volume is fixed to prevent overtraining. "
            f"Rest period extended from {rest_seconds}s to {new_rest_seconds}s to optimize ATP-CP phosphagen re-synthesis."
        )

    elif avg_rpe <= 7.0:
        # High reserve (RPE <= 7.0): Prescribe standard overload increment
        multiplier = 1.025 if is_upper_body else 1.05
        raw_next = current_weight * multiplier
        rounded_next = round_to_nearest(raw_next, rounding_inc)

        # Ensure at least minimal plate increment
        if rounded_next <= current_weight:
            next_weight = round_to_nearest(current_weight + rounding_inc, rounding_inc)
        else:
            next_weight = rounded_next

        new_rest_seconds = rest_seconds
        progression_action = "increment"
        pct_label = "2.5% (upper body)" if is_upper_body else "5.0% (lower body)"
        rationale = (
            f"Submaximal effort detected with high reps-in-reserve (avg RPE {avg_rpe:.1f} <= 7.0). "
            f"Prescribing +{pct_label} progressive overload to {next_weight:.1f}kg (+{next_weight - current_weight:.2f}kg) "
            f"at baseline {new_rest_seconds}s rest interval."
        )

    else:
        # Optimal working hypertrophic zone (avg RPE 7.5 - 9.0 / between 7.0 and 9.5)
        # Linear progression or small increment (+1.0125 or +1.0kg)
        raw_next = current_weight * 1.0125
        candidate_weight = round_to_nearest(raw_next, rounding_inc)

        if candidate_weight > current_weight:
            next_weight = candidate_weight
        else:
            # Fallback to linear +1.0kg or +rounding_increment
            step = max(1.0, rounding_inc)
            next_weight = round_to_nearest(current_weight + step, rounding_inc)

        new_rest_seconds = rest_seconds
        progression_action = "linear_increment"
        rationale = (
            f"Optimal hypertrophic exertion achieved (avg RPE {avg_rpe:.1f} in 7.5-9.0 window). "
            f"Prescribing linear progressive overload to {next_weight:.1f}kg (+{next_weight - current_weight:.2f}kg) "
            f"to sustain mechanical tension without exceeding recovery capacity."
        )

    weight_change = round(next_weight - current_weight, 2)

    return ProgressiveOverloadOutput(
        next_weight=round(next_weight, 2),
        weight_change=weight_change,
        rest_seconds=new_rest_seconds,
        effective_volume=effective_vol,
        total_volume=total_vol,
        estimated_1rm=peak_1rm,
        avg_rpe=avg_rpe,
        reps_failed=reps_failed,
        progression_action=progression_action,
        scientific_rationale=rationale,
        is_upper_body=is_upper_body,
    )
