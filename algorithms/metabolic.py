"""
SmartCoach AI - Adaptive TDEE & Metabolic Engine
=================================================
Mathematical engine implementing dynamic rolling energy expenditure tracking
and precision macronutrient partitioning per BLUEPRINT.md Section 12.

Key Formulas:
- Mass Delta: delta_w = weight_end - weight_start (kg)
- Daily Caloric Imbalance: daily_caloric_imbalance = (delta_w * 7700.0) / 7.0 (kcal)
- True TDEE: true_tdee = avg_daily_calories - daily_caloric_imbalance (kcal)
- Macro Allocation:
    - Hypertrophy: 2.2g protein/kg, moderate fat (1.0g/kg), remaining cals to carbs (+300 kcal surplus)
    - FatLoss: 2.4g protein/kg, moderate fat (0.8g/kg), remaining cals to carbs (-500 kcal deficit)
    - Strength: 2.0g protein/kg, moderate fat (1.0g/kg), remaining cals to carbs (+200 kcal surplus)
    - Endurance: 1.8g protein/kg, moderate fat (0.9g/kg), remaining cals to carbs (maintenance)
    - Maintenance: 2.0g protein/kg, moderate fat (0.9g/kg), remaining cals to carbs (maintenance)
"""

from typing import Union, Dict, Any, Optional
from schemas import (
    AdaptiveTDEEInput,
    AdaptiveTDEEOutput,
)

# 1 kg of adipose human tissue stores ~7700 kcal of energy
ENERGY_DENSITY_PER_KG_KG = 7700.0
DAYS_PER_ROLLING_WINDOW = 7.0

# Evidence-based defaults by athletic goal
GOAL_DEFAULTS = {
    "Hypertrophy": {
        "caloric_delta": 300.0,      # Controlled surplus for muscle protein synthesis
        "protein_g_per_kg": 2.2,     # Maximizes hyper-trophic stimulus (BLUEPRINT.md)
        "fat_g_per_kg": 1.0,         # Optimized hormonal milieu
    },
    "FatLoss": {
        "caloric_delta": -500.0,     # Sustainable deficit ~0.45kg fat loss/week
        "protein_g_per_kg": 2.4,     # Sparing lean body mass in hypocaloric conditions
        "fat_g_per_kg": 0.8,         # Endocrine health floor
    },
    "Strength": {
        "caloric_delta": 200.0,      # Energy support for CNS and maximal force output
        "protein_g_per_kg": 2.0,
        "fat_g_per_kg": 1.0,
    },
    "Endurance": {
        "caloric_delta": 0.0,        # Maintenance / energy replenishment
        "protein_g_per_kg": 1.8,
        "fat_g_per_kg": 0.9,
    },
    "Maintenance": {
        "caloric_delta": 0.0,        # Neutral caloric balance
        "protein_g_per_kg": 2.0,
        "fat_g_per_kg": 0.9,
    },
}


def calculate_weight_delta(weight_start: float, weight_end: float) -> float:
    """
    Calculate weekly rolling body mass delta:
    delta_w = weight_end - weight_start (kg)

    Args:
        weight_start: Initial body weight in kg.
        weight_end: Final body weight in kg.

    Returns:
        Weight change in kg rounded to 3 decimal places.
    """
    return round(float(weight_end) - float(weight_start), 3)


def calculate_daily_caloric_imbalance(delta_w: float) -> float:
    """
    Calculate daily caloric imbalance over a 7-day period:
    daily_caloric_imbalance = (delta_w * 7700.0) / 7.0 (kcal)

    A positive value indicates a daily surplus; a negative value indicates a deficit.

    Args:
        delta_w: Mass change in kg.

    Returns:
        Daily caloric imbalance in kcal rounded to 2 decimal places.
    """
    return round((float(delta_w) * ENERGY_DENSITY_PER_KG_KG) / DAYS_PER_ROLLING_WINDOW, 2)


def calculate_true_tdee(avg_daily_calories: float, daily_caloric_imbalance: float) -> float:
    """
    Calculate true Total Daily Energy Expenditure (TDEE):
    true_tdee = avg_daily_calories - daily_caloric_imbalance (kcal)

    Args:
        avg_daily_calories: Average logged daily intake in kcal.
        daily_caloric_imbalance: Derived daily caloric imbalance in kcal.

    Returns:
        True metabolic expenditure in kcal rounded to 2 decimal places.
    """
    return round(float(avg_daily_calories) - float(daily_caloric_imbalance), 2)


def calculate_macro_split(
    target_calories: float,
    body_weight: float,
    goal: str = "Hypertrophy",
    protein_g_per_kg: Optional[float] = None,
    fat_g_per_kg: Optional[float] = None,
) -> Dict[str, float]:
    """
    Partition target caloric intake into evidence-based macronutrient targets.

    Energy Equivalents:
    - Protein: 4.0 kcal/g
    - Fat: 9.0 kcal/g
    - Carbohydrate: 4.0 kcal/g

    Args:
        target_calories: Prescribed daily caloric intake.
        body_weight: Current body mass in kg.
        goal: Training adaptation objective.
        protein_g_per_kg: Custom protein multiplier override.
        fat_g_per_kg: Custom fat multiplier override.

    Returns:
        Dictionary containing grams, calories, and multiplier metrics.
    """
    defaults = GOAL_DEFAULTS.get(goal, GOAL_DEFAULTS["Hypertrophy"])
    prot_mult = protein_g_per_kg if protein_g_per_kg is not None else defaults["protein_g_per_kg"]
    fat_mult = fat_g_per_kg if fat_g_per_kg is not None else defaults["fat_g_per_kg"]

    # Calculate protein
    protein_grams = round(prot_mult * body_weight, 1)
    protein_calories = round(protein_grams * 4.0, 1)

    # Calculate fat
    fat_grams = round(fat_mult * body_weight, 1)
    fat_calories = round(fat_grams * 9.0, 1)

    # Carbohydrates get remainder of calories
    remaining_calories = target_calories - protein_calories - fat_calories

    # Safety constraint: If remaining cals are too low, adjust fat down to essential floor (0.6g/kg)
    if remaining_calories < (body_weight * 1.5 * 4.0):  # Ensure at least ~1.5g carbs/kg if possible
        floor_fat_mult = 0.6
        if fat_mult > floor_fat_mult:
            fat_mult = floor_fat_mult
            fat_grams = round(fat_mult * body_weight, 1)
            fat_calories = round(fat_grams * 9.0, 1)
            remaining_calories = target_calories - protein_calories - fat_calories

    remaining_calories = max(0.0, remaining_calories)
    carb_grams = round(remaining_calories / 4.0, 1)
    carb_calories = round(carb_grams * 4.0, 1)

    return {
        "protein_grams": protein_grams,
        "protein_calories": protein_calories,
        "protein_g_per_kg": round(prot_mult, 2),
        "fat_grams": fat_grams,
        "fat_calories": fat_calories,
        "fat_g_per_kg": round(fat_mult, 2),
        "carb_grams": carb_grams,
        "carb_calories": carb_calories,
    }


def calculate_adaptive_tdee(
    input_data: Union[AdaptiveTDEEInput, dict]
) -> AdaptiveTDEEOutput:
    """
    Execute weekly adaptive metabolic expenditure calculation and prescribe
    adjusted caloric targets and macronutrient distributions.

    Per BLUEPRINT.md Section 12:
    - delta_w = weight_end - weight_start (kg)
    - daily_caloric_imbalance = (delta_w * 7700.0) / 7.0 (kcal)
    - true_tdee = avg_daily_calories - daily_caloric_imbalance (kcal)
    - Prescribe new caloric target and macronutrient distribution.

    Args:
        input_data: AdaptiveTDEEInput instance or dictionary.

    Returns:
        AdaptiveTDEEOutput with true TDEE, target calories, and macro breakdown.
    """
    if isinstance(input_data, dict):
        payload = AdaptiveTDEEInput(**input_data)
    else:
        payload = input_data

    # 1. Delta mass calculation
    delta_w = calculate_weight_delta(payload.weight_start, payload.weight_end)

    # 2. Daily caloric imbalance
    daily_imbalance = calculate_daily_caloric_imbalance(delta_w)

    # 3. True metabolic expenditure
    true_tdee = calculate_true_tdee(payload.avg_daily_calories, daily_imbalance)

    # 4. Target calories determined by goal
    goal = payload.goal
    defaults = GOAL_DEFAULTS.get(goal, GOAL_DEFAULTS["Hypertrophy"])

    if payload.target_surplus_deficit_kcal is not None:
        caloric_adj = payload.target_surplus_deficit_kcal
    else:
        caloric_adj = defaults["caloric_delta"]

    target_calories = round(true_tdee + caloric_adj, 1)

    # 5. Body weight reference
    current_bw = payload.body_weight if payload.body_weight is not None else payload.weight_end

    # 6. Macronutrient breakdown
    macros = calculate_macro_split(
        target_calories=target_calories,
        body_weight=current_bw,
        goal=goal,
        protein_g_per_kg=payload.protein_g_per_kg,
        fat_g_per_kg=payload.fat_g_per_kg,
    )

    # 7. Scientific rationale synthesis
    if delta_w > 0:
        trend_desc = f"+{delta_w:.2f}kg mass gain (+{daily_imbalance:.1f} kcal/day surplus)"
    elif delta_w < 0:
        trend_desc = f"{delta_w:.2f}kg mass loss ({daily_imbalance:.1f} kcal/day deficit)"
    else:
        trend_desc = "stable body mass (neutral caloric balance)"

    adj_sign = f"+{caloric_adj:.0f}" if caloric_adj > 0 else f"{caloric_adj:.0f}"
    rationale = (
        f"7-day metabolic rolling audit shows {trend_desc} against an average logged intake of {payload.avg_daily_calories:.0f} kcal. "
        f"Calculated True TDEE is {true_tdee:.0f} kcal. To advance towards {goal}, a {adj_sign} kcal adjustment is prescribed, "
        f"yielding a target intake of {target_calories:.0f} kcal. Macronutrients partitioned at {macros['protein_g_per_kg']}g/kg protein "
        f"({macros['protein_grams']}g) and {macros['fat_g_per_kg']}g/kg fat ({macros['fat_grams']}g) to optimize endocrine signaling "
        f"and lean tissue preservation, with remaining energy allocated to glycogen support ({macros['carb_grams']}g carbs)."
    )

    return AdaptiveTDEEOutput(
        delta_w=delta_w,
        daily_caloric_imbalance=daily_imbalance,
        true_tdee=true_tdee,
        target_calories=target_calories,
        target_protein_grams=macros["protein_grams"],
        target_carb_grams=macros["carb_grams"],
        target_fat_grams=macros["fat_grams"],
        protein_calories=macros["protein_calories"],
        carb_calories=macros["carb_calories"],
        fat_calories=macros["fat_calories"],
        protein_g_per_kg=macros["protein_g_per_kg"],
        fat_g_per_kg=macros["fat_g_per_kg"],
        goal=goal,
        scientific_rationale=rationale,
    )
