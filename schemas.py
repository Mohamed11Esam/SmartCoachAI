"""
SmartCoach AI - Domain Schemas (Pydantic v2)
=============================================
Standardized data contracts for athletic intelligence, workout planning,
progressive overload prescription, and adaptive metabolic modeling.

Conforms to BLUEPRINT.md Section 15 and Section 12 specifications.
Supports dual camelCase (TypeScript/NestJS compatibility) and snake_case (Pythonic).
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, AliasChoices


# ==============================================================================
# SECTION 15 DOMAIN CONTRACTS: WORKOUT & MASTER PLAN
# ==============================================================================

class ExerciseSetTarget(BaseModel):
    """
    Contract for a prescribed exercise set target.
    Matches TypeScript `ExerciseSetTarget` interface in BLUEPRINT.md Section 15.
    """
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    set_number: int = Field(
        ...,
        validation_alias=AliasChoices("setNumber", "set_number"),
        serialization_alias="setNumber",
        description="Index of the set in sequence (1-indexed)",
        ge=1,
    )
    reps: int = Field(
        ...,
        description="Target repetitions to perform",
        ge=1,
    )
    target_rpe: float = Field(
        ...,
        validation_alias=AliasChoices("targetRpe", "target_rpe"),
        serialization_alias="targetRpe",
        description="Target Rate of Perceived Exertion (1.0 to 10.0)",
        ge=1.0,
        le=10.0,
    )
    suggested_weight_kg: Optional[float] = Field(
        None,
        validation_alias=AliasChoices("suggestedWeightKg", "suggested_weight_kg"),
        serialization_alias="suggestedWeightKg",
        description="Suggested load in kilograms",
        ge=0.0,
    )
    rest_seconds: int = Field(
        ...,
        validation_alias=AliasChoices("restSeconds", "rest_seconds"),
        serialization_alias="restSeconds",
        description="Prescribed rest period in seconds between sets",
        ge=0,
    )


class StructuredWorkoutExercise(BaseModel):
    """
    Contract for an exercise within a structured workout day.
    """
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    exercise_id: str = Field(
        ...,
        validation_alias=AliasChoices("exerciseId", "exercise_id"),
        serialization_alias="exerciseId",
        description="Unique identifier for the exercise",
    )
    exercise_name: str = Field(
        ...,
        validation_alias=AliasChoices("exerciseName", "exercise_name"),
        serialization_alias="exerciseName",
        description="Human-readable name of the exercise",
    )
    target_sets: List[ExerciseSetTarget] = Field(
        ...,
        validation_alias=AliasChoices("targetSets", "target_sets"),
        serialization_alias="targetSets",
        description="Target sets programming",
    )
    technique_cue: str = Field(
        ...,
        validation_alias=AliasChoices("techniqueCue", "technique_cue"),
        serialization_alias="techniqueCue",
        description="Biomechanical coaching cue for execution",
    )
    alternative_exercise_id: Optional[str] = Field(
        None,
        validation_alias=AliasChoices("alternativeExerciseId", "alternative_exercise_id"),
        serialization_alias="alternativeExerciseId",
        description="Optional substitute exercise ID if equipment is unavailable",
    )


class StructuredWorkoutDay(BaseModel):
    """
    Contract for a complete daily workout session.
    Matches TypeScript `StructuredWorkoutDay` interface in BLUEPRINT.md Section 15.
    """
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    day_name: str = Field(
        ...,
        validation_alias=AliasChoices("dayName", "day_name"),
        serialization_alias="dayName",
        description="Name of the training session (e.g. Upper Body A, Push Heavy)",
    )
    focus_muscle_groups: List[str] = Field(
        ...,
        validation_alias=AliasChoices("focusMuscleGroups", "focus_muscle_groups"),
        serialization_alias="focusMuscleGroups",
        description="Target muscle groups trained in this session",
    )
    warmup_minutes: int = Field(
        ...,
        validation_alias=AliasChoices("warmupMinutes", "warmup_minutes"),
        serialization_alias="warmupMinutes",
        description="Duration of dynamic warmup in minutes",
        ge=0,
    )
    exercises: List[StructuredWorkoutExercise] = Field(
        ...,
        description="Ordered list of exercises prescribed for the day",
    )


class MasterPlanPayload(BaseModel):
    """
    Contract for a full mesocycle master training plan.
    Matches TypeScript `MasterPlanPayload` interface in BLUEPRINT.md Section 15.
    """
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    user_id: str = Field(
        ...,
        validation_alias=AliasChoices("userId", "user_id"),
        serialization_alias="userId",
        description="Unique athlete ID",
    )
    cycle_weeks: int = Field(
        ...,
        validation_alias=AliasChoices("cycleWeeks", "cycle_weeks"),
        serialization_alias="cycleWeeks",
        description="Duration of the mesocycle in weeks",
        ge=1,
    )
    goal: Literal["Hypertrophy", "Strength", "FatLoss", "Endurance"] = Field(
        ...,
        description="Primary training adaptation objective",
    )
    split_type: Literal["UpperLower", "PushPullLegs", "FullBody"] = Field(
        ...,
        validation_alias=AliasChoices("splitType", "split_type"),
        serialization_alias="splitType",
        description="Microcycle split structure",
    )
    weekly_schedule: List[StructuredWorkoutDay] = Field(
        ...,
        validation_alias=AliasChoices("weeklySchedule", "weekly_schedule"),
        serialization_alias="weeklySchedule",
        description="Day-by-day workout programming",
    )
    scientific_rationale: str = Field(
        ...,
        validation_alias=AliasChoices("scientificRationale", "scientific_rationale"),
        serialization_alias="scientificRationale",
        description="Evidence-based reasoning behind split and exercise selection",
    )


# ==============================================================================
# SECTION 12 ALGORITHMIC DOMAIN CONTRACTS: PROGRESSIVE OVERLOAD
# ==============================================================================

class LoggedSet(BaseModel):
    """
    Performance log for a single executed set by the athlete.
    """
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    set_number: Optional[int] = Field(
        None,
        validation_alias=AliasChoices("setNumber", "set_number"),
        serialization_alias="setNumber",
        description="Set index in the sequence",
    )
    reps: int = Field(
        ...,
        description="Actual repetitions completed",
        ge=0,
    )
    weight: float = Field(
        ...,
        description="Load lifted in kilograms",
        ge=0.0,
    )
    rpe: float = Field(
        ...,
        description="Rate of Perceived Exertion (1.0 to 10.0)",
        ge=1.0,
        le=10.0,
    )
    target_reps: Optional[int] = Field(
        None,
        validation_alias=AliasChoices("targetReps", "target_reps"),
        serialization_alias="targetReps",
        description="Prescribed target repetitions",
    )
    failed: Optional[bool] = Field(
        False,
        description="Whether the athlete failed to achieve target reps or reached muscular failure",
    )


class ProgressiveOverloadInput(BaseModel):
    """
    Input telemetry payload for the progressive overload mathematical engine.
    """
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    exercise_id: Optional[str] = Field(
        None,
        validation_alias=AliasChoices("exerciseId", "exercise_id"),
        serialization_alias="exerciseId",
        description="Exercise identifier",
    )
    exercise_name: Optional[str] = Field(
        None,
        validation_alias=AliasChoices("exerciseName", "exercise_name"),
        serialization_alias="exerciseName",
        description="Exercise name",
    )
    is_upper_body: bool = Field(
        True,
        validation_alias=AliasChoices("isUpperBody", "is_upper_body"),
        serialization_alias="isUpperBody",
        description="True if upper-body movement (+2.5%), False if lower-body movement (+5.0%)",
    )
    current_weight: Optional[float] = Field(
        None,
        validation_alias=AliasChoices("currentWeight", "current_weight"),
        serialization_alias="currentWeight",
        description="Baseline working weight in kg (defaults to max working weight if None)",
        ge=0.0,
    )
    current_rest_seconds: int = Field(
        90,
        validation_alias=AliasChoices("currentRestSeconds", "current_rest_seconds"),
        serialization_alias="currentRestSeconds",
        description="Current baseline inter-set rest timer in seconds",
        ge=0,
    )
    sets: List[LoggedSet] = Field(
        ...,
        description="Completed working sets from the training session",
    )
    rounding_increment: float = Field(
        0.5,
        validation_alias=AliasChoices("roundingIncrement", "rounding_increment"),
        serialization_alias="roundingIncrement",
        description="Weight rounding increment in kg (e.g., 0.5kg or 2.5kg)",
        gt=0.0,
    )


class ProgressiveOverloadOutput(BaseModel):
    """
    Prescription output payload generated by the progressive overload engine.
    """
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    next_weight: float = Field(
        ...,
        validation_alias=AliasChoices("nextWeight", "next_weight"),
        serialization_alias="nextWeight",
        description="Prescribed working load for the subsequent session in kg",
    )
    weight_change: float = Field(
        ...,
        validation_alias=AliasChoices("weightChange", "weight_change"),
        serialization_alias="weightChange",
        description="Delta between next_weight and current_weight in kg",
    )
    rest_seconds: int = Field(
        ...,
        validation_alias=AliasChoices("restSeconds", "rest_seconds"),
        serialization_alias="restSeconds",
        description="Prescribed rest interval in seconds between sets",
    )
    effective_volume: float = Field(
        ...,
        validation_alias=AliasChoices("effectiveVolume", "effective_volume"),
        serialization_alias="effectiveVolume",
        description="Stimulating hypertrophic tonnage where RPE >= 7.0 (kg)",
    )
    total_volume: float = Field(
        ...,
        validation_alias=AliasChoices("totalVolume", "total_volume"),
        serialization_alias="totalVolume",
        description="Total tonnage executed across all completed sets (kg)",
    )
    estimated_1rm: float = Field(
        ...,
        validation_alias=AliasChoices("estimated1rm", "estimated_1rm"),
        serialization_alias="estimated1rm",
        description="Peak estimated 1-Rep Max calculated via Epley formula (kg)",
    )
    avg_rpe: float = Field(
        ...,
        validation_alias=AliasChoices("avgRpe", "avg_rpe"),
        serialization_alias="avgRpe",
        description="Average Rate of Perceived Exertion across logged sets",
    )
    reps_failed: bool = Field(
        ...,
        validation_alias=AliasChoices("repsFailed", "reps_failed"),
        serialization_alias="repsFailed",
        description="Flag indicating if reps failed or high fatigue (RPE >= 9.5) occurred",
    )
    progression_action: str = Field(
        ...,
        validation_alias=AliasChoices("progressionAction", "progression_action"),
        serialization_alias="progressionAction",
        description="Action category: 'increment', 'linear_increment', or 'hold_and_extend_rest'",
    )
    scientific_rationale: str = Field(
        ...,
        validation_alias=AliasChoices("scientificRationale", "scientific_rationale"),
        serialization_alias="scientificRationale",
        description="Biomechanical and physiological justification for prescription",
    )
    is_upper_body: bool = Field(
        ...,
        validation_alias=AliasChoices("isUpperBody", "is_upper_body"),
        serialization_alias="isUpperBody",
    )


# ==============================================================================
# SECTION 12 ALGORITHMIC DOMAIN CONTRACTS: ADAPTIVE TDEE & NUTRITION
# ==============================================================================

class AdaptiveTDEEInput(BaseModel):
    """
    Input telemetry payload for the Adaptive TDEE Energy Expenditure Engine.
    """
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    weight_start: float = Field(
        ...,
        validation_alias=AliasChoices("weightStart", "weight_start"),
        serialization_alias="weightStart",
        description="Initial body weight at beginning of 7-day rolling window (kg)",
        gt=0.0,
    )
    weight_end: float = Field(
        ...,
        validation_alias=AliasChoices("weightEnd", "weight_end"),
        serialization_alias="weightEnd",
        description="Final body weight at end of 7-day rolling window (kg)",
        gt=0.0,
    )
    avg_daily_calories: float = Field(
        ...,
        validation_alias=AliasChoices("avgDailyCalories", "avg_daily_calories"),
        serialization_alias="avgDailyCalories",
        description="Average daily logged caloric intake over 7 days (kcal)",
        gt=0.0,
    )
    goal: Literal["Hypertrophy", "Strength", "FatLoss", "Endurance", "Maintenance"] = Field(
        "Hypertrophy",
        description="Athletic adaptation objective governing caloric adjustment and protein requirements",
    )
    body_weight: Optional[float] = Field(
        None,
        validation_alias=AliasChoices("bodyWeight", "body_weight"),
        serialization_alias="bodyWeight",
        description="Current body weight for macro multipliers (defaults to weight_end)",
        gt=0.0,
    )
    target_surplus_deficit_kcal: Optional[float] = Field(
        None,
        validation_alias=AliasChoices("targetSurplusDeficitKcal", "target_surplus_deficit_kcal"),
        serialization_alias="targetSurplusDeficitKcal",
        description="Optional manual override for caloric surplus (+) or deficit (-) in kcal",
    )
    protein_g_per_kg: Optional[float] = Field(
        None,
        validation_alias=AliasChoices("proteinGPerKg", "protein_g_per_kg"),
        serialization_alias="proteinGPerKg",
        description="Optional custom protein target in grams per kg of body weight",
        gt=0.0,
    )
    fat_g_per_kg: Optional[float] = Field(
        None,
        validation_alias=AliasChoices("fatGPerKg", "fat_g_per_kg"),
        serialization_alias="fatGPerKg",
        description="Optional custom fat target in grams per kg of body weight",
        gt=0.0,
    )


class AdaptiveTDEEOutput(BaseModel):
    """
    Output prescription payload generated by the Adaptive TDEE Engine.
    """
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    delta_w: float = Field(
        ...,
        validation_alias=AliasChoices("deltaW", "delta_w"),
        serialization_alias="deltaW",
        description="Weekly rolling body mass change: weight_end - weight_start (kg)",
    )
    daily_caloric_imbalance: float = Field(
        ...,
        validation_alias=AliasChoices("dailyCaloricImbalance", "daily_caloric_imbalance"),
        serialization_alias="dailyCaloricImbalance",
        description="Daily caloric imbalance = (delta_w * 7700.0) / 7.0 (kcal)",
    )
    true_tdee: float = Field(
        ...,
        validation_alias=AliasChoices("trueTdee", "true_tdee"),
        serialization_alias="trueTdee",
        description="True metabolic expenditure = avg_daily_calories - daily_caloric_imbalance (kcal)",
    )
    target_calories: float = Field(
        ...,
        validation_alias=AliasChoices("targetCalories", "target_calories"),
        serialization_alias="targetCalories",
        description="Prescribed daily caloric intake based on metabolic rate and goal (kcal)",
    )
    target_protein_grams: float = Field(
        ...,
        validation_alias=AliasChoices("targetProteinGrams", "target_protein_grams"),
        serialization_alias="targetProteinGrams",
        description="Target daily dietary protein intake (grams)",
    )
    target_carb_grams: float = Field(
        ...,
        validation_alias=AliasChoices("targetCarbGrams", "target_carb_grams"),
        serialization_alias="targetCarbGrams",
        description="Target daily dietary carbohydrate intake (grams)",
    )
    target_fat_grams: float = Field(
        ...,
        validation_alias=AliasChoices("targetFatGrams", "target_fat_grams"),
        serialization_alias="targetFatGrams",
        description="Target daily dietary fat intake (grams)",
    )
    protein_calories: float = Field(
        ...,
        validation_alias=AliasChoices("proteinCalories", "protein_calories"),
        serialization_alias="proteinCalories",
        description="Caloric contribution from protein (protein_grams * 4.0 kcal)",
    )
    carb_calories: float = Field(
        ...,
        validation_alias=AliasChoices("carbCalories", "carb_calories"),
        serialization_alias="carbCalories",
        description="Caloric contribution from carbohydrates (carb_grams * 4.0 kcal)",
    )
    fat_calories: float = Field(
        ...,
        validation_alias=AliasChoices("fatCalories", "fat_calories"),
        serialization_alias="fatCalories",
        description="Caloric contribution from fats (fat_grams * 9.0 kcal)",
    )
    protein_g_per_kg: float = Field(
        ...,
        validation_alias=AliasChoices("proteinGPerKg", "protein_g_per_kg"),
        serialization_alias="proteinGPerKg",
        description="Applied protein multiplier (g/kg body weight)",
    )
    fat_g_per_kg: float = Field(
        ...,
        validation_alias=AliasChoices("fatGPerKg", "fat_g_per_kg"),
        serialization_alias="fatGPerKg",
        description="Applied dietary fat multiplier (g/kg body weight)",
    )
    goal: str = Field(
        ...,
        description="Active fitness adaptation objective",
    )
    scientific_rationale: str = Field(
        ...,
        validation_alias=AliasChoices("scientificRationale", "scientific_rationale"),
        serialization_alias="scientificRationale",
        description="Sports science rationale for the metabolic prescription and macro distribution",
    )
