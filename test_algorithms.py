"""
Verification Test Suite for Sports Science & Algorithmic Engines
================================================================
Tests progressive overload, Epley 1RM, volume calculations,
adaptive TDEE, macronutrient partitioning, and Pydantic v2 schemas.
"""

import sys
import math
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from schemas import (
    ExerciseSetTarget,
    StructuredWorkoutExercise,
    StructuredWorkoutDay,
    MasterPlanPayload,
    LoggedSet,
    ProgressiveOverloadInput,
    ProgressiveOverloadOutput,
    AdaptiveTDEEInput,
    AdaptiveTDEEOutput,
)
from algorithms.overload import (
    calculate_epley_1rm,
    round_to_nearest,
    calculate_effective_volume,
    calculate_total_volume,
    calculate_progressive_overload,
)
from algorithms.metabolic import (
    calculate_weight_delta,
    calculate_daily_caloric_imbalance,
    calculate_true_tdee,
    calculate_macro_split,
    calculate_adaptive_tdee,
)


def test_epley_1rm():
    """Verify exact Epley 1RM formula calculations."""
    print("Running test_epley_1rm...")
    # Formula: 1RM = Weight * (1 + Reps / 30.0)
    # 100 kg for 10 reps -> 100 * (1 + 10/30) = 133.333... -> 133.33 kg
    val1 = calculate_epley_1rm(100.0, 10)
    assert abs(val1 - 133.33) < 0.01, f"Expected 133.33, got {val1}"

    # 140 kg for 5 reps -> 140 * (1 + 5/30) = 140 * 1.16666... = 163.33 kg
    val2 = calculate_epley_1rm(140.0, 5)
    assert abs(val2 - 163.33) < 0.01, f"Expected 163.33, got {val2}"

    # 200 kg for 1 rep -> 200 * (1 + 1/30) = 206.67 kg
    val3 = calculate_epley_1rm(200.0, 1)
    assert abs(val3 - 206.67) < 0.01, f"Expected 206.67, got {val3}"

    # Zero or negative handling
    assert calculate_epley_1rm(0.0, 10) == 0.0
    assert calculate_epley_1rm(100.0, 0) == 0.0
    print("[PASS] test_epley_1rm passed")


def test_rounding_increments():
    """Verify rounding to nearest plate increments (0.5kg and 2.5kg)."""
    print("Running test_rounding_increments...")
    assert round_to_nearest(102.3, 0.5) == 102.5
    assert round_to_nearest(102.1, 0.5) == 102.0
    assert round_to_nearest(101.25, 0.5) == 101.5
    assert round_to_nearest(142.0, 2.5) == 142.5
    assert round_to_nearest(141.0, 2.5) == 140.0
    assert round_to_nearest(147.0, 2.5) == 147.5
    print("[PASS] test_rounding_increments passed")


def test_progressive_overload_upper_body_low_rpe():
    """Verify upper-body overload (+2.5%) when average RPE <= 7.0."""
    print("Running test_progressive_overload_upper_body_low_rpe...")
    # Athlete bench presses 100kg for 3 sets of 8 with easy RPEs
    input_payload = ProgressiveOverloadInput(
        exercise_id="bench-press-1",
        exercise_name="Barbell Flat Bench Press",
        is_upper_body=True,
        current_weight=100.0,
        current_rest_seconds=90,
        rounding_increment=0.5,
        sets=[
            LoggedSet(set_number=1, reps=8, weight=100.0, rpe=6.5),
            LoggedSet(set_number=2, reps=8, weight=100.0, rpe=7.0),
            LoggedSet(set_number=3, reps=8, weight=100.0, rpe=7.0),
        ],
    )
    result = calculate_progressive_overload(input_payload)

    # avg_rpe = (6.5 + 7.0 + 7.0) / 3 = 6.83 <= 7.0
    assert result.avg_rpe == 6.83
    assert result.reps_failed is False
    assert result.progression_action == "increment"
    # 100 * 1.025 = 102.5 kg
    assert result.next_weight == 102.5
    assert result.weight_change == 2.5
    assert result.rest_seconds == 90
    # Effective volume (RPE >= 7.0): Set 2 (8*100) + Set 3 (8*100) = 1600kg
    assert result.effective_volume == 1600.0
    # Total volume: 800 + 800 + 800 = 2400kg
    assert result.total_volume == 2400.0
    # Estimated 1RM from 100kg x 8 reps = 100 * (1 + 8/30) = 126.67kg
    assert abs(result.estimated_1rm - 126.67) < 0.01
    print("[PASS] test_progressive_overload_upper_body_low_rpe passed")


def test_progressive_overload_lower_body_low_rpe():
    """Verify lower-body overload (+5.0%) when average RPE <= 7.0."""
    print("Running test_progressive_overload_lower_body_low_rpe...")
    # Athlete squats 140kg for 3 sets of 6 with RPE <= 7.0
    input_payload = ProgressiveOverloadInput(
        exercise_id="barbell-squat-1",
        exercise_name="Barbell Back Squat",
        is_upper_body=False,  # Lower body
        current_weight=140.0,
        current_rest_seconds=150,
        rounding_increment=2.5,
        sets=[
            LoggedSet(set_number=1, reps=6, weight=140.0, rpe=6.5),
            LoggedSet(set_number=2, reps=6, weight=140.0, rpe=6.5),
            LoggedSet(set_number=3, reps=6, weight=140.0, rpe=7.0),
        ],
    )
    result = calculate_progressive_overload(input_payload)

    # avg_rpe = (6.5 + 6.5 + 7.0) / 3 = 6.67 <= 7.0
    assert result.avg_rpe == 6.67
    assert result.reps_failed is False
    assert result.progression_action == "increment"
    # 140 * 1.05 = 147.0 kg, rounded to nearest 2.5kg -> 147.5 kg
    assert result.next_weight == 147.5
    assert result.weight_change == 7.5
    assert result.rest_seconds == 150
    print("[PASS] test_progressive_overload_lower_body_low_rpe passed")


def test_progressive_overload_linear_optimal_rpe():
    """Verify linear progression (+1.0125 or +1.0kg) when RPE in 7.5-9.0 range."""
    print("Running test_progressive_overload_linear_optimal_rpe...")
    input_payload = ProgressiveOverloadInput(
        exercise_id="ohp-1",
        exercise_name="Standing Overhead Press",
        is_upper_body=True,
        current_weight=60.0,
        current_rest_seconds=90,
        rounding_increment=0.5,
        sets=[
            LoggedSet(set_number=1, reps=8, weight=60.0, rpe=8.0),
            LoggedSet(set_number=2, reps=8, weight=60.0, rpe=8.5),
            LoggedSet(set_number=3, reps=8, weight=60.0, rpe=8.5),
        ],
    )
    result = calculate_progressive_overload(input_payload)

    # avg_rpe = (8.0 + 8.5 + 8.5) / 3 = 8.33
    assert result.avg_rpe == 8.33
    assert result.reps_failed is False
    assert result.progression_action == "linear_increment"
    # 60 * 1.0125 = 60.75 -> rounded to 0.5kg is 61.0kg (+1.0kg)
    assert result.next_weight == 61.0
    assert result.weight_change == 1.0
    assert result.rest_seconds == 90
    print("[PASS] test_progressive_overload_linear_optimal_rpe passed")


def test_progressive_overload_high_rpe_plateau():
    """Verify load lock and rest timer extension (+30s) when RPE >= 9.5."""
    print("Running test_progressive_overload_high_rpe_plateau...")
    input_payload = ProgressiveOverloadInput(
        exercise_id="deadlift-1",
        exercise_name="Conventional Deadlift",
        is_upper_body=False,
        current_weight=180.0,
        current_rest_seconds=180,
        rounding_increment=2.5,
        sets=[
            LoggedSet(set_number=1, reps=5, weight=180.0, rpe=9.5),
            LoggedSet(set_number=2, reps=5, weight=180.0, rpe=9.5),
            LoggedSet(set_number=3, reps=5, weight=180.0, rpe=10.0),
        ],
    )
    result = calculate_progressive_overload(input_payload)

    # avg_rpe = (9.5 + 9.5 + 10.0) / 3 = 9.67 >= 9.5
    assert result.avg_rpe == 9.67
    assert result.progression_action == "hold_and_extend_rest"
    # Weight locked at 180.0kg
    assert result.next_weight == 180.0
    assert result.weight_change == 0.0
    # Rest extended by 30 seconds: 180 + 30 = 210s
    assert result.rest_seconds == 210
    print("[PASS] test_progressive_overload_high_rpe_plateau passed")


def test_progressive_overload_failed_reps():
    """Verify load lock and rest timer extension (+30s) when athlete fails target reps."""
    print("Running test_progressive_overload_failed_reps...")
    input_payload = ProgressiveOverloadInput(
        exercise_id="incline-db-1",
        exercise_name="Incline DB Press",
        is_upper_body=True,
        current_weight=36.0,
        current_rest_seconds=90,
        rounding_increment=1.0,
        sets=[
            LoggedSet(set_number=1, reps=10, weight=36.0, rpe=8.0, target_reps=10),
            LoggedSet(set_number=2, reps=10, weight=36.0, rpe=8.5, target_reps=10),
            LoggedSet(set_number=3, reps=7, weight=36.0, rpe=9.0, target_reps=10, failed=True),
        ],
    )
    result = calculate_progressive_overload(input_payload)

    assert result.reps_failed is True
    assert result.progression_action == "hold_and_extend_rest"
    assert result.next_weight == 36.0
    assert result.weight_change == 0.0
    assert result.rest_seconds == 120  # 90 + 30
    print("[PASS] test_progressive_overload_failed_reps passed")


def test_volume_calculations():
    """Verify effective stimulating volume vs total volume formulas."""
    print("Running test_volume_calculations...")
    sets = [
        # Warmup set @ RPE 5.5: 10 reps @ 60kg = 600kg (not stimulating)
        LoggedSet(set_number=1, reps=10, weight=60.0, rpe=5.5),
        # Working set 1 @ RPE 7.5: 8 reps @ 100kg = 800kg
        LoggedSet(set_number=2, reps=8, weight=100.0, rpe=7.5),
        # Working set 2 @ RPE 8.5: 8 reps @ 100kg = 800kg
        LoggedSet(set_number=3, reps=8, weight=100.0, rpe=8.5),
    ]
    eff_vol = calculate_effective_volume(sets)
    tot_vol = calculate_total_volume(sets)

    # Effective volume should only include sets with RPE >= 7.0: 800 + 800 = 1600 kg
    assert eff_vol == 1600.0, f"Expected 1600.0, got {eff_vol}"
    # Total volume includes all sets: 600 + 800 + 800 = 2200 kg
    assert tot_vol == 2200.0, f"Expected 2200.0, got {tot_vol}"
    print("[PASS] test_volume_calculations passed")


def test_adaptive_tdee_hypertrophy():
    """
    Verify Adaptive TDEE Engine in a Hypertrophy scenario:
    Athlete gained 0.2kg eating 2800 kcal/day.
    - delta_w = +0.2 kg
    - daily_caloric_imbalance = (0.2 * 7700.0) / 7.0 = +220.0 kcal
    - true_tdee = 2800.0 - 220.0 = 2580.0 kcal
    - Target for Hypertrophy (+300 kcal) = 2880.0 kcal
    - Protein: 2.2 g/kg
    """
    print("Running test_adaptive_tdee_hypertrophy...")
    delta_w = calculate_weight_delta(80.0, 80.2)
    assert delta_w == 0.2

    imbalance = calculate_daily_caloric_imbalance(delta_w)
    assert imbalance == 220.0

    true_tdee = calculate_true_tdee(2800.0, imbalance)
    assert true_tdee == 2580.0

    input_data = AdaptiveTDEEInput(
        weight_start=80.0,
        weight_end=80.2,
        avg_daily_calories=2800.0,
        goal="Hypertrophy",
    )
    result = calculate_adaptive_tdee(input_data)

    assert result.delta_w == 0.2
    assert result.daily_caloric_imbalance == 220.0
    assert result.true_tdee == 2580.0
    assert result.target_calories == 2880.0
    # Protein: 2.2g/kg * 80.2kg = 176.4g
    assert result.protein_g_per_kg == 2.2
    assert result.target_protein_grams == 176.4
    assert result.protein_calories == round(176.4 * 4.0, 1)  # 705.6 kcal
    # Fat: 1.0g/kg * 80.2kg = 80.2g
    assert result.target_fat_grams == 80.2
    assert result.fat_calories == round(80.2 * 9.0, 1)  # 721.8 kcal
    # Carbs: remainder = 2880.0 - 705.6 - 721.8 = 1452.6 kcal / 4.0 = 363.2g
    assert abs(result.target_carb_grams - 363.2) <= 0.2
    print("[PASS] test_adaptive_tdee_hypertrophy passed")


def test_adaptive_tdee_fatloss():
    """
    Verify Adaptive TDEE Engine in a FatLoss scenario:
    Athlete lost 0.6kg eating 2200 kcal/day.
    - delta_w = -0.6 kg
    - daily_caloric_imbalance = (-0.6 * 7700.0) / 7.0 = -660.0 kcal
    - true_tdee = 2200.0 - (-660.0) = 2860.0 kcal
    - Target for FatLoss (-500 kcal) = 2360.0 kcal
    - Protein: 2.4 g/kg (elevated to spare lean mass)
    """
    print("Running test_adaptive_tdee_fatloss...")
    delta_w = calculate_weight_delta(90.0, 89.4)
    assert abs(delta_w - (-0.6)) < 0.001

    imbalance = calculate_daily_caloric_imbalance(delta_w)
    assert imbalance == -660.0

    true_tdee = calculate_true_tdee(2200.0, imbalance)
    assert true_tdee == 2860.0

    input_data = AdaptiveTDEEInput(
        weight_start=90.0,
        weight_end=89.4,
        avg_daily_calories=2200.0,
        goal="FatLoss",
    )
    result = calculate_adaptive_tdee(input_data)

    assert result.delta_w == -0.6
    assert result.daily_caloric_imbalance == -660.0
    assert result.true_tdee == 2860.0
    assert result.target_calories == 2360.0
    # Protein: 2.4g/kg * 89.4kg = 214.6g
    assert result.protein_g_per_kg == 2.4
    assert result.target_protein_grams == 214.6
    assert result.protein_calories == round(214.6 * 4.0, 1)  # 858.4 kcal
    # Fat: 0.8g/kg * 89.4kg = 71.5g
    assert result.target_fat_grams == 71.5
    assert result.fat_calories == round(71.5 * 9.0, 1)  # 643.5 kcal
    # Carbs: remainder = 2360.0 - 858.4 - 643.5 = 858.1 kcal / 4.0 = 214.5g
    assert abs(result.target_carb_grams - 214.5) <= 0.2
    print("[PASS] test_adaptive_tdee_fatloss passed")


def test_adaptive_tdee_maintenance():
    """Verify Adaptive TDEE Engine when mass is stable."""
    print("Running test_adaptive_tdee_maintenance...")
    input_data = AdaptiveTDEEInput(
        weight_start=75.0,
        weight_end=75.0,
        avg_daily_calories=2500.0,
        goal="Maintenance",
    )
    result = calculate_adaptive_tdee(input_data)

    assert result.delta_w == 0.0
    assert result.daily_caloric_imbalance == 0.0
    assert result.true_tdee == 2500.0
    assert result.target_calories == 2500.0
    assert result.protein_g_per_kg == 2.0
    assert result.target_protein_grams == 150.0  # 2.0 * 75.0
    print("[PASS] test_adaptive_tdee_maintenance passed")


def test_pydantic_v2_schemas_section_15():
    """Verify strict validation and dual camelCase/snake_case serialization."""
    print("Running test_pydantic_v2_schemas_section_15...")
    # 1. ExerciseSetTarget camelCase input (NestJS / TypeScript format)
    raw_set = {
        "setNumber": 1,
        "reps": 8,
        "targetRpe": 8.0,
        "suggestedWeightKg": 100.0,
        "restSeconds": 90,
    }
    set_obj = ExerciseSetTarget.model_validate(raw_set)
    assert set_obj.set_number == 1
    assert set_obj.reps == 8
    assert set_obj.target_rpe == 8.0
    assert set_obj.suggested_weight_kg == 100.0
    assert set_obj.rest_seconds == 90

    # Serialization with camelCase aliases
    serialized = set_obj.model_dump(by_alias=True)
    assert serialized["setNumber"] == 1
    assert serialized["targetRpe"] == 8.0
    assert serialized["suggestedWeightKg"] == 100.0
    assert serialized["restSeconds"] == 90

    # 2. StructuredWorkoutDay and MasterPlanPayload
    plan_raw = {
        "userId": "user-elite-007",
        "cycleWeeks": 8,
        "goal": "Hypertrophy",
        "splitType": "UpperLower",
        "scientificRationale": "Mesocycle targeting mechanical tension and progressive volume accumulation.",
        "weeklySchedule": [
            {
                "dayName": "Upper Body Heavy",
                "focusMuscleGroups": ["Chest", "Upper Back", "Deltoids"],
                "warmupMinutes": 10,
                "exercises": [
                    {
                        "exerciseId": "ex-bench-01",
                        "exerciseName": "Barbell Incline Press",
                        "techniqueCue": "Retract scapulae and drive through midfoot",
                        "targetSets": [raw_set],
                    }
                ],
            }
        ],
    }
    plan = MasterPlanPayload.model_validate(plan_raw)
    assert plan.user_id == "user-elite-007"
    assert plan.cycle_weeks == 8
    assert plan.goal == "Hypertrophy"
    assert plan.split_type == "UpperLower"
    assert len(plan.weekly_schedule) == 1
    assert plan.weekly_schedule[0].exercises[0].exercise_name == "Barbell Incline Press"

    dumped_plan = plan.model_dump(by_alias=True)
    assert dumped_plan["userId"] == "user-elite-007"
    assert dumped_plan["cycleWeeks"] == 8
    assert dumped_plan["splitType"] == "UpperLower"
    assert dumped_plan["weeklySchedule"][0]["exercises"][0]["targetSets"][0]["setNumber"] == 1
    print("[PASS] test_pydantic_v2_schemas_section_15 passed")


def test_api_endpoints():
    """Verify HTTP endpoint routing for overload and tdee using FastAPI TestClient."""
    print("Running test_api_endpoints...")
    from starlette.testclient import TestClient
    from main import app
    client = TestClient(app)

    # 1. Test /algorithms/overload with camelCase payload
    overload_payload = {
        "exerciseId": "bench-press-1",
        "exerciseName": "Barbell Flat Bench Press",
        "isUpperBody": True,
        "currentWeight": 100.0,
        "currentRestSeconds": 90,
        "roundingIncrement": 0.5,
        "sets": [
            {"setNumber": 1, "reps": 8, "weight": 100.0, "rpe": 6.5},
            {"setNumber": 2, "reps": 8, "weight": 100.0, "rpe": 7.0},
            {"setNumber": 3, "reps": 8, "weight": 100.0, "rpe": 7.0},
        ],
    }
    resp = client.post("/algorithms/overload", json=overload_payload)
    assert resp.status_code == 200, f"Overload failed: {resp.text}"
    data = resp.json()
    assert data["nextWeight"] == 102.5
    assert data["weightChange"] == 2.5
    assert data["progressionAction"] == "increment"
    assert data["effectiveVolume"] == 1600.0

    # 2. Test /algorithms/tdee with camelCase payload
    tdee_payload = {
        "weightStart": 80.0,
        "weightEnd": 80.2,
        "avgDailyCalories": 2800.0,
        "goal": "Hypertrophy",
    }
    resp_tdee = client.post("/algorithms/tdee", json=tdee_payload)
    assert resp_tdee.status_code == 200, f"TDEE failed: {resp_tdee.text}"
    data_tdee = resp_tdee.json()
    assert data_tdee["deltaW"] == 0.2
    assert data_tdee["trueTdee"] == 2580.0
    assert data_tdee["targetCalories"] == 2880.0
    assert data_tdee["proteinGPerKg"] == 2.2
    print("[PASS] test_api_endpoints passed")


def run_all_tests():
    print("==================================================")
    print("STARTING SPORTS SCIENCE & ALGORITHMIC ENGINE TESTS")
    print("==================================================")
    test_epley_1rm()
    test_rounding_increments()
    test_progressive_overload_upper_body_low_rpe()
    test_progressive_overload_lower_body_low_rpe()
    test_progressive_overload_linear_optimal_rpe()
    test_progressive_overload_high_rpe_plateau()
    test_progressive_overload_failed_reps()
    test_volume_calculations()
    test_adaptive_tdee_hypertrophy()
    test_adaptive_tdee_fatloss()
    test_adaptive_tdee_maintenance()
    test_pydantic_v2_schemas_section_15()
    test_api_endpoints()
    print("==================================================")
    print("ALL 13 TESTS PASSED PERFECTLY WITH 100% PRECISION!")
    print("==================================================")


if __name__ == "__main__":
    run_all_tests()
