"""
Verification Test Suite for SmartCoachAI Structured Plan Generator
===================================================================
Tests PlanGenerator, Instructor integration, Pydantic v2 MasterPlanPayload
contracts, progressive overload algorithms, and adaptive nutrition planning.
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from generator import PlanGenerator
from schemas import (
    MasterPlanPayload,
    StructuredWorkoutDay,
    StructuredWorkoutExercise,
    ExerciseSetTarget,
)


def test_generator_initialization():
    """Verify PlanGenerator loads knowledge bases cleanly."""
    print("\n--- Test 1: PlanGenerator Initialization ---")
    gen = PlanGenerator()
    assert len(gen.exercises) >= 800, f"Expected >= 800 exercises, got {len(gen.exercises)}"
    assert len(gen.meals) >= 20, f"Expected >= 20 meals, got {len(gen.meals)}"
    assert "chest" in gen.exercises_by_muscle
    assert "quadriceps" in gen.exercises_by_muscle
    print(f"[PASS] Successfully loaded {len(gen.exercises)} exercises and {len(gen.meals)} meal profiles.")


def test_generate_master_plan_upper_lower():
    """Verify generate_master_plan produces a valid MasterPlanPayload for UpperLower split."""
    print("\n--- Test 2: Master Plan UpperLower Split (Hypertrophy) ---")
    gen = PlanGenerator()
    plan = gen.generate_master_plan(
        user_id="athlete_001",
        goal="Hypertrophy",
        fitness_level="intermediate",
        split_type="UpperLower",
        cycle_weeks=4,
    )

    # 1. Type and Pydantic validation
    assert isinstance(plan, MasterPlanPayload), "Result must be an instance of MasterPlanPayload"
    assert plan.user_id == "athlete_001"
    assert plan.goal == "Hypertrophy"
    assert plan.split_type == "UpperLower"
    assert plan.cycle_weeks == 4

    # 2. Schedule structure: 4 days
    assert len(plan.weekly_schedule) == 4, f"UpperLower split must have 4 days, got {len(plan.weekly_schedule)}"
    day_names = [d.day_name for d in plan.weekly_schedule]
    assert any("Upper Body A" in name for name in day_names)
    assert any("Lower Body A" in name for name in day_names)
    assert any("Upper Body B" in name for name in day_names)
    assert any("Lower Body B" in name for name in day_names)

    # 3. Exercises and set parameters
    for day in plan.weekly_schedule:
        assert len(day.exercises) >= 4, f"Each day should have >= 4 exercises, got {len(day.exercises)}"
        assert day.warmup_minutes >= 5
        for ex in day.exercises:
            assert ex.exercise_id, "Exercise ID must not be empty"
            assert ex.exercise_name, "Exercise name must not be empty"
            assert ex.technique_cue, "Technique cue must not be empty"
            assert len(ex.target_sets) >= 3, "Each exercise must have >= 3 sets"
            for s in ex.target_sets:
                assert s.set_number >= 1
                assert 6 <= s.reps <= 15
                assert 1.0 <= s.target_rpe <= 10.0
                assert s.suggested_weight_kg is not None and s.suggested_weight_kg >= 0.0
                assert s.rest_seconds >= 30

    # 4. Scientific rationale presence
    assert len(plan.scientific_rationale) > 50, "Scientific rationale must be substantive"
    assert "progressive overload" in plan.scientific_rationale.lower()

    # 5. Serialization and re-validation via Pydantic v2
    dumped_alias = plan.model_dump(by_alias=True)
    revalidated = MasterPlanPayload.model_validate(dumped_alias)
    assert revalidated.user_id == plan.user_id
    assert len(revalidated.weekly_schedule) == 4

    print(f"[PASS] UpperLower MasterPlanPayload generated and validated with {len(plan.weekly_schedule)} days.")


def test_generate_master_plan_push_pull_legs():
    """Verify generate_master_plan for PushPullLegs (Strength, Advanced)."""
    print("\n--- Test 3: Master Plan PushPullLegs Split (Strength, Advanced) ---")
    gen = PlanGenerator()
    plan = gen.generate_master_plan(
        user_id="athlete_strength",
        goal="Strength",
        fitness_level="advanced",
        split_type="PushPullLegs",
        cycle_weeks=6,
    )

    assert isinstance(plan, MasterPlanPayload)
    assert plan.goal == "Strength"
    assert plan.split_type == "PushPullLegs"
    assert plan.cycle_weeks == 6
    assert len(plan.weekly_schedule) == 3

    # Check strength programming: lower reps (4-8), longer rest (120-180s)
    for day in plan.weekly_schedule:
        for ex in day.exercises:
            assert len(ex.target_sets) == 4, "Advanced strength programming should prescribe 4 sets"
            first_set = ex.target_sets[0]
            assert first_set.reps <= 8, f"Strength reps should be <= 8, got {first_set.reps}"
            assert first_set.rest_seconds >= 120, f"Strength rest should be >= 120s, got {first_set.rest_seconds}"

    print(f"[PASS] PushPullLegs Strength plan validated with 3 training sessions and heavy load rest periods.")


def test_generate_master_plan_full_body():
    """Verify generate_master_plan for FullBody (FatLoss, Beginner)."""
    print("\n--- Test 4: Master Plan FullBody Split (FatLoss, Beginner) ---")
    gen = PlanGenerator()
    plan = gen.generate_master_plan(
        user_id="athlete_beginner",
        goal="FatLoss",
        fitness_level="beginner",
        split_type="FullBody",
        cycle_weeks=4,
    )

    assert isinstance(plan, MasterPlanPayload)
    assert plan.goal == "FatLoss"
    assert plan.split_type == "FullBody"
    assert len(plan.weekly_schedule) == 3

    # Beginner parameters: 10 min warmup, 3 working sets
    for day in plan.weekly_schedule:
        assert day.warmup_minutes == 10
        for ex in day.exercises:
            assert len(ex.target_sets) == 3
            first_set = ex.target_sets[0]
            assert first_set.reps >= 10, f"FatLoss reps should be >= 10, got {first_set.reps}"
            assert first_set.rest_seconds <= 90, f"FatLoss rest should be <= 90s, got {first_set.rest_seconds}"

    print(f"[PASS] FullBody FatLoss plan validated with 10 min warmup and metabolic rest timers.")


def test_equipment_and_injury_filtering():
    """Verify safety guard filters out contraindications and restricts equipment."""
    print("\n--- Test 5: Equipment and Injury Filtering ---")
    gen = PlanGenerator()
    plan = gen.generate_master_plan(
        user_id="athlete_injured",
        goal="Hypertrophy",
        fitness_level="intermediate",
        split_type="UpperLower",
        equipment=["dumbbell", "body only"],
        injuries=["knee", "shoulder"],
    )

    for day in plan.weekly_schedule:
        for ex in day.exercises:
            name_lower = ex.exercise_name.lower()
            # Knee contraindications check
            assert "sissy squat" not in name_lower, f"Contraindicated knee exercise found: {ex.exercise_name}"
            # Shoulder contraindications check
            assert "behind the neck" not in name_lower, f"Contraindicated shoulder exercise found: {ex.exercise_name}"
            # Check equipment restriction in DB
            db_ex = gen.exercises_by_id.get(ex.exercise_id)
            if db_ex:
                eq = str(db_ex.get("equipment", "")).lower()
                assert eq in ["dumbbell", "body only", "none", "other", "bands"], f"Disallowed equipment: {eq}"

    print("[PASS] Successfully enforced equipment boundaries and bypassed knee/shoulder contraindications.")


def test_llm_fallback_resilience():
    """Verify generator falls back gracefully when invalid LLM key is provided."""
    print("\n--- Test 6: LLM Fallback Resilience ---")
    gen = PlanGenerator(gemini_api_key="invalid_test_key")
    plan = gen.generate_master_plan(
        user_id="athlete_fallback",
        goal="Hypertrophy",
        fitness_level="intermediate",
    )
    assert isinstance(plan, MasterPlanPayload), "Fallback must yield valid MasterPlanPayload"
    assert plan.user_id == "athlete_fallback"
    assert len(plan.weekly_schedule) == 4
    print("[PASS] Graceful fallback to deterministic engine verified on LLM failure.")


def test_generate_nutrition_plan_hypertrophy():
    """Verify generate_nutrition_plan calculates TDEE surplus and macro splits for Hypertrophy."""
    print("\n--- Test 7: Nutrition Plan (Hypertrophy) ---")
    gen = PlanGenerator()
    nut = gen.generate_nutrition_plan(
        user_id="athlete_nut_01",
        goal="Hypertrophy",
        current_weight_kg=80.0,
        target_weight_kg=84.0,
    )

    # 1. Verification of top-level contracts
    assert nut["user_id"] == "athlete_nut_01"
    assert nut["goal"] == "Hypertrophy"
    assert nut["current_weight_kg"] == 80.0
    assert nut["target_weight_kg"] == 84.0

    # 2. Baseline TDEE: 80 * 33 = 2640; Surplus: +300 -> 2940 kcal
    expected_cals = round(80.0 * 33.0 + 300.0, 1)
    assert nut["target_calories"] == expected_cals, f"Expected {expected_cals}, got {nut['target_calories']}"

    # 3. Protein target: 2.2 g/kg * 80 kg = 176.0 g
    assert nut["target_protein_grams"] == 176.0
    assert nut["macro_split"]["protein"]["g_per_kg"] == 2.2

    # 4. Fat target: 1.0 g/kg * 80 kg = 80.0 g
    assert nut["target_fat_grams"] == 80.0
    assert nut["macro_split"]["fat"]["g_per_kg"] == 1.0

    # 5. Carbohydrate remainder
    assert nut["target_carb_grams"] > 0
    calculated_cals = (
        nut["target_protein_grams"] * 4.0
        + nut["target_fat_grams"] * 9.0
        + nut["target_carb_grams"] * 4.0
    )
    assert abs(calculated_cals - nut["target_calories"]) < 10.0

    # 6. Structured meals
    assert len(nut["meals"]) == 4, f"Expected 4 meals, got {len(nut['meals'])}"
    meal_types = [m["meal_type"] for m in nut["meals"]]
    assert "breakfast" in meal_types
    assert "lunch" in meal_types
    assert "dinner" in meal_types
    assert "snack" in meal_types

    assert nut["total_meal_calories"] > 0
    assert nut["total_meal_protein_grams"] > 0
    assert len(nut["scientific_rationale"]) > 50

    print(f"[PASS] Nutrition plan generated: {nut['target_calories']} kcal ({nut['target_protein_grams']}g protein, {nut['target_carb_grams']}g carbs, {nut['target_fat_grams']}g fat).")


def test_generate_nutrition_plan_fat_loss_vegetarian():
    """Verify generate_nutrition_plan for FatLoss with vegetarian restriction."""
    print("\n--- Test 8: Nutrition Plan (FatLoss & Vegetarian) ---")
    gen = PlanGenerator()
    nut = gen.generate_nutrition_plan(
        user_id="athlete_veg",
        goal="FatLoss",
        current_weight_kg=90.0,
        target_weight_kg=82.0,
        dietary_preferences=["vegetarian"],
    )

    # Deficit: 90 * 33 = 2970; Deficit: -500 -> 2470 kcal
    expected_cals = round(90.0 * 33.0 - 500.0, 1)
    assert nut["target_calories"] == expected_cals
    # Protein in hypocaloric conditions: 2.4 g/kg * 90 = 216.0 g
    assert nut["target_protein_grams"] == 216.0

    # Verify vegetarian ingredients in selected meals
    meat_tokens = ["chicken", "beef", "turkey", "salmon", "cod", "tuna", "meatballs"]
    for meal in nut["meals"]:
        ing_text = " ".join(meal["ingredients"]).lower() + " " + meal["meal_name"].lower()
        for meat in meat_tokens:
            assert meat not in ing_text, f"Vegetarian plan contained {meat} in {meal['meal_name']}"

    print(f"[PASS] Vegetarian FatLoss plan validated with {nut['target_calories']} kcal and no meat ingredients.")


def test_generate_nutrition_plan_custom_calories():
    """Verify manual caloric override."""
    print("\n--- Test 9: Nutrition Plan with Manual Caloric Override ---")
    gen = PlanGenerator()
    nut = gen.generate_nutrition_plan(
        user_id="athlete_custom",
        goal="Strength",
        current_weight_kg=75.0,
        target_weight_kg=78.0,
        daily_calories=3100,
    )
    assert nut["target_calories"] == 3100.0
    print(f"[PASS] Manual caloric override (3100 kcal) applied correctly.")


def test_typescript_blueprint_alias_conformance():
    """Verify serialization conforms exactly to BLUEPRINT.md Section 15 TypeScript interfaces."""
    print("\n--- Test 10: TypeScript BLUEPRINT.md Section 15 Conformance ---")
    gen = PlanGenerator()
    plan = gen.generate_master_plan(
        user_id="ts_contract_test",
        goal="Hypertrophy",
        fitness_level="intermediate",
        split_type="UpperLower",
    )

    dumped = plan.model_dump(by_alias=True)

    # MasterPlanPayload interface keys
    expected_root_keys = {"userId", "cycleWeeks", "goal", "splitType", "weeklySchedule", "scientificRationale"}
    assert expected_root_keys.issubset(dumped.keys()), f"Missing root keys: {expected_root_keys - set(dumped.keys())}"

    # StructuredWorkoutDay interface keys
    day = dumped["weeklySchedule"][0]
    expected_day_keys = {"dayName", "focusMuscleGroups", "warmupMinutes", "exercises"}
    assert expected_day_keys.issubset(day.keys()), f"Missing day keys: {expected_day_keys - set(day.keys())}"

    # StructuredWorkoutExercise interface keys
    ex = day["exercises"][0]
    expected_ex_keys = {"exerciseId", "exerciseName", "targetSets", "techniqueCue"}
    assert expected_ex_keys.issubset(ex.keys()), f"Missing exercise keys: {expected_ex_keys - set(ex.keys())}"

    # ExerciseSetTarget interface keys
    s = ex["targetSets"][0]
    expected_set_keys = {"setNumber", "reps", "targetRpe", "suggestedWeightKg", "restSeconds"}
    assert expected_set_keys.issubset(s.keys()), f"Missing set keys: {expected_set_keys - set(s.keys())}"

    print("[PASS] Full dual camelCase serialization exactly matches BLUEPRINT.md Section 15 TypeScript interfaces.")


def run_all_tests():
    """Run all verification tests."""
    print("=" * 70)
    print("SMARTCOACH AI - PLAN GENERATOR VERIFICATION SUITE")
    print("=" * 70)

    test_generator_initialization()
    test_generate_master_plan_upper_lower()
    test_generate_master_plan_push_pull_legs()
    test_generate_master_plan_full_body()
    test_equipment_and_injury_filtering()
    test_llm_fallback_resilience()
    test_generate_nutrition_plan_hypertrophy()
    test_generate_nutrition_plan_fat_loss_vegetarian()
    test_generate_nutrition_plan_custom_calories()
    test_typescript_blueprint_alias_conformance()

    print("\n" + "=" * 70)
    print("ALL 10 VERIFICATION TESTS PASSED SUCCESSFULLY (100% SUCCESS RATE)")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
