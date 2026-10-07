"""
Apex Athletic / SmartCoach AI - Phase 2 End-to-End Verification Suite
======================================================================
Tests all Phase 2 API endpoints via fastapi.testclient.TestClient:
1. Health check endpoint (/health)
2. Semantic exercise search (/exercises/search)
3. Semantic nutrition search (/nutrition/search)
4. Semantic RAG query chat (/rag/query)
5. Master plan generation (/generator/master-plan & /rag/master-plan)
6. Nutrition plan generation (/generator/nutrition-plan & /rag/nutrition-plan)
7. Progressive overload calculation (/algorithms/overload)
8. Adaptive TDEE calculation (/algorithms/tdee)
"""

import math
import sys
from fastapi.testclient import TestClient

from main import app
from schemas import (
    AdaptiveTDEEOutput,
    MasterPlanPayload,
    ProgressiveOverloadOutput,
)

client = TestClient(app)


def test_01_health_check():
    """Verify GET /health returns operational readiness for all subsystems."""
    print("\n" + "=" * 70)
    print("TEST 1: Health Check Endpoint (/health)")
    print("=" * 70)

    response = client.get("/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    print("Health response:", data)

    assert data.get("status") == "healthy", f"Status not healthy: {data.get('status')}"
    assert data.get("qdrant_connected") is True, "Qdrant not connected"
    assert data.get("vector_engine_ready") is True, "VectorEngine not ready"
    assert data.get("generator_ready") is True, "PlanGenerator not ready"
    print("[PASS] Test 1: All core subsystems report healthy & ready.")


def test_02_semantic_exercise_search():
    """Verify GET /exercises/search with dense vector search and metadata filters."""
    print("\n" + "=" * 70)
    print("TEST 2: Semantic Exercise Search (/exercises/search)")
    print("=" * 70)

    # 1. Search 'bench' with muscle='chest'
    response = client.get("/exercises/search?q=bench&muscle=chest&limit=5")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert data.get("count", 0) > 0, "No exercises returned for 'bench' with muscle='chest'"
    results = data.get("results", [])
    print(f"Returned {len(results)} exercises for 'bench' + chest:")
    for ex in results:
        print(f"  • {ex.get('name')} (Score: {ex.get('score'):.4f}, Primary: {ex.get('primaryMuscles')}, Eq: {ex.get('equipment')})")
        assert "score" in ex, "Exercise missing similarity score"
        assert "name" in ex, "Exercise missing name"

    # 2. Search with equipment and level filters
    response_quads = client.get("/exercises/search?muscle=quadriceps&equipment=barbell&level=beginner&limit=3")
    assert response_quads.status_code == 200
    quad_data = response_quads.json()
    assert quad_data.get("count", 0) > 0, "Expected barbell quad beginner exercises"
    print(f"Returned {quad_data['count']} filtered barbell quad exercises.")

    # 3. Validation error on empty parameters
    response_empty = client.get("/exercises/search")
    assert response_empty.status_code == 400, f"Expected 400 for empty search, got {response_empty.status_code}"
    print("[PASS] Test 2: Semantic exercise search and filtering verified successfully.")


def test_03_semantic_nutrition_search():
    """Verify GET /nutrition/search across meals and protocols."""
    print("\n" + "=" * 70)
    print("TEST 3: Semantic Nutrition Search (/nutrition/search)")
    print("=" * 70)

    # 1. Search 'high protein'
    response = client.get("/nutrition/search?q=high%20protein&limit=5")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert data.get("count", 0) > 0, "No nutrition documents returned for 'high protein'"
    results = data.get("results", [])
    print(f"Returned {len(results)} nutrition results for 'high protein':")
    for nut in results:
        print(f"  • {nut.get('name')} (Type: {nut.get('type')}, Score: {nut.get('score'):.4f})")
        assert "score" in nut, "Nutrition item missing similarity score"
        assert "name" in nut, "Nutrition item missing name"

    # 2. Validation error on missing/empty query
    response_empty = client.get("/nutrition/search?q=")
    assert response_empty.status_code == 400, f"Expected 400, got {response_empty.status_code}"
    print("[PASS] Test 3: Semantic nutrition search verified successfully.")


def test_04_rag_query_chat():
    """Verify POST /rag/query powered by vector search and contextual synthesis."""
    print("\n" + "=" * 70)
    print("TEST 4: Semantic RAG Query Chat (/rag/query)")
    print("=" * 70)

    payload = {
        "query": "How should I structure progressive overload for bench press and what should I eat for recovery?",
        "user_id": "athlete_101",
    }
    response = client.post("/rag/query", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()

    print("Response text snippet:", data.get("response")[:180], "...")
    print("Cited sources:", data.get("sources"))

    assert len(data.get("response", "")) > 25, "RAG response text is too short or empty"
    assert isinstance(data.get("sources"), list), "Sources should be a list"
    assert len(data.get("sources")) > 0, "Expected at least one retrieved source citation"

    # Validation: empty query rejected
    bad_res = client.post("/rag/query", json={"query": ""})
    assert bad_res.status_code == 400, f"Expected 400 for empty query, got {bad_res.status_code}"
    print("[PASS] Test 4: RAG query chat returns grounded answer with citations.")


def test_05_master_plan_generation():
    """Verify POST /generator/master-plan and alias /rag/master-plan."""
    print("\n" + "=" * 70)
    print("TEST 5: Master Plan Generation (/generator/master-plan & /rag/master-plan)")
    print("=" * 70)

    request_payload = {
        "user_id": "athlete_hypertrophy_01",
        "goal": "Hypertrophy",
        "fitness_level": "intermediate",
        "split_type": "UpperLower",
        "cycle_weeks": 4,
        "equipment": ["barbell", "dumbbell", "cable", "bench"],
        "injuries": ["shoulder_impingement"],
    }

    # 1. Primary endpoint: /generator/master-plan
    response = client.post("/generator/master-plan", json=request_payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()

    # Validate against strict Pydantic v2 MasterPlanPayload schema
    validated_plan = MasterPlanPayload.model_validate(data)
    assert validated_plan.goal == "Hypertrophy"
    assert len(validated_plan.weekly_schedule) == 4, f"Expected 4 days in weekly_schedule, got {len(validated_plan.weekly_schedule)}"

    # Verify structure of first training day and exercise
    day1 = validated_plan.weekly_schedule[0]
    assert day1.day_name, "Day missing day_name"
    assert len(day1.exercises) >= 3, f"Expected at least 3 exercises, got {len(day1.exercises)}"
    ex1 = day1.exercises[0]
    assert ex1.exercise_name, "Exercise missing exercise_name"
    assert len(ex1.target_sets) >= 2, "Exercise missing target_sets"
    print(f"  • Generated Plan: {validated_plan.cycle_weeks}-week {validated_plan.split_type} with {len(validated_plan.weekly_schedule)} training sessions.")
    print(f"  • Day 1 ({day1.day_name}): {len(day1.exercises)} exercises. First: {ex1.exercise_name} ({len(ex1.target_sets)} sets @ RPE {ex1.target_sets[0].target_rpe})")

    # 2. Alias endpoint: /rag/master-plan
    alias_payload = {
        "user_id": "athlete_strength_02",
        "goal": "Strength",
        "fitness_level": "advanced",
        "split_type": "PushPullLegs",
        "cycle_weeks": 6,
    }
    alias_res = client.post("/rag/master-plan", json=alias_payload)
    assert alias_res.status_code == 200, f"Expected 200 for alias /rag/master-plan, got {alias_res.status_code}"
    alias_validated = MasterPlanPayload.model_validate(alias_res.json())
    assert alias_validated.goal == "Strength"
    print(f"  • Alias /rag/master-plan: Validated {alias_validated.cycle_weeks}-week {alias_validated.split_type} plan.")

    # 3. Bad request validation
    bad_req = client.post("/generator/master-plan", json={"user_id": "", "goal": "Strength", "fitness_level": "beginner"})
    assert bad_req.status_code == 400, f"Expected 400 for empty user_id, got {bad_req.status_code}"
    print("[PASS] Test 5: MasterPlanPayload validated across both endpoints.")


def test_06_nutrition_plan_generation():
    """Verify POST /generator/nutrition-plan and alias /rag/nutrition-plan."""
    print("\n" + "=" * 70)
    print("TEST 6: Adaptive Nutrition Plan Generation (/generator/nutrition-plan)")
    print("=" * 70)

    # 1. Primary endpoint: /generator/nutrition-plan
    nut_payload = {
        "user_id": "athlete_nutrition_01",
        "goal": "Hypertrophy",
        "current_weight_kg": 80.0,
        "target_weight_kg": 84.0,
        "dietary_preferences": ["high_protein"],
    }
    response = client.post("/generator/nutrition-plan", json=nut_payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()

    print(f"  • Nutrition Plan Calories: {data.get('daily_target_calories')} kcal")
    print(f"  • Macros: Protein={data.get('macros', {}).get('protein_grams')}g, Carbs={data.get('macros', {}).get('carb_grams')}g, Fat={data.get('macros', {}).get('fat_grams')}g")
    print(f"  • Meals prescribed: {len(data.get('meals', []))}")

    assert data.get("daily_target_calories", 0) > 0, "Expected positive daily calories"
    assert data.get("macros", {}).get("protein_grams", 0) > 0, "Expected positive protein target"
    assert len(data.get("meals", [])) >= 3, "Expected at least 3 prescribed meals"

    # 2. Alias endpoint: /rag/nutrition-plan with Vegetarian preference
    alias_payload = {
        "user_id": "athlete_nutrition_02",
        "goal": "FatLoss",
        "current_weight_kg": 90.0,
        "target_weight_kg": 82.0,
        "dietary_preferences": ["vegetarian"],
    }
    alias_res = client.post("/rag/nutrition-plan", json=alias_payload)
    assert alias_res.status_code == 200, f"Expected 200 for alias /rag/nutrition-plan, got {alias_res.status_code}"
    alias_data = alias_res.json()
    assert alias_data.get("daily_target_calories", 0) > 0
    assert "vegetarian" in alias_data.get("dietary_preferences", [])
    print(f"  • Alias /rag/nutrition-plan: Vegetarian FatLoss plan validated ({alias_data.get('daily_target_calories')} kcal).")

    # 3. Bad request validation (negative weight rejected by schema or handler)
    bad_req = client.post("/generator/nutrition-plan", json={"user_id": "u1", "goal": "FatLoss", "current_weight_kg": -10, "target_weight_kg": 75})
    assert bad_req.status_code in (400, 422), f"Expected 400 or 422, got {bad_req.status_code}"
    print("[PASS] Test 6: Adaptive nutrition plan generation verified successfully.")


def test_07_progressive_overload_calculation():
    """Verify POST /algorithms/overload evaluates sets and RPE, computes 1RM, volume, and next load."""
    print("\n" + "=" * 70)
    print("TEST 7: Progressive Overload Calculation (/algorithms/overload)")
    print("=" * 70)

    # 3 sets performed cleanly below target RPE 8 -> should increment load
    overload_input = {
        "exercise_id": "barbell_bench_press",
        "exercise_name": "Barbell Bench Press",
        "is_upper_body": True,
        "current_weight": 100.0,
        "current_rest_seconds": 90,
        "rounding_increment": 0.5,
        "sets": [
            {"set_number": 1, "weight": 100.0, "reps": 8, "rpe": 7.0, "failed": False},
            {"set_number": 2, "weight": 100.0, "reps": 8, "rpe": 7.0, "failed": False},
            {"set_number": 3, "weight": 100.0, "reps": 8, "rpe": 7.0, "failed": False},
        ],
    }

    response = client.post("/algorithms/overload", json=overload_input)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()

    # Validate schema
    validated = ProgressiveOverloadOutput.model_validate(data)

    print(f"  • Estimated 1RM (Epley): {validated.estimated_1rm} kg")
    print(f"  • Next Suggested Weight: {validated.next_weight} kg (from {overload_input['current_weight']} kg)")
    print(f"  • Effective Volume: {validated.effective_volume} kg")
    print(f"  • Action: {validated.progression_action}")
    print(f"  • Rationale: {validated.scientific_rationale}")

    # Epley 1RM formula check: 100 * (1 + 8/30) = 126.67
    expected_1rm = 100.0 * (1.0 + 8.0 / 30.0)
    assert math.isclose(validated.estimated_1rm, expected_1rm, rel_tol=1e-2), f"1RM mismatch: {validated.estimated_1rm} vs {expected_1rm}"
    # Next load should increase by 2.5% rounded to 0.5kg (102.5kg)
    assert validated.next_weight == 102.5, f"Expected 102.5kg, got {validated.next_weight}"
    assert validated.effective_volume == 2400.0, f"Expected 2400 effective volume, got {validated.effective_volume}"
    assert validated.total_volume == 2400.0, f"Expected 2400 total volume, got {validated.total_volume}"
    print("[PASS] Test 7: Progressive overload algorithmic engine verified with sports science precision.")


def test_08_adaptive_tdee_calculation():
    """Verify POST /algorithms/tdee computes true TDEE, delta_w, daily imbalance, and macro splits."""
    print("\n" + "=" * 70)
    print("TEST 8: Adaptive TDEE Calculation (/algorithms/tdee)")
    print("=" * 70)

    # Weekly weight change: 80.0kg -> 80.5kg (+0.5kg) on 2800 kcal avg intake
    tdee_input = {
        "weight_start": 80.0,
        "weight_end": 80.5,
        "avg_daily_calories": 2800.0,
        "goal": "Hypertrophy",
    }

    response = client.post("/algorithms/tdee", json=tdee_input)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()

    # Validate schema
    validated = AdaptiveTDEEOutput.model_validate(data)

    print(f"  • Weight delta (delta_w): {validated.delta_w} kg")
    print(f"  • Daily Caloric Imbalance: {validated.daily_caloric_imbalance} kcal")
    print(f"  • True TDEE: {validated.true_tdee} kcal")
    print(f"  • Prescribed Target Calories: {validated.target_calories} kcal")
    print(f"  • Target Macros: {validated.target_protein_grams}g P / {validated.target_carb_grams}g C / {validated.target_fat_grams}g F")

    # Math verifications per Section 12:
    # delta_w = 80.5 - 80.0 = 0.5 kg
    assert math.isclose(validated.delta_w, 0.5, abs_tol=1e-3), f"delta_w mismatch: {validated.delta_w}"
    # daily_caloric_imbalance = (0.5 * 7700.0) / 7.0 = 550.0 kcal
    assert math.isclose(validated.daily_caloric_imbalance, 550.0, abs_tol=1e-3), f"daily imbalance mismatch: {validated.daily_caloric_imbalance}"
    # true_tdee = 2800.0 - 550.0 = 2250.0 kcal
    assert math.isclose(validated.true_tdee, 2250.0, abs_tol=1e-3), f"true_tdee mismatch: {validated.true_tdee}"
    # Prescribed target for Hypertrophy should be true_tdee + surplus (2250 + 300 = 2550)
    assert math.isclose(validated.target_calories, 2550.0, abs_tol=1e-3), f"target_calories mismatch: {validated.target_calories}"
    assert validated.target_protein_grams > 0
    assert validated.target_carb_grams > 0
    assert validated.target_fat_grams > 0
    print("[PASS] Test 8: Adaptive TDEE calculations and macro distributions verified.")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("STARTING SMARTCOACH AI PHASE 2 END-TO-END VERIFICATION")
    print("=" * 70)

    test_01_health_check()
    test_02_semantic_exercise_search()
    test_03_semantic_nutrition_search()
    test_04_rag_query_chat()
    test_05_master_plan_generation()
    test_06_nutrition_plan_generation()
    test_07_progressive_overload_calculation()
    test_08_adaptive_tdee_calculation()

    print("\n" + "=" * 70)
    print("ALL 8 PHASE 2 END-TO-END TESTS PASSED WITH 100% SUCCESS RATE!")
    print("=" * 70 + "\n")
