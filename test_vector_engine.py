"""
Verification Suite for Apex Athletic / SmartCoachAI Vector Engine
==================================================================
Verifies the Qdrant + SentenceTransformers RAG vector engine:
1. Total exercises indexed is exactly 876 into 'apex_exercises'.
2. Searching for 'bench press' returns barbell chest compound exercises.
3. Searching for 'squat' returns quadriceps leg exercises.
4. Searching for 'high protein' returns nutrition documents.
5. Metadata filtering (muscle, equipment, level) functions correctly.
"""

import sys
import logging
from typing import List, Dict, Any

from vector_engine import VectorEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("TestVectorEngine")


def test_index_count(engine: VectorEngine) -> bool:
    print("\n" + "=" * 65)
    print("TEST 1: Verify Exercise & Nutrition Index Counts")
    print("=" * 65)

    stats = engine.get_collection_stats()
    print("Collection Statistics:", stats)

    ex_stats = stats.get("apex_exercises", {})
    nut_stats = stats.get("apex_nutrition", {})

    total_exercises = ex_stats.get("points_count", 0)
    total_nutrition = nut_stats.get("points_count", 0)

    print(f"Exercises indexed: {total_exercises} (Target: 876)")
    print(f"Nutrition docs indexed: {total_nutrition} (Target: >= 20)")

    assert total_exercises == 876, f"Expected 876 exercises, got {total_exercises}"
    assert total_nutrition >= 20, f"Expected at least 20 nutrition documents, got {total_nutrition}"
    print("[PASS] Test 1: Exercise index count is exactly 876.")
    return True


def test_search_bench_press(engine: VectorEngine) -> bool:
    print("\n" + "=" * 65)
    print("TEST 2: Verify Search 'bench press' -> Barbell Chest Compound")
    print("=" * 65)

    # Search top 10 results for bench press
    results = engine.search_exercises("bench press", limit=10)
    print(f"Search 'bench press' returned {len(results)} results:")
    for idx, r in enumerate(results, 1):
        print(
            f"  {idx:2d}. {r['name']} (Score: {r['score']:.4f}) | "
            f"Equipment: {r.get('equipment')} | "
            f"Primary: {r.get('primaryMuscles')} | "
            f"Secondary: {r.get('secondaryMuscles')} | "
            f"Mechanic: {r.get('mechanic')}"
        )

    assert len(results) > 0, "No results returned for 'bench press'"

    # Identify barbell exercises targeting chest with compound mechanics
    barbell_chest_compound = []
    for r in results:
        all_muscles = [m.lower() for m in (r.get("primaryMuscles", []) + r.get("secondaryMuscles", []))]
        is_barbell = r.get("equipment") == "barbell"
        is_chest = "chest" in all_muscles
        is_compound = r.get("mechanic") == "compound"
        if is_barbell and is_chest and is_compound:
            barbell_chest_compound.append(r)

    print(f"\nMatching Barbell Chest Compound exercises found: {len(barbell_chest_compound)}")
    for match in barbell_chest_compound:
        print(f"  -> Match: {match['name']} | Primary: {match.get('primaryMuscles')} | Equip: {match.get('equipment')}")

    assert len(barbell_chest_compound) > 0, (
        "Expected at least one barbell chest compound exercise in results for 'bench press'"
    )

    # Also verify primary chest barbell press is discovered
    primary_chest_matches = [
        r for r in barbell_chest_compound
        if "chest" in [m.lower() for m in r.get("primaryMuscles", [])]
    ]
    print(f"Primary Chest Barbell Compound matches: {len(primary_chest_matches)}")
    for m in primary_chest_matches:
        print(f"  -> Primary Chest Match: {m['name']}")
    assert len(primary_chest_matches) > 0, "Expected at least one primary chest barbell compound exercise"

    print("[PASS] Test 2: 'bench press' successfully retrieved barbell chest compound exercises.")
    return True


def test_search_squat(engine: VectorEngine) -> bool:
    print("\n" + "=" * 65)
    print("TEST 3: Verify Search 'squat' -> Quadriceps Leg Exercises")
    print("=" * 65)

    results = engine.search_exercises("squat", limit=5)
    print(f"Search 'squat' returned {len(results)} results:")
    for idx, r in enumerate(results, 1):
        print(
            f"  {idx:2d}. {r['name']} (Score: {r['score']:.4f}) | "
            f"Equipment: {r.get('equipment')} | "
            f"Primary Muscles: {r.get('primaryMuscles')} | "
            f"Mechanic: {r.get('mechanic')}"
        )

    assert len(results) > 0, "No results returned for 'squat'"

    # Verify that top results target quadriceps
    quadriceps_exercises = [
        r for r in results
        if "quadriceps" in [m.lower() for m in r.get("primaryMuscles", [])]
    ]

    print(f"\nMatching Quadriceps exercises found in top 5: {len(quadriceps_exercises)}")
    for match in quadriceps_exercises:
        print(f"  -> Match: {match['name']} | Primary: {match.get('primaryMuscles')}")

    assert len(quadriceps_exercises) > 0, (
        "Expected at least one quadriceps leg exercise in top results for 'squat'"
    )
    print("[PASS] Test 3: 'squat' successfully retrieved quadriceps leg exercises.")
    return True


def test_search_high_protein(engine: VectorEngine) -> bool:
    print("\n" + "=" * 65)
    print("TEST 4: Verify Search 'high protein' -> Nutrition Documents")
    print("=" * 65)

    results = engine.search_nutrition("high protein", limit=5)
    print(f"Search 'high protein' returned {len(results)} results:")
    for idx, r in enumerate(results, 1):
        print(
            f"  {idx:2d}. {r['name']} (Score: {r['score']:.4f}) | "
            f"Type: {r.get('type')} | Category: {r.get('category')} | "
            f"Protein: {r.get('protein', 'N/A')}g | Summary: {r.get('summary', '')[:65]}"
        )

    assert len(results) > 0, "No results returned for 'high protein'"

    # Verify documents are valid nutrition items
    nutrition_docs = [
        r for r in results
        if r.get("category") in ("meal", "protocol", "nutrition", "recovery")
        or (r.get("protein") and r.get("protein", 0) > 0)
        or "protein" in r.get("name", "").lower()
    ]

    print(f"\nValid nutrition documents returned: {len(nutrition_docs)}/{len(results)}")
    assert len(nutrition_docs) > 0, "Expected valid nutrition documents for 'high protein'"
    print("[PASS] Test 4: 'high protein' successfully returned nutrition documents.")
    return True


def test_metadata_filtering(engine: VectorEngine) -> bool:
    print("\n" + "=" * 65)
    print("TEST 5: Verify Metadata Filtering (Muscle, Equipment, Level)")
    print("=" * 65)

    # Test 5a: Filter by muscle='chest' and equipment='barbell'
    chest_barbell = engine.search_exercises("press", muscle="chest", equipment="barbell", limit=5)
    print(f"Filtered (muscle=chest, equipment=barbell) returned {len(chest_barbell)} results:")
    for r in chest_barbell:
        print(f"  - {r['name']} | Eq: {r.get('equipment')} | Primary: {r.get('primaryMuscles')} | Secondary: {r.get('secondaryMuscles')}")
        assert r.get("equipment") == "barbell", f"Expected barbell, got {r.get('equipment')}"
        all_muscles = [m.lower() for m in r.get("primaryMuscles", []) + r.get("secondaryMuscles", [])]
        assert "chest" in all_muscles, f"Expected chest in muscles, got {all_muscles}"

    # Test 5b: Filter by level='beginner'
    beginner_moves = engine.search_exercises("squat", level="beginner", limit=5)
    print(f"\nFiltered (level=beginner) returned {len(beginner_moves)} results:")
    for r in beginner_moves:
        print(f"  - {r['name']} | Level: {r.get('level')}")
        assert r.get("level") == "beginner", f"Expected beginner, got {r.get('level')}"

    print("[PASS] Test 5: Metadata filtering verified successfully.")
    return True


def main():
    print("Starting Apex Athletic Vector Engine Verification Suite...")
    engine = VectorEngine(storage_path="./qdrant_storage", force_reindex=False)

    tests = [
        ("Exercise & Nutrition Counts", test_index_count),
        ("Search 'bench press' (Barbell Chest Compound)", test_search_bench_press),
        ("Search 'squat' (Quadriceps Leg)", test_search_squat),
        ("Search 'high protein' (Nutrition)", test_search_high_protein),
        ("Metadata Filtering", test_metadata_filtering),
    ]

    passed = 0
    failed = 0

    for test_name, test_fn in tests:
        try:
            test_fn(engine)
            passed += 1
        except Exception as e:
            print(f"\n[FAIL] {test_name}: {e}")
            failed += 1

    print("\n" + "=" * 65)
    print(f"SUMMARY: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 65)

    engine.close()

    if failed > 0:
        sys.exit(1)
    else:
        print("\nALL VERIFICATIONS PASSED SUCCESSFULLY!")
        sys.exit(0)


if __name__ == "__main__":
    main()
