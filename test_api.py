"""Quick test script for the AI service"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    print("\n=== Testing Health Endpoint ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_chat():
    print("\n=== Testing Chat Query ===")
    data = {
        "query": "What is a good chest workout for beginners?",
        "user_id": "test123"
    }
    response = requests.post(f"{BASE_URL}/rag/query", json=data)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {result['response'][:500]}...")
    print(f"Sources found: {len(result.get('sources', []))}")
    return response.status_code == 200

def test_workout_plan():
    print("\n=== Testing Workout Plan Generation ===")
    data = {
        "user_id": "test123",
        "fitness_level": "beginner",
        "goal": "build_muscle",
        "days_per_week": 3
    }
    response = requests.post(f"{BASE_URL}/rag/workout-plan", json=data)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Plan: {json.dumps(result, indent=2)[:500]}...")
    return response.status_code == 200

def test_meal_plan():
    print("\n=== Testing Meal Plan Generation ===")
    data = {
        "user_id": "test123",
        "goal": "build_muscle",
        "calories_target": 2500,
        "meals_per_day": 4
    }
    response = requests.post(f"{BASE_URL}/rag/meal-plan", json=data)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Plan: {json.dumps(result, indent=2)[:500]}...")
    return response.status_code == 200

if __name__ == "__main__":
    print("SmartCoach AI Service - API Tests")
    print("=" * 50)

    results = []
    results.append(("Health", test_health()))
    results.append(("Chat", test_chat()))
    results.append(("Workout Plan", test_workout_plan()))
    results.append(("Meal Plan", test_meal_plan()))

    print("\n" + "=" * 50)
    print("RESULTS:")
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
