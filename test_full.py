"""Full test with complete responses"""
import requests
import json

BASE_URL = "http://localhost:8000"

print("=" * 60)
print("SMARTCOACH AI - FULL TEST")
print("=" * 60)

# Test 1: Chat
print("\n\n### TEST 1: CHAT QUERY ###")
print("-" * 40)
data = {"query": "What is a good chest workout for beginners?", "user_id": "test123"}
response = requests.post(f"{BASE_URL}/rag/query", json=data)
result = response.json()
print(f"Status: {response.status_code}")
print(f"\nFULL RESPONSE:\n{result['response']}")
print(f"\nSources found: {len(result.get('sources', []))}")

# Test 2: Workout Plan
print("\n\n### TEST 2: WORKOUT PLAN ###")
print("-" * 40)
data = {
    "user_id": "test123",
    "fitness_level": "beginner",
    "goal": "build_muscle",
    "days_per_week": 3
}
response = requests.post(f"{BASE_URL}/rag/workout-plan", json=data)
result = response.json()
print(f"Status: {response.status_code}")
print(f"\nFULL PLAN:\n{json.dumps(result['plan'], indent=2)}")

# Test 3: Meal Plan
print("\n\n### TEST 3: MEAL PLAN ###")
print("-" * 40)
data = {
    "user_id": "test123",
    "goal": "build_muscle",
    "calories_target": 2500,
    "meals_per_day": 3
}
response = requests.post(f"{BASE_URL}/rag/meal-plan", json=data)
result = response.json()
print(f"Status: {response.status_code}")
print(f"\nFULL PLAN:\n{json.dumps(result['plan'], indent=2)}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
