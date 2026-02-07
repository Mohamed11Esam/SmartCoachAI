"""
RAG System - Simple Version
===========================
Uses TF-IDF for search and Google Gemini for generation.
No complex dependencies - works out of the box!
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

# Try to import the new google-genai package
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class RAGSystem:
    """
    Simple RAG system for fitness and nutrition knowledge.

    How it works:
    1. Load fitness data (exercises, meals, tips)
    2. When user asks a question, search for relevant information using TF-IDF
    3. Combine the question + relevant info and send to Gemini
    4. Return the response
    """

    def __init__(self):
        self.is_initialized = False
        self.documents = []
        self.doc_metadata = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.client = None

        self._setup_gemini()
        self._load_knowledge_base()
        self._build_search_index()
        self.is_initialized = True

    def _setup_gemini(self):
        """Configure Google Gemini API"""
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            print("[WARNING] GEMINI_API_KEY not set. Using mock responses.")
            self.client = None
            return

        if not GENAI_AVAILABLE:
            print("[WARNING] google-genai package not available. Using mock responses.")
            self.client = None
            return

        try:
            self.client = genai.Client(api_key=api_key)
            print("[OK] Gemini API configured")
        except Exception as e:
            print(f"[ERROR] Gemini setup error: {e}")
            self.client = None

    def _load_knowledge_base(self):
        """Load fitness data from JSON files"""
        data_path = Path(__file__).parent / "data"

        # Load workouts
        workouts_file = data_path / "workouts.json"
        if workouts_file.exists():
            with open(workouts_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for exercise in data.get("exercises", []):
                doc = f"""
Exercise: {exercise['name']}
Muscle Group: {exercise.get('muscle_group', 'N/A')}
Difficulty: {exercise.get('difficulty', 'N/A')}
Description: {exercise.get('description', '')}
Instructions: {exercise.get('instructions', '')}
Sets: {exercise.get('sets', 'N/A')} | Reps: {exercise.get('reps', 'N/A')}
Equipment: {exercise.get('equipment', 'None')}
"""
                self.documents.append(doc)
                self.doc_metadata.append({"type": "exercise", "name": exercise['name']})
            print(f"  [OK] Loaded {len(data.get('exercises', []))} exercises")

        # Load nutrition
        nutrition_file = data_path / "nutrition.json"
        if nutrition_file.exists():
            with open(nutrition_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for meal in data.get("meals", []):
                doc = f"""
Meal: {meal['name']}
Type: {meal.get('type', 'N/A')}
Calories: {meal.get('calories', 'N/A')} kcal
Protein: {meal.get('protein', 'N/A')}g | Carbs: {meal.get('carbs', 'N/A')}g | Fat: {meal.get('fat', 'N/A')}g
Ingredients: {', '.join(meal.get('ingredients', []))}
Good for: {', '.join(meal.get('goal', []))}
Preparation: {meal.get('preparation', '')}
"""
                self.documents.append(doc)
                self.doc_metadata.append({"type": "meal", "name": meal['name']})
            print(f"  [OK] Loaded {len(data.get('meals', []))} meals")

        # Load tips
        tips_file = data_path / "tips.json"
        if tips_file.exists():
            with open(tips_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for tip in data.get("tips", []):
                doc = f"""
Topic: {tip.get('topic', 'General')}
Category: {tip.get('category', 'general')}
Tip: {tip['content']}
"""
                self.documents.append(doc)
                self.doc_metadata.append({"type": "tip", "topic": tip.get('topic', '')})
            print(f"  [OK] Loaded {len(data.get('tips', []))} tips")

        print(f"[OK] Knowledge base loaded: {len(self.documents)} total documents")

    def _build_search_index(self):
        """Build TF-IDF search index"""
        if not self.documents:
            print("[WARNING] No documents to index")
            return

        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        print("[OK] Search index built")

    def _search(self, query: str, n_results: int = 5) -> List[str]:
        """Search for relevant documents using TF-IDF similarity"""
        if self.vectorizer is None or self.tfidf_matrix is None:
            return []

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Get top N results
        top_indices = similarities.argsort()[-n_results:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0.05:  # Only include if similarity > threshold
                results.append(self.documents[idx])

        return results

    def _generate_with_llm(self, prompt: str) -> str:
        """Generate response using Gemini LLM"""
        if not self.client:
            return self._mock_response(prompt)

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"[ERROR] LLM Error: {e}")
            return f"I apologize, but I encountered an error. Error: {str(e)}"

    def _mock_response(self, prompt: str) -> str:
        """Mock response when Gemini is not configured"""
        return """I'm SmartCoach AI! To enable full AI responses:
1. Get a free Gemini API key from https://aistudio.google.com/app/apikey
2. Add it to your .env file as GEMINI_API_KEY=your-key

For now, here's a tip: Stay consistent with your workouts and focus on proper nutrition!"""

    def query(self, question: str, context: Optional[Dict] = None) -> Tuple[str, List[str]]:
        """
        Answer a fitness/nutrition question using RAG
        """
        # Search for relevant information
        relevant_docs = self._search(question, n_results=5)

        # Build the prompt
        context_text = "\n---\n".join(relevant_docs) if relevant_docs else "No specific information found in knowledge base."

        prompt = f"""You are SmartCoach, a knowledgeable and friendly fitness and nutrition AI assistant.

RELEVANT INFORMATION FROM KNOWLEDGE BASE:
{context_text}

USER QUESTION: {question}

INSTRUCTIONS:
- Provide a helpful, accurate, and encouraging response
- Use the relevant information above when applicable
- If the information doesn't fully answer the question, use your general fitness knowledge
- Keep the response concise but complete
- Use bullet points or numbered lists for clarity when appropriate
- Always encourage safe fitness practices

YOUR RESPONSE:"""

        response = self._generate_with_llm(prompt)
        return response, relevant_docs[:3]

    def generate_fitness_plan(
        self,
        goal: str,
        fitness_level: str,
        preferences: Optional[Dict] = None,
        duration_weeks: int = 4
    ) -> Dict[str, Any]:
        """Generate a complete fitness plan"""

        workout_docs = self._search(f"{fitness_level} {goal} workout exercises", n_results=8)
        nutrition_docs = self._search(f"{goal} nutrition meals", n_results=8)

        prompt = f"""You are SmartCoach AI creating a personalized fitness plan.

USER PROFILE:
- Goal: {goal}
- Fitness Level: {fitness_level}
- Duration: {duration_weeks} weeks
- Preferences: {preferences or 'None specified'}

AVAILABLE EXERCISES:
{chr(10).join(workout_docs[:5])}

AVAILABLE MEALS:
{chr(10).join(nutrition_docs[:5])}

Create a structured {duration_weeks}-week fitness plan. Return ONLY valid JSON:
{{
    "plan_name": "Name of the plan",
    "goal": "{goal}",
    "duration_weeks": {duration_weeks},
    "weekly_schedule": [
        {{
            "day": "Monday",
            "workout": {{
                "name": "Workout name",
                "exercises": [
                    {{"name": "Exercise", "sets": 3, "reps": "10-12"}}
                ],
                "duration_minutes": 45
            }},
            "nutrition": {{
                "calories_target": 2000,
                "meals": ["Breakfast suggestion", "Lunch", "Dinner"]
            }}
        }}
    ],
    "tips": ["Tip 1", "Tip 2"]
}}"""

        response = self._generate_with_llm(prompt)
        return self._parse_json_response(response, {
            "plan_name": f"{goal.replace('_', ' ').title()} Plan",
            "goal": goal,
            "duration_weeks": duration_weeks,
            "description": response
        })

    def generate_workout_plan(
        self,
        fitness_level: str,
        goal: str,
        equipment: Optional[List[str]] = None,
        duration: int = 45,
        days_per_week: int = 3
    ) -> Dict[str, Any]:
        """Generate a workout plan"""

        equipment_str = ', '.join(equipment) if equipment else 'bodyweight only'
        relevant_docs = self._search(f"{fitness_level} {goal} exercises {equipment_str}", n_results=10)

        prompt = f"""Create a workout plan:
- Fitness Level: {fitness_level}
- Goal: {goal}
- Equipment: {equipment_str}
- Duration: {duration} minutes per workout
- Days per Week: {days_per_week}

EXERCISE DATABASE:
{chr(10).join(relevant_docs[:7])}

Return ONLY valid JSON:
{{
    "name": "Plan name",
    "days_per_week": {days_per_week},
    "workouts": [
        {{
            "day": 1,
            "name": "Day name",
            "duration_minutes": {duration},
            "exercises": [
                {{"name": "Exercise", "sets": 3, "reps": "10-12", "rest_seconds": 60}}
            ],
            "warmup": "5 min cardio",
            "cooldown": "5 min stretching"
        }}
    ]
}}"""

        response = self._generate_with_llm(prompt)
        return self._parse_json_response(response, {
            "name": f"{goal.title()} Workout Plan",
            "days_per_week": days_per_week,
            "description": response
        })

    def generate_meal_plan(
        self,
        goal: str,
        restrictions: Optional[List[str]] = None,
        calories: Optional[int] = None,
        meals_per_day: int = 3
    ) -> Dict[str, Any]:
        """Generate a meal/nutrition plan"""

        restrictions_str = ', '.join(restrictions) if restrictions else 'none'
        calories_str = f"{calories} kcal" if calories else "appropriate for goal"
        relevant_docs = self._search(f"{goal} meals nutrition {restrictions_str}", n_results=10)

        prompt = f"""Create a daily meal plan:
- Goal: {goal}
- Dietary Restrictions: {restrictions_str}
- Target Calories: {calories_str}
- Meals per Day: {meals_per_day}

MEAL DATABASE:
{chr(10).join(relevant_docs[:7])}

Return ONLY valid JSON:
{{
    "name": "Plan name",
    "goal": "{goal}",
    "daily_calories": {calories or 2000},
    "meals_per_day": {meals_per_day},
    "daily_plan": [
        {{
            "meal": "Breakfast",
            "time": "7:00 AM",
            "name": "Meal name",
            "calories": 500,
            "protein_g": 30,
            "ingredients": ["ingredient1", "ingredient2"]
        }}
    ],
    "tips": ["Nutrition tip"]
}}"""

        response = self._generate_with_llm(prompt)
        return self._parse_json_response(response, {
            "name": f"{goal.title()} Meal Plan",
            "goal": goal,
            "meals_per_day": meals_per_day,
            "description": response
        })

    def _parse_json_response(self, response: str, fallback: Dict) -> Dict:
        """Parse JSON from LLM response, with fallback"""
        try:
            clean = response.strip()
            # Remove markdown code blocks if present
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            return json.loads(clean)
        except (json.JSONDecodeError, IndexError):
            return fallback
