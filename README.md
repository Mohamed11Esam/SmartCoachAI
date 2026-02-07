---
title: SmartCoach AI
emoji: 💪
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
license: mit
---

# SmartCoach AI Service

RAG-powered AI service for fitness and nutrition advice.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed health status |
| `/docs` | GET | Swagger API documentation |
| `/rag/query` | POST | Ask any fitness question |
| `/rag/plan` | POST | Generate complete fitness plan |
| `/rag/workout-plan` | POST | Generate workout plan |
| `/rag/meal-plan` | POST | Generate meal plan |

## Example Requests

### Chat Query
```json
POST /rag/query
{
  "query": "What's a good workout for beginners?",
  "user_id": "user123"
}
```

### Generate Workout Plan
```json
POST /rag/workout-plan
{
  "user_id": "user123",
  "fitness_level": "beginner",
  "goal": "build_muscle",
  "days_per_week": 3
}
```

### Generate Meal Plan
```json
POST /rag/meal-plan
{
  "user_id": "user123",
  "goal": "build_muscle",
  "calories_target": 2500,
  "meals_per_day": 4
}
```

## How RAG Works

```
User Question → Search Database → Build Prompt → Gemini AI → Response
```

The system uses TF-IDF to find relevant exercises, meals, and tips from the knowledge base, then sends them to Gemini for generating personalized responses.
