# SmartCoach AI Service

RAG-powered AI service for fitness and nutrition advice.

## Quick Start

### 1. Get Gemini API Key (FREE)

1. Go to: https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key

### 2. Setup

```bash
cd ai-service

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
copy .env.example .env    # Windows
cp .env.example .env      # Mac/Linux

# Add your Gemini API key to .env
```

### 3. Run

```bash
uvicorn main:app --reload --port 8000
```

### 4. Test

- Open: http://localhost:8000/docs
- Try the endpoints!

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/rag/query` | POST | Ask any fitness question |
| `/rag/plan` | POST | Generate complete fitness plan |
| `/rag/workout-plan` | POST | Generate workout plan |
| `/rag/meal-plan` | POST | Generate meal plan |
| `/health` | GET | Health check |

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
User Question
     │
     ▼
┌─────────────────┐
│ Search Database │ ← Finds relevant exercises/meals/tips
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Build Prompt    │ ← Combines question + relevant info
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Gemini AI       │ ← Generates personalized response
└────────┬────────┘
         │
         ▼
    Response
```

## Knowledge Base

The `data/` folder contains:
- `workouts.json` - 25 exercises
- `nutrition.json` - 20 meals
- `tips.json` - 25 fitness tips

Feel free to add more data to improve responses!

## Integration with NestJS

The NestJS backend calls this service at `AI_SERVICE_URL`.

In NestJS `.env`:
```
AI_SERVICE_URL=http://localhost:8000
```

That's it! The integration is already set up.
