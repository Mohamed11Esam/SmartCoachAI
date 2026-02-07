"""
SmartCoach AI Service - FastAPI + RAG
=====================================
A simple RAG-based AI service for fitness and nutrition advice.

Run with: uvicorn main:app --reload --port 8000
Docs at: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from rag import RAGSystem
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="SmartCoach AI Service",
    description="RAG-powered fitness and nutrition AI assistant",
    version="1.0.0"
)

# CORS - Allow NestJS backend to call this service
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, set specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG System
rag = RAGSystem()


# ==========================================
# Request/Response Models
# ==========================================

class QueryRequest(BaseModel):
    """Request for general chat/questions"""
    query: str
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    """Response for chat queries"""
    response: str
    sources: Optional[List[str]] = None

class PlanRequest(BaseModel):
    """Request for generating plans"""
    user_id: str
    goal: str  # "lose_weight", "build_muscle", "stay_fit", "increase_endurance"
    fitness_level: str  # "beginner", "intermediate", "advanced"
    preferences: Optional[Dict[str, Any]] = None
    duration_weeks: Optional[int] = 4

class WorkoutPlanRequest(BaseModel):
    """Request for workout plan generation"""
    user_id: str
    fitness_level: str  # "beginner", "intermediate", "advanced"
    goal: str  # "strength", "cardio", "flexibility", "muscle_building"
    available_equipment: Optional[List[str]] = None
    duration_minutes: Optional[int] = 45
    days_per_week: Optional[int] = 3

class MealPlanRequest(BaseModel):
    """Request for meal plan generation"""
    user_id: str
    goal: str  # "lose_weight", "build_muscle", "maintain"
    dietary_restrictions: Optional[List[str]] = None  # "vegetarian", "vegan", "gluten_free"
    calories_target: Optional[int] = None
    meals_per_day: Optional[int] = 3


# ==========================================
# API Endpoints
# ==========================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "service": "SmartCoach AI"}


@app.get("/health")
async def health():
    """Health check for monitoring"""
    return {"status": "healthy", "rag_initialized": rag.is_initialized}


@app.post("/rag/query", response_model=QueryResponse)
async def chat_query(request: QueryRequest):
    """
    General chat endpoint - Ask any fitness/nutrition question

    Example:
    {
        "query": "What's a good chest workout for beginners?",
        "user_id": "user123"
    }
    """
    try:
        response, sources = rag.query(request.query, request.context)
        return QueryResponse(response=response, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/plan")
async def generate_fitness_plan(request: PlanRequest):
    """
    Generate a complete fitness plan (workout + nutrition)

    Example:
    {
        "user_id": "user123",
        "goal": "build_muscle",
        "fitness_level": "intermediate",
        "duration_weeks": 4
    }
    """
    try:
        plan = rag.generate_fitness_plan(
            goal=request.goal,
            fitness_level=request.fitness_level,
            preferences=request.preferences,
            duration_weeks=request.duration_weeks
        )
        return {"plan": plan, "user_id": request.user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/workout-plan")
async def generate_workout_plan(request: WorkoutPlanRequest):
    """
    Generate a workout plan

    Example:
    {
        "user_id": "user123",
        "fitness_level": "beginner",
        "goal": "strength",
        "days_per_week": 3,
        "duration_minutes": 45
    }
    """
    try:
        plan = rag.generate_workout_plan(
            fitness_level=request.fitness_level,
            goal=request.goal,
            equipment=request.available_equipment,
            duration=request.duration_minutes,
            days_per_week=request.days_per_week
        )
        return {"plan": plan, "user_id": request.user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/meal-plan")
async def generate_meal_plan(request: MealPlanRequest):
    """
    Generate a meal/nutrition plan

    Example:
    {
        "user_id": "user123",
        "goal": "build_muscle",
        "calories_target": 2500,
        "meals_per_day": 4
    }
    """
    try:
        plan = rag.generate_meal_plan(
            goal=request.goal,
            restrictions=request.dietary_restrictions,
            calories=request.calories_target,
            meals_per_day=request.meals_per_day
        )
        return {"plan": plan, "user_id": request.user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# Run Server
# ==========================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
