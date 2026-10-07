"""
SmartCoach AI Service - FastAPI + Qdrant Vector Engine & Algorithmic RAG
========================================================================
High-performance athletic intelligence platform exposing:
- System Health Diagnostics (/health)
- Dense Vector RAG Chat (/rag/query)
- Evidence-Based Master Plan Mesocycles (/generator/master-plan & /rag/master-plan)
- Adaptive Metabolic Nutrition Plans (/generator/nutrition-plan & /rag/nutrition-plan)
- Section 12 Progressive Overload & Autoregulation (/algorithms/overload)
- Section 12 Adaptive TDEE & Macro Partitioning (/algorithms/tdee)
- Semantic Vector Exercise Search (/exercises/search)
- Semantic Vector Nutrition Search (/nutrition/search)

Run with: uvicorn main:app --reload --port 8000
Docs at: http://localhost:8000/docs
"""

import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from algorithms import (
    calculate_adaptive_tdee,
    calculate_progressive_overload,
)
from generator import PlanGenerator
from rag import RAGSystem
from schemas import (
    AdaptiveTDEEInput,
    AdaptiveTDEEOutput,
    MasterPlanPayload,
    ProgressiveOverloadInput,
    ProgressiveOverloadOutput,
)
from vector_engine import VectorEngine, get_vector_engine

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("SmartCoachAI")

# Initialize FastAPI app
app = FastAPI(
    title="SmartCoach AI Service",
    description="Vector-powered athletic intelligence, training mesocycles, and metabolic nutrition service",
    version="2.0.0",
)

# CORS - Allow NestJS backend & frontend to call this service
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core AI Subsystems
logger.info("Initializing SmartCoach AI subsystems...")
vector_engine: VectorEngine = get_vector_engine()
plan_generator: PlanGenerator = PlanGenerator()
rag: RAGSystem = RAGSystem()
logger.info("SmartCoach AI core subsystems initialized successfully.")


# ==============================================================================
# Request/Response Schemas
# ==============================================================================

class QueryRequest(BaseModel):
    """Request for semantic RAG chat query."""
    query: str = Field(..., description="The user's fitness or nutrition inquiry")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    context: Optional[Dict[str, Any]] = Field(None, description="Optional conversation context")


class QueryResponse(BaseModel):
    """Response containing AI advice and vector knowledge citations."""
    response: str
    sources: Optional[List[str]] = None


class PlanRequest(BaseModel):
    """Legacy request for generating plans."""
    user_id: str
    goal: str
    fitness_level: str
    preferences: Optional[Dict[str, Any]] = None
    duration_weeks: Optional[int] = 4


class WorkoutPlanRequest(BaseModel):
    """Legacy request for workout plan generation."""
    user_id: str
    fitness_level: str
    goal: str
    available_equipment: Optional[List[str]] = None
    duration_minutes: Optional[int] = 45
    days_per_week: Optional[int] = 3


class MealPlanRequest(BaseModel):
    """Legacy request for meal plan generation."""
    user_id: str
    goal: str
    dietary_restrictions: Optional[List[str]] = None
    calories_target: Optional[int] = None
    meals_per_day: Optional[int] = 3


class MasterPlanRequest(BaseModel):
    """Request for generating a structured MasterPlan mesocycle conforming to Section 15."""
    user_id: str = Field(..., description="Unique user identifier")
    goal: str = Field(..., description="Target adaptation: Hypertrophy, Strength, FatLoss, Endurance")
    fitness_level: str = Field(..., description="Athletic level: beginner, intermediate, advanced")
    split_type: Optional[str] = Field("UpperLower", description="Mesocycle split: UpperLower, PushPullLegs, FullBody")
    cycle_weeks: Optional[int] = Field(4, description="Mesocycle duration in weeks")
    equipment: Optional[List[str]] = Field(None, description="Available training equipment")
    injuries: Optional[List[str]] = Field(None, description="Reported injuries or joint contraindications")


class NutritionPlanRequest(BaseModel):
    """Request for generating an adaptive metabolic nutrition plan."""
    user_id: str = Field(..., description="Unique user identifier")
    goal: str = Field(..., description="Nutritional goal: Hypertrophy, FatLoss, Maintenance, Endurance")
    current_weight_kg: float = Field(..., description="Current body weight in kg", gt=0.0)
    target_weight_kg: float = Field(..., description="Target body weight in kg", gt=0.0)
    dietary_preferences: Optional[List[str]] = Field(None, description="Preferences e.g. vegetarian, vegan, high_protein")
    daily_calories: Optional[int] = Field(None, description="Optional manual caloric intake override")


# ==============================================================================
# Health Check & Diagnostics
# ==============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Root status endpoint."""
    return {"status": "ok", "service": "SmartCoach AI", "version": "2.0.0"}


@app.get("/health", tags=["Health"])
async def health():
    """
    Comprehensive health check for monitoring:
    Verifies Qdrant connectivity, VectorEngine readiness, and PlanGenerator state.
    """
    try:
        qdrant_connected = False
        vector_engine_ready = False

        if vector_engine is not None and hasattr(vector_engine, "client") and vector_engine.client is not None:
            try:
                # Query collections to verify active Qdrant communication
                vector_engine.client.get_collections()
                qdrant_connected = True
                vector_engine_ready = True
            except Exception as q_err:
                logger.warning(f"Qdrant health check warning: {q_err}")
                qdrant_connected = False
                vector_engine_ready = False

        generator_ready = plan_generator is not None

        return {
            "status": "healthy",
            "qdrant_connected": qdrant_connected,
            "vector_engine_ready": vector_engine_ready,
            "generator_ready": generator_ready,
            "rag_initialized": rag.is_initialized if rag else False,
        }
    except Exception as e:
        logger.error(f"Health check encountered error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ==============================================================================
# Semantic Search Endpoints (Exercises & Nutrition)
# ==============================================================================

@app.get("/exercises/search", tags=["Vector Search"])
async def search_exercises_endpoint(
    q: str = Query("", description="Semantic search query (e.g. 'bench', 'squat', 'pull up')"),
    muscle: Optional[str] = Query(None, description="Target muscle group filter (e.g. 'chest', 'quadriceps')"),
    equipment: Optional[str] = Query(None, description="Equipment filter (e.g. 'barbell', 'dumbbell', 'body only')"),
    level: Optional[str] = Query(None, description="Fitness level filter ('beginner', 'intermediate', 'expert')"),
    limit: int = Query(5, ge=1, le=50, description="Maximum number of exercises to return"),
):
    """
    Search 876 indexed exercises using dense semantic vector similarity and metadata filtering.
    Powered by vector_engine.search_exercises.
    """
    if not q.strip() and not muscle and not equipment and not level:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one query parameter ('q', 'muscle', 'equipment', or 'level') must be provided.",
        )

    try:
        effective_query = q.strip() if q.strip() else (muscle or "compound exercise")
        results = vector_engine.search_exercises(
            query=effective_query,
            muscle=muscle,
            equipment=equipment,
            level=level,
            limit=limit,
        )
        return {
            "query": q,
            "muscle": muscle,
            "equipment": equipment,
            "level": level,
            "count": len(results),
            "results": results,
        }
    except Exception as e:
        logger.error(f"Exercise search error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/nutrition/search", tags=["Vector Search"])
async def search_nutrition_endpoint(
    q: str = Query(..., description="Nutritional inquiry (e.g. 'high protein', 'caloric deficit', 'post workout')"),
    limit: int = Query(5, ge=1, le=50, description="Maximum number of items to return"),
):
    """
    Search nutritional meal plans and protocols using dense semantic vector similarity.
    Powered by vector_engine.search_nutrition.
    """
    if not q or not q.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query parameter 'q' cannot be empty.",
        )

    try:
        results = vector_engine.search_nutrition(query=q.strip(), limit=limit)
        return {
            "query": q,
            "count": len(results),
            "results": results,
        }
    except Exception as e:
        logger.error(f"Nutrition search error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ==============================================================================
# Semantic RAG Query Chat
# ==============================================================================

@app.post("/rag/query", response_model=QueryResponse, tags=["RAG"])
async def chat_query(request: QueryRequest):
    """
    Semantic RAG chat query powered by vector_engine.search_exercises
    and vector_engine.search_nutrition.
    """
    clean_query = (request.query or "").strip()
    if not clean_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty.",
        )

    try:
        # Retrieve relevant exercises and nutrition context via dense vector similarity
        exercise_results = vector_engine.search_exercises(clean_query, limit=3)
        nutrition_results = vector_engine.search_nutrition(clean_query, limit=3)

        sources: List[str] = []
        context_snippets: List[str] = []

        for ex in exercise_results:
            ex_name = ex.get("name", "Exercise")
            muscles = ", ".join(ex.get("primaryMuscles", []))
            eq = ex.get("equipment") or "bodyweight"
            sources.append(f"Exercise: {ex_name} ({muscles})")
            context_snippets.append(
                f"[Exercise] {ex_name}: Target muscles: {muscles}. Equipment: {eq}. Level: {ex.get('level', 'all')}."
            )

        for nut in nutrition_results:
            n_name = nut.get("name") or nut.get("topic") or "Nutrition"
            n_type = nut.get("type", "nutrition")
            sources.append(f"Nutrition: {n_name} ({n_type})")
            if nut.get("type") == "meal":
                context_snippets.append(
                    f"[Meal] {n_name}: {nut.get('calories', 0)} kcal, {nut.get('protein', 0)}g protein, {nut.get('carbs', 0)}g carbs."
                )
            else:
                context_snippets.append(
                    f"[Protocol] {n_name}: {nut.get('content', nut.get('summary', ''))}"
                )

        combined_context = "\n".join(context_snippets)

        # Generate response using LLM if configured; fallback to structured sports science response
        prompt = f"""You are SmartCoach APEX AI, an elite evidence-based fitness and sports nutrition advisor.

RELEVANT KNOWLEDGE BASE CONTEXT:
{combined_context}

USER QUESTION: {clean_query}

Provide a concise, scientifically grounded, and actionable recommendation based on the knowledge above."""

        response_text: Optional[str] = None
        if rag and getattr(rag, "client", None):
            try:
                response_text = rag._generate_with_llm(prompt)
            except Exception as llm_err:
                logger.warning(f"LLM query generation failed, using sports science synthesizer: {llm_err}")
                response_text = None

        if not response_text or "I apologize" in response_text or "mock responses" in response_text:
            lines = [
                f"Based on Apex Athletic sports science guidelines for '{clean_query}':\n"
            ]
            if exercise_results:
                lines.append("Recommended Exercises:")
                for ex in exercise_results:
                    primary = ", ".join(ex.get("primaryMuscles", []))
                    lines.append(f"• {ex.get('name')} - Focus: {primary} ({ex.get('equipment', 'bodyweight')}, {ex.get('level', 'all')} level)")
            if nutrition_results:
                lines.append("\nNutritional Recommendations:")
                for nut in nutrition_results:
                    if nut.get("type") == "meal":
                        lines.append(f"• {nut.get('name')}: {nut.get('calories', 0)} kcal, {nut.get('protein', 0)}g protein, {nut.get('carbs', 0)}g carbs")
                    else:
                        lines.append(f"• {nut.get('name')}: {nut.get('content', nut.get('summary', ''))[:120]}...")
            lines.append("\nAutoregulation Note: Calibrate working sets at RPE 7-8 and apply progressive overload systematically.")
            response_text = "\n".join(lines)

        return QueryResponse(response=response_text, sources=sources)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error handling /rag/query: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ==============================================================================
# Master Plan Mesocycle Generation (Conforming to BLUEPRINT.md Section 15)
# ==============================================================================

@app.post("/generator/master-plan", response_model=MasterPlanPayload, tags=["Generator"])
@app.post("/rag/master-plan", response_model=MasterPlanPayload, tags=["Generator"])
async def generate_master_plan_endpoint(request: MasterPlanRequest):
    """
    Generate a strictly typed, evidence-based mesocycle training plan.
    Conforms to BLUEPRINT.md Section 15 MasterPlanPayload.
    """
    if not request.user_id or not request.user_id.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="user_id is required.")
    if not request.goal or not request.goal.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="goal is required.")
    if not request.fitness_level or not request.fitness_level.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="fitness_level is required.")

    try:
        return plan_generator.generate_master_plan(
            user_id=request.user_id.strip(),
            goal=request.goal.strip(),
            fitness_level=request.fitness_level.strip(),
            split_type=request.split_type or "UpperLower",
            cycle_weeks=request.cycle_weeks or 4,
            equipment=request.equipment,
            injuries=request.injuries,
        )
    except Exception as e:
        logger.error(f"Failed to generate master plan: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ==============================================================================
# Adaptive Nutrition Plan Generation
# ==============================================================================

@app.post("/generator/nutrition-plan", tags=["Generator"])
@app.post("/rag/nutrition-plan", tags=["Generator"])
async def generate_nutrition_plan_endpoint(request: NutritionPlanRequest):
    """
    Generate a structured adaptive nutrition plan matching algorithms.metabolic
    and the nutrition database.
    """
    if not request.user_id or not request.user_id.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="user_id is required.")
    if not request.goal or not request.goal.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="goal is required.")
    if request.current_weight_kg <= 0 or request.target_weight_kg <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Weight values must be positive.")

    try:
        plan = plan_generator.generate_nutrition_plan(
            user_id=request.user_id.strip(),
            goal=request.goal.strip(),
            current_weight_kg=request.current_weight_kg,
            target_weight_kg=request.target_weight_kg,
            dietary_preferences=request.dietary_preferences,
            daily_calories=request.daily_calories,
        )
        if "daily_target_calories" not in plan and "target_calories" in plan:
            plan["daily_target_calories"] = plan["target_calories"]
        if "macros" not in plan:
            plan["macros"] = {
                "protein_grams": plan.get("target_protein_grams", 0),
                "carb_grams": plan.get("target_carb_grams", 0),
                "fat_grams": plan.get("target_fat_grams", 0),
            }
        return plan
    except Exception as e:
        logger.error(f"Failed to generate nutrition plan: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ==============================================================================
# Algorithmic Engines: Section 12 Progressive Overload & Adaptive TDEE
# ==============================================================================

@app.post("/algorithms/overload", response_model=ProgressiveOverloadOutput, tags=["Algorithms"])
async def progressive_overload_endpoint(request: ProgressiveOverloadInput):
    """
    Calculate Progressive Overload prescription based on logged sets and RPE.
    Per BLUEPRINT.md Section 12.
    """
    try:
        return calculate_progressive_overload(request)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Error computing progressive overload: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/algorithms/tdee", response_model=AdaptiveTDEEOutput, tags=["Algorithms"])
async def adaptive_tdee_endpoint(request: AdaptiveTDEEInput):
    """
    Calculate Adaptive TDEE energy expenditure and macronutrient partitioning.
    Per BLUEPRINT.md Section 12.
    """
    try:
        return calculate_adaptive_tdee(request)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Error computing adaptive TDEE: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ==============================================================================
# Legacy RAG Endpoints (Preserved for Backward Compatibility)
# ==============================================================================

@app.post("/rag/plan", tags=["Legacy RAG"])
async def generate_fitness_plan(request: PlanRequest):
    """Generate a complete fitness plan (workout + nutrition)."""
    try:
        plan = rag.generate_fitness_plan(
            goal=request.goal,
            fitness_level=request.fitness_level,
            preferences=request.preferences,
            duration_weeks=request.duration_weeks or 4,
        )
        return {"plan": plan, "user_id": request.user_id}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/rag/workout-plan", tags=["Legacy RAG"])
async def generate_workout_plan(request: WorkoutPlanRequest):
    """Generate a workout plan."""
    try:
        plan = rag.generate_workout_plan(
            fitness_level=request.fitness_level,
            goal=request.goal,
            equipment=request.available_equipment,
            duration=request.duration_minutes or 45,
            days_per_week=request.days_per_week or 3,
        )
        return {"plan": plan, "user_id": request.user_id}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/rag/meal-plan", tags=["Legacy RAG"])
async def generate_meal_plan(request: MealPlanRequest):
    """Generate a meal/nutrition plan."""
    try:
        plan = rag.generate_meal_plan(
            goal=request.goal,
            restrictions=request.dietary_restrictions,
            calories=request.calories_target,
            meals_per_day=request.meals_per_day or 3,
        )
        return {"plan": plan, "user_id": request.user_id}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ==============================================================================
# Server Entrypoint
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
