"""
Vector and RAG Engine for Apex Athletic / SmartCoachAI
======================================================
High-performance semantic vector search engine powered by Qdrant
and SentenceTransformers (all-MiniLM-L6-v2).

Features:
- Indexes 876 comprehensive exercises into collection 'apex_exercises'
- Indexes nutrition meals and protocols into collection 'apex_nutrition'
- Flexible metadata filtering (muscle, equipment, fitness level)
- Automatic fallback from persistent Qdrant storage to in-memory mode if locked
- Clean lifecycle management avoiding Python 3.13 teardown warnings
"""

import atexit
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import qdrant_client
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("VectorEngine")

# Common muscle term normalization mapping
MUSCLE_ALIASES: Dict[str, str] = {
    "quad": "quadriceps",
    "quads": "quadriceps",
    "leg": "quadriceps",
    "legs": "quadriceps",
    "ab": "abdominals",
    "abs": "abdominals",
    "core": "abdominals",
    "glute": "glutes",
    "hamstring": "hamstrings",
    "calf": "calves",
    "trap": "traps",
    "shoulder": "shoulders",
    "tricep": "triceps",
    "bicep": "biceps",
    "back": "middle back",
    "lat": "lats",
}


class VectorEngine:
    """
    Qdrant-backed Vector Engine for Apex Athletic.
    Provides dense vector search with metadata filtering for exercises and nutrition.
    """

    EXERCISES_COLLECTION = "apex_exercises"
    NUTRITION_COLLECTION = "apex_nutrition"
    DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(
        self,
        storage_path: Optional[str] = "./qdrant_storage",
        model_name: str = DEFAULT_MODEL_NAME,
        force_reindex: bool = False,
    ):
        """
        Initialize the VectorEngine.

        :param storage_path: Directory path for persistent Qdrant storage, or ':memory:'.
        :param model_name: SentenceTransformers model identifier.
        :param force_reindex: If True, drops existing collections and re-indexes from source data.
        """
        self.storage_path = storage_path
        self.model_name = model_name
        self.data_dir = Path(__file__).resolve().parent / "data"

        # Initialize Embedding Model
        logger.info(f"Loading SentenceTransformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.vector_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded. Dimension: {self.vector_dim}")

        # Initialize Qdrant Client
        self.client = self._init_qdrant_client(storage_path)

        # Register cleanup on program exit
        atexit.register(self.close)

        # Populate or verify collections
        self.init_collections(force_reindex=force_reindex)

    def _init_qdrant_client(self, storage_path: Optional[str]) -> QdrantClient:
        """
        Initialize QdrantClient with safe fallback to in-memory mode if path is locked.
        """
        if not storage_path or storage_path == ":memory:":
            logger.info("Initializing in-memory Qdrant client (:memory:)...")
            return QdrantClient(":memory:")

        resolved_path = str(Path(storage_path).resolve())
        try:
            os.makedirs(resolved_path, exist_ok=True)
            logger.info(f"Connecting to persistent Qdrant storage at: {resolved_path}")
            client = QdrantClient(path=resolved_path)
            # Verify client is responsive
            client.get_collections()
            return client
        except Exception as e:
            logger.warning(
                f"Failed to initialize persistent Qdrant at '{resolved_path}' ({e}). "
                f"Falling back to ':memory:' Qdrant instance."
            )
            return QdrantClient(":memory:")

    def get_embedding(self, text: str) -> List[float]:
        """Generate dense embedding for a single text string."""
        emb = self.model.encode(text, show_progress_bar=False, convert_to_numpy=True)
        return emb.tolist()

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Generate dense embeddings for a batch of text strings."""
        embs = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
        return embs.tolist()

    def init_collections(self, force_reindex: bool = False):
        """Ensure collections exist and are populated with domain data."""
        self.index_exercises(force_reindex=force_reindex)
        self.index_nutrition(force_reindex=force_reindex)

    def index_exercises(
        self,
        exercises_file: Optional[Path] = None,
        force_reindex: bool = False,
    ) -> int:
        """
        Index all 876 exercises into 'apex_exercises' collection.

        :param exercises_file: Path to exercises JSON (defaults to data/exercises_800.json).
        :param force_reindex: Whether to re-populate collection if it already exists.
        :return: Count of indexed exercises.
        """
        if exercises_file is None:
            exercises_file = self.data_dir / "exercises_800.json"

        if not exercises_file.exists():
            raise FileNotFoundError(f"Exercises data file not found at: {exercises_file}")

        collection_name = self.EXERCISES_COLLECTION

        # Check existing collection state
        if self.client.collection_exists(collection_name):
            if force_reindex:
                logger.info(f"Reindexing requested. Deleting collection '{collection_name}'...")
                self.client.delete_collection(collection_name)
            else:
                info = self.client.get_collection(collection_name)
                if info.points_count >= 876:
                    logger.info(
                        f"Collection '{collection_name}' already populated with {info.points_count} points. Skipping indexing."
                    )
                    return info.points_count
                else:
                    logger.info(
                        f"Collection '{collection_name}' has {info.points_count} points (expected 876). Rebuilding..."
                    )
                    self.client.delete_collection(collection_name)

        logger.info(f"Loading exercises from: {exercises_file}")
        with open(exercises_file, "r", encoding="utf-8") as f:
            exercises_data = json.load(f)

        total_exercises = len(exercises_data)
        logger.info(f"Loaded {total_exercises} exercises. Building embeddings and payloads...")

        # Create collection
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
        )

        texts = []
        payloads = []

        for ex in exercises_data:
            name = ex.get("name", "")
            force = ex.get("force") or "none"
            level = ex.get("level", "")
            mechanic = ex.get("mechanic") or "none"
            equipment = ex.get("equipment") or "none"
            primary_muscles = ex.get("primaryMuscles", [])
            secondary_muscles = ex.get("secondaryMuscles", [])
            primary_str = ", ".join(primary_muscles)
            secondary_str = ", ".join(secondary_muscles)
            instructions = ex.get("instructions", [])
            instr_str = " ".join(instructions) if isinstance(instructions, list) else str(instructions)
            category = ex.get("category", "")

            # Concise summary of instructions to keep token focus on core identity
            overview = instr_str[:220].strip()

            # Construct balanced, semantically rich representation prioritizing exercise title, muscles & equipment
            embed_text = (
                f"{name}. {name}. Target muscles: {primary_str}. Secondary: {secondary_str}. "
                f"Equipment: {equipment}. Mechanic: {mechanic} exercise. Category: {category}. "
                f"Force: {force}. Level: {level}. Instructions: {overview}"
            )
            texts.append(embed_text)

            # Store complete rich payload
            payload = {
                "id": ex.get("id"),
                "name": name,
                "force": ex.get("force"),
                "level": level,
                "mechanic": ex.get("mechanic"),
                "equipment": ex.get("equipment"),
                "primaryMuscles": primary_muscles,
                "secondaryMuscles": secondary_muscles,
                "instructions": instructions,
                "category": category,
            }
            payloads.append(payload)

        # Batch encode vectors
        logger.info("Computing dense embeddings for all exercises...")
        vectors = self.get_embeddings_batch(texts, batch_size=64)

        # Upsert into Qdrant in batches
        batch_size = 200
        logger.info(f"Upserting {total_exercises} exercise points to '{collection_name}'...")
        for i in range(0, total_exercises, batch_size):
            end_idx = min(i + batch_size, total_exercises)
            batch_points = [
                PointStruct(
                    id=j,
                    vector=vectors[j],
                    payload=payloads[j],
                )
                for j in range(i, end_idx)
            ]
            self.client.upsert(collection_name=collection_name, points=batch_points)

        info = self.client.get_collection(collection_name)
        logger.info(f"Successfully indexed {info.points_count} exercises into '{collection_name}'.")
        return info.points_count

    def index_nutrition(
        self,
        nutrition_file: Optional[Path] = None,
        tips_file: Optional[Path] = None,
        force_reindex: bool = False,
    ) -> int:
        """
        Index nutrition meal plans and nutritional protocols into 'apex_nutrition'.

        :param nutrition_file: Path to nutrition.json (defaults to data/nutrition.json).
        :param tips_file: Path to tips.json for protocols (defaults to data/tips.json).
        :param force_reindex: Whether to re-populate collection.
        :return: Count of indexed nutrition items.
        """
        if nutrition_file is None:
            nutrition_file = self.data_dir / "nutrition.json"
        if tips_file is None:
            tips_file = self.data_dir / "tips.json"

        collection_name = self.NUTRITION_COLLECTION

        if self.client.collection_exists(collection_name):
            if force_reindex:
                logger.info(f"Reindexing requested. Deleting collection '{collection_name}'...")
                self.client.delete_collection(collection_name)
            else:
                info = self.client.get_collection(collection_name)
                if info.points_count > 0:
                    logger.info(
                        f"Collection '{collection_name}' already populated with {info.points_count} points. Skipping."
                    )
                    return info.points_count

        logger.info(f"Creating collection '{collection_name}'...")
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
        )

        texts = []
        payloads = []
        item_id = 0

        # 1. Load Meals from nutrition.json
        if nutrition_file.exists():
            with open(nutrition_file, "r", encoding="utf-8") as f:
                nutrition_data = json.load(f)

            for meal in nutrition_data.get("meals", []):
                name = meal.get("name", "")
                m_type = meal.get("type", "meal")
                calories = meal.get("calories", 0)
                protein = meal.get("protein", 0)
                carbs = meal.get("carbs", 0)
                fat = meal.get("fat", 0)
                ingredients = meal.get("ingredients", [])
                goal = meal.get("goal", [])
                prep = meal.get("preparation", "")

                ingr_str = ", ".join(ingredients)
                goal_str = ", ".join(goal)

                embed_text = (
                    f"Nutritional Meal: {name}. High protein fitness meal. "
                    f"Meal Type: {m_type}. Nutrition: {protein}g protein, {calories} calories, "
                    f"{carbs}g carbs, {fat}g fat. "
                    f"Ingredients: {ingr_str}. Goals: {goal_str}. Preparation: {prep}"
                )
                texts.append(embed_text)

                payload = {
                    "id": f"meal_{item_id}",
                    "name": name,
                    "type": m_type,
                    "category": "meal",
                    "calories": calories,
                    "protein": protein,
                    "carbs": carbs,
                    "fat": fat,
                    "ingredients": ingredients,
                    "goal": goal,
                    "preparation": prep,
                    "summary": f"{name} ({calories} kcal, {protein}g protein)",
                }
                payloads.append(payload)
                item_id += 1

        # 2. Load Nutrition Protocols & Tips from tips.json
        if tips_file.exists():
            with open(tips_file, "r", encoding="utf-8") as f:
                tips_data = json.load(f)

            for tip in tips_data.get("tips", []):
                cat = tip.get("category", "")
                # Focus on nutrition and recovery protocols
                if cat in ("nutrition", "recovery"):
                    topic = tip.get("topic", "")
                    content = tip.get("content", "")

                    embed_text = (
                        f"Nutritional Protocol: {topic}. High protein and dietary guidelines. "
                        f"Category: {cat}. Protocol: {content}"
                    )
                    texts.append(embed_text)

                    payload = {
                        "id": f"protocol_{item_id}",
                        "name": topic,
                        "topic": topic,
                        "type": "protocol",
                        "category": cat,
                        "content": content,
                        "summary": f"{topic}: {content[:80]}...",
                    }
                    payloads.append(payload)
                    item_id += 1

        total_items = len(texts)
        if total_items > 0:
            logger.info(f"Computing embeddings for {total_items} nutrition items...")
            vectors = self.get_embeddings_batch(texts, batch_size=32)

            points = [
                PointStruct(id=i, vector=vectors[i], payload=payloads[i])
                for i in range(total_items)
            ]
            self.client.upsert(collection_name=collection_name, points=points)

        info = self.client.get_collection(collection_name)
        logger.info(f"Successfully indexed {info.points_count} nutrition documents into '{collection_name}'.")
        return info.points_count

    def search_exercises(
        self,
        query: str,
        muscle: Optional[str] = None,
        equipment: Optional[str] = None,
        level: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search exercises using dense vector similarity with optional metadata filtering.

        :param query: Free-text search query (e.g. 'bench press', 'squat', 'core strengthening').
        :param muscle: Optional target muscle filter (e.g. 'chest', 'quadriceps', 'quads', 'biceps').
        :param equipment: Optional equipment filter (e.g. 'barbell', 'dumbbell', 'body only').
        :param level: Optional fitness level filter ('beginner', 'intermediate', 'expert').
        :param limit: Number of top results to return (default 5).
        :return: List of exercise dictionaries ranked by similarity score.
        """
        query_vector = self.get_embedding(query)

        must_conditions: List[Union[Filter, FieldCondition]] = []

        # Filter by muscle (checks both primaryMuscles and secondaryMuscles)
        if muscle and muscle.strip():
            raw_muscle = muscle.strip().lower()
            normalized = MUSCLE_ALIASES.get(raw_muscle, raw_muscle)
            candidate_muscles = list({raw_muscle, normalized})

            muscle_or_conditions: List[FieldCondition] = []
            for m in candidate_muscles:
                muscle_or_conditions.append(
                    FieldCondition(key="primaryMuscles", match=MatchValue(value=m))
                )
                muscle_or_conditions.append(
                    FieldCondition(key="secondaryMuscles", match=MatchValue(value=m))
                )

            must_conditions.append(Filter(should=muscle_or_conditions))

        # Filter by equipment
        if equipment and equipment.strip():
            eq_val = equipment.strip().lower()
            must_conditions.append(
                FieldCondition(key="equipment", match=MatchValue(value=eq_val))
            )

        # Filter by level
        if level and level.strip():
            lvl_val = level.strip().lower()
            must_conditions.append(
                FieldCondition(key="level", match=MatchValue(value=lvl_val))
            )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        results = self.client.query_points(
            collection_name=self.EXERCISES_COLLECTION,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        formatted_results = []
        for point in results.points:
            record = dict(point.payload or {})
            record["score"] = float(point.score)
            record["point_id"] = point.id
            formatted_results.append(record)

        return formatted_results

    def search_nutrition(
        self,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search nutrition protocols and meal plans using dense vector similarity.

        :param query: Nutritional inquiry (e.g. 'high protein', 'caloric deficit', 'muscle building breakfast').
        :param limit: Maximum results to return (default 5).
        :return: List of nutrition documents ranked by similarity score.
        """
        query_vector = self.get_embedding(query)

        results = self.client.query_points(
            collection_name=self.NUTRITION_COLLECTION,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )

        formatted_results = []
        for point in results.points:
            record = dict(point.payload or {})
            record["score"] = float(point.score)
            record["point_id"] = point.id
            formatted_results.append(record)

        return formatted_results

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return diagnostic point counts and collection status."""
        stats = {}
        for coll in [self.EXERCISES_COLLECTION, self.NUTRITION_COLLECTION]:
            if self.client.collection_exists(coll):
                info = self.client.get_collection(coll)
                stats[coll] = {
                    "exists": True,
                    "points_count": info.points_count,
                    "status": str(info.status),
                }
            else:
                stats[coll] = {"exists": False, "points_count": 0}
        return stats

    def close(self):
        """Safely close QdrantClient instance."""
        if hasattr(self, "client") and self.client:
            try:
                self.client.close()
            except Exception:
                pass


# Global singleton instance helper
_engine_instance: Optional[VectorEngine] = None


def get_vector_engine(storage_path: Optional[str] = "./qdrant_storage") -> VectorEngine:
    """Retrieve or create singleton VectorEngine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = VectorEngine(storage_path=storage_path)
    return _engine_instance


if __name__ == "__main__":
    print("Testing VectorEngine standalone initialization...")
    engine = VectorEngine(storage_path="./qdrant_storage", force_reindex=True)
    stats = engine.get_collection_stats()
    print("Collection Stats:", stats)

    print("\nSearch Test: 'bench press'")
    exercises = engine.search_exercises("bench press", limit=3)
    for ex in exercises:
        print(f" - {ex['name']} | score: {ex['score']:.4f} | eq: {ex.get('equipment')} | muscles: {ex.get('primaryMuscles')}")

    print("\nSearch Test: 'high protein'")
    meals = engine.search_nutrition("high protein", limit=3)
    for m in meals:
        print(f" - {m.get('name')} | score: {m['score']:.4f} | type: {m.get('type')}")
    
    engine.close()
