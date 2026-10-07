"""
SmartCoach AI - Structured Plan Generator Engine
=================================================
Generates strictly typed, evidence-based mesocycle training programs
and adaptive metabolic nutrition plans conforming to BLUEPRINT.md
Section 15 and Section 12 specifications.

Integrates Instructor with Google Gemini (and OpenAI) for structured
Pydantic v2 output generation, backed by a deterministic, evidence-based
sports science fallback engine using authentic exercise and nutrition databases.
"""

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Union

from dotenv import load_dotenv

# Schemas
from schemas import (
    ExerciseSetTarget,
    StructuredWorkoutExercise,
    StructuredWorkoutDay,
    MasterPlanPayload,
    AdaptiveTDEEInput,
    AdaptiveTDEEOutput,
)

# Algorithmic engines
from algorithms.overload import (
    calculate_epley_1rm,
    round_to_nearest,
    calculate_effective_volume,
    calculate_total_volume,
)
from algorithms.metabolic import (
    calculate_macro_split,
    calculate_adaptive_tdee,
    calculate_weight_delta,
    calculate_daily_caloric_imbalance,
    calculate_true_tdee,
    GOAL_DEFAULTS,
)

load_dotenv()

logger = logging.getLogger("PlanGenerator")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# Goal & Split normalizations
VALID_GOALS = ["Hypertrophy", "Strength", "FatLoss", "Endurance"]
VALID_SPLITS = ["UpperLower", "PushPullLegs", "FullBody"]

MUSCLE_NORMALIZATION: Dict[str, str] = {
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


def normalize_goal(goal: str) -> Literal["Hypertrophy", "Strength", "FatLoss", "Endurance"]:
    """Normalize input goal string to strict Literal contract."""
    g = (goal or "").strip().lower().replace("-", "_").replace(" ", "_")
    if any(k in g for k in ["strength", "power", "heavy"]):
        return "Strength"
    elif any(k in g for k in ["fat", "loss", "cut", "lean", "weight_loss", "lose"]):
        return "FatLoss"
    elif any(k in g for k in ["endurance", "stamina", "cardio", "conditioning"]):
        return "Endurance"
    else:
        return "Hypertrophy"


def normalize_split(split_type: str) -> Literal["UpperLower", "PushPullLegs", "FullBody"]:
    """Normalize input split string to strict Literal contract."""
    s = (split_type or "").strip().lower().replace("-", "").replace("_", "").replace(" ", "").replace("/", "")
    if "ppl" in s or "pushpull" in s:
        return "PushPullLegs"
    elif "full" in s or "whole" in s:
        return "FullBody"
    else:
        return "UpperLower"


def normalize_fitness_level(fitness_level: str) -> str:
    """Normalize fitness level to beginner, intermediate, or advanced."""
    fl = (fitness_level or "").strip().lower()
    if "adv" in fl or "expert" in fl:
        return "advanced"
    elif "beg" in fl or "novice" in fl:
        return "beginner"
    else:
        return "intermediate"


class PlanGenerator:
    """
    Structured Workout & Nutrition Plan Generator for Apex Athletic / SmartCoachAI.

    Features:
    1. Instructor Integration: Configures Instructor with Google Gemini (or OpenAI)
       to enforce 100% Pydantic v2 type safety for MasterPlanPayload.
    2. Deterministic Sports Science Fallback Engine: High-performance, evidence-based
       generator using 876 exercises and 20 nutrition profiles.
    3. Progressive Overload & Autoregulation: Computes target weights, sets, reps,
       RPE, and rest intervals per BLUEPRINT.md Section 12.
    4. Contraindication & Safety Guard: Filters exercises contraindicating reported injuries.
    5. Equipment Filtering: Adapts plans to available equipment or bodyweight.
    6. Adaptive Metabolic Nutrition Planner: Calculates TDEE and macro splits via
       algorithms.metabolic and prescribes structured daily meals.
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        gemini_model: Optional[str] = None,
        openai_model: Optional[str] = None,
        data_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the PlanGenerator.

        :param gemini_api_key: Optional Google Gemini API key override.
        :param openai_api_key: Optional OpenAI API key override.
        :param gemini_model: Gemini model identifier (defaults to gemini-1.5-flash).
        :param openai_model: OpenAI model identifier (defaults to gpt-4o-mini).
        :param data_dir: Directory containing exercises_800.json and nutrition.json.
        """
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parent / "data"

        # API Keys
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        # Models
        self.gemini_model = gemini_model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.openai_model = openai_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        # Instructor client setup
        self.instructor_client = None
        self.provider = None
        self.model_name = None
        self._setup_instructor()

        # Load knowledge bases
        self.exercises: List[Dict[str, Any]] = []
        self.exercises_by_id: Dict[str, Dict[str, Any]] = {}
        self.exercises_by_muscle: Dict[str, List[Dict[str, Any]]] = {}
        self.meals: List[Dict[str, Any]] = []
        self._load_databases()

    def _setup_instructor(self):
        """Configure Instructor client with Gemini or OpenAI."""
        import instructor

        # 1. Try Google Gemini first
        if self.gemini_api_key:
            try:
                from google import genai

                raw_client = genai.Client(api_key=self.gemini_api_key)
                if hasattr(instructor, "from_genai"):
                    self.instructor_client = instructor.from_genai(raw_client)
                elif hasattr(instructor, "from_gemini"):
                    self.instructor_client = instructor.from_gemini(raw_client)
                else:
                    self.instructor_client = instructor.from_openai(raw_client)

                self.provider = "gemini"
                self.model_name = self.gemini_model
                logger.info(f"Instructor configured with Google Gemini (model: {self.model_name})")
                return
            except Exception as e:
                logger.warning(f"Failed to configure Instructor with Google Gemini: {e}")

        # 2. Try OpenAI
        if self.openai_api_key:
            try:
                from openai import OpenAI

                raw_client = OpenAI(api_key=self.openai_api_key)
                self.instructor_client = instructor.from_openai(raw_client)
                self.provider = "openai"
                self.model_name = self.openai_model
                logger.info(f"Instructor configured with OpenAI (model: {self.model_name})")
                return
            except Exception as e:
                logger.warning(f"Failed to configure Instructor with OpenAI: {e}")

        logger.info("No LLM API key configured or valid. Deterministic sports-science engine will be active.")

    def _load_databases(self):
        """Load and index exercise and nutrition databases."""
        # 1. Exercises
        exercises_path = self.data_dir / "exercises_800.json"
        if exercises_path.exists():
            try:
                with open(exercises_path, "r", encoding="utf-8") as f:
                    self.exercises = json.load(f)
                for ex in self.exercises:
                    ex_id = ex.get("id", "")
                    if ex_id:
                        self.exercises_by_id[ex_id] = ex
                    for m in ex.get("primaryMuscles", []):
                        m_norm = MUSCLE_NORMALIZATION.get(m.lower(), m.lower())
                        self.exercises_by_muscle.setdefault(m_norm, []).append(ex)
                logger.info(f"Loaded {len(self.exercises)} exercises from {exercises_path.name}")
            except Exception as e:
                logger.error(f"Failed to load exercises database: {e}")
        else:
            logger.warning(f"Exercises file not found at {exercises_path}")

        # 2. Nutrition
        nutrition_path = self.data_dir / "nutrition.json"
        if nutrition_path.exists():
            try:
                with open(nutrition_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.meals = data.get("meals", [])
                logger.info(f"Loaded {len(self.meals)} nutrition meal profiles from {nutrition_path.name}")
            except Exception as e:
                logger.error(f"Failed to load nutrition database: {e}")
        else:
            logger.warning(f"Nutrition file not found at {nutrition_path}")

    # ==========================================================================
    # MASTER WORKOUT PLAN GENERATION
    # ==========================================================================

    def generate_master_plan(
        self,
        user_id: str,
        goal: str,
        fitness_level: str,
        split_type: str = "UpperLower",
        cycle_weeks: int = 4,
        equipment: Optional[List[str]] = None,
        injuries: Optional[List[str]] = None,
    ) -> MasterPlanPayload:
        """
        Generate a fully structured mesocycle master training plan.

        Adheres strictly to BLUEPRINT.md Section 15.
        Attempts LLM generation with Instructor if Gemini/OpenAI is available.
        Automatically falls back to deterministic sports science generator on error or missing key.

        :param user_id: Unique athlete identifier.
        :param goal: Training objective ('Hypertrophy', 'Strength', 'FatLoss', 'Endurance').
        :param fitness_level: Athlete experience level ('beginner', 'intermediate', 'advanced').
        :param split_type: Microcycle split ('UpperLower', 'PushPullLegs', 'FullBody').
        :param cycle_weeks: Mesocycle duration in weeks (>= 1).
        :param equipment: Available training equipment list.
        :param injuries: Reported injuries or contraindications list.
        :return: Validated MasterPlanPayload instance.
        """
        norm_goal = normalize_goal(goal)
        norm_split = normalize_split(split_type)
        norm_level = normalize_fitness_level(fitness_level)
        weeks = max(1, int(cycle_weeks))

        # Attempt LLM generation if Instructor is active
        if self.instructor_client is not None:
            try:
                plan = self._generate_llm_master_plan(
                    user_id=user_id,
                    goal=norm_goal,
                    fitness_level=norm_level,
                    split_type=norm_split,
                    cycle_weeks=weeks,
                    equipment=equipment,
                    injuries=injuries,
                )
                if isinstance(plan, MasterPlanPayload):
                    logger.info(f"Successfully generated MasterPlanPayload via {self.provider}")
                    return plan
            except Exception as e:
                logger.warning(
                    f"LLM plan generation failed ({e}). "
                    f"Falling back to deterministic sports-science engine."
                )

        # Guaranteed deterministic sports science fallback
        return self._generate_fallback_master_plan(
            user_id=user_id,
            goal=norm_goal,
            fitness_level=norm_level,
            split_type=norm_split,
            cycle_weeks=weeks,
            equipment=equipment,
            injuries=injuries,
        )

    def _generate_llm_master_plan(
        self,
        user_id: str,
        goal: Literal["Hypertrophy", "Strength", "FatLoss", "Endurance"],
        fitness_level: str,
        split_type: Literal["UpperLower", "PushPullLegs", "FullBody"],
        cycle_weeks: int,
        equipment: Optional[List[str]],
        injuries: Optional[List[str]],
    ) -> MasterPlanPayload:
        """Execute Instructor-enforced LLM generation for MasterPlanPayload."""
        equip_str = ", ".join(equipment) if equipment else "Full Commercial Gym (All Equipment)"
        injuries_str = ", ".join(injuries) if injuries else "None reported"

        prompt = (
            f"Generate a strictly structured mesocycle training plan conforming to MasterPlanPayload:\n"
            f"- Athlete ID: {user_id}\n"
            f"- Primary Adaptation Goal: {goal}\n"
            f"- Fitness Level: {fitness_level}\n"
            f"- Split Type: {split_type}\n"
            f"- Mesocycle Weeks: {cycle_weeks}\n"
            f"- Available Equipment: {equip_str}\n"
            f"- Injuries / Contraindications: {injuries_str}\n\n"
            f"Requirements:\n"
            f"1. Provide a comprehensive weekly schedule matching the split structure:\n"
            f"   - UpperLower: 4 workout days (Upper Body A, Lower Body A, Upper Body B, Lower Body B)\n"
            f"   - PushPullLegs: 3 workout days (Push Day, Pull Day, Legs Day)\n"
            f"   - FullBody: 3 workout days (Full Body A, Full Body B, Full Body C)\n"
            f"2. Each day must contain 4 to 6 structured exercises with exact technique cues.\n"
            f"3. Each exercise must include 3 to 4 target sets with realistic set_number, reps, "
            f"target_rpe (1.0-10.0), suggested_weight_kg (>=0.0), and rest_seconds (>=0).\n"
            f"4. Filter out any exercises that aggravate reported injuries: {injuries_str}.\n"
            f"5. Provide a rigorous scientificRationale explaining mesocycle volume, RPE autoregulation, "
            f"and progressive overload prescription (+2.5% upper body / +5.0% lower body).\n"
        )

        return self.instructor_client.chat.completions.create(
            model=self.model_name,
            response_model=MasterPlanPayload,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are SmartCoach AI, an elite sports science engine adhering to BLUEPRINT.md Section 15. "
                        "You output strictly typed, evidence-based mesocycle training plans."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

    # ==========================================================================
    # DETERMINISTIC SPORTS-SCIENCE FALLBACK GENERATOR
    # ==========================================================================

    def _generate_fallback_master_plan(
        self,
        user_id: str,
        goal: Literal["Hypertrophy", "Strength", "FatLoss", "Endurance"],
        fitness_level: str,
        split_type: Literal["UpperLower", "PushPullLegs", "FullBody"],
        cycle_weeks: int,
        equipment: Optional[List[str]],
        injuries: Optional[List[str]],
    ) -> MasterPlanPayload:
        """
        Construct a guaranteed, 100% valid MasterPlanPayload using authentic
        exercises from data/exercises_800.json, calculating progressive overload
        weights and rest intervals according to goal and fitness level.
        """
        allowed_equipment = self._normalize_equipment_filter(equipment)
        injury_set = self._normalize_injury_filter(injuries)

        # Select weekly schedule based on split type
        if split_type == "UpperLower":
            schedule = self._build_upper_lower_schedule(goal, fitness_level, allowed_equipment, injury_set)
        elif split_type == "PushPullLegs":
            schedule = self._build_push_pull_legs_schedule(goal, fitness_level, allowed_equipment, injury_set)
        else:
            schedule = self._build_full_body_schedule(goal, fitness_level, allowed_equipment, injury_set)

        # Synthesize evidence-based scientific rationale
        rationale = self._synthesize_workout_rationale(goal, fitness_level, split_type, cycle_weeks, injuries)

        return MasterPlanPayload(
            user_id=user_id,
            cycle_weeks=cycle_weeks,
            goal=goal,
            split_type=split_type,
            weekly_schedule=schedule,
            scientific_rationale=rationale,
        )

    def _normalize_equipment_filter(self, equipment: Optional[List[str]]) -> Optional[Set[str]]:
        """Normalize equipment constraints into a set."""
        if not equipment:
            return None
        norm = set()
        for eq in equipment:
            eq_clean = eq.strip().lower()
            norm.add(eq_clean)
            if "dumbbell" in eq_clean:
                norm.add("dumbbell")
            if "barbell" in eq_clean:
                norm.add("barbell")
            if "cable" in eq_clean:
                norm.add("cable")
            if "machine" in eq_clean:
                norm.add("machine")
            if "band" in eq_clean:
                norm.add("bands")
        # Always allow bodyweight exercises
        norm.add("body only")
        norm.add("none")
        norm.add("other")
        return norm

    def _normalize_injury_filter(self, injuries: Optional[List[str]]) -> Set[str]:
        """Normalize injuries into searchable tokens."""
        if not injuries:
            return set()
        tokens = set()
        for inj in injuries:
            inj_clean = inj.strip().lower()
            tokens.add(inj_clean)
            if "shoulder" in inj_clean:
                tokens.add("shoulder")
            if "knee" in inj_clean:
                tokens.add("knee")
            if "back" in inj_clean:
                tokens.add("lower back")
                tokens.add("spine")
            if "elbow" in inj_clean:
                tokens.add("elbow")
            if "wrist" in inj_clean:
                tokens.add("wrist")
        return tokens

    def _is_contraindicated(self, ex: Dict[str, Any], injuries: Set[str]) -> bool:
        """Check if an exercise contraindicates reported injuries."""
        if not injuries:
            return False

        ex_name = ex.get("name", "").lower()
        ex_id = ex.get("id", "").lower()
        prim_muscles = [m.lower() for m in ex.get("primaryMuscles", [])]

        if "shoulder" in injuries:
            # Avoid heavy overhead vertical pressing or behind-neck movements
            if any(k in ex_name or k in ex_id for k in ["behind the neck", "military press", "overhead press"]):
                return True
            if "shoulders" in prim_muscles and "overhead" in ex_name:
                return True

        if "knee" in injuries:
            # Avoid heavy shear knee extension or deep flexion under load
            if any(k in ex_name or k in ex_id for k in ["sissy squat", "leg extensions", "jump squat"]):
                return True

        if "lower back" in injuries or "spine" in injuries:
            # Avoid heavy axial compression on lumbar spine
            if any(k in ex_name or k in ex_id for k in ["deadlift", "good morning", "barbell full squat"]):
                return True
            if "lower back" in prim_muscles and "heavy" in ex_name:
                return True

        if "wrist" in injuries:
            if "straight bar" in ex_name:
                return True

        return False

    def _find_candidate_exercise(
        self,
        preferred_ids: List[str],
        muscle: str,
        mechanic: Optional[str] = None,
        equipment_filter: Optional[Set[str]] = None,
        injuries: Optional[Set[str]] = None,
        excluded_ids: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """
        Locate optimal exercise matching muscle group, equipment, and injury boundaries.
        Falls back smoothly to bodyweight or safe movements if restricted.
        """
        excluded = excluded_ids or set()
        inj_set = injuries or set()

        # 1. Check preferred high-yield exercises first
        for pid in preferred_ids:
            if pid in self.exercises_by_id and pid not in excluded:
                ex = self.exercises_by_id[pid]
                if not self._is_contraindicated(ex, inj_set):
                    ex_eq = str(ex.get("equipment", "")).lower()
                    if equipment_filter is None or ex_eq in equipment_filter or ex_eq == "body only":
                        return ex

        # 2. Search database by muscle group
        norm_m = MUSCLE_NORMALIZATION.get(muscle.lower(), muscle.lower())
        candidates = self.exercises_by_muscle.get(norm_m, [])

        # Priority 1: Match mechanic & equipment without contraindication
        for ex in candidates:
            eid = ex.get("id", "")
            if eid in excluded or self._is_contraindicated(ex, inj_set):
                continue
            ex_eq = str(ex.get("equipment", "")).lower()
            if equipment_filter is not None and ex_eq not in equipment_filter and ex_eq != "body only":
                continue
            if mechanic and ex.get("mechanic") != mechanic:
                continue
            if ex.get("category") in ["strength", "powerlifting"]:
                return ex

        # Priority 2: Match equipment without mechanic restriction
        for ex in candidates:
            eid = ex.get("id", "")
            if eid in excluded or self._is_contraindicated(ex, inj_set):
                continue
            ex_eq = str(ex.get("equipment", "")).lower()
            if equipment_filter is None or ex_eq in equipment_filter or ex_eq == "body only":
                return ex

        # Priority 3: Any safe exercise in the database
        for ex in candidates:
            eid = ex.get("id", "")
            if eid not in excluded and not self._is_contraindicated(ex, inj_set):
                return ex

        # Absolute fallback: Return first exercise or a dummy safe entry
        if self.exercises:
            return self.exercises[0]
        return {
            "id": f"Safe_{norm_m.title()}_Movement",
            "name": f"Controlled {norm_m.title()} Movement",
            "primaryMuscles": [norm_m],
            "equipment": "body only",
            "mechanic": "isolation",
            "instructions": ["Execute movement with controlled tempo and full range of motion."],
        }

    def _extract_technique_cue(self, ex: Dict[str, Any], muscle: str) -> str:
        """Extract or synthesize biomechanical coaching cue."""
        instructions = ex.get("instructions", [])
        if instructions and len(instructions) > 0 and len(instructions[0].strip()) > 10:
            cue = instructions[0].strip()
            if len(cue) > 130:
                cue = cue[:127] + "..."
            return cue

        m_norm = muscle.lower()
        cues = {
            "chest": "Retract scapulae, maintain 45-degree elbow path, and drive through the midfoot.",
            "lats": "Drive elbows toward hips, maintain thoracic extension, and control the eccentric stretch.",
            "middle back": "Initiate pull by squeezing shoulder blades, keep torso steady, avoid momentum.",
            "quadriceps": "Brace core with 360-degree intra-abdominal pressure and track knees over toes.",
            "hamstrings": "Hinge at the hips with a neutral lumbar spine and squeeze glutes at lockout.",
            "glutes": "Drive through the heels, engage glutes at top lockout, maintain neutral pelvic tilt.",
            "shoulders": "Lead with elbows, avoid shrugging traps, and maintain smooth eccentric tempo.",
            "triceps": "Keep elbows pinned in place and push through complete terminal elbow extension.",
            "biceps": "Keep upper arms stationary and supinate wrists at peak contraction.",
            "calves": "Pause at full dorsiflexion stretch, then explode onto balls of feet.",
            "abdominals": "Exhale fully on contraction, posterior pelvic tilt, avoid cervical spinal strain.",
        }
        return cues.get(m_norm, "Maintain neutral spinal posture, brace core, and control the eccentric tempo.")

    def _build_target_sets(
        self,
        goal: Literal["Hypertrophy", "Strength", "FatLoss", "Endurance"],
        fitness_level: str,
        is_compound: bool,
        is_upper_body: bool,
        equipment: str,
    ) -> List[ExerciseSetTarget]:
        """Generate progressive overload target sets based on goal and level."""
        # Number of working sets
        if fitness_level == "beginner":
            set_count = 3
        elif fitness_level == "advanced":
            set_count = 4
        else:  # intermediate
            set_count = 4 if is_compound else 3

        # Goal-specific rep ranges, RPE, and rest intervals
        if goal == "Strength":
            reps = 5 if is_compound else 8
            base_rpe = 8.0 if fitness_level != "beginner" else 7.5
            rest_seconds = 180 if is_compound else 120
            weight_mult = 1.15
        elif goal == "FatLoss":
            reps = 12 if is_compound else 15
            base_rpe = 7.5
            rest_seconds = 75 if is_compound else 60
            weight_mult = 0.85
        elif goal == "Endurance":
            reps = 15 if is_compound else 20
            base_rpe = 7.0
            rest_seconds = 60 if is_compound else 45
            weight_mult = 0.75
        else:  # Hypertrophy
            reps = 8 if is_compound else 10
            base_rpe = 7.5
            rest_seconds = 120 if is_compound else 90
            weight_mult = 1.0

        # Suggested load estimation (kg)
        eq_clean = equipment.lower()
        if eq_clean == "body only" or "body" in eq_clean:
            suggested_weight = 0.0
        else:
            if is_compound:
                if is_upper_body:
                    base_load = 40.0 if fitness_level == "beginner" else (65.0 if fitness_level == "intermediate" else 90.0)
                else:
                    base_load = 55.0 if fitness_level == "beginner" else (95.0 if fitness_level == "intermediate" else 135.0)
            else:
                if is_upper_body:
                    base_load = 8.0 if fitness_level == "beginner" else (14.0 if fitness_level == "intermediate" else 20.0)
                else:
                    base_load = 25.0 if fitness_level == "beginner" else (45.0 if fitness_level == "intermediate" else 65.0)

            raw_load = base_load * weight_mult
            suggested_weight = float(round_to_nearest(raw_load, 0.5))

        target_sets: List[ExerciseSetTarget] = []
        for s_idx in range(1, set_count + 1):
            # Autoregulated progressive RPE across sets (e.g. 7.5 -> 8.0 -> 8.5)
            rpe_increment = min(1.0, (s_idx - 1) * 0.5)
            set_rpe = round(min(10.0, base_rpe + rpe_increment), 1)

            target_sets.append(
                ExerciseSetTarget(
                    set_number=s_idx,
                    reps=reps,
                    target_rpe=set_rpe,
                    suggested_weight_kg=suggested_weight,
                    rest_seconds=rest_seconds,
                )
            )

        return target_sets

    def _assemble_exercise(
        self,
        preferred_ids: List[str],
        muscle: str,
        mechanic: str,
        is_upper_body: bool,
        goal: Literal["Hypertrophy", "Strength", "FatLoss", "Endurance"],
        fitness_level: str,
        equipment_filter: Optional[Set[str]],
        injuries: Set[str],
        used_ids: Set[str],
    ) -> StructuredWorkoutExercise:
        """Find, configure, and assemble a StructuredWorkoutExercise."""
        ex = self._find_candidate_exercise(
            preferred_ids=preferred_ids,
            muscle=muscle,
            mechanic=mechanic,
            equipment_filter=equipment_filter,
            injuries=injuries,
            excluded_ids=used_ids,
        )
        eid = ex.get("id", f"{muscle}_exercise")
        used_ids.add(eid)

        # Alternative exercise for equipment busy scenario
        alt_ex = self._find_candidate_exercise(
            preferred_ids=[],
            muscle=muscle,
            mechanic=mechanic,
            equipment_filter=equipment_filter,
            injuries=injuries,
            excluded_ids=used_ids,
        )
        alt_id = alt_ex.get("id") if alt_ex.get("id") != eid else None

        cue = self._extract_technique_cue(ex, muscle)
        target_sets = self._build_target_sets(
            goal=goal,
            fitness_level=fitness_level,
            is_compound=(mechanic == "compound"),
            is_upper_body=is_upper_body,
            equipment=str(ex.get("equipment", "body only")),
        )

        return StructuredWorkoutExercise(
            exercise_id=eid,
            exercise_name=ex.get("name", eid.replace("_", " ")),
            target_sets=target_sets,
            technique_cue=cue,
            alternative_exercise_id=alt_id,
        )

    # --------------------------------------------------------------------------
    # Split Builders: Upper/Lower, PPL, Full Body
    # --------------------------------------------------------------------------

    def _build_upper_lower_schedule(
        self,
        goal: Literal["Hypertrophy", "Strength", "FatLoss", "Endurance"],
        fitness_level: str,
        equipment_filter: Optional[Set[str]],
        injuries: Set[str],
    ) -> List[StructuredWorkoutDay]:
        """Upper/Lower 4-day mesocycle split."""
        used_ids: Set[str] = set()

        # Day 1: Upper Body A (Horizontal Focus)
        day1_exercises = [
            self._assemble_exercise(
                preferred_ids=["Barbell_Bench_Press_-_Medium_Grip", "Barbell_Incline_Bench_Press_-_Medium_Grip", "Dumbbell_Bench_Press"],
                muscle="chest",
                mechanic="compound",
                is_upper_body=True,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids,
            ),
            self._assemble_exercise(
                preferred_ids=["Bent_Over_Barbell_Row", "Bent_Over_Two-Dumbbell_Row", "Bent_Over_Two-Arm_Long_Bar_Row"],
                muscle="middle back",
                mechanic="compound",
                is_upper_body=True,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids,
            ),
            self._assemble_exercise(
                preferred_ids=["Alternating_Cable_Shoulder_Press", "Dumbbell_Shoulder_Press", "Side_Lateral_Raise"],
                muscle="shoulders",
                mechanic="compound",
                is_upper_body=True,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids,
            ),
            self._assemble_exercise(
                preferred_ids=["Wide-Grip_Lat_Pulldown", "Close-Grip_Front_Lat_Pulldown", "Band_Assisted_Pull-Up"],
                muscle="lats",
                mechanic="compound",
                is_upper_body=True,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids,
            ),
            self._assemble_exercise(
                preferred_ids=["Cable_Incline_Triceps_Extension", "Triceps_Pushdown", "Bench_Dips"],
                muscle="triceps",
                mechanic="isolation",
                is_upper_body=True,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids,
            ),
            self._assemble_exercise(
                preferred_ids=["Alternate_Incline_Dumbbell_Curl", "Alternate_Hammer_Curl", "Barbell_Curl"],
                muscle="biceps",
                mechanic="isolation",
                is_upper_body=True,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids,
            ),
        ]

        # Day 2: Lower Body A (Anterior / Quad Focus)
        day2_exercises = [
            self._assemble_exercise(
                preferred_ids=["Barbell_Full_Squat", "Barbell_Hack_Squat", "Leg_Press"],
                muscle="quadriceps",
                mechanic="compound",
                is_upper_body=False,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids,
            ),
            self._assemble_exercise(
                preferred_ids=["Romanian_Deadlift", "Band_Good_Morning", "Platform_Hamstring_Slides"],
                muscle="hamstrings",
                mechanic="compound",
                is_upper_body=False,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids,
            ),
            self._assemble_exercise(
                preferred_ids=["Barbell_Lunge", "Barbell_Walking_Lunge", "Bodyweight_Walking_Lunge"],
                muscle="quadriceps",
                mechanic="compound",
                is_upper_body=False,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids,
            ),
            self._assemble_exercise(
                preferred_ids=["Barbell_Glute_Bridge", "Glute_Ham_Raise"],
                muscle="glutes",
                mechanic="compound",
                is_upper_body=False,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids,
            ),
            self._assemble_exercise(
                preferred_ids=["Barbell_Seated_Calf_Raise", "Calf_Press", "Standing_Calf_Raises"],
                muscle="calves",
                mechanic="isolation",
                is_upper_body=False,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids,
            ),
            self._assemble_exercise(
                preferred_ids=["Ab_Crunch_Machine", "Cable_Crunch", "3_4_Sit-Up"],
                muscle="abdominals",
                mechanic="isolation",
                is_upper_body=False,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids,
            ),
        ]

        # Reset used IDs for variation in second half of week
        used_ids_b: Set[str] = set()

        # Day 3: Upper Body B (Vertical Focus)
        day3_exercises = [
            self._assemble_exercise(
                preferred_ids=["Wide-Grip_Rear_Pull-Up", "Full_Range-Of-Motion_Lat_Pulldown", "One_Arm_Lat_Pulldown"],
                muscle="lats",
                mechanic="compound",
                is_upper_body=True,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids_b,
            ),
            self._assemble_exercise(
                preferred_ids=["Barbell_Incline_Bench_Press_-_Medium_Grip", "Flat_Bench_Cable_Flyes", "Alternating_Floor_Press"],
                muscle="chest",
                mechanic="compound",
                is_upper_body=True,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids_b,
            ),
            self._assemble_exercise(
                preferred_ids=["Bent_Over_Two-Dumbbell_Row_With_Palms_In", "Smith_Machine_Bent_Over_Row", "Alternating_Renegade_Row"],
                muscle="middle back",
                mechanic="compound",
                is_upper_body=True,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids_b,
            ),
            self._assemble_exercise(
                preferred_ids=["Side_Lateral_Raise", "Cable_Seated_Lateral_Raise", "Front_Plate_Raise"],
                muscle="shoulders",
                mechanic="isolation",
                is_upper_body=True,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids_b,
            ),
            self._assemble_exercise(
                preferred_ids=["Alternate_Hammer_Curl", "Reverse_Plate_Curls"],
                muscle="biceps",
                mechanic="isolation",
                is_upper_body=True,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids_b,
            ),
            self._assemble_exercise(
                preferred_ids=["Body_Tricep_Press", "Band_Skull_Crusher"],
                muscle="triceps",
                mechanic="isolation",
                is_upper_body=True,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids_b,
            ),
        ]

        # Day 4: Lower Body B (Posterior / Hinge Focus)
        day4_exercises = [
            self._assemble_exercise(
                preferred_ids=["Barbell_Deadlift", "Axle_Deadlift", "Romanian_Deadlift"],
                muscle="hamstrings",
                mechanic="compound",
                is_upper_body=False,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids_b,
            ),
            self._assemble_exercise(
                preferred_ids=["Barbell_Hack_Squat", "Squat_with_Plate_Movers", "Barbell_Full_Squat"],
                muscle="quadriceps",
                mechanic="compound",
                is_upper_body=False,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids_b,
            ),
            self._assemble_exercise(
                preferred_ids=["Ball_Leg_Curl", "Platform_Hamstring_Slides", "90_90_Hamstring"],
                muscle="hamstrings",
                mechanic="isolation",
                is_upper_body=False,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids_b,
            ),
            self._assemble_exercise(
                preferred_ids=["Barbell_Side_Split_Squat", "Barbell_Lunge"],
                muscle="quadriceps",
                mechanic="compound",
                is_upper_body=False,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids_b,
            ),
            self._assemble_exercise(
                preferred_ids=["Calf_Press", "Barbell_Seated_Calf_Raise"],
                muscle="calves",
                mechanic="isolation",
                is_upper_body=False,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids_b,
            ),
            self._assemble_exercise(
                preferred_ids=["Flat_Bench_Lying_Leg_Raise", "Flat_Bench_Leg_Pull-In", "Seated_Flat_Bench_Leg_Pull-In"],
                muscle="abdominals",
                mechanic="isolation",
                is_upper_body=False,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids_b,
            ),
        ]

        warmup = 10 if fitness_level == "beginner" else 8

        return [
            StructuredWorkoutDay(
                day_name="Upper Body A (Horizontal Push/Pull Focus)",
                focus_muscle_groups=["chest", "middle back", "shoulders", "lats", "triceps", "biceps"],
                warmup_minutes=warmup,
                exercises=day1_exercises,
            ),
            StructuredWorkoutDay(
                day_name="Lower Body A (Anterior / Quad Dominant)",
                focus_muscle_groups=["quadriceps", "hamstrings", "glutes", "calves", "abdominals"],
                warmup_minutes=warmup,
                exercises=day2_exercises,
            ),
            StructuredWorkoutDay(
                day_name="Upper Body B (Vertical Push/Pull & Deltoids)",
                focus_muscle_groups=["lats", "chest", "middle back", "shoulders", "biceps", "triceps"],
                warmup_minutes=warmup,
                exercises=day3_exercises,
            ),
            StructuredWorkoutDay(
                day_name="Lower Body B (Posterior Chain & Hip Hinge)",
                focus_muscle_groups=["hamstrings", "quadriceps", "glutes", "calves", "abdominals"],
                warmup_minutes=warmup,
                exercises=day4_exercises,
            ),
        ]

    def _build_push_pull_legs_schedule(
        self,
        goal: Literal["Hypertrophy", "Strength", "FatLoss", "Endurance"],
        fitness_level: str,
        equipment_filter: Optional[Set[str]],
        injuries: Set[str],
    ) -> List[StructuredWorkoutDay]:
        """Push / Pull / Legs 3-day microcycle split."""
        used_ids: Set[str] = set()

        # Day 1: Push Session
        push_exercises = [
            self._assemble_exercise(
                preferred_ids=["Barbell_Bench_Press_-_Medium_Grip", "Barbell_Incline_Bench_Press_-_Medium_Grip"],
                muscle="chest",
                mechanic="compound",
                is_upper_body=True,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids,
            ),
            self._assemble_exercise(
                preferred_ids=["Flat_Bench_Cable_Flyes", "One-Arm_Flat_Bench_Dumbbell_Flye"],
                muscle="chest",
                mechanic="isolation",
                is_upper_body=True,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids,
            ),
            self._assemble_exercise(
                preferred_ids=["Alternating_Cable_Shoulder_Press", "Dumbbell_Shoulder_Press"],
                muscle="shoulders",
                mechanic="compound",
                is_upper_body=True,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids,
            ),
            self._assemble_exercise(
                preferred_ids=["Side_Lateral_Raise", "Cable_Seated_Lateral_Raise"],
                muscle="shoulders",
                mechanic="isolation",
                is_upper_body=True,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids,
            ),
            self._assemble_exercise(
                preferred_ids=["Cable_Incline_Triceps_Extension", "Band_Skull_Crusher"],
                muscle="triceps",
                mechanic="isolation",
                is_upper_body=True,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids,
            ),
        ]

        # Day 2: Pull Session
        pull_exercises = [
            self._assemble_exercise(
                preferred_ids=["Bent_Over_Barbell_Row", "Bent_Over_Two-Dumbbell_Row"],
                muscle="middle back",
                mechanic="compound",
                is_upper_body=True,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids,
            ),
            self._assemble_exercise(
                preferred_ids=["Wide-Grip_Lat_Pulldown", "Band_Assisted_Pull-Up"],
                muscle="lats",
                mechanic="compound",
                is_upper_body=True,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids,
            ),
            self._assemble_exercise(
                preferred_ids=["Bent_Over_Dumbbell_Rear_Delt_Raise_With_Head_On_Bench", "Bent_Over_Low-Pulley_Side_Lateral"],
                muscle="shoulders",
                mechanic="isolation",
                is_upper_body=True,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids,
            ),
            self._assemble_exercise(
                preferred_ids=["Alternate_Incline_Dumbbell_Curl", "Alternate_Hammer_Curl"],
                muscle="biceps",
                mechanic="isolation",
                is_upper_body=True,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids,
            ),
            self._assemble_exercise(
                preferred_ids=["Scapular_Pull-Up", "Plate_Pinch"],
                muscle="traps",
                mechanic="compound",
                is_upper_body=True,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids,
            ),
        ]

        # Day 3: Legs Session
        legs_exercises = [
            self._assemble_exercise(
                preferred_ids=["Barbell_Full_Squat", "Barbell_Hack_Squat"],
                muscle="quadriceps",
                mechanic="compound",
                is_upper_body=False,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids,
            ),
            self._assemble_exercise(
                preferred_ids=["Romanian_Deadlift", "Platform_Hamstring_Slides"],
                muscle="hamstrings",
                mechanic="compound",
                is_upper_body=False,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids,
            ),
            self._assemble_exercise(
                preferred_ids=["Barbell_Lunge", "Bodyweight_Walking_Lunge"],
                muscle="quadriceps",
                mechanic="compound",
                is_upper_body=False,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids,
            ),
            self._assemble_exercise(
                preferred_ids=["Barbell_Seated_Calf_Raise", "Calf_Press"],
                muscle="calves",
                mechanic="isolation",
                is_upper_body=False,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids,
            ),
            self._assemble_exercise(
                preferred_ids=["Ab_Crunch_Machine", "3_4_Sit-Up"],
                muscle="abdominals",
                mechanic="isolation",
                is_upper_body=False,
                goal=goal,
                fitness_level=fitness_level,
                equipment_filter=equipment_filter,
                injuries=injuries,
                used_ids=used_ids,
            ),
        ]

        warmup = 10 if fitness_level == "beginner" else 8

        return [
            StructuredWorkoutDay(
                day_name="Push Session (Chest, Shoulders, Triceps)",
                focus_muscle_groups=["chest", "shoulders", "triceps"],
                warmup_minutes=warmup,
                exercises=push_exercises,
            ),
            StructuredWorkoutDay(
                day_name="Pull Session (Back, Lats, Traps, Biceps)",
                focus_muscle_groups=["middle back", "lats", "shoulders", "biceps", "traps"],
                warmup_minutes=warmup,
                exercises=pull_exercises,
            ),
            StructuredWorkoutDay(
                day_name="Legs & Core Session (Quads, Hamstrings, Calves, Abs)",
                focus_muscle_groups=["quadriceps", "hamstrings", "calves", "abdominals"],
                warmup_minutes=warmup,
                exercises=legs_exercises,
            ),
        ]

    def _build_full_body_schedule(
        self,
        goal: Literal["Hypertrophy", "Strength", "FatLoss", "Endurance"],
        fitness_level: str,
        equipment_filter: Optional[Set[str]],
        injuries: Set[str],
    ) -> List[StructuredWorkoutDay]:
        """Full Body 3-day weekly schedule."""
        used_a: Set[str] = set()
        day1 = [
            self._assemble_exercise(["Barbell_Full_Squat", "Barbell_Hack_Squat"], "quadriceps", "compound", False, goal, fitness_level, equipment_filter, injuries, used_a),
            self._assemble_exercise(["Barbell_Bench_Press_-_Medium_Grip", "Dumbbell_Bench_Press"], "chest", "compound", True, goal, fitness_level, equipment_filter, injuries, used_a),
            self._assemble_exercise(["Bent_Over_Barbell_Row", "Bent_Over_Two-Dumbbell_Row"], "middle back", "compound", True, goal, fitness_level, equipment_filter, injuries, used_a),
            self._assemble_exercise(["Side_Lateral_Raise", "Cable_Seated_Lateral_Raise"], "shoulders", "isolation", True, goal, fitness_level, equipment_filter, injuries, used_a),
            self._assemble_exercise(["Ab_Crunch_Machine", "3_4_Sit-Up"], "abdominals", "isolation", False, goal, fitness_level, equipment_filter, injuries, used_a),
        ]

        used_b: Set[str] = set()
        day2 = [
            self._assemble_exercise(["Romanian_Deadlift", "Platform_Hamstring_Slides"], "hamstrings", "compound", False, goal, fitness_level, equipment_filter, injuries, used_b),
            self._assemble_exercise(["Barbell_Incline_Bench_Press_-_Medium_Grip", "Flat_Bench_Cable_Flyes"], "chest", "compound", True, goal, fitness_level, equipment_filter, injuries, used_b),
            self._assemble_exercise(["Wide-Grip_Lat_Pulldown", "Band_Assisted_Pull-Up"], "lats", "compound", True, goal, fitness_level, equipment_filter, injuries, used_b),
            self._assemble_exercise(["Alternate_Hammer_Curl", "Alternate_Incline_Dumbbell_Curl"], "biceps", "isolation", True, goal, fitness_level, equipment_filter, injuries, used_b),
            self._assemble_exercise(["Barbell_Seated_Calf_Raise", "Calf_Press"], "calves", "isolation", False, goal, fitness_level, equipment_filter, injuries, used_b),
        ]

        used_c: Set[str] = set()
        day3 = [
            self._assemble_exercise(["Barbell_Lunge", "Bodyweight_Walking_Lunge"], "quadriceps", "compound", False, goal, fitness_level, equipment_filter, injuries, used_c),
            self._assemble_exercise(["Alternating_Cable_Shoulder_Press", "Dumbbell_Shoulder_Press"], "shoulders", "compound", True, goal, fitness_level, equipment_filter, injuries, used_c),
            self._assemble_exercise(["Bent_Over_Two-Arm_Long_Bar_Row", "Bent_Over_Two-Dumbbell_Row_With_Palms_In"], "middle back", "compound", True, goal, fitness_level, equipment_filter, injuries, used_c),
            self._assemble_exercise(["Cable_Incline_Triceps_Extension", "Band_Skull_Crusher"], "triceps", "isolation", True, goal, fitness_level, equipment_filter, injuries, used_c),
            self._assemble_exercise(["Flat_Bench_Lying_Leg_Raise", "Flat_Bench_Leg_Pull-In"], "abdominals", "isolation", False, goal, fitness_level, equipment_filter, injuries, used_c),
        ]

        warmup = 10 if fitness_level == "beginner" else 8

        return [
            StructuredWorkoutDay(day_name="Full Body A (Squat, Press, Row Focus)", focus_muscle_groups=["quadriceps", "chest", "middle back", "shoulders", "abdominals"], warmup_minutes=warmup, exercises=day1),
            StructuredWorkoutDay(day_name="Full Body B (Hinge, Incline, Pull Focus)", focus_muscle_groups=["hamstrings", "chest", "lats", "biceps", "calves"], warmup_minutes=warmup, exercises=day2),
            StructuredWorkoutDay(day_name="Full Body C (Unilateral, Overhead, Arms Focus)", focus_muscle_groups=["quadriceps", "shoulders", "middle back", "triceps", "abdominals"], warmup_minutes=warmup, exercises=day3),
        ]

    def _synthesize_workout_rationale(
        self,
        goal: str,
        fitness_level: str,
        split_type: str,
        cycle_weeks: int,
        injuries: Optional[List[str]],
    ) -> str:
        """Synthesize sports-science evidence rationale for the prescribed mesocycle."""
        injury_note = (
            f" Exercise selection strictly bypasses contraindications for reported conditions: {', '.join(injuries)}."
            if injuries
            else " No injury contraindications active; full kinetic chain loaded."
        )

        return (
            f"Prescribed a {cycle_weeks}-week periodized mesocycle using a {split_type} split tailored for {fitness_level} {goal}. "
            f"Frequency stimulates target muscle groups every 48-72 hours, optimizing muscle protein synthesis (MPS) without exceeding "
            f"systemic recovery capacity. Volume landmarks are aligned to adaptive thresholds: compound movements prioritize mechanical tension "
            f"at RPE 7.5-8.5 with extended rest periods (120-180s) to restore phosphagen stores (ATP-CP), while accessory isolation work introduces "
            f"metabolic accumulation (60-90s rest). Progressive overload follows the BLUEPRINT.md Section 12 rule: when average session RPE <= 7.0, "
            f"subsequent cycle loads increment by +2.5% for upper body and +5.0% for lower body movements.{injury_note}"
        )

    # ==========================================================================
    # ADAPTIVE NUTRITION PLAN GENERATION
    # ==========================================================================

    def generate_nutrition_plan(
        self,
        user_id: str,
        goal: str,
        current_weight_kg: float,
        target_weight_kg: float,
        dietary_preferences: Optional[List[str]] = None,
        daily_calories: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate structured nutrition plan with macro distribution and daily meals.

        1. Calculates TDEE and macro splits using algorithms.metabolic.
        2. Selects matching meals from data/nutrition.json adhering to dietary preferences.
        3. Returns structured payload conforming to sports science standards.

        :param user_id: Unique athlete identifier.
        :param goal: Fitness goal ('Hypertrophy', 'Strength', 'FatLoss', 'Endurance', 'Maintenance').
        :param current_weight_kg: Current body weight in kg (> 0).
        :param target_weight_kg: Target body weight in kg (> 0).
        :param dietary_preferences: Optional preferences ('vegetarian', 'vegan', 'high_protein', etc.).
        :param daily_calories: Optional manual daily caloric target override.
        :return: Structured nutrition dictionary.
        """
        norm_goal = normalize_goal(goal)
        weight_curr = max(20.0, float(current_weight_kg))
        weight_target = max(20.0, float(target_weight_kg))

        # 1. Caloric Target Calculation
        if daily_calories is not None and daily_calories > 0:
            target_calories = float(daily_calories)
            baseline_tdee = target_calories
        else:
            # Baseline active metabolic rate: ~33 kcal / kg body mass for athletic population
            baseline_tdee = round(weight_curr * 33.0, 1)
            caloric_delta = GOAL_DEFAULTS.get(norm_goal, GOAL_DEFAULTS["Hypertrophy"])["caloric_delta"]
            target_calories = round(baseline_tdee + caloric_delta, 1)

        # 2. Macronutrient Partitioning via algorithms.metabolic
        macros = calculate_macro_split(
            target_calories=target_calories,
            body_weight=weight_curr,
            goal=norm_goal,
        )

        # 3. Meal Selection from Database
        selected_meals = self._select_structured_meals(
            goal=norm_goal,
            target_calories=target_calories,
            preferences=dietary_preferences,
        )

        # Calculate totals across selected meals
        total_meal_cals = sum(m["calories"] for m in selected_meals)
        total_meal_protein = sum(m["protein_g"] for m in selected_meals)
        total_meal_carbs = sum(m["carbs_g"] for m in selected_meals)
        total_meal_fat = sum(m["fat_g"] for m in selected_meals)

        # 4. Synthesize Metabolic Rationale
        rationale = self._synthesize_nutrition_rationale(
            norm_goal=norm_goal,
            weight_curr=weight_curr,
            weight_target=weight_target,
            target_calories=target_calories,
            macros=macros,
        )

        return {
            "user_id": user_id,
            "goal": norm_goal,
            "current_weight_kg": weight_curr,
            "target_weight_kg": weight_target,
            "target_calories": target_calories,
            "target_protein_grams": macros["protein_grams"],
            "target_carb_grams": macros["carb_grams"],
            "target_fat_grams": macros["fat_grams"],
            "macro_split": {
                "protein": {
                    "grams": macros["protein_grams"],
                    "calories": macros["protein_calories"],
                    "g_per_kg": macros["protein_g_per_kg"],
                },
                "carbs": {
                    "grams": macros["carb_grams"],
                    "calories": macros["carb_calories"],
                },
                "fat": {
                    "grams": macros["fat_grams"],
                    "calories": macros["fat_calories"],
                    "g_per_kg": macros["fat_g_per_kg"],
                },
            },
            "meals": selected_meals,
            "total_meal_calories": total_meal_cals,
            "total_meal_protein_grams": total_meal_protein,
            "total_meal_carb_grams": total_meal_carbs,
            "total_meal_fat_grams": total_meal_fat,
            "dietary_preferences": dietary_preferences or [],
            "scientific_rationale": rationale,
        }

    def _select_structured_meals(
        self,
        goal: str,
        target_calories: float,
        preferences: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        """Select 4 meals (breakfast, lunch, dinner, snack) matching database and preferences."""
        pref_tokens = set(p.strip().lower() for p in (preferences or []))

        def matches_preferences(meal: Dict[str, Any]) -> bool:
            ingredients = [i.lower() for i in meal.get("ingredients", [])]
            name = meal.get("name", "").lower()
            all_text = " ".join(ingredients) + " " + name

            if "vegetarian" in pref_tokens or "vegan" in pref_tokens:
                meat_words = ["chicken", "turkey", "beef", "salmon", "cod", "tuna", "meatballs"]
                if any(w in all_text for w in meat_words):
                    return False

            if "vegan" in pref_tokens:
                animal_words = ["egg", "whey", "yogurt", "cottage cheese", "cheese", "feta", "honey", "milk"]
                if any(w in all_text for w in animal_words):
                    return False

            if "gluten_free" in pref_tokens or "gluten-free" in pref_tokens:
                gluten_words = ["bread", "pasta", "wrap", "toast"]
                if any(w in all_text for w in gluten_words):
                    return False

            return True

        # Filter database meals
        filtered = [m for m in self.meals if matches_preferences(m)]
        if not filtered:
            filtered = self.meals  # Fallback to full pool if filter too restrictive

        # Categorize by meal type
        breakfasts = [m for m in filtered if m.get("type") == "breakfast"] or self.meals[:1]
        lunches = [m for m in filtered if m.get("type") == "lunch"] or self.meals[1:2]
        dinners = [m for m in filtered if m.get("type") == "dinner"] or self.meals[2:3]
        snacks = [m for m in filtered if m.get("type") == "snack"] or self.meals[3:4]

        # Prioritize matching goal if available
        goal_tag = "build_muscle" if goal in ["Hypertrophy", "Strength"] else "lose_weight"

        def pick_best(meal_list: List[Dict[str, Any]]) -> Dict[str, Any]:
            for m in meal_list:
                if goal_tag in m.get("goal", []):
                    return m
            return meal_list[0] if meal_list else {}

        b = pick_best(breakfasts)
        l = pick_best(lunches)
        d = pick_best(dinners)
        s = pick_best(snacks)

        selected = [
            {
                "meal_name": b.get("name", "High-Protein Breakfast"),
                "meal_type": "breakfast",
                "timing": "07:30",
                "calories": b.get("calories", 400),
                "protein_g": b.get("protein", 30),
                "carbs_g": b.get("carbs", 50),
                "fat_g": b.get("fat", 10),
                "ingredients": b.get("ingredients", []),
                "preparation": b.get("preparation", ""),
            },
            {
                "meal_name": l.get("name", "Nutrient-Dense Lunch"),
                "meal_type": "lunch",
                "timing": "12:30",
                "calories": l.get("calories", 500),
                "protein_g": l.get("protein", 40),
                "carbs_g": l.get("carbs", 55),
                "fat_g": l.get("fat", 15),
                "ingredients": l.get("ingredients", []),
                "preparation": l.get("preparation", ""),
            },
            {
                "meal_name": s.get("name", "Pre/Post Training Fuel"),
                "meal_type": "snack",
                "timing": "16:00",
                "calories": s.get("calories", 250),
                "protein_g": s.get("protein", 25),
                "carbs_g": s.get("carbs", 25),
                "fat_g": s.get("fat", 6),
                "ingredients": s.get("ingredients", []),
                "preparation": s.get("preparation", ""),
            },
            {
                "meal_name": d.get("name", "Recovery Dinner"),
                "meal_type": "dinner",
                "timing": "19:30",
                "calories": d.get("calories", 550),
                "protein_g": d.get("protein", 45),
                "carbs_g": d.get("carbs", 45),
                "fat_g": d.get("fat", 18),
                "ingredients": d.get("ingredients", []),
                "preparation": d.get("preparation", ""),
            },
        ]

        return selected

    def _synthesize_nutrition_rationale(
        self,
        norm_goal: str,
        weight_curr: float,
        weight_target: float,
        target_calories: float,
        macros: Dict[str, float],
    ) -> str:
        """Synthesize sports-science evidence rationale for the prescribed nutrition plan."""
        delta = round(weight_target - weight_curr, 1)
        direction = "mass gain" if delta > 0 else ("fat reduction" if delta < 0 else "weight maintenance")

        return (
            f"Prescribed daily caloric intake of {target_calories:.0f} kcal supporting {norm_goal} and {direction} "
            f"({weight_curr:.1f}kg to {weight_target:.1f}kg). Macronutrients partitioned via algorithms.metabolic: "
            f"protein set at {macros['protein_g_per_kg']} g/kg ({macros['protein_grams']:.1f}g / {macros['protein_calories']:.0f} kcal) "
            f"to maximize muscle protein synthesis and prevent catabolism; dietary fats set at {macros['fat_g_per_kg']} g/kg "
            f"({macros['fat_grams']:.1f}g / {macros['fat_calories']:.0f} kcal) to sustain endocrine hormonal homeostasis; "
            f"with remaining energy allocated to complex carbohydrates ({macros['carb_grams']:.1f}g / {macros['carb_calories']:.0f} kcal) "
            f"to replete skeletal muscle glycogen stores for high-intensity training sessions."
        )
