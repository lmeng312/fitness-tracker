"""
AI Health Tracker - Backend
Multi-agent system for fitness data analysis and health insights

Adapted from AI Trip Planner architecture for health/fitness domain.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime, date, timedelta
from pathlib import Path
import os
import json
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from sqlalchemy.orm import Session
from sqlalchemy import and_

load_dotenv(find_dotenv())

# Database imports
from database import init_db, get_db_session
from models import User, Workout, Nutrition, UserPreferences, InfluencerSource
from strava_client import StravaClient, STRAVA_WEBHOOK_VERIFY_TOKEN

# Observability (Arize AX - OpenInference)
try:
    from arize.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.litellm import LiteLLMInstrumentor
    from openinference.instrumentation import using_prompt_template, using_attributes
    from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    _TRACING = True
    print("✅ Arize AX observability enabled")
except Exception as e:
    print(f"⚠️  Arize AX not available: {e}")
    def using_prompt_template(**kwargs):
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
    def using_attributes(*args, **kwargs):
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
    _TRACING = False

# LangGraph + LangChain
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from functools import lru_cache
import hashlib
import httpx


# ============================================================================
# CACHING & PERFORMANCE
# ============================================================================

def hash_data(data: dict) -> str:
    """Create hash of data for caching"""
    # Custom JSON encoder to handle date objects
    def default_encoder(obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    return hashlib.md5(json.dumps(data, sort_keys=True, default=default_encoder).encode()).hexdigest()


# Simple in-memory cache for analysis results
_analysis_cache = {}
CACHE_ENABLED = os.getenv("ENABLE_CACHE", "1").lower() not in {"0", "false", "no"}
CACHE_MAX_AGE = int(os.getenv("CACHE_MAX_AGE_SECONDS", "3600"))  # 1 hour default
LITE_MODE = os.getenv("LITE_MODE", "0").lower() not in {"0", "false", "no"}  # Skip expensive correlations
COACH_MODE = os.getenv("COACH_MODE", "0").lower() not in {"0", "false", "no"}


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class NutritionEntry(BaseModel):
    """Nutrition log entry"""
    date: date
    calories: int = Field(ge=0, le=10000, description="Total calories consumed")
    protein_g: Optional[int] = Field(None, ge=0, le=1000)
    carbs_g: Optional[int] = Field(None, ge=0, le=2000)
    fat_g: Optional[int] = Field(None, ge=0, le=1000)
    method: str = Field(default="manual", description="ai_estimate or manual")
    source_description: Optional[str] = None  # Original meal description for AI
    notes: Optional[str] = None
    
    @field_validator('method')
    @classmethod
    def validate_method(cls, v):
        if v not in ['ai_estimate', 'manual']:
            raise ValueError('method must be ai_estimate or manual')
        return v


class WorkoutData(BaseModel):
    """Single workout entry"""
    date: date
    activity_type: str = Field(description="running, cycling, swimming, strength, rest, etc.")
    duration_min: Optional[int] = None
    distance_km: Optional[float] = None
    avg_pace_min_per_km: Optional[float] = None
    avg_speed_kph: Optional[float] = None
    avg_power_watts: Optional[int] = None
    avg_hr_bpm: Optional[int] = Field(None, ge=30, le=220)
    max_hr_bpm: Optional[int] = Field(None, ge=30, le=220)
    calories_burned: Optional[int] = None
    notes: Optional[str] = None


class RecoveryData(BaseModel):
    """Daily recovery metrics"""
    date: date
    sleep_hours: Optional[float] = Field(None, ge=0, le=14)
    sleep_quality_score: Optional[int] = Field(None, ge=1, le=10)
    hrv_ms: Optional[int] = Field(None, ge=10, le=200, description="Heart Rate Variability")
    rhr_bpm: Optional[int] = Field(None, ge=30, le=100, description="Resting Heart Rate")
    weight_kg: Optional[float] = None
    perceived_energy: Optional[int] = Field(None, ge=1, le=10)
    notes: Optional[str] = None


class HealthRequest(BaseModel):
    """Request for health analysis"""
    user_id: Optional[str] = "default_user"
    date_range_start: Optional[date] = None
    date_range_end: Optional[date] = None
    workouts: List[WorkoutData] = []
    recovery: List[RecoveryData] = []
    nutrition: List[NutritionEntry] = []
    goals: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None


# ------------------------------
# Coach Mode: Pydantic models
# ------------------------------

class PreferencesUpsert(BaseModel):
    equipment: List[str] = []
    dietary_prefs: List[str] = []


class InfluencerEntry(BaseModel):
    platform: Optional[str] = None
    handle: Optional[str] = None
    url: Optional[str] = None
    tags: List[str] = []


class InfluencersUpsert(BaseModel):
    sources: List[InfluencerEntry]


class SuggestWorkoutRequest(BaseModel):
    request: str = Field(description="Natural language intent, e.g., '45 min strength for knee strengthening'")
    equipment: Optional[List[str]] = None
    focus: Optional[str] = None
    duration_min: Optional[int] = None


class WorkoutBlock(BaseModel):
    name: str
    duration_min: int
    instructions: str
    exercises: Optional[List[Dict[str, Any]]] = None  # [{name, sets, reps, tempo, rest_sec, rpe, cues}]


class WorkoutPlanResponse(BaseModel):
    title: str
    total_duration_min: int
    focus: Optional[str] = None
    equipment: List[str] = []
    blocks: List[WorkoutBlock] = []
    rationale: Optional[str] = None
    more_about_this_plan: List[str] = []
    readiness_score: Optional[int] = None
    recommended_intensity: Optional[str] = None
    pace_zones: Optional[Dict[str, Any]] = None


class SuggestMealsRequest(BaseModel):
    workout_type: str
    duration_min: int
    timing: str = Field(description="pre or post")
    dietary_prefs: List[str] = []


class MealSuggestion(BaseModel):
    name: str
    calories: int
    protein_g: int
    carbs_g: int
    fat_g: int
    notes: Optional[str] = None


class MealSuggestionsResponse(BaseModel):
    pre: List[MealSuggestion] = []
    post: List[MealSuggestion] = []
    rationale: Optional[str] = None
    more_about_this_plan: List[str] = []


class HealthResponse(BaseModel):
    """Response with health insights"""
    readiness_score: int = Field(ge=1, le=10)
    activity_summary: str
    recovery_summary: str
    nutrition_summary: str
    insights: List[str]
    correlations: List[Dict[str, Any]] = []
    recommendations: List[str]
    tool_calls: List[Dict[str, Any]] = []


class NutritionEstimateRequest(BaseModel):
    """Request for AI nutrition estimation"""
    meal_description: str = Field(description="Natural language description of meals eaten")
    date: Optional[date] = None


class NutritionEstimateResponse(BaseModel):
    """AI-generated nutrition estimate"""
    calories: int
    protein_g: int
    carbs_g: int
    fat_g: int
    confidence_score: float = Field(ge=0, le=1)
    method: str = "ai_estimate"


# ============================================================================
# LLM INITIALIZATION
# ============================================================================

def _init_llm():
    """Initialize LLM (OpenAI or OpenRouter)"""
    class _Fake:
        def __init__(self):
            pass
        def bind_tools(self, tools):
            return self
        def invoke(self, messages):
            class _Msg:
                content = "Test health analysis"
                tool_calls: List[Dict[str, Any]] = []
            return _Msg()

    if os.getenv("TEST_MODE"):
        return _Fake()
    
    # Get model settings from env (allows easy switching)
    model_name = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1500"))  # Reduced from 2000
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.5"))  # Reduced for faster, focused responses
    
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(
            model=model_name.replace("openai/", ""),  # Remove openrouter prefix
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=30  # 30 second timeout
        )
    elif os.getenv("OPENROUTER_API_KEY"):
        return ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=30
        )
    else:
        raise ValueError("Please set OPENAI_API_KEY or OPENROUTER_API_KEY in your .env")


llm = _init_llm()


# ============================================================================
# HEALTH-SPECIFIC TOOLS
# ============================================================================

@tool
def estimate_nutrition_from_description(meal_description: str) -> dict:
    """
    Estimate calories and macronutrients from natural language meal description.
    Uses LLM to analyze and provide realistic estimates.
    
    Args:
        meal_description: Natural language description of meals (e.g., "oatmeal for breakfast, chicken salad for lunch")
    
    Returns:
        Dict with calories, protein_g, carbs_g, fat_g, and confidence_score
    """
    prompt = f"""You are a nutrition estimation expert. Analyze this meal description and estimate nutrition values for a typical active adult.

Meal description: {meal_description}

Estimation guidelines:
- Breakfast: 300-600 kcal typical
- Lunch: 500-800 kcal typical
- Dinner: 600-1000 kcal typical
- Snacks: 100-300 kcal typical
- Use standard portion sizes unless specified
- Protein target: ~20-25% of calories (4 cal/g)
- Carbs: ~45-55% of calories (4 cal/g)
- Fat: ~25-30% of calories (9 cal/g)

Return ONLY a JSON object with this exact structure (no other text):
{{
  "calories": <int>,
  "protein_g": <int>,
  "carbs_g": <int>,
  "fat_g": <int>,
  "confidence_score": <float 0-1>
}}"""
    
    try:
        response = llm.invoke([
            SystemMessage(content="You are a nutrition estimation expert. Always return valid JSON."),
            HumanMessage(content=prompt)
        ])
        
        # Parse JSON response
        content = response.content.strip()
        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        estimate = json.loads(content)
        
        # Validate and return
        return {
            "calories": int(estimate.get("calories", 2000)),
            "protein_g": int(estimate.get("protein_g", 100)),
            "carbs_g": int(estimate.get("carbs_g", 250)),
            "fat_g": int(estimate.get("fat_g", 70)),
            "confidence_score": float(estimate.get("confidence_score", 0.7))
        }
    
    except Exception as e:
        # Fallback to rough estimate
        word_count = len(meal_description.split())
        estimated_calories = min(3000, max(1500, word_count * 100))
        return {
            "calories": estimated_calories,
            "protein_g": int(estimated_calories * 0.21 / 4),
            "carbs_g": int(estimated_calories * 0.50 / 4),
            "fat_g": int(estimated_calories * 0.29 / 9),
            "confidence_score": 0.5
        }


@tool
def calculate_readiness_score(sleep_hours: float, hrv_ms: int, rhr_bpm: int, 
                              training_load: str, days_since_rest: int) -> dict:
    """
    Calculate daily readiness score (1-10) based on recovery metrics.
    
    Args:
        sleep_hours: Hours of sleep last night
        hrv_ms: Heart Rate Variability in milliseconds
        rhr_bpm: Resting Heart Rate in beats per minute
        training_load: Training load level (low, moderate, high)
        days_since_rest: Number of consecutive training days
    
    Returns:
        Dict with readiness_score and explanation
    """
    score = 5.0  # Start at middle
    factors = []
    
    # Sleep impact (0-3 points)
    if sleep_hours >= 8:
        score += 2.5
        factors.append("Excellent sleep (8+ hours)")
    elif sleep_hours >= 7:
        score += 1.5
        factors.append("Good sleep (7+ hours)")
    elif sleep_hours >= 6:
        score += 0.5
        factors.append("Adequate sleep (6+ hours)")
    else:
        score -= 1
        factors.append("Insufficient sleep (<6 hours)")
    
    # HRV impact (0-2 points)
    # Assuming 60ms is baseline for active adult
    if hrv_ms >= 70:
        score += 2
        factors.append("High HRV (excellent recovery)")
    elif hrv_ms >= 55:
        score += 1
        factors.append("Normal HRV (good recovery)")
    elif hrv_ms >= 45:
        score += 0
        factors.append("Below average HRV")
    else:
        score -= 1
        factors.append("Low HRV (poor recovery)")
    
    # RHR impact (0-1 points)
    # Lower RHR typically better for athletes
    if rhr_bpm <= 50:
        score += 1
        factors.append("Excellent resting heart rate")
    elif rhr_bpm <= 60:
        score += 0.5
        factors.append("Good resting heart rate")
    
    # Training load impact (-2 to 0 points)
    if training_load == "high" and days_since_rest >= 4:
        score -= 2
        factors.append("High training load without adequate rest")
    elif training_load == "high":
        score -= 1
        factors.append("High training load")
    elif training_load == "moderate":
        score -= 0.5
        factors.append("Moderate training load")
    
    # Days since rest impact
    if days_since_rest >= 5:
        score -= 1.5
        factors.append(f"{days_since_rest} days without rest (fatigue risk)")
    elif days_since_rest >= 3:
        score -= 0.5
        factors.append(f"{days_since_rest} consecutive training days")
    
    # Clamp to 1-10 range
    final_score = max(1, min(10, round(score)))
    
    # Generate recommendation
    if final_score >= 8:
        recommendation = "Excellent readiness - good day for high-intensity training"
    elif final_score >= 6:
        recommendation = "Good readiness - suitable for moderate training"
    elif final_score >= 4:
        recommendation = "Below optimal - consider light/active recovery"
    else:
        recommendation = "Low readiness - rest day recommended"
    
    return {
        "readiness_score": final_score,
        "recommendation": recommendation,
        "factors": factors
    }


@tool
def analyze_performance_trends(workouts: List[dict], metric: str = "pace") -> dict:
    """
    Analyze performance trends over time for a specific metric.
    
    Args:
        workouts: List of workout dictionaries with date, activity_type, and metrics
        metric: Metric to analyze (pace, power, distance, etc.)
    
    Returns:
        Dict with trend analysis including improvement percentage, best/worst, patterns
    """
    if not workouts:
        return {"trend": "insufficient_data", "message": "Not enough data for trend analysis"}
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(workouts)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Filter by activity type if analyzing pace/power
    if metric == "pace":
        df = df[df['activity_type'] == 'running']
        metric_col = 'avg_pace_min_per_km'
    elif metric == "power":
        df = df[df['activity_type'] == 'cycling']
        metric_col = 'avg_power_watts'
    else:
        metric_col = metric
    
    if metric_col not in df.columns or df[metric_col].isna().all():
        return {"trend": "no_data", "message": f"No data available for {metric}"}
    
    # Remove NaN values
    df_clean = df[df[metric_col].notna()]
    
    if len(df_clean) < 3:
        return {"trend": "insufficient_data", "message": "Need at least 3 data points"}
    
    # Calculate statistics
    first_value = df_clean[metric_col].iloc[0]
    last_value = df_clean[metric_col].iloc[-1]
    best_value = df_clean[metric_col].min() if metric == "pace" else df_clean[metric_col].max()
    worst_value = df_clean[metric_col].max() if metric == "pace" else df_clean[metric_col].min()
    
    # Calculate improvement (pace: lower is better, power: higher is better)
    if metric == "pace":
        improvement_pct = ((first_value - last_value) / first_value) * 100
        improving = last_value < first_value
    else:
        improvement_pct = ((last_value - first_value) / first_value) * 100
        improving = last_value > first_value
    
    return {
        "metric": metric,
        "trend": "improving" if improving else "declining",
        "improvement_percentage": round(improvement_pct, 1),
        "current_value": round(last_value, 2),
        "starting_value": round(first_value, 2),
        "best_value": round(best_value, 2),
        "data_points": len(df_clean),
        "date_range": f"{df_clean['date'].min().strftime('%Y-%m-%d')} to {df_clean['date'].max().strftime('%Y-%m-%d')}"
    }


@tool
def detect_correlations(performance_data: List[dict], recovery_data: List[dict], 
                        nutrition_data: List[dict]) -> List[dict]:
    """
    Detect statistical correlations between recovery/nutrition factors and performance.
    
    Args:
        performance_data: List of workout performance metrics
        recovery_data: List of sleep/HRV data
        nutrition_data: List of nutrition logs
    
    Returns:
        List of correlation findings with strength and insights
    """
    if len(performance_data) < 20:
        return [{
            "factor": "insufficient_data",
            "correlation": 0,
            "message": "Need at least 20 workouts for reliable correlation analysis"
        }]
    
    try:
        # Create DataFrames
        perf_df = pd.DataFrame(performance_data)
        rec_df = pd.DataFrame(recovery_data)
        nutr_df = pd.DataFrame(nutrition_data)
        
        # Merge on date
        perf_df['date'] = pd.to_datetime(perf_df['date'])
        rec_df['date'] = pd.to_datetime(rec_df['date'])
        nutr_df['date'] = pd.to_datetime(nutr_df['date'])
        
        # Merge recovery with performance (use previous night's data)
        rec_df['next_date'] = rec_df['date'] + pd.Timedelta(days=1)
        merged = pd.merge(perf_df, rec_df, left_on='date', right_on='next_date', how='inner', suffixes=('', '_rec'))
        merged = pd.merge(merged, nutr_df, on='date', how='left', suffixes=('', '_nutr'))
        
        if len(merged) < 10:
            return [{"factor": "insufficient_overlap", "correlation": 0, 
                    "message": "Not enough overlapping data points"}]
        
        correlations = []
        
        # Analyze sleep → performance
        if 'sleep_hours' in merged.columns and 'avg_pace_min_per_km' in merged.columns:
            sleep_pace = merged[['sleep_hours', 'avg_pace_min_per_km']].dropna()
            if len(sleep_pace) >= 10:
                corr = sleep_pace.corr().iloc[0, 1]
                correlations.append({
                    "factor": "Sleep Duration",
                    "correlation": round(corr, 3),
                    "strength": "strong" if abs(corr) > 0.6 else "moderate" if abs(corr) > 0.4 else "weak",
                    "insight": f"Sleep hours correlates with running pace (r={corr:.2f})",
                    "sample_size": len(sleep_pace)
                })
        
        # Analyze HRV → performance
        if 'hrv_ms' in merged.columns and 'avg_pace_min_per_km' in merged.columns:
            hrv_pace = merged[['hrv_ms', 'avg_pace_min_per_km']].dropna()
            if len(hrv_pace) >= 10:
                corr = hrv_pace.corr().iloc[0, 1]
                correlations.append({
                    "factor": "Heart Rate Variability",
                    "correlation": round(corr, 3),
                    "strength": "strong" if abs(corr) > 0.6 else "moderate" if abs(corr) > 0.4 else "weak",
                    "insight": f"HRV correlates with running pace (r={corr:.2f})",
                    "sample_size": len(hrv_pace)
                })
        
        # Analyze carbs → performance
        if 'carbs_g' in merged.columns and 'avg_pace_min_per_km' in merged.columns:
            carbs_pace = merged[['carbs_g', 'avg_pace_min_per_km']].dropna()
            if len(carbs_pace) >= 10:
                corr = carbs_pace.corr().iloc[0, 1]
                correlations.append({
                    "factor": "Carbohydrate Intake",
                    "correlation": round(corr, 3),
                    "strength": "strong" if abs(corr) > 0.6 else "moderate" if abs(corr) > 0.4 else "weak",
                    "insight": f"Carb intake correlates with running pace (r={corr:.2f})",
                    "sample_size": len(carbs_pace)
                })
        
        # Sort by correlation strength
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return correlations[:5]  # Return top 5
    
    except Exception as e:
        return [{"error": str(e), "correlation": 0}]


# ------------------------------
# Coach Mode Tools
# ------------------------------

@tool
def generate_workout_plan(intent: str, equipment: List[str], focus: Optional[str] = None, duration_min: Optional[int] = None) -> dict:
    """Generate a structured workout plan from a natural-language intent, tailored to available equipment.
    Returns a dict with title, total_duration_min, focus, equipment, blocks[{name,duration_min,instructions,exercises[]}], rationale.
    """
    try:
        duration_hint = f"Target duration: {duration_min} minutes." if duration_min else ""
        eq = ", ".join(equipment) if equipment else "bodyweight"
        prompt = f"""
You are a certified strength coach. Create a time-boxed, safe and progressive workout.

Intent: {intent}
Focus: {focus or 'general strength'}
Equipment available: {eq}
{duration_hint}

Rules:
- Keep total time within ±5 minutes of target if provided
- Use only available equipment; default to bodyweight otherwise
- Include warm-up, main sets (with reps/tempo), and cooldown/mobility
- Emphasize joint-friendly progressions if knee focus is requested
 - For Main Set, provide 3–5 exercises each with: sets, reps, tempo, rest_sec, target RPE, and 1–2 coaching cues

Return ONLY valid JSON with keys: title,total_duration_min,focus,equipment,blocks(name,duration_min,instructions,exercises[{name,sets,reps,tempo,rest_sec,rpe,cues}]),rationale
"""
        response = llm.invoke([
            SystemMessage(content="You return JSON only."),
            HumanMessage(content=prompt)
        ])
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        plan = json.loads(content)
        return plan
    except Exception:
        # Simple fallback template
        total = duration_min or 45
        warm = max(5, int(total * 0.15))
        cool = max(5, int(total * 0.15))
        main = max(10, total - warm - cool)
        return {
            "title": "Structured Strength Session",
            "total_duration_min": total,
            "focus": focus or "strength",
            "equipment": equipment or ["bodyweight"],
            "blocks": [
                {
                    "name": "Warm-up",
                    "duration_min": warm,
                    "instructions": "Light cardio + dynamic mobility (hips, knees, ankles)",
                    "exercises": [
                        {"name":"Bike/Easy Cardio","sets":1,"reps":warm*60,"tempo":"steady","rest_sec":0,"rpe":3,"cues":"Nasal breathing, light sweat"},
                        {"name":"Leg Swings + Ankle Rocks","sets":1,"reps":10,"tempo":"controlled","rest_sec":0,"rpe":2,"cues":"Tall posture, smooth range"}
                    ]
                },
                {
                    "name": "Main Set",
                    "duration_min": main,
                    "instructions": "3 rounds, rest 60–90s between exercises",
                    "exercises": [
                        {"name":"Split Squat (DB)","sets":3,"reps":8,"tempo":"3-1-1","rest_sec":60,"rpe":7,"cues":"Slow down, knee tracks toes"},
                        {"name":"Romanian Deadlift (DB/KB)","sets":3,"reps":10,"tempo":"3-0-1","rest_sec":60,"rpe":7,"cues":"Hinge at hips, long spine"},
                        {"name":"Step-up (knee-friendly)","sets":3,"reps":10,"tempo":"2-1-1","rest_sec":45,"rpe":6,"cues":"Soft landing, drive through mid-foot"},
                        {"name":"Side Plank","sets":3,"reps":30,"tempo":"hold","rest_sec":30,"rpe":6,"cues":"Ribs down, glutes on"}
                    ]
                },
                {
                    "name": "Cooldown",
                    "duration_min": cool,
                    "instructions": "Easy walk + quad/hamstring/calf stretches and breathing",
                    "exercises": [
                        {"name":"Quad/Calf Stretch","sets":1,"reps":60,"tempo":"hold","rest_sec":0,"rpe":2,"cues":"Gentle, no pain"}
                    ]
                }
            ],
            "rationale": "Fallback template respecting equipment and duration."
        }


@tool
def suggest_meals(workout_type: str, duration_min: int, dietary_prefs: List[str]) -> dict:
    """Suggest pre and post-workout meals tailored to workout type, duration, and dietary preferences.
    Returns dict with pre[], post[], rationale. Each meal has name, calories, protein_g, carbs_g, fat_g, notes.
    """
    try:
        prefs = ", ".join(dietary_prefs) if dietary_prefs else "none"
        prompt = f"""
You are a sports nutritionist. Propose 2 pre-workout and 2 post-workout meal options.

Workout: {workout_type}, Duration: {duration_min} min
Dietary preferences: {prefs}

Guidelines:
- Pre: focus carbs + some protein, light fat; easy to digest, 60–90 min before
- Post: prioritize protein 25–40g + carbs 0.8–1.2 g/kg
- Keep options simple and common foods; include calories and macros

Return JSON with pre[], post[], rationale (no extra text).
"""
        response = llm.invoke([
            SystemMessage(content="You return JSON only."),
            HumanMessage(content=prompt)
        ])
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content)
    except Exception:
        # Fallback simple suggestions
        pre = [
            {"name": "Banana + Greek yogurt", "calories": 250, "protein_g": 15, "carbs_g": 38, "fat_g": 4, "notes": "Easy digesting pre-workout"},
            {"name": "Toast with jam + whey shake", "calories": 300, "protein_g": 20, "carbs_g": 45, "fat_g": 4}
        ]
        post = [
            {"name": "Chicken rice bowl", "calories": 600, "protein_g": 40, "carbs_g": 80, "fat_g": 12},
            {"name": "Tofu stir-fry + noodles", "calories": 550, "protein_g": 35, "carbs_g": 75, "fat_g": 14}
        ]
        return {"pre": pre, "post": post, "rationale": "Fallback balanced options."}


@tool
def fetch_influencer_content(handles_or_urls: List[str], tags: Optional[List[str]] = None) -> List[dict]:
    """Fetch or summarize influencer content from provided handles/URLs. If no API keys, generate short summaries via LLM.
    Returns list of {source, title, url, summary}.
    """
    try:
        # No external API keys in this project; summarize inputs
        items = []
        for src in handles_or_urls:
            prompt = f"Summarize why this source is relevant to training or recovery in 2 sentences: {src}"
            res = llm.invoke([HumanMessage(content=prompt)])
            items.append({"source": src, "title": "Recommended source", "url": src if src.startswith("http") else None, "summary": res.content[:200]})
        return items
    except Exception:
        return [{"source": s, "title": "Source", "url": s if s.startswith("http") else None, "summary": "User-provided source."} for s in handles_or_urls]


def search_youtube_videos(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search YouTube for workout videos using Data API v3.
    Returns list of {video_id, title, description, thumbnail_url, channel_title, duration, view_count, url}.
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        return []
    
    try:
        # Build search query
        search_url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": max_results,
            "key": api_key,
            "order": "relevance"
        }
        
        with httpx.Client(timeout=10.0) as client:
            response = client.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
        
        video_ids = [item["id"]["videoId"] for item in data.get("items", [])]
        if not video_ids:
            return []
        
        # Get detailed info (duration, view count)
        details_url = "https://www.googleapis.com/youtube/v3/videos"
        details_params = {
            "part": "contentDetails,statistics,snippet",
            "id": ",".join(video_ids),
            "key": api_key
        }
        
        with httpx.Client(timeout=10.0) as client:
            details_response = client.get(details_url, params=details_params)
            details_response.raise_for_status()
            details_data = details_response.json()
        
        videos = []
        for item in details_data.get("items", []):
            # Parse duration (ISO 8601 format like PT45M30S)
            duration_str = item.get("contentDetails", {}).get("duration", "PT0M")
            duration_min = 0
            if "H" in duration_str:
                hours = int(duration_str.split("H")[0].replace("PT", ""))
                duration_min += hours * 60
            if "M" in duration_str:
                minutes_part = duration_str.split("M")[0]
                if "H" in minutes_part:
                    minutes = int(minutes_part.split("H")[1])
                else:
                    minutes = int(minutes_part.replace("PT", ""))
                duration_min += minutes
            
            videos.append({
                "video_id": item["id"],
                "title": item["snippet"]["title"],
                "description": item["snippet"]["description"][:500],  # First 500 chars
                "thumbnail_url": item["snippet"]["thumbnails"].get("medium", {}).get("url", ""),
                "channel_title": item["snippet"]["channelTitle"],
                "duration_min": duration_min,
                "view_count": int(item.get("statistics", {}).get("viewCount", 0)),
                "url": f"https://www.youtube.com/watch?v={item['id']}"
            })
        
        return videos
    except Exception as e:
        print(f"YouTube search error: {e}")
        return []


def extract_workout_from_description(description: str, title: str) -> Optional[Dict[str, Any]]:
    """Use LLM to extract structured workout plan from YouTube video description/title.
    Returns dict with blocks/exercises if successful, None otherwise.
    """
    try:
        prompt = f"""Extract a structured workout plan from this YouTube video information.

Title: {title}
Description: {description[:1000]}

Return ONLY valid JSON with this structure (or null if not a workout video):
{{
  "title": "extracted workout title",
  "total_duration_min": <int>,
  "blocks": [
    {{
      "name": "Warm-up",
      "duration_min": <int>,
      "instructions": "brief description",
      "exercises": [{{"name": "...", "sets": <int>, "reps": <int>, "tempo": "...", "rest_sec": <int>, "rpe": <int>, "cues": "..."}}]
    }},
    {{
      "name": "Main Set",
      "duration_min": <int>,
      "instructions": "brief description",
      "exercises": [...]
    }},
    {{
      "name": "Cool-down",
      "duration_min": <int>,
      "instructions": "brief description",
      "exercises": [...]
    }}
  ]
}}

If this is not a structured workout video, return null."""
        
        response = llm.invoke([
            SystemMessage(content="You are a workout extraction expert. Return only valid JSON."),
            HumanMessage(content=prompt)
        ])
        
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        plan = json.loads(content)
        if plan and isinstance(plan, dict) and "blocks" in plan:
            return plan
        return None
    except Exception:
        return None

# ============================================================================
# HEALTH STATE (LangGraph State)
# ============================================================================

class HealthState(TypedDict):
    """State for health analysis workflow"""
    messages: Annotated[List[BaseMessage], operator.add]
    health_request: Dict[str, Any]
    activity_analysis: Optional[str]
    recovery_analysis: Optional[str]
    nutrition_analysis: Optional[str]
    final_insights: Optional[str]
    readiness_score: Optional[int]
    correlations: Optional[List[Dict[str, Any]]]
    tool_calls: Annotated[List[Dict[str, Any]], operator.add]
    # Coach Mode additions
    suggested_workout_plan: Optional[Dict[str, Any]]
    suggested_meals_pre: Optional[List[Dict[str, Any]]]
    suggested_meals_post: Optional[List[Dict[str, Any]]]
    influencer_refs: Annotated[List[Dict[str, Any]], operator.add]


# ============================================================================
# MULTI-AGENT SYSTEM
# ============================================================================

def activity_agent(state: HealthState) -> HealthState:
    """
    Activity Agent - Analyzes workouts and performance trends.
    Replaces research_agent from trip planner.
    """
    req = state["health_request"]
    workouts = req.get("workouts", [])
    
    prompt_t = """You are an activity analysis expert for athletes and fitness enthusiasts.

Analyze the workout data and provide insights on:
- Training volume and frequency
- Performance trends (are they improving?)
- Training load assessment
- Activity patterns and consistency
- Recommendations for training optimization

Use the available tools to analyze performance trends.

Workout data summary:
- Total workouts: {workout_count}
- Date range: {date_range}
- Activity types: {activity_types}
"""
    
    # Prepare workout summary
    workout_count = len(workouts)
    activity_types = list(set([w.get('activity_type', 'unknown') for w in workouts]))
    dates = [w.get('date') for w in workouts if w.get('date')]
    date_range = f"{min(dates)} to {max(dates)}" if dates else "No dates"
    
    vars_ = {
        "workout_count": workout_count,
        "date_range": date_range,
        "activity_types": ", ".join(activity_types)
    }
    
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [analyze_performance_trends]
    agent = llm.bind_tools(tools)
    
    calls: List[Dict[str, Any]] = []
    
    # Agent execution with Arize AX observability
    with using_attributes(
        tags=["activity", "performance_analysis"],
        session_id=req.get("user_id", "default"),
        user_id=req.get("user_id", "default"),
        metadata={
            "agent_name": "activity_agent",
            "agent_type": "activity",
            "workout_count": workout_count,
            "activity_types": activity_types
        }
    ):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                # OpenInference semantic attributes
                current_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.AGENT.value)
                current_span.set_attribute("agent.name", "activity_agent")
                current_span.set_attribute("agent.type", "activity")
                current_span.set_attribute("agent.role", "performance_analyzer")
                current_span.set_attribute("data.workout_count", workout_count)
                current_span.set_attribute("data.activity_types", activity_types)
                current_span.set_attribute(SpanAttributes.INPUT_VALUE, f"Analyzing {workout_count} workouts")
        
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1.0"):
            try:
                res = agent.invoke(messages)
                if _TRACING and trace.get_current_span():
                    trace.get_current_span().set_attribute(SpanAttributes.OUTPUT_VALUE, res.content[:200])
                    trace.get_current_span().set_status(Status(StatusCode.OK))
            except Exception as e:
                if _TRACING and trace.get_current_span():
                    trace.get_current_span().record_exception(e)
                    trace.get_current_span().set_status(Status(StatusCode.ERROR))
                raise
    
    # Execute tools if called
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "activity", "tool": c["name"], "args": c.get("args", {})})
        
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})
        
        messages.append(res)
        messages.extend(tr["messages"])
        
        synthesis_prompt = "Based on the analysis above, provide a comprehensive activity summary with actionable recommendations."
        messages.append(SystemMessage(content=synthesis_prompt))
        
        final_res = llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content
    
    return {
        "messages": [SystemMessage(content=out)],
        "activity_analysis": out,
        "tool_calls": calls
    }


def recovery_agent(state: HealthState) -> HealthState:
    """
    Recovery Agent - Analyzes sleep, HRV, and calculates readiness score.
    Replaces budget_agent from trip planner.
    """
    req = state["health_request"]
    recovery_data = req.get("recovery", [])
    workouts = req.get("workouts", [])
    
    prompt_t = """You are a recovery and sleep analysis expert.

Analyze the recovery metrics and calculate today's readiness score:
- Sleep quality and duration patterns
- Heart Rate Variability (HRV) trends
- Resting Heart Rate patterns
- Signs of overtraining or good recovery
- Recovery recommendations

Use the calculate_readiness_score tool to assess today's readiness.

Recovery data summary:
- Data points: {recovery_count}
- Recent sleep average: {sleep_avg}h
- Recent HRV average: {hrv_avg}ms
"""
    
    # Calculate averages
    recovery_count = len(recovery_data)
    sleep_values = [r.get('sleep_hours') for r in recovery_data if r.get('sleep_hours')]
    hrv_values = [r.get('hrv_ms') for r in recovery_data if r.get('hrv_ms')]
    
    sleep_avg = round(sum(sleep_values) / len(sleep_values), 1) if sleep_values else 0
    hrv_avg = round(sum(hrv_values) / len(hrv_values)) if hrv_values else 0
    
    vars_ = {
        "recovery_count": recovery_count,
        "sleep_avg": sleep_avg,
        "hrv_avg": hrv_avg
    }
    
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    tools = [calculate_readiness_score]
    agent = llm.bind_tools(tools)
    
    calls: List[Dict[str, Any]] = []
    
    with using_attributes(
        tags=["recovery", "readiness"],
        session_id=req.get("user_id", "default"),
        user_id=req.get("user_id", "default"),
        metadata={
            "agent_name": "recovery_agent",
            "agent_type": "recovery",
            "recovery_count": recovery_count,
            "sleep_avg": sleep_avg,
            "hrv_avg": hrv_avg
        }
    ):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.AGENT.value)
                current_span.set_attribute("agent.name", "recovery_agent")
                current_span.set_attribute("agent.type", "recovery")
                current_span.set_attribute("agent.role", "readiness_calculator")
                current_span.set_attribute("data.recovery_count", recovery_count)
                current_span.set_attribute("data.sleep_avg", sleep_avg)
                current_span.set_attribute("data.hrv_avg", hrv_avg)
                current_span.set_attribute(SpanAttributes.INPUT_VALUE, f"Assessing {recovery_count} recovery data points")
        
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1.0"):
            try:
                res = agent.invoke(messages)
                if _TRACING and trace.get_current_span():
                    trace.get_current_span().set_attribute(SpanAttributes.OUTPUT_VALUE, res.content[:200])
                    trace.get_current_span().set_status(Status(StatusCode.OK))
            except Exception as e:
                if _TRACING and trace.get_current_span():
                    trace.get_current_span().record_exception(e)
                    trace.get_current_span().set_status(Status(StatusCode.ERROR))
                raise
    
    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent": "recovery", "tool": c["name"], "args": c.get("args", {})})
        
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})
        
        messages.append(res)
        messages.extend(tr["messages"])
        
        synthesis_prompt = "Provide a recovery summary with specific recommendations for today."
        messages.append(SystemMessage(content=synthesis_prompt))
        
        final_res = llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content
    
    # Extract readiness score from tool calls
    readiness_score = 7  # Default
    for call in calls:
        if call["tool"] == "calculate_readiness_score":
            # Tool was called, result should be in messages
            pass
    
    return {
        "messages": [SystemMessage(content=out)],
        "recovery_analysis": out,
        "readiness_score": readiness_score,
        "tool_calls": calls
    }


def nutrition_agent(state: HealthState) -> HealthState:
    """
    Nutrition Agent - Analyzes nutrition patterns and energy balance.
    Replaces local_agent from trip planner.
    """
    req = state["health_request"]
    nutrition_data = req.get("nutrition", [])
    workouts = req.get("workouts", [])
    
    prompt_t = """You are a sports nutrition expert.

Analyze nutrition intake patterns and energy balance:
- Calorie intake vs. expenditure
- Macronutrient ratios (protein, carbs, fat)
- Fueling patterns relative to training
- Nutrition adequacy for performance goals
- Recommendations for optimization

Nutrition data summary:
- Logged days: {nutrition_count}
- Average calories: {avg_calories} kcal/day
- Average protein: {avg_protein}g
- Average carbs: {avg_carbs}g
"""
    
    nutrition_count = len(nutrition_data)
    calories = [n.get('calories') for n in nutrition_data if n.get('calories')]
    protein = [n.get('protein_g') for n in nutrition_data if n.get('protein_g')]
    carbs = [n.get('carbs_g') for n in nutrition_data if n.get('carbs_g')]
    
    avg_calories = round(sum(calories) / len(calories)) if calories else 0
    avg_protein = round(sum(protein) / len(protein)) if protein else 0
    avg_carbs = round(sum(carbs) / len(carbs)) if carbs else 0
    
    vars_ = {
        "nutrition_count": nutrition_count,
        "avg_calories": avg_calories,
        "avg_protein": avg_protein,
        "avg_carbs": avg_carbs
    }
    
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    agent = llm  # No specific tools for nutrition agent in MVP
    
    with using_attributes(
        tags=["nutrition", "energy_balance"],
        session_id=req.get("user_id", "default"),
        user_id=req.get("user_id", "default"),
        metadata={
            "agent_name": "nutrition_agent",
            "agent_type": "nutrition",
            "nutrition_count": nutrition_count,
            "avg_calories": avg_calories,
            "avg_protein": avg_protein
        }
    ):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.AGENT.value)
                current_span.set_attribute("agent.name", "nutrition_agent")
                current_span.set_attribute("agent.type", "nutrition")
                current_span.set_attribute("agent.role", "nutrition_analyzer")
                current_span.set_attribute("data.nutrition_count", nutrition_count)
                current_span.set_attribute("data.avg_calories", avg_calories)
                current_span.set_attribute("data.avg_protein", avg_protein)
                current_span.set_attribute("data.avg_carbs", avg_carbs)
                current_span.set_attribute(SpanAttributes.INPUT_VALUE, f"Analyzing {nutrition_count} nutrition entries")
        
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1.0"):
            try:
                res = agent.invoke(messages)
                if _TRACING and trace.get_current_span():
                    trace.get_current_span().set_attribute(SpanAttributes.OUTPUT_VALUE, res.content[:200])
                    trace.get_current_span().set_status(Status(StatusCode.OK))
            except Exception as e:
                if _TRACING and trace.get_current_span():
                    trace.get_current_span().record_exception(e)
                    trace.get_current_span().set_status(Status(StatusCode.ERROR))
                raise
    
    return {
        "messages": [SystemMessage(content=res.content)],
        "nutrition_analysis": res.content,
        "tool_calls": []
    }


def coach_workout_agent(state: HealthState) -> HealthState:
    """
    Coach Workout Agent - Generates suggested workout plan from natural language intent
    """
    req = state["health_request"]
    coach_req = req.get("coach_request", {})
    intent = coach_req.get("request", "45 min full-body strength")
    equipment = coach_req.get("equipment", [])
    focus = coach_req.get("focus")
    duration = coach_req.get("duration_min")

    tools = [generate_workout_plan]
    agent = llm.bind_tools(tools)
    calls: List[Dict[str, Any]] = []

    prompt = f"Create a structured plan for: {intent}."
    with using_attributes(tags=["coach","workout"], metadata={"agent_name":"coach_workout_agent"}):
        res = agent.invoke([HumanMessage(content=prompt)])

    if getattr(res, "tool_calls", None):
        for c in res.tool_calls:
            calls.append({"agent":"coach_workout","tool":c["name"],"args":c.get("args",{})})
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages":[res]})
        messages = [res] + tr["messages"]
        final = llm.invoke(messages)
        out_text = final.content
    else:
        # Direct tool invocation fallback
        plan = generate_workout_plan.invoke({
            "intent": intent,
            "equipment": equipment,
            "focus": focus,
            "duration_min": duration
        })
        out_text = json.dumps(plan)

    # Try to parse plan from last tool output if available
    try:
        plan = json.loads(out_text)
    except Exception:
        plan = {
            "title": "Suggested Workout",
            "total_duration_min": duration or 45,
            "focus": focus,
            "equipment": equipment,
            "blocks": [],
            "rationale": out_text[:200]
        }

    return {
        "messages": [SystemMessage(content="coach_workout_plan_ready")],
        "suggested_workout_plan": plan,
        "tool_calls": calls
    }


def coach_meal_agent(state: HealthState) -> HealthState:
    """
    Coach Meal Agent - Suggests pre/post-workout meals based on context and prefs
    """
    req = state["health_request"]
    coach_req = req.get("coach_request", {})
    workout_type = coach_req.get("workout_type", "strength")
    duration = coach_req.get("duration_min", 45)
    dietary_prefs = coach_req.get("dietary_prefs", [])

    tools = [suggest_meals]
    agent = llm.bind_tools(tools)
    calls: List[Dict[str, Any]] = []

    prompt = f"Suggest pre and post-workout meals for a {duration} min {workout_type} session."
    with using_attributes(tags=["coach","meals"], metadata={"agent_name":"coach_meal_agent"}):
        res = agent.invoke([HumanMessage(content=prompt)])

    suggestions = suggest_meals.invoke({
        "workout_type": workout_type,
        "duration_min": duration,
        "dietary_prefs": dietary_prefs
    })

    return {
        "messages": [SystemMessage(content="coach_meals_ready")],
        "suggested_meals_pre": suggestions.get("pre", []),
        "suggested_meals_post": suggestions.get("post", []),
        "tool_calls": calls
    }

def insights_agent(state: HealthState) -> HealthState:
    """
    Insights Agent - Synthesizes all analyses and detects correlations.
    Replaces itinerary_agent from trip planner.
    """
    req = state["health_request"]
    
    prompt_t = """You are an AI health insights synthesizer.

Combine all analyses to provide comprehensive health insights:

Activity Analysis:
{activity_analysis}

Recovery Analysis:
{recovery_analysis}

Nutrition Analysis:
{nutrition_analysis}

Create a unified health summary with:
1. Overall assessment
2. Key insights and patterns
3. Specific actionable recommendations
4. Correlation findings (use detect_correlations tool if enough data)
5. What to focus on for improvement
"""
    
    vars_ = {
        "activity_analysis": state.get("activity_analysis", "No activity data")[:500],
        "recovery_analysis": state.get("recovery_analysis", "No recovery data")[:500],
        "nutrition_analysis": state.get("nutrition_analysis", "No nutrition data")[:500]
    }
    
    messages = [SystemMessage(content=prompt_t.format(**vars_))]
    # Skip correlations in lite mode for faster analysis
    tools = [detect_correlations] if not LITE_MODE else []
    agent = llm.bind_tools(tools) if tools else llm
    
    calls: List[Dict[str, Any]] = []
    correlations_found = []
    
    with using_attributes(
        tags=["insights", "synthesis", "final"],
        session_id=req.get("user_id", "default"),
        user_id=req.get("user_id", "default"),
        metadata={
            "agent_name": "insights_agent",
            "agent_type": "insights",
            "has_activity": bool(state.get("activity_analysis")),
            "has_recovery": bool(state.get("recovery_analysis")),
            "has_nutrition": bool(state.get("nutrition_analysis"))
        }
    ):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.AGENT.value)
                current_span.set_attribute("agent.name", "insights_agent")
                current_span.set_attribute("agent.type", "insights")
                current_span.set_attribute("agent.role", "synthesis")
                current_span.set_attribute("workflow.parallel_agents", 3)
                current_span.set_attribute("workflow.has_activity", bool(state.get("activity_analysis")))
                current_span.set_attribute("workflow.has_recovery", bool(state.get("recovery_analysis")))
                current_span.set_attribute("workflow.has_nutrition", bool(state.get("nutrition_analysis")))
                current_span.set_attribute(SpanAttributes.INPUT_VALUE, "Synthesizing multi-agent insights")
        
        with using_prompt_template(template=prompt_t, variables=vars_, version="v1.0"):
            try:
                res = agent.invoke(messages)
                if _TRACING and trace.get_current_span():
                    trace.get_current_span().set_attribute(SpanAttributes.OUTPUT_VALUE, res.content[:200])
                    trace.get_current_span().set_status(Status(StatusCode.OK))
            except Exception as e:
                if _TRACING and trace.get_current_span():
                    trace.get_current_span().record_exception(e)
                    trace.get_current_span().set_status(Status(StatusCode.ERROR))
                raise
    
    if getattr(res, "tool_calls", None) and tools:
        for c in res.tool_calls:
            calls.append({"agent": "insights", "tool": c["name"], "args": c.get("args", {})})
        
        tool_node = ToolNode(tools)
        tr = tool_node.invoke({"messages": [res]})
        
        messages.append(res)
        messages.extend(tr["messages"])
        
        synthesis_prompt = "Synthesize all findings into actionable insights."
        messages.append(SystemMessage(content=synthesis_prompt))
        
        final_res = llm.invoke(messages)
        out = final_res.content
    else:
        out = res.content
    
    return {
        "messages": [SystemMessage(content=out)],
        "final_insights": out,
        "correlations": correlations_found,
        "tool_calls": calls
    }


# ============================================================================
# LANGGRAPH WORKFLOW
# ============================================================================

def build_health_graph():
    """Build LangGraph workflow for health analysis"""
    g = StateGraph(HealthState)
    
    # Add agent nodes
    g.add_node("activity_node", activity_agent)
    g.add_node("recovery_node", recovery_agent)
    g.add_node("nutrition_node", nutrition_agent)
    g.add_node("insights_node", insights_agent)
    if COACH_MODE:
        g.add_node("coach_workout_node", coach_workout_agent)
        g.add_node("coach_meal_node", coach_meal_agent)
    
    # Parallel execution: Activity, Recovery, Nutrition run simultaneously
    g.add_edge(START, "activity_node")
    g.add_edge(START, "recovery_node")
    g.add_edge(START, "nutrition_node")
    if COACH_MODE:
        g.add_edge(START, "coach_workout_node")
        g.add_edge(START, "coach_meal_node")
    
    # All three feed into Insights agent
    g.add_edge("activity_node", "insights_node")
    g.add_edge("recovery_node", "insights_node")
    g.add_edge("nutrition_node", "insights_node")
    if COACH_MODE:
        g.add_edge("coach_workout_node", "insights_node")
        g.add_edge("coach_meal_node", "insights_node")
    
    g.add_edge("insights_node", END)
    
    # Compile without checkpointer (fresh state per request)
    return g.compile()


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(title="AI Health Tracker API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Startup event to initialize database
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    print("🚀 Initializing database...")
    init_db()
    print("✅ Database ready")


# Initialize Arize AX tracing once at startup
if _TRACING:
    try:
        space_id = os.getenv("ARIZE_SPACE_ID")
        api_key = os.getenv("ARIZE_API_KEY")
        if space_id and api_key:
            print(f"🔗 Connecting to Arize AX (Space: {space_id[:8]}...)")
            tp = register(
                space_id=space_id, 
                api_key=api_key, 
                project_name="ai-health-tracker"
            )
            # Auto-instrument LangChain and LiteLLM
            LangChainInstrumentor().instrument(
                tracer_provider=tp, 
                include_chains=True, 
                include_agents=True, 
                include_tools=True
            )
            LiteLLMInstrumentor().instrument(
                tracer_provider=tp, 
                skip_dep_check=True
            )
            print("✅ Arize AX instrumentation complete")
        else:
            print("⚠️  Arize credentials not found - tracing disabled")
            _TRACING = False
    except Exception as e:
        print(f"❌ Arize AX setup failed: {e}")
        _TRACING = False


@app.get("/")
def root():
    """Serve Dashboard page"""
    dashboard_path = Path(__file__).parent.parent / "frontend" / "dashboard.html"
    if dashboard_path.exists():
        return FileResponse(dashboard_path)
    return {"message": "AI Health Tracker API", "version": "1.0.0"}


@app.get("/dashboard.html")
def dashboard():
    """Serve dashboard page"""
    dashboard_path = Path(__file__).parent.parent / "frontend" / "dashboard.html"
    if dashboard_path.exists():
        return FileResponse(dashboard_path)
    return {"error": "Dashboard page not found"}


@app.get("/upload.html")
def upload_page():
    """Serve upload page"""
    upload_path = Path(__file__).parent.parent / "frontend" / "upload.html"
    if upload_path.exists():
        return FileResponse(upload_path)
    return {"error": "Upload page not found"}


@app.get("/ai-insights.html")
def ai_insights():
    """Serve AI insights page"""
    insights_path = Path(__file__).parent.parent / "frontend" / "ai-insights.html"
    if insights_path.exists():
        return FileResponse(insights_path)
    return {"error": "AI insights page not found"}


@app.get("/settings.html")
def settings_page():
    """Serve settings page"""
    settings_path = Path(__file__).parent.parent / "frontend" / "settings.html"
    if settings_path.exists():
        return FileResponse(settings_path)
    return {"error": "Settings page not found"}


@app.get("/backend/sample_data.json")
def sample_data():
    """Serve sample data for frontend testing"""
    sample_path = Path(__file__).parent / "sample_data.json"
    if sample_path.exists():
        return FileResponse(sample_path)
    return {"error": "Sample data not found"}


@app.get("/backend/my_strava_data.json")
def my_strava_data():
    """Serve user's real Strava data"""
    strava_path = Path(__file__).parent / "my_strava_data.json"
    if strava_path.exists():
        return FileResponse(strava_path)
    # Fall back to sample data if user data doesn't exist
    sample_path = Path(__file__).parent / "sample_data.json"
    if sample_path.exists():
        return FileResponse(sample_path)
    return {"error": "No data found"}


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ai-health-tracker"}


@app.post("/api/cache/clear")
def clear_cache():
    """Clear the analysis cache"""
    global _analysis_cache
    cache_size = len(_analysis_cache)
    _analysis_cache.clear()
    return {
        "status": "success",
        "message": f"Cleared {cache_size} cached entries",
        "cache_enabled": CACHE_ENABLED
    }


@app.get("/api/cache/stats")
def cache_stats():
    """Get cache statistics"""
    return {
        "cache_enabled": CACHE_ENABLED,
        "cached_entries": len(_analysis_cache),
        "max_age_seconds": CACHE_MAX_AGE,
        "lite_mode": LITE_MODE,
        "max_workouts": os.getenv("MAX_WORKOUTS_PER_ANALYSIS", "30"),
        "model": os.getenv("LLM_MODEL", "openai/gpt-4o-mini"),
        "max_tokens": os.getenv("LLM_MAX_TOKENS", "1500")
    }


# =========================================================================
# COACH MODE: Preferences & Influencers API
# =========================================================================

def _csv_to_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [s.strip() for s in value.split(",") if s.strip()]


def _list_to_csv(values: Optional[List[str]]) -> Optional[str]:
    if not values:
        return None
    return ", ".join([v.strip() for v in values if v and v.strip()])


@app.get("/api/preferences")
def get_preferences(db: Session = Depends(get_db_session)):
    if not COACH_MODE:
        return {"enabled": False}
    pref = db.query(UserPreferences).filter(UserPreferences.user_id == 1).first()
    return {
        "equipment": _csv_to_list(pref.equipment_csv) if pref else [],
        "dietary_prefs": _csv_to_list(pref.dietary_prefs_csv) if pref else []
    }


@app.post("/api/preferences")
def upsert_preferences(req: PreferencesUpsert, db: Session = Depends(get_db_session)):
    if not COACH_MODE:
        raise HTTPException(status_code=400, detail="Coach mode is disabled")
    pref = db.query(UserPreferences).filter(UserPreferences.user_id == 1).first()
    if not pref:
        pref = UserPreferences(user_id=1)
        db.add(pref)
    pref.equipment_csv = _list_to_csv(req.equipment)
    pref.dietary_prefs_csv = _list_to_csv(req.dietary_prefs)
    db.commit()
    db.refresh(pref)
    return {"status": "success"}


@app.get("/api/influencers")
def get_influencers(db: Session = Depends(get_db_session)):
    if not COACH_MODE:
        return {"enabled": False}
    rows = db.query(InfluencerSource).filter(InfluencerSource.user_id == 1).all()
    out = []
    for r in rows:
        out.append({
            "platform": r.platform,
            "handle": r.handle,
            "url": r.url,
            "tags": _csv_to_list(r.tags_csv)
        })
    return {"sources": out}


@app.post("/api/influencers")
def upsert_influencers(req: InfluencersUpsert, db: Session = Depends(get_db_session)):
    if not COACH_MODE:
        raise HTTPException(status_code=400, detail="Coach mode is disabled")
    # Replace current list for simplicity
    db.query(InfluencerSource).filter(InfluencerSource.user_id == 1).delete()
    for s in req.sources:
        row = InfluencerSource(
            user_id=1,
            platform=s.platform,
            handle=s.handle,
            url=s.url,
            tags_csv=_list_to_csv(s.tags)
        )
        db.add(row)
    db.commit()
    return {"status": "success", "count": len(req.sources)}


# =========================================================================
# COACH MODE: Suggestion Endpoints
# =========================================================================

@app.post("/api/coach/workouts/suggest", response_model=WorkoutPlanResponse)
def coach_suggest_workout(req: SuggestWorkoutRequest, db: Session = Depends(get_db_session)):
    if not COACH_MODE:
        raise HTTPException(status_code=400, detail="Coach mode is disabled")
    # Load stored equipment as default
    if not req.equipment:
        pref = db.query(UserPreferences).filter(UserPreferences.user_id == 1).first()
        req.equipment = _csv_to_list(pref.equipment_csv) if pref else []
    plan = generate_workout_plan.invoke({
        "intent": req.request,
        "equipment": req.equipment or [],
        "focus": req.focus,
        "duration_min": req.duration_min
    })
    # Build contextual bullets from recent workouts
    bullets = _build_more_about_plan(db)
    
    # Calculate readiness and pace zones
    readiness = calculate_readiness_from_strava(db)
    pace_zones_data = None
    
    # Include pace zones for endurance goals
    focus_str = str(req.focus or "").lower()
    request_str = str(req.request or "").lower()
    if "endurance" in focus_str or "endurance" in request_str:
        pace_zones_data = identify_pace_zones(db)
    
    # Adjust workout intensity based on readiness
    if readiness["recommended_intensity"] == "recovery" and req.duration_min:
        # Reduce duration for recovery days
        req.duration_min = max(20, int(req.duration_min * 0.7))
    elif readiness["recommended_intensity"] == "low":
        # Slightly reduce intensity cues
        req.duration_min = max(30, int(req.duration_min * 0.85))
    
    # Map to response model constraints
    blocks = []
    for b in plan.get("blocks", [])[:10]:
        try:
            exercises = []
            for ex in b.get("exercises", []):
                exercises.append({
                    "name": ex.get("name", "Exercise"),
                    "sets": ex.get("sets", 1),
                    "reps": ex.get("reps", ""),
                    "tempo": ex.get("tempo"),
                    "rest_sec": ex.get("rest_sec"),
                    "rpe": ex.get("rpe"),
                    "cues": ex.get("cues")
                })
            blocks.append(WorkoutBlock(
                name=b.get("name","Block"),
                duration_min=int(b.get("duration_min", 10)),
                instructions=b.get("instructions",""),
                exercises=exercises
            ))
        except Exception:
            continue
    # Guard: ensure blocks exist (LLM may omit). Synthesize defaults based on total duration
    if not blocks:
        total_minutes = int(plan.get("total_duration_min", req.duration_min or 45))
        warm = max(5, int(round(total_minutes * 0.15)))
        cool = max(5, int(round(total_minutes * 0.15)))
        main = max(10, total_minutes - warm - cool)
        blocks = [
            WorkoutBlock(
                name="Warm-up",
                duration_min=warm,
                instructions="5–10 min easy cardio + dynamic knee/ankle/hip mobility"
            ),
            WorkoutBlock(
                name="Main Set",
                duration_min=main,
                instructions="3–4 rounds: split squats (controlled eccentrics), hip hinge (RDL), step-ups, core bracing. Use available equipment."
            ),
            WorkoutBlock(
                name="Cool-down",
                duration_min=cool,
                instructions="Easy walk + quad/hamstring/calf stretches and breathing to lower HR"
            )
        ]
    
    # Include readiness info in bullets
    if readiness.get("intensity_note"):
        bullets.insert(0, readiness["intensity_note"])
    
    return WorkoutPlanResponse(
        title=plan.get("title", "Suggested Workout"),
        total_duration_min=int(plan.get("total_duration_min", req.duration_min or 45)),
        focus=plan.get("focus"),
        equipment=plan.get("equipment", req.equipment or []),
        blocks=blocks,
        rationale=plan.get("rationale"),
        more_about_this_plan=bullets,
        readiness_score=readiness.get("readiness_score"),
        recommended_intensity=readiness.get("recommended_intensity"),
        pace_zones=pace_zones_data if pace_zones_data and pace_zones_data.get("has_zones") else None
    )


@app.post("/api/coach/meals/suggest", response_model=MealSuggestionsResponse)
def coach_suggest_meals(req: SuggestMealsRequest, db: Session = Depends(get_db_session)):
    if not COACH_MODE:
        raise HTTPException(status_code=400, detail="Coach mode is disabled")
    # Merge stored dietary prefs
    if not req.dietary_prefs:
        pref = db.query(UserPreferences).filter(UserPreferences.user_id == 1).first()
        req.dietary_prefs = _csv_to_list(pref.dietary_prefs_csv) if pref else []
    sug = suggest_meals.invoke({
        "workout_type": req.workout_type,
        "duration_min": req.duration_min,
        "dietary_prefs": req.dietary_prefs
    })
    bullets = _build_more_about_plan(db)
    def _map_list(lst):
        out = []
        for m in lst[:5]:
            try:
                out.append(MealSuggestion(
                    name=m.get("name","Meal"),
                    calories=int(m.get("calories", 400)),
                    protein_g=int(m.get("protein_g", 25)),
                    carbs_g=int(m.get("carbs_g", 50)),
                    fat_g=int(m.get("fat_g", 10)),
                    notes=m.get("notes")
                ))
            except Exception:
                continue
        return out
    return MealSuggestionsResponse(
        pre=_map_list(sug.get("pre", [])),
        post=_map_list(sug.get("post", [])),
        rationale=sug.get("rationale"),
        more_about_this_plan=bullets
    )


class YouTubeSearchRequest(BaseModel):
    goal: str
    equipment: List[str] = []
    duration_min: int


class YouTubeExtractRequest(BaseModel):
    video_id: str
    title: str
    description: str


@app.post("/api/coach/youtube/search")
def coach_search_youtube(req: YouTubeSearchRequest):
    """Search YouTube for workout videos matching goal/equipment/duration."""
    if not COACH_MODE:
        raise HTTPException(status_code=400, detail="Coach mode is disabled")
    
    # Build search query
    goal_map = {
        "strength_building": "strength training",
        "endurance": "endurance workout",
        "recovery_mobility": "recovery mobility",
        "general_fitness": "full body workout",
        "weight_loss": "fat burning workout"
    }
    goal_term = goal_map.get(req.goal, "workout")
    
    equipment_str = " ".join(req.equipment[:3]) if req.equipment else ""
    query = f"{req.duration_min} minute {goal_term} {equipment_str}".strip()
    
    videos = search_youtube_videos(query, max_results=5)
    return {"videos": videos}


@app.get("/api/coach/readiness")
def coach_get_readiness(db: Session = Depends(get_db_session)):
    """Get readiness score and recommended intensity based on recent Strava data."""
    if not COACH_MODE:
        raise HTTPException(status_code=400, detail="Coach mode is disabled")
    return calculate_readiness_from_strava(db)


@app.get("/api/coach/pace-zones")
def coach_get_pace_zones(db: Session = Depends(get_db_session)):
    """Get pace zones calculated from recent running workouts."""
    if not COACH_MODE:
        raise HTTPException(status_code=400, detail="Coach mode is disabled")
    return identify_pace_zones(db)


@app.post("/api/coach/youtube/extract", response_model=WorkoutPlanResponse)
def coach_extract_workout(req: YouTubeExtractRequest):
    """Extract structured workout plan from YouTube video description."""
    if not COACH_MODE:
        raise HTTPException(status_code=400, detail="Coach mode is disabled")
    
    plan = extract_workout_from_description(req.description, req.title)
    if not plan:
        raise HTTPException(status_code=400, detail="Could not extract workout structure from video")
    
    # Map to response model
    blocks = []
    for b in plan.get("blocks", [])[:10]:
        try:
            exercises = []
            for ex in b.get("exercises", []):
                exercises.append({
                    "name": ex.get("name", "Exercise"),
                    "sets": ex.get("sets", 1),
                    "reps": ex.get("reps", ""),
                    "tempo": ex.get("tempo"),
                    "rest_sec": ex.get("rest_sec"),
                    "rpe": ex.get("rpe"),
                    "cues": ex.get("cues")
                })
            blocks.append(WorkoutBlock(
                name=b.get("name", "Block"),
                duration_min=int(b.get("duration_min", 10)),
                instructions=b.get("instructions", ""),
                exercises=exercises
            ))
        except Exception:
            continue
    
    return WorkoutPlanResponse(
        title=plan.get("title", "YouTube Workout"),
        total_duration_min=int(plan.get("total_duration_min", 45)),
        focus=None,
        equipment=[],
        blocks=blocks,
        rationale="Extracted from YouTube video",
        more_about_this_plan=[]
    )


def calculate_readiness_from_strava(db: Session) -> Dict[str, Any]:
    """Calculate readiness score (1-10) and recommended intensity based on recent Strava workouts.
    Returns dict with readiness_score, recommended_intensity, and factors.
    """
    try:
        from sqlalchemy import desc
        today = datetime.now().date()
        cutoff_14 = today - timedelta(days=14)
        
        workouts = db.query(Workout).filter(
            and_(Workout.user_id == 1, Workout.date >= cutoff_14)
        ).order_by(desc(Workout.date)).all()
        
        if not workouts or len(workouts) < 3:
            return {
                "readiness_score": 7,
                "recommended_intensity": "moderate",
                "factors": ["Limited recent data: using moderate intensity"],
                "days_since_last_workout": None,
                "days_since_hard_workout": None,
                "weekly_load_change": None
            }
        
        # Calculate metrics
        last_workout_days = (today - workouts[0].date).days if workouts else None
        
        # Find last "hard" workout (long duration or high HR)
        last_hard_days = None
        for w in workouts:
            is_hard = (w.duration_min and w.duration_min >= 60) or (w.heart_rate_avg and w.heart_rate_avg >= 155)
            if is_hard:
                last_hard_days = (today - w.date).days
                break
        
        # Calculate 7-day vs prior 7-day load
        def sum_duration_for_range(start_offset: int, end_offset: int) -> float:
            start = today - timedelta(days=start_offset)
            end = today - timedelta(days=end_offset)
            total = 0.0
            for w in workouts:
                if end <= w.date <= start and w.duration_min:
                    total += w.duration_min
            return total
        
        last7 = sum_duration_for_range(0, 7)
        prev7 = sum_duration_for_range(7, 14)
        load_change = ((last7 - prev7) / prev7 * 100) if prev7 > 0 else None
        
        # Calculate readiness score (1-10)
        score = 7.0  # Start at neutral
        factors = []
        
        # Days since last workout
        if last_workout_days == 0:
            score -= 0.5
            factors.append("Workout today: consider lighter intensity")
        elif last_workout_days >= 3:
            score += 1.0
            factors.append(f"Rest day: {last_workout_days}d since last workout")
        
        # Days since hard workout
        if last_hard_days is not None:
            if last_hard_days == 0:
                score -= 1.5
                factors.append("Hard workout today: recovery recommended")
            elif last_hard_days == 1:
                score -= 1.0
                factors.append("Hard workout yesterday: easier day suggested")
            elif last_hard_days >= 3:
                score += 0.5
                factors.append(f"Recovered: {last_hard_days}d since last hard session")
        
        # Weekly load change
        if load_change is not None:
            if load_change > 20:
                score -= 1.0
                factors.append(f"High load increase: {load_change:.0f}% (risk of overtraining)")
            elif load_change < -30:
                score += 0.5
                factors.append(f"Reduced load: {load_change:.0f}% (good for recovery)")
        
        # Determine recommended intensity
        score = max(1, min(10, round(score)))
        if score >= 8:
            intensity = "high"
            intensity_note = "Ready for high-intensity training"
        elif score >= 6:
            intensity = "moderate"
            intensity_note = "Good for moderate training"
        elif score >= 4:
            intensity = "low"
            intensity_note = "Consider lighter intensity or active recovery"
        else:
            intensity = "recovery"
            intensity_note = "Recovery day recommended"
        
        return {
            "readiness_score": score,
            "recommended_intensity": intensity,
            "intensity_note": intensity_note,
            "factors": factors[:3],
            "days_since_last_workout": last_workout_days,
            "days_since_hard_workout": last_hard_days,
            "weekly_load_change": round(load_change, 1) if load_change else None
        }
    except Exception as e:
        print(f"Readiness calculation error: {e}")
        return {
            "readiness_score": 7,
            "recommended_intensity": "moderate",
            "factors": ["Unable to calculate readiness"],
            "days_since_last_workout": None,
            "days_since_hard_workout": None,
            "weekly_load_change": None
        }


def identify_pace_zones(db: Session) -> Dict[str, Any]:
    """Identify pace zones from recent running workouts.
    Returns dict with easy, tempo, threshold, and interval paces.
    """
    try:
        from sqlalchemy import desc
        today = datetime.now().date()
        cutoff_90 = today - timedelta(days=90)
        
        # Get running workouts with pace data
        runs = db.query(Workout).filter(
            and_(
                Workout.user_id == 1,
                Workout.date >= cutoff_90,
                Workout.type.in_(["Run", "run", "Running", "running"]),
                Workout.pace_min_mi.isnot(None)
            )
        ).order_by(desc(Workout.date)).all()
        
        if not runs or len(runs) < 5:
            return {
                "has_zones": False,
                "message": "Need at least 5 recent runs to estimate pace zones"
            }
        
        # Extract paces (convert to min/mile if needed)
        paces = [r.pace_min_mi for r in runs if r.pace_min_mi and r.pace_min_mi > 0]
        if not paces:
            return {
                "has_zones": False,
                "message": "No pace data available in recent runs"
            }
        
        # Find best (fastest) pace (most recent 30 days preferred)
        recent_runs = [r for r in runs if (today - r.date).days <= 30]
        if recent_runs:
            best_pace = min([r.pace_min_mi for r in recent_runs if r.pace_min_mi and r.pace_min_mi > 0])
        else:
            best_pace = min(paces)
        
        # Estimate zones using common formulas
        # Easy pace: ~20-30% slower than 5K pace (best pace)
        # Tempo: ~10-15% slower than 5K pace
        # Threshold: ~5-8% slower than 5K pace
        # Interval: ~0-3% slower than 5K pace (close to best)
        
        easy_pace = round(best_pace * 1.25, 1)  # 25% slower
        tempo_pace = round(best_pace * 1.12, 1)  # 12% slower
        threshold_pace = round(best_pace * 1.06, 1)  # 6% slower
        interval_pace = round(best_pace * 1.02, 1)  # 2% slower
        
        # Calculate average pace for reference
        avg_pace = round(sum(paces) / len(paces), 1)
        
        return {
            "has_zones": True,
            "best_pace_min_mi": round(best_pace, 1),
            "avg_pace_min_mi": avg_pace,
            "zones": {
                "easy": {"min": easy_pace, "max": round(easy_pace * 1.1, 1), "description": "Conversational pace, aerobic base"},
                "tempo": {"min": tempo_pace, "max": round(tempo_pace * 1.05, 1), "description": "Comfortably hard, sustainable 20-30min"},
                "threshold": {"min": threshold_pace, "max": round(threshold_pace * 1.03, 1), "description": "Lactate threshold, sustainable 10-20min"},
                "interval": {"min": interval_pace, "max": round(best_pace * 0.98, 1), "description": "Hard effort, 3-5min intervals"}
            },
            "sample_size": len(paces),
            "data_range_days": (today - runs[-1].date).days
        }
    except Exception as e:
        print(f"Pace zone calculation error: {e}")
        return {
            "has_zones": False,
            "message": "Error calculating pace zones"
        }


def _build_more_about_plan(db: Session) -> List[str]:
    """Compute concise bullets explaining why this plan fits today's context.
    Uses recent workouts to create up to 3 actionable lines.
    """
    try:
        from sqlalchemy import desc
        # Last 28 days workouts
        cutoff_28 = datetime.now().date() - timedelta(days=28)
        ws = db.query(Workout).filter(
            and_(Workout.user_id == 1, Workout.date >= cutoff_28)
        ).order_by(desc(Workout.date)).all()
        if not ws:
            return [
                "Limited recent data: using conservative volume and knee-friendly progressions."
            ]

        # Aggregate durations by day
        by_day: Dict[str, float] = {}
        types: Dict[str, int] = {}
        last_hard_day_delta = None
        today = datetime.now().date()
        for w in ws:
            key = w.date.isoformat()
            by_day[key] = by_day.get(key, 0.0) + (w.duration_min or 0.0)
            types[w.type or "Other"] = types.get(w.type or "Other", 0) + 1
            # Simple "hard" heuristic
            if last_hard_day_delta is None and ((w.duration_min or 0) >= 60 or (w.heart_rate_avg or 0) >= 155):
                last_hard_day_delta = (today - w.date).days

        # 7-day vs prior 7-day load
        def sum_range(start_offset: int, end_offset: int) -> float:
            start = today - timedelta(days=start_offset)
            end = today - timedelta(days=end_offset)
            total = 0.0
            for i in range((start - end).days):
                d = (end + timedelta(days=i)).isoformat()
                total += by_day.get(d, 0.0)
            return total

        last7 = sum_range(0, 7)
        prev7 = sum_range(7, 14)
        ratio = (last7 / prev7) if prev7 > 0 else None
        load_line = None
        if ratio is not None:
            delta_pct = round((ratio - 1.0) * 100)
            load_line = f"Weekly load change: {delta_pct:+d}% (last 7d vs prior)"
        else:
            load_line = f"Weekly load (last 7d): {int(round(last7))} min"

        # Modality breakdown
        top_mod = sorted(types.items(), key=lambda x: x[1], reverse=True)[:1]
        mod_line = f"Recent focus: {top_mod[0][0]} sessions" if top_mod else None

        hard_line = f"Last hard session: {last_hard_day_delta}d ago" if last_hard_day_delta is not None else None

        bullets = [b for b in [hard_line, load_line, mod_line] if b]
        return bullets[:3] if bullets else ["Balanced session based on recent training."]
    except Exception:
        return ["Context unavailable: using safe defaults."]



class QuestionRequest(BaseModel):
    question: str
    user_data: Dict[str, Any]


@app.post("/api/ask-question")
def ask_question(req: QuestionRequest):
    """
    Interactive AI chat endpoint - ask questions about your health data
    """
    try:
        # Get LLM
        llm = _init_llm()
        
        # Prepare context from user data
        workouts = req.user_data.get("workouts", [])
        recovery = req.user_data.get("recovery", [])
        nutrition = req.user_data.get("nutrition", [])
        
        # Create context summary
        context = f"""
        User's Health Data Summary:
        - Total workouts: {len(workouts)}
        - Recovery data points: {len(recovery)}
        - Nutrition entries: {len(nutrition)}
        
        Recent workouts: {workouts[-5:] if workouts else "None"}
        Recent recovery: {recovery[-3:] if recovery else "None"}
        Recent nutrition: {nutrition[-3:] if nutrition else "None"}
        """
        
        # Create prompt for question answering
        prompt = f"""You are an AI health coach analyzing a user's fitness data. 

        {context}
        
        User Question: {req.question}
        
        Provide a concise, well-formatted answer following this structure:

        **Key Insight:** [1-2 sentences summarizing the main finding]

        **Details:**
        • [Specific data point or trend]
        • [Another relevant insight]
        • [Additional context if needed]

        **Recommendation:** [1 clear, actionable next step]

        Guidelines:
        - Keep total response under 150 words
        - Use bullet points (•) for lists
        - Include specific numbers from their data
        - Use emojis sparingly (🏃 for running, 🥗 for nutrition, 📈 for trends)
        - Be direct and actionable
        """
        
        # Get AI response
        with using_attributes(
            tags=["chat", "question_answering"],
            session_id=req.user_data.get("user_id", "default"),
            metadata={
                "question_type": "user_question",
                "data_points": len(workouts) + len(recovery) + len(nutrition)
            }
        ):
            if _TRACING:
                current_span = trace.get_current_span()
                if current_span:
                    current_span.set_attribute("chat.question", req.question)
                    current_span.set_attribute("chat.data_points", len(workouts) + len(recovery) + len(nutrition))
            
            with using_prompt_template(template=prompt, variables={"question": req.question}, version="v1.0"):
                response = llm.invoke([HumanMessage(content=prompt)])
        
        return {
            "answer": response.content,
            "question": req.question,
            "data_summary": {
                "workouts": len(workouts),
                "recovery": len(recovery),
                "nutrition": len(nutrition)
            }
        }
        
    except Exception as e:
        return {
            "answer": f"Sorry, I encountered an error analyzing your data: {str(e)}. Please try asking a simpler question.",
            "question": req.question,
            "error": str(e)
        }


@app.post("/api/analyze-health", response_model=HealthResponse)
def analyze_health(req: HealthRequest):
    """
    Main health analysis endpoint - runs multi-agent analysis
    
    Performance optimizations:
    - Limits data to most recent 100 workouts
    - Caches results for 1 hour
    - Uses faster LLM settings
    """
    try:
        # Limit data for performance (most recent 30 workouts for faster analysis)
        MAX_WORKOUTS = int(os.getenv("MAX_WORKOUTS_PER_ANALYSIS", "30"))
        req_dict = req.model_dump()
        
        if len(req_dict.get("workouts", [])) > MAX_WORKOUTS:
            # Sort by date and take most recent
            workouts = sorted(req_dict["workouts"], key=lambda x: x.get("date", ""), reverse=True)
            req_dict["workouts"] = workouts[:MAX_WORKOUTS]
        
        # Check cache
        if CACHE_ENABLED:
            cache_key = hash_data(req_dict)
            cached_result = _analysis_cache.get(cache_key)
            
            if cached_result:
                cached_time, cached_response_dict = cached_result
                if (datetime.now().timestamp() - cached_time) < CACHE_MAX_AGE:
                    # Return cached result (reconstruct from dict)
                    return HealthResponse(**cached_response_dict)
        
        # Run analysis with observability
        graph = build_health_graph()
        
        state = {
            "messages": [],
            "health_request": req_dict,
            "tool_calls": []
        }
        
        # Wrap the entire multi-agent workflow with observability
        with using_attributes(
            session_id=req_dict.get("user_id", "default"),
            user_id=req_dict.get("user_id", "default"),
            tags=["health_analysis", "multi_agent", "workflow"],
            metadata={
                "workflow_type": "health_analysis",
                "num_workouts": len(req_dict.get("workouts", [])),
                "num_recovery": len(req_dict.get("recovery", [])),
                "num_nutrition": len(req_dict.get("nutrition", [])),
                "parallel_agents": 3
            }
        ):
            if _TRACING:
                current_span = trace.get_current_span()
                if current_span:
                    current_span.set_attribute("workflow.name", "health_analysis")
                    current_span.set_attribute("workflow.type", "langgraph_parallel")
                    current_span.set_attribute("workflow.agents", "activity,recovery,nutrition,insights")
                    current_span.set_attribute("data.workout_count", len(req_dict.get("workouts", [])))
                    current_span.set_attribute("data.recovery_count", len(req_dict.get("recovery", [])))
                    current_span.set_attribute("data.nutrition_count", len(req_dict.get("nutrition", [])))
            
            result = graph.invoke(state)
        
        # Build response
        response = HealthResponse(
            readiness_score=result.get("readiness_score", 7),
            activity_summary=result.get("activity_analysis", "No activity data"),
            recovery_summary=result.get("recovery_analysis", "No recovery data"),
            nutrition_summary=result.get("nutrition_analysis", "No nutrition data"),
            insights=[result.get("final_insights", "Analysis complete")],
            correlations=result.get("correlations", []),
            recommendations=[],
            tool_calls=result.get("tool_calls", [])
        )
        
        # Cache the result (store as dict to avoid serialization issues)
        if CACHE_ENABLED:
            _analysis_cache[cache_key] = (datetime.now().timestamp(), response.model_dump())
            
            # Cleanup old cache entries (keep cache size reasonable)
            if len(_analysis_cache) > 100:
                # Remove oldest entries
                sorted_cache = sorted(_analysis_cache.items(), key=lambda x: x[1][0])
                for old_key, _ in sorted_cache[:50]:
                    del _analysis_cache[old_key]
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/nutrition/estimate", response_model=NutritionEstimateResponse)
def estimate_nutrition(req: NutritionEstimateRequest):
    """
    AI-powered nutrition estimation from meal description
    """
    try:
        result = estimate_nutrition_from_description.invoke({"meal_description": req.meal_description})
        
        return NutritionEstimateResponse(
            calories=result["calories"],
            protein_g=result["protein_g"],
            carbs_g=result["carbs_g"],
            fat_g=result["fat_g"],
            confidence_score=result["confidence_score"],
            method="ai_estimate"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Estimation failed: {str(e)}")


@app.post("/api/nutrition/log")
def log_nutrition(entry: NutritionEntry, db: Session = Depends(get_db_session)):
    """
    Log nutrition entry (manual or AI estimate) to database
    """
    try:
        # Create nutrition entry (single-user mode: user_id=1)
        # Pydantic already converts ISO date strings to date objects
        nutrition = Nutrition(
            user_id=1,
            date=entry.date,
            method=entry.method,
            source_description=entry.source_description or (entry.notes if entry.notes else "Manual entry"),
            calories=entry.calories,
            protein_g=float(entry.protein_g) if entry.protein_g is not None else None,
            carbs_g=float(entry.carbs_g) if entry.carbs_g is not None else None,
            fat_g=float(entry.fat_g) if entry.fat_g is not None else None
        )
        
        db.add(nutrition)
        db.commit()
        db.refresh(nutrition)
        
        return {
            "status": "success",
            "message": "Nutrition logged successfully",
            "entry": {
                "id": nutrition.id,
                "date": nutrition.date.isoformat(),
                "method": nutrition.method,
                "source_description": nutrition.source_description,
                "calories": nutrition.calories,
                "protein_g": nutrition.protein_g,
                "carbs_g": nutrition.carbs_g,
                "fat_g": nutrition.fat_g
            }
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error saving nutrition entry: {str(e)}")


# ============================================================================
# NUTRITION LOGGING FILE I/O FUNCTIONS
# ============================================================================

def load_nutrition_logs() -> dict:
    """Load nutrition logs from JSON file, create if doesn't exist"""
    nutrition_file = Path(__file__).parent / "my_nutrition_logs.json"
    
    if not nutrition_file.exists():
        # Create empty structure
        default_data = {
            "user_id": "default_user",
            "entries": []
        }
        with open(nutrition_file, 'w') as f:
            json.dump(default_data, f, indent=2)
        return default_data
    
    try:
        with open(nutrition_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading nutrition logs: {e}")
        return {"user_id": "default_user", "entries": []}


def save_nutrition_log(entry: dict) -> bool:
    """Save nutrition log entry to JSON file"""
    try:
        nutrition_file = Path(__file__).parent / "my_nutrition_logs.json"
        
        # Load existing data
        data = load_nutrition_logs()
        
        # Add timestamp to entry
        entry["timestamp"] = datetime.now().isoformat()
        
        # Append new entry
        data["entries"].append(entry)
        
        # Save back to file
        with open(nutrition_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error saving nutrition log: {e}")
        return False


def get_recent_logs(days: int = 7) -> List[dict]:
    """Get recent nutrition log entries"""
    data = load_nutrition_logs()
    entries = data.get("entries", [])
    
    if not entries:
        return []
    
    # Filter by date (last N days)
    cutoff_date = datetime.now() - pd.Timedelta(days=days)
    
    recent_entries = []
    for entry in entries:
        try:
            entry_date = datetime.fromisoformat(entry["date"])
            if entry_date >= cutoff_date:
                recent_entries.append(entry)
        except (ValueError, KeyError):
            # Skip entries with invalid dates
            continue
    
    # Sort by date (newest first)
    recent_entries.sort(key=lambda x: x.get("date", ""), reverse=True)
    
    return recent_entries


# ============================================================================
# NUTRITION LOGGING API ENDPOINTS
# ============================================================================

class NutritionSaveRequest(BaseModel):
    """Request to save nutrition entry"""
    date: str
    meal_description: str
    calories: int
    protein_g: int
    carbs_g: int
    fat_g: int
    confidence_score: float


@app.post("/api/nutrition/save")
def save_nutrition_entry(req: NutritionSaveRequest, db: Session = Depends(get_db_session)):
    """
    Save nutrition entry to database (AI estimate)
    """
    try:
        # Parse date
        entry_date = datetime.fromisoformat(req.date).date()
        
        # Create nutrition entry (single-user mode: user_id=1)
        nutrition = Nutrition(
            user_id=1,
            date=entry_date,
            method="ai_estimate",
            source_description=req.meal_description,
            calories=req.calories,
            protein_g=req.protein_g,
            carbs_g=req.carbs_g,
            fat_g=req.fat_g
        )
        
        db.add(nutrition)
        db.commit()
        db.refresh(nutrition)
        
        return {
            "status": "success",
            "message": "Nutrition entry saved successfully",
            "entry": {
                "id": nutrition.id,
                "date": nutrition.date.isoformat(),
                "meal_description": req.meal_description,
                "calories": req.calories,
                "protein_g": req.protein_g,
                "carbs_g": req.carbs_g,
                "fat_g": req.fat_g,
                "confidence_score": req.confidence_score,
                "method": "ai_estimate"
            }
        }
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error saving nutrition entry: {str(e)}")


@app.get("/api/nutrition/logs")
def get_nutrition_logs(days: int = 7, db: Session = Depends(get_db_session)):
    """
    Get recent nutrition log entries from database
    """
    try:
        # Calculate cutoff date
        cutoff_date = datetime.now().date() - timedelta(days=days)
        
        # Query nutrition logs for user_id=1 (single-user mode)
        nutrition_logs = db.query(Nutrition).filter(
            and_(
                Nutrition.user_id == 1,
                Nutrition.date >= cutoff_date
            )
        ).order_by(Nutrition.date.desc()).all()
        
        # Convert to dict format
        entries = []
        for log in nutrition_logs:
            entries.append({
                "id": log.id,
                "date": log.date.isoformat(),
                "method": log.method,
                "source_description": log.source_description,
                "meal_description": log.source_description,  # Alias for compatibility
                "calories": log.calories,
                "protein_g": log.protein_g,
                "carbs_g": log.carbs_g,
                "fat_g": log.fat_g,
                "timestamp": log.created_at.isoformat() if log.created_at else None
            })
        
        return {
            "status": "success",
            "entries": entries,
            "count": len(entries)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading nutrition logs: {str(e)}")




# ============================================================================
# WORKOUT API ENDPOINTS
# ============================================================================

@app.get("/api/workouts")
def get_workouts(days: int = 90, db: Session = Depends(get_db_session)):
    """
    Get recent workouts from database
    Returns format compatible with existing frontend (my_strava_data.json format)
    Default to 90 days to capture more historical data
    """
    try:
        # Calculate cutoff date
        cutoff_date = datetime.now().date() - timedelta(days=days)
        
        # Query workouts for user_id=1 (single-user mode)
        workouts = db.query(Workout).filter(
            and_(
                Workout.user_id == 1,
                Workout.date >= cutoff_date
            )
        ).order_by(Workout.date.desc()).all()
        
        # Convert to dict format compatible with frontend
        workouts_list = []
        for workout in workouts:
            workouts_list.append({
                "date": workout.date.isoformat(),
                "type": workout.type,
                "duration_min": workout.duration_min,
                "distance_mi": workout.distance_mi,
                "pace_min_mi": workout.pace_min_mi,
                "calories_burned": workout.calories_burned,
                "elevation_gain_ft": workout.elevation_gain_ft,
                "heart_rate_avg": workout.heart_rate_avg,
                "strava_activity_id": workout.strava_activity_id
            })
        
        return {
            "workouts": workouts_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading workouts: {str(e)}")


# ============================================================================
# STRAVA OAUTH & INTEGRATION ENDPOINTS
# ============================================================================

@app.get("/auth/strava/login")
def strava_login():
    """
    Redirect user to Strava OAuth authorization page
    """
    # Get the base URL from environment or request
    base_url = os.getenv("BASE_URL", "http://localhost:8001")
    redirect_uri = f"{base_url}/auth/strava/callback"
    
    auth_url = StravaClient.get_authorization_url(redirect_uri, state="auth")
    return RedirectResponse(url=auth_url)


@app.get("/auth/strava/callback")
def strava_callback(code: str, db: Session = Depends(get_db_session)):
    """
    Handle Strava OAuth callback
    Exchange code for tokens and save to database
    """
    try:
        # Exchange code for tokens
        token_data = StravaClient.exchange_code_for_tokens(code)
        
        # Get or create user (single-user mode)
        user = db.query(User).filter(User.id == 1).first()
        if not user:
            user = User(id=1)
            db.add(user)
        
        # Update user with Strava data
        user.strava_user_id = token_data["athlete"]["id"]
        user.access_token = token_data["access_token"]
        user.refresh_token = token_data["refresh_token"]
        user.token_expires_at = token_data["expires_at"]
        
        db.commit()
        
        # Trigger initial sync of activities
        try:
            StravaClient.sync_recent_activities(user, db, days=30)
        except Exception as e:
            print(f"⚠️  Error syncing activities: {e}")
        
        # Redirect to settings page with success message
        return RedirectResponse(url="/settings.html?strava=connected")
    
    except Exception as e:
        print(f"❌ Error in Strava callback: {e}")
        return RedirectResponse(url="/settings.html?strava=error")


@app.get("/auth/strava/status")
def strava_status(db: Session = Depends(get_db_session)):
    """
    Check if user is connected to Strava
    """
    try:
        user = db.query(User).filter(User.id == 1).first()
        
        if user and user.strava_user_id and user.access_token:
            return {
                "connected": True,
                "strava_user_id": user.strava_user_id,
                "token_expires_at": user.token_expires_at
            }
        else:
            return {
                "connected": False
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking Strava status: {str(e)}")


@app.post("/auth/strava/disconnect")
def strava_disconnect(db: Session = Depends(get_db_session)):
    """
    Disconnect Strava account
    """
    try:
        user = db.query(User).filter(User.id == 1).first()
        
        if user:
            user.strava_user_id = None
            user.access_token = None
            user.refresh_token = None
            user.token_expires_at = None
            db.commit()
        
        return {
            "status": "success",
            "message": "Strava disconnected successfully"
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error disconnecting Strava: {str(e)}")


@app.post("/auth/strava/sync")
def strava_sync_now(days: int = Query(default=30), db: Session = Depends(get_db_session)):
    """
    Manually trigger sync of Strava activities
    """
    try:
        user = db.query(User).filter(User.id == 1).first()
        
        if not user or not user.access_token:
            raise HTTPException(status_code=400, detail="Strava not connected")
        
        new_count = StravaClient.sync_recent_activities(user, db, days=days)
        
        return {
            "status": "success",
            "message": f"Synced {new_count} new activities from Strava",
            "new_activities": new_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error syncing Strava activities: {str(e)}")


@app.get("/api/strava/debug")
def strava_debug(db: Session = Depends(get_db_session)):
    """
    Debug endpoint to see raw Strava data and diagnose sync issues
    """
    try:
        user = db.query(User).filter(User.id == 1).first()
        
        if not user or not user.access_token:
            return {
                "error": "Strava not connected",
                "solution": "Visit /auth/strava/login to connect your Strava account"
            }
        
        # Get fresh token
        access_token = StravaClient.ensure_fresh_token(user, db)
        
        # Fetch recent activities from Strava API (summary)
        activities = StravaClient.get_athlete_activities(
            access_token,
            per_page=10  # Just 10 for debugging
        )
        
        # Test detailed fetch for first activity to show calories
        detailed_activity = None
        if activities:
            try:
                detailed_activity = StravaClient.get_activity_details(access_token, activities[0]["id"])
            except Exception as e:
                print(f"Debug: Failed to fetch detailed activity: {e}")
        
        # Check database workouts
        db_workouts = db.query(Workout).filter(
            Workout.user_id == 1
        ).order_by(Workout.date.desc()).limit(10).all()
        
        return {
            "strava_connection": {
                "connected": True,
                "strava_user_id": user.strava_user_id,
                "token_expires_at": user.token_expires_at,
                "token_expired": user.token_expires_at < int(datetime.utcnow().timestamp()) if user.token_expires_at else True
            },
            "raw_strava_activities": [
                {
                    "id": a.get("id"),
                    "name": a.get("name"),
                    "type": a.get("type"),
                    "date": a.get("start_date"),
                    "distance_meters": a.get("distance"),
                    "distance_miles": round(a.get("distance", 0) / 1609.34, 2),
                    "moving_time_sec": a.get("moving_time"),
                    "duration_min": round(a.get("moving_time", 0) / 60, 1),
                    "average_heartrate": a.get("average_heartrate"),
                    "calories": a.get("calories")
                } for a in activities[:10]
            ],
            "detailed_activity_test": {
                "activity_id": detailed_activity.get("id") if detailed_activity else None,
                "calories": detailed_activity.get("calories") if detailed_activity else None,
                "kilojoules": detailed_activity.get("kilojoules") if detailed_activity else None,
                "has_detailed_data": detailed_activity is not None
            },
            "database_workouts": [
                {
                    "id": w.id,
                    "date": str(w.date),
                    "type": w.type,
                    "distance_mi": w.distance_mi,
                    "duration_min": w.duration_min,
                    "pace_min_mi": w.pace_min_mi,
                    "heart_rate_avg": w.heart_rate_avg,
                    "strava_activity_id": w.strava_activity_id
                } for w in db_workouts
            ],
            "diagnosis": {
                "strava_activities_fetched": len(activities),
                "database_workouts_stored": len(db_workouts),
                "total_database_workouts": db.query(Workout).filter(Workout.user_id == 1).count(),
                "issue_detected": _diagnose_sync_issue(len(activities), len(db_workouts)),
                "recommendation": _get_sync_recommendation(len(activities), len(db_workouts))
            }
        }
    except Exception as e:
        return {
            "error": str(e),
            "type": type(e).__name__,
            "solution": "Check server logs for detailed error"
        }


def _diagnose_sync_issue(strava_count: int, db_count: int) -> str:
    """Helper function to diagnose sync issues"""
    if strava_count == 0 and db_count == 0:
        return "No activities found in Strava OR no activities in date range"
    elif strava_count > 0 and db_count == 0:
        return "Activities exist in Strava but not syncing to database"
    elif strava_count == 0 and db_count > 0:
        return "Database has old workouts but no recent Strava activities"
    elif strava_count == db_count:
        return "Sync appears to be working correctly"
    elif db_count < strava_count:
        return "Some activities not synced (check deduplication or sync days parameter)"
    else:
        return "More database workouts than Strava activities (manual uploads?)"


def _get_sync_recommendation(strava_count: int, db_count: int) -> str:
    """Helper function to get sync recommendations"""
    if strava_count == 0 and db_count == 0:
        return "1. Check if you have activities on Strava.com 2. Try increasing days parameter: /auth/strava/sync?days=90"
    elif strava_count > 0 and db_count == 0:
        return "Run manual sync: curl -X POST https://your-domain.com/auth/strava/sync?days=90"
    elif strava_count > db_count:
        return "Run manual sync with more days: curl -X POST https://your-domain.com/auth/strava/sync?days=180"
    else:
        return "Sync is working! Check dashboard at /dashboard.html"


# ============================================================================
# STRAVA WEBHOOK ENDPOINTS
# ============================================================================

@app.get("/webhooks/strava")
def strava_webhook_verify(
    hub_mode: str = Query(alias="hub.mode"),
    hub_verify_token: str = Query(alias="hub.verify_token"),
    hub_challenge: str = Query(alias="hub.challenge")
):
    """
    Verify Strava webhook subscription
    """
    if hub_verify_token == STRAVA_WEBHOOK_VERIFY_TOKEN:
        return JSONResponse(content={"hub.challenge": hub_challenge})
    else:
        raise HTTPException(status_code=403, detail="Invalid verify token")


@app.post("/webhooks/strava")
async def strava_webhook_event(event: Dict[str, Any], db: Session = Depends(get_db_session)):
    """
    Handle Strava webhook events (activity created/updated)
    """
    try:
        object_type = event.get("object_type")
        aspect_type = event.get("aspect_type")
        object_id = event.get("object_id")
        owner_id = event.get("owner_id")
        
        # Only process activity events
        if object_type != "activity":
            return {"status": "ignored", "reason": "not an activity event"}
        
        # Only process create and update events
        if aspect_type not in ["create", "update"]:
            return {"status": "ignored", "reason": f"aspect_type {aspect_type} not handled"}
        
        # Find user by Strava user ID
        user = db.query(User).filter(User.strava_user_id == owner_id).first()
        
        if not user:
            return {"status": "ignored", "reason": "user not found"}
        
        # Get fresh access token
        access_token = StravaClient.ensure_fresh_token(user, db)
        
        # Fetch full activity details
        activity = StravaClient.get_activity_details(access_token, object_id)
        
        # Check if workout already exists
        existing = db.query(Workout).filter(
            Workout.strava_activity_id == str(object_id)
        ).first()
        
        if existing and aspect_type == "update":
            # Update existing workout
            workout_data = StravaClient.activity_to_workout(activity, user.id)
            existing.date = workout_data.date
            existing.type = workout_data.type
            existing.duration_min = workout_data.duration_min
            existing.distance_mi = workout_data.distance_mi
            existing.pace_min_mi = workout_data.pace_min_mi
            existing.calories_burned = workout_data.calories_burned
            existing.elevation_gain_ft = workout_data.elevation_gain_ft
            existing.heart_rate_avg = workout_data.heart_rate_avg
            db.commit()
            return {"status": "updated", "workout_id": existing.id}
        
        elif not existing and aspect_type == "create":
            # Create new workout
            workout = StravaClient.activity_to_workout(activity, user.id)
            db.add(workout)
            db.commit()
            return {"status": "created", "workout_id": workout.id}
        
        else:
            return {"status": "ignored", "reason": "workout already exists or wrong event type"}
    
    except Exception as e:
        print(f"❌ Error processing webhook event: {e}")
        db.rollback()
        # Return 200 to avoid Strava retrying
        return {"status": "error", "message": str(e)}


# ============================================================================
# FILE UPLOAD ENDPOINTS
# ============================================================================

@app.post("/api/upload-data")
async def upload_data(file: UploadFile = File(...)):
    """
    Upload CSV/Excel file with health data
    """
    try:
        # Read file
        contents = await file.read()
        
        # Determine file type
        if file.filename.endswith('.csv'):
            df = pd.read_csv(pd.io.common.BytesIO(contents))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(pd.io.common.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="File must be CSV or Excel")
        
        # Validate required columns
        if 'date' not in df.columns:
            raise HTTPException(status_code=400, detail="Missing required column: date")
        
        # TODO: Parse and store data
        rows_processed = len(df)
        
        return {
            "status": "success",
            "message": f"Processed {rows_processed} rows",
            "filename": file.filename
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)


