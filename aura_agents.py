import logging
import json
import boto3
from typing import Any, Dict, List, Optional
import json
import logging
import re
from typing import Any, Dict

brt = boto3.client("bedrock-runtime")

MODEL_ID = "eu.meta.llama3-2-1b-instruct-v1:0"

logging.basicConfig(level=logging.INFO)


def safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Safely parse JSON from model output.

    - Strips markdown fences and whitespace.
    - Keeps only the substring from first '{' to last '}'.
    - Tries json.loads; if it fails, auto-closes up to 3 missing '}'.
    - Normalizes common LLM mistakes (string instead of list) for some keys.
    - Returns {} on failure.
    """
    if not text:
        return {}

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()

    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first == -1 or last == -1 or last <= first:
        logging.error("No JSON object found in model output: %r", text)
        return {}

    candidate = cleaned[first:last + 1]

    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        open_count = candidate.count("{")
        close_count = candidate.count("}")
        max_extra_braces = 3
        obj = None

        for _ in range(max_extra_braces):
            if close_count >= open_count:
                break
            candidate += "}"
            close_count += 1
            try:
                obj = json.loads(candidate)
                break
            except json.JSONDecodeError:
                obj = None

        if obj is None:
            logging.error("Failed to parse model JSON even after repair; raw=%r", text)
            return {}

    if not isinstance(obj, dict):
        logging.error("Parsed JSON is not an object: %r", obj)
        return {}

    list_like_keys = {"detections", "actions", "interventions", "badges_earned"}
    for key in list_like_keys:
        if key in obj:
            val = obj[key]
            if isinstance(val, str):
                obj[key] = [val]
            elif val is None:
                obj[key] = []
            elif not isinstance(val, list):
                obj[key] = [val]

    return obj


def call_bedrock_converse(user_message: str,
                          model_id: str = MODEL_ID,
                          max_tokens: int = 512,
                          temperature: float = 0.2,
                          top_p: float = 0.9) -> str:
    """
    Send a single user message to Bedrock using the 'converse' API
    and return the raw response text.
    """
    conversation = [
        {
            "role": "user",
            "content": [{"text": user_message}],
        }
    ]

    response = brt.converse(
        modelId=model_id,
        messages=conversation,
        inferenceConfig={
            "maxTokens": max_tokens,
            "temperature": temperature,
            "topP": top_p,
        },
    )

    response_text = response["output"]["message"]["content"][0]["text"]
    logging.debug("Raw model response: %s", response_text)
    return response_text


def run_screen_behavior_agent(screen_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Screen Behavior Agent.

    Input: screen_context (dict) with keys like:
      - time_of_day, weekday, total_screen_time_minutes, session_length_minutes
      - app_usage, scroll_velocity, mood_self_report, recent_social_interaction_flag

    Returns dict with keys:
      - detections: List[str]
      - actions: List[str]
      - interventions: List[str]
      - notify_sleep_agent: Dict[str, Any]
    """
    screen_context_json = json.dumps(screen_context, ensure_ascii=False)

    user_message = f"""
    You are the **Screen Behavior Agent** in the AURA system (Adaptive Unified Routine Assistant).

    Your job:
    - Analyze phone usage behavior.
    - Detect harmful or unhelpful patterns.
    - Decide **concrete actions** AURA should take right now.
    - Optionally notify the Sleep + Recovery Agent if behavior impacts sleep/recovery.

    You receive the following JSON as input:

    {screen_context_json}
    Detection goals (examples, not exhaustive):

    - Doomscrolling events.

    - Late-night overstimulation.

    - Stress-driven compulsive use.

    - Increased usage after stressful social interactions.

    - Any pattern that suggests the user would benefit from a short break or wind-down.

    Available actions (examples, you can choose multiple):

    - "SUGGEST_BREAK" – show a gentle suggestion to pause.

    - "LOCK_SOCIAL_APPS_15_MIN" – temporarily lock social apps.

    - "ENABLE_GREYSCALE" – switch phone display to greyscale.

    - "TRIGGER_BREATHING_EXERCISE"– start a breathing exercise on the watch.

    - "NOTIFY_SLEEP_AGENT" – send a message to the Sleep Agent to adjust bedtime/wind-down.

    - "NO_ACTION" – if everything looks fine.

    Interventions should feel supportive and human, e.g.:

    - "It looks like you’ve been scrolling for a while. Want to try a 30-second breathing break?"

    - "It’s getting late and your feed seems quite stimulating. Should I dim things a bit for you?"

    You must respond with a single JSON object and nothing else.
    Do NOT include any explanations, descriptions, text outside JSON.
    Do NOT include comments or trailing commas.

    Use this exact schema:
    {{
      "detections": [ "string" ],
      "actions": [ "string" ],
      "interventions": [ "string" ],
      "notify_sleep_agent": {{
        "should_notify": true or false,
        "reason": "string",
        "payload": {{}}
      }}
    }}

    Rules:

    If you don’t want to notify the Sleep Agent, set "should_notify" to false and "payload" to an empty object.

    Keep messages short but caring.

    If the input looks totally fine, return "detections": [], "actions": ["NO_ACTION"].
    """

    response_text = call_bedrock_converse(user_message)
    result = safe_json_loads(response_text)

    result.setdefault("detections", [])
    result.setdefault("actions", [])
    result.setdefault("interventions", [])
    result.setdefault("notify_sleep_agent", {
        "should_notify": False,
        "reason": "",
        "payload": {}
    })

    return result
    

def run_sleep_recovery_agent(sleep_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sleep + Recovery Agent.

    Input: sleep_context (dict) with keys like:
      - last_nights_sleep_duration_hours, sleep_interruptions,
        average_sleep_duration_past_7_days_hours, sleep_debt_hours,
        hrv_ms, resting_hr_bpm, activity_level_today, circadian_phase,
        current_time, recent_screen_behavior (dict)

    Returns dict with keys:
      - detections: List[str]
      - actions: List[str]
      - interventions: List[str]
      - notify_screen_agent: Dict[str, Any]
    """
    sleep_context_json = json.dumps(sleep_context, ensure_ascii=False)

    user_message = f"""


    You are the Sleep + Recovery Agent in the AURA system.

    Your job:

    - Evaluate the user’s sleep and recovery state.

    - Detect issues like fatigue risk, sleep debt, poor recovery, stress-related changes.

    - Decide concrete actions to improve recovery and tomorrow’s energy.

    - Coordinate with the Screen Behavior Agent when needed.

    You receive the following JSON as input:

    {sleep_context_json}


    Detection goals:

    - Predicted fatigue (e.g., short sleep duration + high sleep debt + low HRV).

    - Poor recovery (e.g., increased resting HR, decreased HRV).

    - Sleep debt buildup.

    - Stress-related physiological changes.

    Available actions (examples, choose any that fit):

    - "RECOMMEND_EARLIER_BEDTIME"

    - "TRIGGER_WIND_DOWN_ROUTINE"

    - "SILENCE_NOTIFICATIONS"

    - "ADJUST_ALARM_MINUS_20" (wake up 20 minutes later if possible)

    - "SEND_DAY_STRUCTURE_SUGGESTIONS" (tips for pacing the next day)

    - "NOTIFY_SCREEN_AGENT" (to reduce stimulating evening use)

    - "NO_ACTION"

    Interventions must sound human and supportive, e.g.:

    - "Your recovery looks a bit low today. Going to bed 30 minutes earlier could really help."

    - "Sleep debt has built up over the last few days. I’ll keep evenings a bit calmer for you."

    You must respond with a single JSON object and nothing else.
    Do NOT include any explanations, descriptions, text outside JSON.
    Do NOT include comments or trailing commas.
    Use this exact schema:

    {{
      "detections": [ "string" ],
      "actions": [ "string" ],
      "interventions": [ "string" ],
      "notify_screen_agent": {{
        "should_notify": true or false,
        "reason": "string",
        "payload": {{}}
      }}
    }}


    Rules:

    If no issue is detected, use "detections": [], "actions": ["NO_ACTION"].

    If you want Screen Agent to calm down late-night usage, set "should_notify" to true
    and include a small "payload" with what you’d like it to do (e.g., lower stimulation before midnight).
    """

    response_text = call_bedrock_converse(user_message)
    result = safe_json_loads(response_text)

    # Make sure all expected keys exist

    result.setdefault("detections", [])
    result.setdefault("actions", [])
    result.setdefault("interventions", [])
    result.setdefault("notify_screen_agent", {
        "should_notify": False,
        "reason": "",
        "payload": {}
    })

    return result


def compute_aura_points(aura_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aura Points Calculator.

    Input: aura_input (dict) with keys like:
      - sleep: {duration_hours, sleep_debt_hours, interruptions, quality_score}
      - recovery: {hrv_ms, resting_hr_bpm}
      - screen: {total_minutes, doomscrolling_events, long_sessions_over_20_min, late_night_minutes_after_23}
      - activity: {steps, workout_minutes}

    Returns dict with keys:
      - aura_score: int (0-100)
      - subscores: {sleep, recovery, screen_hygiene, activity} each int 0-100
      - level: str
      - badges_earned: List[str]
      - message: str
    """
    aura_input_json = json.dumps(aura_input, ensure_ascii=False)

    user_message = f"""


    You are the Aura Points Calculator for the AURA system.

    Your job:

    - Take the user’s daily metrics.

    - Compute a daily aura_score from 0–100 (higher is better).

    - Provide subscores for sleep, recovery, screen_hygiene, and activity.

    - Optionally assign a "level" (e.g. "Recovering", "Balanced", "Thriving").

    - Optionally assign some badges (e.g. "Screen Break Hero", "Solid Sleeper").

    - Generate one short, kind summary message explaining the score.

    You receive the following JSON as input:

    {aura_input_json}


    Guidelines:

    - A good day with decent sleep, moderate screen use, and some activity should land around 70–85.

    - A very tough day (poor sleep + doomscrolling + low activity) might be 30–50.

    - Never shame the user; always be encouraging and focus on next steps.

    You must respond with a single JSON object and nothing else.
    Do NOT include any explanations, descriptions, text outside JSON.
    Do NOT include comments or trailing commas.
    Use exactly this schema:

    {{
      "aura_score": 0,
      "subscores": {{
        "sleep": 0,
        "recovery": 0,
        "screen_hygiene": 0,
        "activity": 0
      }},
      "level": "string",
      "badges_earned": [ "string" ],
      "message": "string"
    }}


    Rules:

    - All scores are integers in [0, 100].

    - "badges_earned" can be an empty list if nothing special happened.

    - "message" should be 1–2 sentences, supportive and specific.
    """

    response_text = call_bedrock_converse(user_message, temperature=0.3)
    result = safe_json_loads(response_text)

    # Ensure all keys exist with defaults

    result.setdefault("aura_score", 0)
    result.setdefault("subscores", {
        "sleep": 0,
        "recovery": 0,
        "screen_hygiene": 0,
        "activity": 0,
    })
    result.setdefault("level", "Unknown")
    result.setdefault("badges_earned", [])
    result.setdefault("message", "")

    return result
    

def demo_screen_agent() -> None:
    # Example input 
    screen_context = {
        "time_of_day": "23:17",
        "weekday": "Monday",
        "total_screen_time_minutes": 185,
        "session_length_minutes": 27,
        "app_usage": {
            "instagram": 73,
            "tiktok": 54,
            "youtube": 28,
            "other": 30
        },
        "scroll_velocity": "high",  # "low" | "medium" | "high" | None
        "mood_self_report": "tired",  # or None
        "recent_social_interaction_flag": True
    }

    result = run_screen_behavior_agent(screen_context)
    print("=== Screen Behavior Agent ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def demo_sleep_agent() -> None:
    sleep_context = {
        "last_nights_sleep_duration_hours": 5.3,
        "sleep_interruptions": 3,
        "average_sleep_duration_past_7_days_hours": 6.2,
        "sleep_debt_hours": 6.0,
        "hrv_ms": 48,
        "resting_hr_bpm": 72,
        "activity_level_today": "low",  # "low" | "medium" | "high"
        "circadian_phase": "late_evening",
        "current_time": "22:45",
        "recent_screen_behavior": {
            "doomscrolling_detected": True,
            "late_night_use_minutes": 45
        }
    }

    result = run_sleep_recovery_agent(sleep_context)
    print("=== Sleep + Recovery Agent ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def demo_aura_points() -> None:
    aura_input = {
        "day": "2025-11-22",
        "sleep": {
            "duration_hours": 6.5,
            "sleep_debt_hours": 3.5,
            "interruptions": 1,
            "quality_score": 72
        },
        "recovery": {
            "hrv_ms": 55,
            "resting_hr_bpm": 62
        },
        "screen": {
            "total_minutes": 210,
            "doomscrolling_events": 1,
            "long_sessions_over_20_min": 2,
            "late_night_minutes_after_23": 35
        },
        "activity": {
            "steps": 8200,
            "workout_minutes": 25
        }
    }

    result = compute_aura_points(aura_input)
    print("=== Aura Points ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    demo_screen_agent()
    demo_sleep_agent()
    demo_aura_points()