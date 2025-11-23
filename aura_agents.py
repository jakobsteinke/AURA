import os
import json
import re
import logging
from typing import Any, Dict, List, Optional

import boto3
from dotenv import load_dotenv

from db import SessionLocal
from db_models import User, DailyMetrics, AuraAgentOutput

# ---------------------------------------------------------------------
# Config & Bedrock client
# ---------------------------------------------------------------------

MODEL_ID = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"

logging.basicConfig(level=logging.INFO)

load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

brt = boto3.client(
    service_name="bedrock-runtime",
    aws_access_key_id=AWS_ACCESS_KEY_ID,          # only needed if running locally
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,  # only needed if running locally
    region_name="eu-central-1",
)

# ---------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------

def safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Safely parse JSON from model output.

    - Strips markdown fences and whitespace.
    - Keeps only the substring from first '{' to last '}'.
    - Tries json.loads; if it fails, auto-closes up to 3 missing '}'.
    - Returns {} on failure.
    """
    if not text:
        return {}

    cleaned = text.strip()

    # Strip ```json fences if present
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

    return obj

# ---------------------------------------------------------------------
# Bedrock wrapper
# ---------------------------------------------------------------------

def call_bedrock_converse(
    user_message: str,
    model_id: str = MODEL_ID,
    max_tokens: int = 512,
    temperature: Optional[float] = 0.7,
    top_p: Optional[float] = None,
) -> str:
    """
    Send a single user message to Bedrock using the 'converse' API
    and return the raw response text.

    NOTE:
    - For some models (incl. this Anthropic one), `temperature` and `top_p`
      CANNOT both be set at the same time.
    - By default we only use `temperature`. If you explicitly set `top_p`,
      set `temperature=None` when calling this function.
    """
    conversation = [
        {
            "role": "user",
            "content": [{"text": user_message}],
        }
    ]

    # Build inferenceConfig so that we NEVER send both temperature and topP
    inference_config: Dict[str, Any] = {"maxTokens": max_tokens}

    if temperature is not None and top_p is None:
        inference_config["temperature"] = temperature
    elif top_p is not None and temperature is None:
        inference_config["topP"] = top_p
    else:
        # If both are set or both are None, default to temperature if available
        if temperature is not None:
            inference_config["temperature"] = temperature
        elif top_p is not None:
            inference_config["topP"] = top_p

    response = brt.converse(
        modelId=model_id,
        messages=conversation,
        inferenceConfig=inference_config,
    )

    response_text = response["output"]["message"]["content"][0]["text"]
    logging.debug("Raw model response: %s", response_text)
    return response_text

# ---------------------------------------------------------------------
# Single AURA Agent (LLM logic only)
# ---------------------------------------------------------------------

def run_aura_agent(
    current_context: Dict[str, Any],
    history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Unified AURA Agent.

    Input: current_context dict with (any may be None / missing):
      - last_nights_sleep_duration_hours: float | None
      - resting_hr_bpm: int | None
      - total_screen_minutes: int | None
      - steps: int | None
      - long_sessions_over_20_min: int | None
      - residence_location: str | None (e.g. "Munich, Germany")

    history: list (oldest first) of up to 10 previous agent calls.

    Output JSON keys:
      - notification_title: str
      - notification_description: str
      - write_therapist_mail: bool
      - therapist_mail_address: str
      - therapist_mail_title: str
      - therapist_mail_content: str
    """
    if history is None:
        history = []

    history = history[-10:]  # only last 10 entries

    current_context_json = json.dumps(current_context, ensure_ascii=False)
    history_json = json.dumps(history, ensure_ascii=False)

    user_message = f"""
You are the single AURA Agent (Adaptive Unified Routine Assistant).

You receive:
1. The current daily context as JSON:
{current_context_json}

2. The history of your last outputs (oldest first, up to 10 entries).
   Each entry has this shape:
   {{
     "created_at": "... ISO timestamp ...",
     "context": {{ ...metrics at that time... }},
     "agent_output": {{ ...your previous decision... }}
   }}:
{history_json}

Your goals:
- Decide whether to show the user a notification right now.
- If so, make the notification short, specific, and caring.
- Very rarely decide whether to suggest contacting a therapist via email.

Input fields (may be missing or null):
- last_nights_sleep_duration_hours: TOTAL number of hours the user actually slept last night
  (for example 3.2 means they slept 3.2 hours, NOT that they are 3.2 hours short).
- resting_hr_bpm: the user's resting heart rate in beats per minute.
- total_screen_minutes: total minutes spent on screens today.
- steps: total steps walked today.
- long_sessions_over_20_min: number of long screen sessions (> 20 minutes) today.
- residence_location: the city and country where the user lives.

STRICT RULES ABOUT NUMBERS (VERY IMPORTANT):
- You MUST NOT invent any numeric values (hours, minutes, steps, heart rate, counts).
- If you mention a concrete number (like hours of sleep, minutes on screen, or steps),
  it MUST come directly from the CURRENT context or from a specific day in the HISTORY.
- If you mention last_nights_sleep_duration_hours, you MUST use exactly the numeric value
  from current_context["last_nights_sleep_duration_hours"].
- You MUST NOT write phrases like "X hours too short" because the required baseline
  (how many hours the user needs) is NOT provided.
- If a field is null or missing, you MUST NOT invent a numeric value for it.
  In that case, speak qualitatively (e.g., "less sleep than usual") or avoid mentioning numbers.

IMPORTANT: Therapist decision
- "write_therapist_mail" must be true if you detect clearly concerning data or a clear concerning pattern over multiple days.
- Only use this in extreme cases where the user's well-being seems seriously at risk (not just after a single bad day, only after multiple consecutive extremely bad entries in the history).
- If everything looks fine, keep "write_therapist_mail" false.
- When "write_therapist_mail" is true, you MUST fill therapist_mail_address, therapist_mail_title, and therapist_mail_content with plausible values.
- When "write_therapist_mail" is false, those three fields MUST be empty strings.
- The mail to the therapist must be written in first person, as if written by the user. It should explain the situation and ask for an appointment.

IMPORTANT: Do not repeat yourself
- Read all previous "agent_output" objects in the history, especially:
  - notification_title
  - notification_description
- You MUST NOT return exactly the same notification_title and notification_description as any previous history entry.
- If your best advice is similar to something you said before, you MUST rephrase it and/or make it slightly more specific or escalated.
- Over time, as the pattern continues without improvement, your wording should reflect increasing urgency and care (while still being supportive and non-judgmental).

Output fields and meaning (you MUST always include all keys in the JSON object):
- "notification_title": short title for a popup notification to the user (can be empty string "" if no notification is needed)
- "notification_description": 1–2 sentences suggesting what the user could do next (can be empty "" if no notification is needed)
- "write_therapist_mail": boolean
- "therapist_mail_address": string
- "therapist_mail_title": string
- "therapist_mail_content": string

STRICT OUTPUT FORMAT (THIS IS CRITICAL):
- You MUST produce exactly ONE JSON object for the CURRENT situation.
- Your reply MUST start with the character '{{' as the very first character of the response.
- Your reply MUST end with the matching '}}' of that JSON object.
- Do NOT include any text, explanations, labels, or the word "Example" before or after the JSON.
- Do NOT wrap the JSON in markdown fences (no ```).
- The JSON object MUST contain ALL of these keys and NO other keys:
  "notification_title", "notification_description",
  "write_therapist_mail", "therapist_mail_address",
  "therapist_mail_title", "therapist_mail_content".

Now, based ONLY on the CURRENT context and HISTORY, output that single JSON object.
"""

    response_text = call_bedrock_converse(user_message)
    result = safe_json_loads(response_text)

    # Ensure all keys exist with defaults
    result.setdefault("notification_title", "")
    result.setdefault("notification_description", "")
    result.setdefault("write_therapist_mail", False)
    result.setdefault("therapist_mail_address", "")
    result.setdefault("therapist_mail_title", "")
    result.setdefault("therapist_mail_content", "")

    # If we're not writing an email, clear all therapist fields
    if not bool(result.get("write_therapist_mail")):
        result["write_therapist_mail"] = False
        result["therapist_mail_address"] = ""
        result["therapist_mail_title"] = ""
        result["therapist_mail_content"] = ""

    return result


def update_history(
    history: List[Dict[str, Any]],
    new_output: Dict[str, Any],
    new_context: Dict[str, Any],
    created_at: Optional[str] = None,
    max_len: int = 10,
) -> List[Dict[str, Any]]:
    """
    Append a new history entry and keep only the last `max_len` entries.
    For testing: each new entry is +7 days compared to the previous one.
    """
    import datetime as _dt

    # --- Step 1: Determine timestamp for new entry ---
    if created_at:
        ts = _dt.datetime.fromisoformat(created_at)
    else:
        if history:
            last_ts = _dt.datetime.fromisoformat(history[-1]["created_at"])
            ts = last_ts + _dt.timedelta(days=7)
        else:
            ts = _dt.datetime.utcnow()

    # --- Step 2: Construct the new entry ---
    entry = {
        "created_at": ts.isoformat(),
        "context": new_context,
        "agent_output": new_output,
    }

    # --- Step 3: Update history and trim to last 10 entries ---
    history = (history or []) + [entry]
    return history[-max_len:]

# ---------------------------------------------------------------------
# DB-integrated helper: run agent for a given user_id
# ---------------------------------------------------------------------

def run_aura_for_user(
    user_id: int,
    current_context_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    High-level helper:
    - Loads AuraAgentOutput rows as history (including context snapshots),
    - Uses either:
        * the provided current_context_override (from API), OR
        * today's DailyMetrics for the user (legacy fallback),
    - Calls the AURA agent,
    - Uses update_history() to append a new entry with +7 days,
    - Stores only the last 10 outputs in the database,
    - Returns the full agent output (including therapist mail data) to the caller.
    """
    import datetime

    session = SessionLocal()
    try:
        user = session.query(User).filter_by(user_id=user_id).one()

        # 1) Load ALL existing outputs for this user (oldest first)
        past_outputs = (
            session.query(AuraAgentOutput)
            .filter_by(user_id=user_id)
            .order_by(AuraAgentOutput.created_at.asc())
            .all()
        )

        # Build history for the LLM: oldest → newest
        history_for_llm: List[Dict[str, Any]] = []
        for row in past_outputs:
            history_for_llm.append(
                {
                    "created_at": row.created_at.isoformat(),
                    "context": row.context or {},
                    "agent_output": row.output or {},
                }
            )

        # 2) Build current context
        if current_context_override is not None:
            # Use the context provided by the caller (e.g. API)
            current_context = {
                "last_nights_sleep_duration_hours": current_context_override.get("last_nights_sleep_duration_hours"),
                "resting_hr_bpm": current_context_override.get("resting_hr_bpm"),
                "total_screen_minutes": current_context_override.get("total_screen_minutes"),
                "steps": current_context_override.get("steps"),
                "long_sessions_over_20_min": current_context_override.get("long_sessions_over_20_min"),
                # fall back to stored user residence if not provided
                "residence_location": current_context_override.get("residence_location") or user.residence_location,
            }
        else:
            # Legacy behavior: use today's metrics from DailyMetrics
            today = datetime.date.today()

            metrics = (
                session.query(DailyMetrics)
                .filter_by(user_id=user_id, day=today)
                .one_or_none()
            )

            if metrics is None:
                logging.warning(
                    "No DailyMetrics found for user %s on %s, creating empty row.",
                    user_id,
                    today,
                )
                metrics = DailyMetrics(
                    user_id=user_id,
                    day=today,
                    last_nights_sleep_duration_hours=None,
                    resting_hr_bpm=None,
                    total_screen_minutes=None,
                    steps=None,
                    long_sessions_over_20_min=None,
                )
                session.add(metrics)
                session.commit()

            current_context = {
                "last_nights_sleep_duration_hours": metrics.last_nights_sleep_duration_hours,
                "resting_hr_bpm": metrics.resting_hr_bpm,
                "total_screen_minutes": metrics.total_screen_minutes,
                "steps": metrics.steps,
                "long_sessions_over_20_min": metrics.long_sessions_over_20_min,
                "residence_location": user.residence_location,
            }

        # 3) Call the AURA agent with the *old* history
        agent_output = run_aura_agent(current_context, history_for_llm)

        # 4) Use update_history to add the new entry with +7 days
        updated_history = update_history(
            history_for_llm,
            agent_output,    # store full output INCLUDING therapist info
            current_context,
            created_at=None, # let update_history handle +7 days logic
            max_len=10,
        )

        # The last entry is the one we just added
        last_entry = updated_history[-1]
        created_at_str = last_entry["created_at"]
        created_at_dt = datetime.datetime.fromisoformat(created_at_str)
        stored_context = last_entry["context"]
        stored_output = last_entry["agent_output"]

        # 5) Insert new row with synthetic created_at
        new_row = AuraAgentOutput(
            user_id=user.user_id,
            output=stored_output,   # now includes therapist fields
            context=stored_context,
            created_at=created_at_dt,  # override default timestamp
        )
        session.add(new_row)
        session.commit()

        # 6) Enforce: keep only last 10 rows in DB for this user
        all_rows = (
            session.query(AuraAgentOutput)
            .filter_by(user_id=user_id)
            .order_by(AuraAgentOutput.created_at.asc())
            .all()
        )
        if len(all_rows) > 10:
            to_delete = all_rows[0 : len(all_rows) - 10]  # earliest rows
            for r in to_delete:
                session.delete(r)
            session.commit()

        # Return:
        # - agent_output: full result (with therapist info if present)
        # - history_used: what the LLM actually saw (before new entry)
        # - current_context: today's / override context
        return {
            "agent_output": agent_output,
            "history_used": history_for_llm,
            "current_context": current_context,
        }

    finally:
        session.close()

# ---------------------------------------------------------------------
# Simple demo (assumes a user with user_id=1 exists and has metrics)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    try:
        # Demo without override → uses DailyMetrics
        output = run_aura_for_user(user_id=1)
        print("=== AURA Agent Output ===")
        print(json.dumps(output, indent=2, ensure_ascii=False))
    except Exception as e:
        logging.error("Demo failed: %s", e)
