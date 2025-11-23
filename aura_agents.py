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
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
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
# Bedrock wrapper  **UPDATED**
# ---------------------------------------------------------------------

def call_bedrock_converse(
    user_message: str,
    model_id: str = MODEL_ID,
    max_tokens: int = 512,
    temperature: Optional[float] = 0.8,  # <-- now optional
    top_p: Optional[float] = None,       # <-- now optional
) -> str:
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

    # Build inferenceConfig safely — NEVER send both top_p & temperature
    inference_config: Dict[str, Any] = {"maxTokens": max_tokens}

    if temperature is not None and top_p is not None:
        # Option 1: raise explicitly
        raise ValueError(
            "For this model, you must specify *either* temperature OR top_p — not both."
        )

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
- Very rarely, decide whether to suggest contacting a therapist via email.

[... prompt continues unchanged ...]
    """

    response_text = call_bedrock_converse(user_message)
    result = safe_json_loads(response_text)

    result.setdefault("notification_title", "")
    result.setdefault("notification_description", "")
    result.setdefault("write_therapist_mail", False)
    result.setdefault("therapist_mail_address", "")
    result.setdefault("therapist_mail_title", "")
    result.setdefault("therapist_mail_content", "")

    if not bool(result.get("write_therapist_mail")):
        result["write_therapist_mail"] = False
        result["therapist_mail_address"] = ""
        result["therapist_mail_title"] = ""
        result["therapist_mail_content"] = ""

    return result

# ---------------------------------------------------------------------
# History helper
# ---------------------------------------------------------------------

def update_history(
    history: List[Dict[str, Any]],
    new_output: Dict[str, Any],
    new_context: Dict[str, Any],
    created_at: Optional[str] = None,
    max_len: int = 10,
) -> List[Dict[str, Any]]:
    import datetime as _dt

    if created_at:
        ts = _dt.datetime.fromisoformat(created_at)
    else:
        if history:
            last_ts = _dt.datetime.fromisoformat(history[-1]["created_at"])
            ts = last_ts + _dt.timedelta(days=7)
        else:
            ts = _dt.datetime.utcnow()

    entry = {
        "created_at": ts.isoformat(),
        "context": new_context,
        "agent_output": new_output,
    }

    history = (history or []) + [entry]
    return history[-max_len:]

# ---------------------------------------------------------------------
# DB-integrated helper
# ---------------------------------------------------------------------

def run_aura_for_user(
    user_id: int,
    current_context_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    import datetime

    session = SessionLocal()
    try:
        user = session.query(User).filter_by(user_id=user_id).one()
        past_outputs = (
            session.query(AuraAgentOutput)
            .filter_by(user_id=user_id)
            .order_by(AuraAgentOutput.created_at.asc())
            .all()
        )

        history_for_llm = []
        for row in past_outputs:
            history_for_llm.append(
                {
                    "created_at": row.created_at.isoformat(),
                    "context": row.context or {},
                    "agent_output": row.output or {},
                }
            )

        if current_context_override is not None:
            current_context = {
                "last_nights_sleep_duration_hours": current_context_override.get("last_nights_sleep_duration_hours"),
                "resting_hr_bpm": current_context_override.get("resting_hr_bpm"),
                "total_screen_minutes": current_context_override.get("total_screen_minutes"),
                "steps": current_context_override.get("steps"),
                "long_sessions_over_20_min": current_context_override.get("long_sessions_over_20_min"),
                "residence_location": current_context_override.get("residence_location") or user.residence_location,
            }
        else:
            today = datetime.date.today()
            metrics = (
                session.query(DailyMetrics)
                .filter_by(user_id=user_id, day=today)
                .one_or_none()
            )
            if metrics is None:
                logging.warning("No DailyMetrics found for user %s on %s, creating empty row.",
                                user_id, today)
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

        agent_output = run_aura_agent(current_context, history_for_llm)

        updated_history = update_history(
            history_for_llm,
            agent_output,
            current_context,
            created_at=None,
            max_len=10,
        )

        last_entry = updated_history[-1]
        created_at_dt = datetime.datetime.fromisoformat(last_entry["created_at"])

        new_row = AuraAgentOutput(
            user_id=user.user_id,
            output=last_entry["agent_output"],
            context=last_entry["context"],
            created_at=created_at_dt,
        )
        session.add(new_row)
        session.commit()

        all_rows = (
            session.query(AuraAgentOutput)
            .filter_by(user_id=user_id)
            .order_by(AuraAgentOutput.created_at.asc())
            .all()
        )
        if len(all_rows) > 10:
            to_delete = all_rows[0 : len(all_rows) - 10]
            for r in to_delete:
                session.delete(r)
            session.commit()

        return {
            "agent_output": agent_output,
            "history_used": history_for_llm,
            "current_context": current_context,
        }

    finally:
        session.close()

# ---------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------

if __name__ == "__main__":
    try:
        output = run_aura_for_user(user_id=1)
        print("=== AURA Agent Output ===")
        print(json.dumps(output, indent=2, ensure_ascii=False))
    except Exception as e:
        logging.error("Demo failed: %s", e)
