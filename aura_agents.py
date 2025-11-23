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

MODEL_ID = "eu.meta.llama3-2-1b-instruct-v1:0"

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
    temperature: float = 0.2,
    top_p: float = 0.9,
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

    history: list of the last up to 10 previous agent outputs (oldest first).
             Each element is expected to be a dict with the same output schema.

    Output: dict with keys (any may be empty strings / False):
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

    print(history)

    user_message = f"""
You are the single AURA Agent (Adaptive Unified Routine Assistant).

You receive:
1. The current daily context as JSON:
{current_context_json}

2. The history of your last outputs (oldest first, up to 10 entries):
{history_json}

Your goals:
- Decide whether to show the user a notification right now.
- If so, make the notification short, specific, and caring.
- Very rarely, decide whether to suggest contacting a therapist via email.

Input fields (may be missing or null):
- last_nights_sleep_duration_hours
- resting_hr_bpm
- total_screen_minutes
- steps
- long_sessions_over_20_min
- residence_location

Output fields and meaning (you MUST always include all keys in the JSON object):
- "notification_title": short title for a popup notification to the user (can be empty string "")
- "notification_description": 1–2 sentences suggesting what the user could do next (can be empty "")
- "write_therapist_mail": boolean, true only if you think a therapist should be contacted
- "therapist_mail_address": email address of a therapist or mental health service near the residence_location when write_therapist_mail is true, else ""
- "therapist_mail_title": subject line for the therapist email (concise, can be "")
- "therapist_mail_content": content of the therapist email (can be "")

Behavioral rules:
- You may leave notification_title and notification_description as empty strings if nothing is needed.
- You must be conservative with contacting a therapist:
  - Consider the ENTIRE history plus the current context.
  - A single concerning day is *not* enough to contact a therapist.
  - Only set "write_therapist_mail": true if there is a clear pattern of repeated, strongly concerning data over time.
  - If write_therapist_mail is false, "therapist_mail_address", "therapist_mail_title", and "therapist_mail_content" should be empty strings.

Therapist email rule:
- Use residence_location to choose a plausible local therapist or mental health service email.
- If you are unsure, use a generic mental health support email for that city or country (e.g. a local counseling center).

Tone guidelines:
- Notifications should be supportive and non-judgmental.
- Focus on small, realistic next steps (e.g. "take a short walk", "wind down for sleep", "reduce screen time slightly").

Output format:
You must respond with a single JSON object and nothing else.
DO NOT include any explanations, comments, or text outside JSON.
Use exactly this schema:

{{
  "notification_title": "string",
  "notification_description": "string",
  "write_therapist_mail": false,
  "therapist_mail_address": "string",
  "therapist_mail_title": "string",
  "therapist_mail_content": "string"
}}
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

# ---------------------------------------------------------------------
# (Optional) in-memory history helper
# ---------------------------------------------------------------------

def update_history(
    history: List[Dict[str, Any]],
    new_output: Dict[str, Any],
    max_len: int = 10,
) -> List[Dict[str, Any]]:
    """
    Append new_output to history and keep only the last `max_len` entries.
    """
    history = (history or []) + [new_output]
    return history[-max_len:]

# ---------------------------------------------------------------------
# DB-integrated helper: run agent for a given user_id
# ---------------------------------------------------------------------

def run_aura_for_user(user_id: int) -> Dict[str, Any]:
    """
    High-level helper:
    - Loads today's DailyMetrics for the user (assuming they are already written),
    - Loads last 10 AuraAgentOutput rows as history,
    - Calls the AURA agent,
    - Stores a *redacted* version of the output in the database (no therapist info),
    - Returns the full agent output (including therapist mail data) to the caller.
    """
    import datetime

    session = SessionLocal()
    try:
        user = session.query(User).filter_by(user_id=user_id).one()

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

        # Load last 10 outputs (oldest first)
        past_outputs = (
            session.query(AuraAgentOutput)
            .filter_by(user_id=user_id)
            .order_by(AuraAgentOutput.created_at.asc())
            .limit(10)
            .all()
        )
        history = [row.output for row in past_outputs]

        current_context = {
            "last_nights_sleep_duration_hours": metrics.last_nights_sleep_duration_hours,
            "resting_hr_bpm": metrics.resting_hr_bpm,
            "total_screen_minutes": metrics.total_screen_minutes,
            "steps": metrics.steps,
            "long_sessions_over_20_min": metrics.long_sessions_over_20_min,
            "residence_location": user.residence_location,
        }

        agent_output = run_aura_agent(current_context, history)

        # REDACT therapist info before storing in DB
        stored_output = dict(agent_output)
        if stored_output.get("write_therapist_mail"):
            stored_output["therapist_mail_address"] = ""
            stored_output["therapist_mail_title"] = ""
            stored_output["therapist_mail_content"] = ""

        new_row = AuraAgentOutput(
            user_id=user.user_id,
            output=stored_output,
        )
        session.add(new_row)
        session.commit()

        # We return the full output so the caller can actually send the email / show notification.
        #return agent_output

        return {
            "agent_output": agent_output,
            "history_used": history,   # <-- visible in Postman!
            "current_context": current_context,
        }


    finally:
        session.close()

# ---------------------------------------------------------------------
# Simple demo (assumes a user with user_id=1 exists and has metrics)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    try:
        output = run_aura_for_user(user_id=1)
        print("=== AURA Agent Output ===")
        print(json.dumps(output, indent=2, ensure_ascii=False))
    except Exception as e:
        logging.error("Demo failed: %s", e)
