import os
import json
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
# Bedrock wrapper
# ---------------------------------------------------------------------

def call_bedrock_converse(
    user_message: str,
    model_id: str = MODEL_ID,
    max_tokens: int = 512,
    temperature: float = 0.7,
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

    history: list (oldest first) of up to 10 previous agent calls.
             Each element has the form:
             {
               "created_at": "... ISO timestamp ...",
               "context": {... metrics at that time ...},
               "agent_output": {... stored_output ...}
             }

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

    # Only last 10 entries for sanity
    history = history[-10:]

    current_context_json = json.dumps(current_context, ensure_ascii=False)
    history_json = json.dumps(history, ensure_ascii=False)

    # Simple deterministic style seed → more variety but reproducible
    style_index = len(history) % 4
    if style_index == 0:
        style_profile = "warm and encouraging coach, focusing on gentle suggestions."
    elif style_index == 1:
        style_profile = "calm and factual observer, focusing on clear, neutral wording."
    elif style_index == 2:
        style_profile = "gentle but slightly more firm reminder, emphasizing follow-through."
    else:
        style_profile = "practical guide, suggesting 1–2 concrete next steps."

    # Examples deliberately avoid specific numeric values to reduce copying
    examples = r"""
Example 1 (everything fine, no notification, no therapist):
{
  "notification_title": "",
  "notification_description": "",
  "write_therapist_mail": false,
  "therapist_mail_address": "",
  "therapist_mail_title": "",
  "therapist_mail_content": ""
}

Example 2 (mildly concerning day, notification only):
{
  "notification_title": "Gentle reminder to rest",
  "notification_description": "You seem to have slept less than usual and spent a lot of time on screens. Try to take a short break and wind down a bit earlier today.",
  "write_therapist_mail": false,
  "therapist_mail_address": "",
  "therapist_mail_title": "",
  "therapist_mail_content": ""
}

Example 3 (repeated concerning pattern, suggest therapist):
{
  "notification_title": "Consider getting extra support",
  "notification_description": "Your recent days show very poor sleep and high screen time. It might help to talk to a professional about how you’re feeling.",
  "write_therapist_mail": true,
  "therapist_mail_address": "info@psychotherapie-muenchen.de",
  "therapist_mail_title": "Request for an initial consultation",
  "therapist_mail_content": "Dear therapist,\n\nI have been experiencing poor sleep and high stress for several days in a row. I would like to ask for an initial consultation to discuss my situation and possible next steps.\n\nKind regards,\nA concerned patient"
}
"""

    user_message = f"""
You are the single AURA Agent (Adaptive Unified Routine Assistant).

You receive:
1. The CURRENT daily context as JSON:
{current_context_json}

2. The HISTORY of your last outputs (oldest first, up to 10 entries).
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
- Only when clearly justified, suggest contacting a therapist via email.

STYLE PROFILE FOR THIS RESPONSE:
- For this response, write like a {style_profile}

------------------- INPUT FIELD SEMANTICS -------------------
Input fields (may be missing or null):
- last_nights_sleep_duration_hours:
    TOTAL number of hours the user actually slept last night.
    Example: 3.2 means they slept 3.2 hours, NOT that they are 3.2 hours short.
- resting_hr_bpm:
    The user's resting heart rate in beats per minute.
- total_screen_minutes:
    Total minutes spent on screens today.
- steps:
    Total steps walked today.
- long_sessions_over_20_min:
    Number of long screen sessions (> 20 minutes) today.
- residence_location:
    The city and country where the user lives.

------------------- RISK CLASSIFICATION RULES -------------------
MENTALLY, classify days like this (do NOT output the labels, just use them internally):

A day is SEVERE if ANY of these hold:
- last_nights_sleep_duration_hours is not null AND < 2.0
- steps is not null AND < 500
- total_screen_minutes is not null AND > 720
- long_sessions_over_20_min is not null AND >= 20

A day is CONCERNING (but not necessarily SEVERE) if ANY of these hold:
- last_nights_sleep_duration_hours is not null AND < 4.0
- steps is not null AND < 2000
- total_screen_minutes is not null AND > 480
- long_sessions_over_20_min is not null AND >= 12

Otherwise the day is OK.

Look at the 5 most recent days: up to the last 4 entries from the history (newest first) plus the current day.

------------------- THERAPIST DECISION LOGIC -------------------
Use the following logic for write_therapist_mail:

- Let window_5 be the last 5 days (up to 4 history entries + current day).
- Count how many days in window_5 are SEVERE and how many are CONCERNING.

Set write_therapist_mail = true if ANY of these are true:
1) There are at least 3 days that are CONCERNING or SEVERE in window_5.
2) There are at least 2 SEVERE days in window_5.
3) Today is SEVERE AND there is at least 1 other CONCERNING or SEVERE day in window_5.

Otherwise set write_therapist_mail = false.

When write_therapist_mail is true:
- You MUST fill therapist_mail_address, therapist_mail_title, and therapist_mail_content with plausible values.
- therapist_mail_content should be a short, respectful email body that the user could send.

When write_therapist_mail is false:
- therapist_mail_address, therapist_mail_title, therapist_mail_content MUST be empty strings.

------------------- STRICT RULES ABOUT NUMBERS -------------------
THIS IS VERY IMPORTANT:

- You MUST NOT invent any numeric values (hours, minutes, steps, heart rate, counts).
- If you mention a concrete number (like hours of sleep, minutes on screen, or steps),
  it MUST come directly from the CURRENT context or from a specific day in the HISTORY.
- If you mention last_nights_sleep_duration_hours in the CURRENT day, you MUST use exactly
  the numeric value from current_context["last_nights_sleep_duration_hours"].
- You MUST NOT write phrases like "X hours too short" because the required baseline sleep
  (how many hours the user needs) is NOT provided.
- If a field is null or missing, you MUST NOT invent a numeric value for it.
  In that case, speak qualitatively (e.g., "less sleep than usual" if history supports it)
  or avoid mentioning numbers at all.

------------------- VARIETY & NON-REPETITION -------------------
- Read all previous "agent_output" objects in the history, especially:
    - notification_title
    - notification_description
- You MUST NOT return exactly the same notification_title and notification_description
  as any previous history entry.
- Avoid repeating short generic phrases like "Consider getting extra support" over and over:
  rephrase them or give slightly more specific guidance.
- If your best advice is similar to something you said before, rephrase it and/or make it
  more specific, or slightly escalated (for example, add a concrete next step or timing).
- Over time, as the pattern continues without improvement, your wording should reflect
  increasing urgency and care (while still being supportive and non-judgmental).

------------------- OUTPUT SCHEMA -------------------
You MUST return exactly one JSON object with ALL of these keys and NO other keys:

- "notification_title": short title for a popup notification to the user
    (can be empty string "" if no notification is needed)
- "notification_description": 1–2 sentences suggesting what the user could do next
    (can be empty "" if no notification is needed)
- "write_therapist_mail": boolean
- "therapist_mail_address": string
- "therapist_mail_title": string
- "therapist_mail_content": string

Here are EXAMPLES of VALID output JSON objects (they are only examples, do NOT copy them literally, adapt to the current data):
{examples}

------------------- STRICT OUTPUT FORMAT (CRITICAL) -------------------
- You MUST produce exactly ONE JSON object for the CURRENT situation.
- Your reply MUST start with the character '{{' as the very first character of the response.
- Your reply MUST end with the matching '}}' of that JSON object.
- Do NOT include any text, explanations, labels, or the word "Example" before or after the JSON.
- Do NOT wrap the JSON in markdown fences (no ```).
- The JSON object MUST contain ALL of these keys and NO other keys:
  "notification_title", "notification_description",
  "write_therapist_mail", "therapist_mail_address",
  "therapist_mail_title", "therapist_mail_content".

Now, think step by step silently (do NOT output your reasoning), then output that single JSON object.
"""

    response_text = call_bedrock_converse(user_message)

    # Simple JSON parsing using the strict format instructions
    try:
        result = json.loads(response_text)
    except json.JSONDecodeError:
        logging.error("Failed to parse model JSON: %r", response_text)
        result = {}

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
        output = run_aura_for_user(user_id=1)
        print("=== AURA Agent Output ===")
        print(json.dumps(output, indent=2, ensure_ascii=False))
    except Exception as e:
        logging.error("Demo failed: %s", e)
