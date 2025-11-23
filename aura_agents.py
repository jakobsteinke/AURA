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
# Bedrock wrapper
# ---------------------------------------------------------------------

def call_bedrock_converse(
    user_message: str,
    model_id: str = MODEL_ID,
    max_tokens: int = 512,
    temperature: float = 0.8,
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

    # Few-shot examples of VALID outputs for the model
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
  "notification_description": "You slept only 4.0 hours last night and spent a lot of time on screens. Try to take a short break and wind down a bit earlier today.",
  "write_therapist_mail": false,
  "therapist_mail_address": "",
  "therapist_mail_title": "",
  "therapist_mail_content": ""
}

Example 3 (repeated concerning pattern, suggest therapist):
{
  "notification_title": "Consider getting extra support",
  "notification_description": "Your recent days show very little sleep and high screen time. It might help to talk to a professional about how you’re feeling.",
  "write_therapist_mail": true,
  "therapist_mail_address": "info@psychotherapie-muenchen.de",
  "therapist_mail_title": "Request for an initial consultation",
  "therapist_mail_content": "Dear therapist,\n\nI have been experiencing poor sleep and high stress for several days in a row. I would like to ask for an initial consultation to discuss my situation and possible next steps.\n\nKind regards,\nA concerned patient"
}
"""

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

Input fields (may be missing or null):
- last_nights_sleep_duration_hours: TOTAL number of hours the user actually slept last night (for example 3.2 means they slept 3.2 hours, NOT that they are 3.2 hours short).
- resting_hr_bpm: the user's resting heart rate in beats per minute.
- total_screen_minutes: total minutes spent on screens today.
- steps: total steps walked today.
- long_sessions_over_20_min: number of long screen sessions (> 20 minutes) today.
- residence_location: the city and country where the user lives.

Here are EXAMPLES of VALID output JSON objects (they are only examples, do NOT copy them literally, adapt to the current data):
{examples}

IMPORTANT: Therapist decision
- "write_therapist_mail" must be true if you detect slighlty concerning data or a clear concerning pattern over several days.
- If everything looks fine, keep "write_therapist_mail" false.
- When "write_therapist_mail" is true, you MUST fill therapist_mail_address, therapist_mail_title, and therapist_mail_content with plausible values.
- When "write_therapist_mail" is false, those three fields MUST be empty strings.

IMPORTANT: Do not repeat yourself
- Read all previous "agent_output" objects in the history, especially:
  - notification_title
  - notification_description
- You MUST NOT return exactly the same notification_title and notification_description as any previous history entry.
- If your best advice is similar to something you said before, you MUST rephrase it and/or make it slightly more specific or escalated (for example, suggest a concrete next step or timing).
- Over time, as the pattern continues without improvement, your wording should reflect increasing urgency and care (while still being supportive and non-judgmental).

Output fields and meaning (you MUST always include all keys in the JSON object):
- "notification_title": short title for a popup notification to the user (can be empty string "" if no notification is needed because everything is fine)
- "notification_description": 1–2 sentences suggesting what the user could do next (can be empty "" if no notification is needed because everything is fine)
- "write_therapist_mail": boolean, true only if you think a therapist should be contacted (set it to true if the user repeatedly ignores your advice and does not improve over time)
- "therapist_mail_address": email address of a therapist or mental health service near the residence_location when write_therapist_mail is true, else ""
- "therapist_mail_title": subject line of the email that should be sent to the therapist (concise, can be "")
- "therapist_mail_content": content of the email that should be sent to the therapist (can be "")

Behavioral rules:
- You may leave notification_title and notification_description as empty strings if nothing is needed.
- Consider the ENTIRE history plus the current context.
- A single concerning day is not enough to contact a therapist, but do contact if there are multiple. IMPORTANT: do not choose this conservatively, if you notice concerning data, use this feature. You can also choose it if the history already contains therapist mails.
- Set "write_therapist_mail": true if there is a clear pattern of repeated, concerning data over time.
- If write_therapist_mail is false, "therapist_mail_address", "therapist_mail_title", and "therapist_mail_content" should be empty strings.

Therapist email rule:
- Use residence_location to choose a plausible local therapist or mental health service email.
- If you are unsure, use a generic mental health support email for that city or country (e.g. a local counseling center).

Tone guidelines:
- Notifications should be supportive and non-judgmental.
- Focus on small, realistic next steps (e.g. "take a short walk", "wind down for sleep", "reduce screen time slightly").

Output format (VERY IMPORTANT):
- You MUST respond with a single valid JSON object.
- Do NOT wrap it in markdown fences.
- Do NOT add any text before or after the JSON.
- The JSON MUST have exactly these keys:
  "notification_title", "notification_description", "write_therapist_mail",
  "therapist_mail_address", "therapist_mail_title", "therapist_mail_content".
"""

    response_text = call_bedrock_converse(user_message)

    # Simple JSON parsing using the few-shot guidance
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
        # if explicitly given, use it
        ts = _dt.datetime.fromisoformat(created_at)
    else:
        if history:
            # Take the last history timestamp and add 7 days for testing purposes
            last_ts = _dt.datetime.fromisoformat(history[-1]["created_at"])
            ts = last_ts + _dt.timedelta(days=7)
        else:
            # First entry → use now()
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
