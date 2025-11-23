from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from aura_agents import run_aura_for_user

app = FastAPI(title="AURA Backend", version="0.2.0")

origins = [
    "http://localhost:3000",  
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AuraContext(BaseModel):
    last_nights_sleep_duration_hours: float | None = None
    resting_hr_bpm: int | None = None
    total_screen_minutes: int | None = None
    steps: int | None = None
    long_sessions_over_20_min: int | None = None
    residence_location: str | None = None

class AuraRequest(BaseModel):
    user_id: int
    context: AuraContext


@app.post("/run-aura")
def run_aura_endpoint(req: AuraRequest) -> Dict[str, Any]:
    """
    Call the unified AURA agent for a given user_id and an explicit context.

    Assumes:
    - The user with this user_id exists in the `users` table.
    - History is loaded from `aura_agent_outputs`.
    - The *current* context is provided by the client in the request body.
    """
    try:
        current_context = req.context.dict()
        result = run_aura_for_user(req.user_id, current_context_override=current_context)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
