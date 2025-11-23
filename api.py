# api.py

from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from aura_agents import run_aura_for_user

# -------------------------------
# FastAPI App Setup
# -------------------------------

app = FastAPI(title="AURA Backend", version="0.2.0")

origins = [
    "http://localhost:3000",  # React / Lovable
    "http://localhost:5173",  # Vite
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Request Models
# -------------------------------

class AuraRequest(BaseModel):
    user_id: int


# -------------------------------
# API Endpoints
# -------------------------------

@app.post("/run-aura")
def run_aura_endpoint(req: AuraRequest) -> Dict[str, Any]:
    """
    Call the unified AURA agent for a given user_id.

    Assumes:
    - The user with this user_id exists in the `users` table.
    - Today's metrics for this user exist in `daily_metrics`
      (or will be created empty by run_aura_for_user).
    - History is loaded from `aura_agent_outputs`.

    Returns:
    - The full agent output (including notification + optional therapist mail),
      but therapist details are NOT stored in the database.
    """
    try:
        result = run_aura_for_user(req.user_id)
        return result
    except Exception as e:
        # Optional: log the error in more detail
        raise HTTPException(status_code=500, detail=str(e))
