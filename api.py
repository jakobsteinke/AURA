# api.py

from typing import Any, Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aura_agents import (
    run_screen_behavior_agent,
    run_sleep_recovery_agent,
    compute_aura_points,
)

# -------------------------------
# FastAPI App Setup
# -------------------------------

app = FastAPI(title="AURA Backend", version="0.1.0")

origins = [
    "http://localhost:3000",  # Lovable / React
    "http://localhost:5173",  # Vite
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # <= IMPORTANT for frontend calls
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# API Endpoints
# -------------------------------

@app.post("/screen-agent")
def screen_agent_endpoint(screen_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call the Screen Behavior Agent.
    The frontend sends JSON – we directly pass it to the agent.
    """
    return run_screen_behavior_agent(screen_context)


@app.post("/sleep-agent")
def sleep_agent_endpoint(sleep_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call the Sleep + Recovery Agent.
    """
    return run_sleep_recovery_agent(sleep_context)


@app.post("/aura-points")
def aura_points_endpoint(aura_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute Aura Points / gamification summary.
    """
    return compute_aura_points(aura_input)
