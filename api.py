from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel

from aura_agents import (
    run_screen_behavior_agent,
    run_sleep_recovery_agent,
    compute_aura_points,
)

app = FastAPI(title="AURA Backend", version="0.1.0")


class ScreenContext(BaseModel):
    __root__: Dict[str, Any]


class SleepContext(BaseModel):
    __root__: Dict[str, Any]


class AuraInput(BaseModel):
    __root__: Dict[str, Any]


@app.post("/screen-agent")
def screen_agent_endpoint(body: ScreenContext) -> Dict[str, Any]:
    """
    Call the Screen Behavior Agent.
    Frontend sends the screen_context as JSON.
    """
    screen_context = body.__root__
    result = run_screen_behavior_agent(screen_context)
    return result


@app.post("/sleep-agent")
def sleep_agent_endpoint(body: SleepContext) -> Dict[str, Any]:
    """
    Call the Sleep + Recovery Agent.
    """
    sleep_context = body.__root__
    result = run_sleep_recovery_agent(sleep_context)
    return result


@app.post("/aura-points")
def aura_points_endpoint(body: AuraInput) -> Dict[str, Any]:
    """
    Compute Aura Points / gamification summary.
    """
    aura_input = body.__root__
    result = compute_aura_points(aura_input)
    return result
