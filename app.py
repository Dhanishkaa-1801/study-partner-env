"""
app.py — FastAPI wrapper around the StudyEnv environment.
This is what judges ping. Every OpenEnv call goes through here.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Dict
import uvicorn

from models import StudyAction, StudyObservation, StepResult

# We import lazily so the server starts even if environment.py
# is still being written by Person A — swap the stub below once
# Person A's environment.py is ready.
try:
    from environment import StudyEnv
    ENV_READY = True
except ImportError:
    ENV_READY = False

app = FastAPI(
    title="Adaptive Study Partner",
    description="OpenEnv environment — personalized study planning RL agent",
    version="1.0.0",
)

# One env instance per named session
# Fine for hackathon scale (single-agent evaluation)
envs: Dict[str, "StudyEnv"] = {}


# ─────────────────────────────────────────────
# Health + meta
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "env_ready": ENV_READY}


@app.get("/")
def root():
    return {
        "name": "adaptive-study-partner",
        "version": "1.0.0",
        "endpoints": ["/health", "/reset", "/step", "/state", "/tasks"],
    }


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": "task_1",
                "name": "Single Topic Recommendation",
                "difficulty": "easy",
                "max_steps": 5,
            },
            {
                "id": "task_2",
                "name": "Weekly Study Plan",
                "difficulty": "medium",
                "max_steps": 10,
            },
            {
                "id": "task_3",
                "name": "Exam Crunch Triage",
                "difficulty": "hard",
                "max_steps": 15,
            },
        ]
    }


# ─────────────────────────────────────────────
# OpenEnv core endpoints
# ─────────────────────────────────────────────

@app.post("/reset")
def reset(task_id: str = Query(default="task_1")):
    """
    Start a new episode for the given task.
    Returns the initial observation.
    """
    if not ENV_READY:
        raise HTTPException(503, "Environment not ready yet — environment.py missing")

    if task_id not in ("task_1", "task_2", "task_3"):
        raise HTTPException(400, f"Unknown task_id: {task_id}. Use task_1, task_2, or task_3")

    env = StudyEnv(task_id=task_id)
    envs["default"] = env
    obs = env.reset()
    return obs.model_dump()


@app.post("/step")
def step(action: StudyAction):
    """
    Take one action in the current episode.
    Returns observation, reward, done, info.
    """
    env = envs.get("default")
    if not env:
        raise HTTPException(400, "No active episode. Call /reset first.")

    try:
        result = env.step(action)
    except ValueError as e:
        raise HTTPException(422, str(e))

    return result.model_dump()


@app.get("/state")
def state():
    """
    Return the full current state without advancing the episode.
    """
    env = envs.get("default")
    if not env:
        raise HTTPException(400, "No active episode. Call /reset first.")

    return env.state().model_dump()


# ─────────────────────────────────────────────
# Run directly for local testing
# ─────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)