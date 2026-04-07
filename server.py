"""FastAPI server for JobSwitchEnv with session-based instances."""

import uuid
import time
from collections import OrderedDict
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Optional

from env.environment import JobSwitchEnvironment
from tasks.task1_straightforward import Task1
from tasks.task2_cold_network import Task2
from tasks.task3_competing_pressures import Task3

app = FastAPI(title="JobSwitchEnv", version="0.1.0")

TASK_REGISTRY = {1: Task1, 2: Task2, 3: Task3}
MAX_SESSIONS = 500

# Session-based environment instances with task tracking
sessions: OrderedDict[str, dict] = OrderedDict()


def _evict_if_needed():
    """Remove oldest sessions if over capacity."""
    while len(sessions) > MAX_SESSIONS:
        sessions.popitem(last=False)


class ResetRequest(BaseModel):
    seed: int = 42
    task_id: int = 1
    max_steps: int = 30
    session_id: Optional[str] = None


class StepRequest(BaseModel):
    action_type: str
    target: Optional[str] = None
    parameters: dict = {}
    reasoning: str = ""
    session_id: str


@app.get("/")
async def root():
    return {"status": "ok", "environment": "jobswitch-env", "version": "0.1.0"}


@app.get("/health")
async def health():
    return {"status": "ok", "environment": "jobswitch-env", "version": "0.1.0"}


@app.post("/reset")
async def reset(request: Optional[ResetRequest] = Body(default=None)):
    if request is None:
        request = ResetRequest()
    sid = request.session_id or str(uuid.uuid4())
    env = JobSwitchEnvironment()

    task_class = TASK_REGISTRY.get(request.task_id)
    task = None
    if task_class:
        task = task_class(env)
        obs = await task.reset()
    else:
        obs = await env.reset(
            seed=request.seed,
            task_id=request.task_id,
            max_steps=request.max_steps,
        )

    sessions[sid] = {"env": env, "task": task, "created": time.time()}
    _evict_if_needed()
    return {"observation": obs, "session_id": sid}


@app.post("/step")
async def step(request: StepRequest):
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")
    session = sessions[request.session_id]
    env = session["env"]
    action = {
        "action_type": request.action_type,
        "target": request.target,
        "parameters": request.parameters,
        "reasoning": request.reasoning,
    }
    try:
        result = await env.step(action)
        result["session_id"] = request.session_id

        # Include task grade when episode ends
        if result.get("done") and session.get("task"):
            result["info"]["task_grade"] = session["task"].grade()

        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state/{session_id}")
async def state(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return await sessions[session_id]["env"].get_state()


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "deleted"}
