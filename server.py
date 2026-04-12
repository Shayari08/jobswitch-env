"""FastAPI server for JobSwitchEnv with session-based instances."""

import uuid
import time
from collections import OrderedDict
from fastapi import FastAPI, HTTPException, Body, Request
from pydantic import BaseModel
from typing import Any, Dict, Optional

from env.environment import JobSwitchEnvironment
from env.models import Action, Observation
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


# ---------------------------------------------------------------------------
# OpenEnv standard endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {"status": "ok", "environment": "jobswitch-env", "version": "0.1.0"}


@app.get("/health")
async def health():
    """OpenEnv required: status must be 'healthy'."""
    return {"status": "healthy", "environment": "jobswitch-env", "version": "0.1.0"}


@app.get("/metadata")
async def metadata():
    """OpenEnv required: returns name and description of the environment."""
    return {
        "name": "jobswitch-env",
        "description": (
            "End-to-end job search navigation environment for RL training. "
            "One episode = one complete job search across five phases: "
            "market targeting, network activation, application optimization, "
            "interview pipeline management, and offer negotiation."
        ),
        "version": "0.1.0",
        "author": "Shayari Bagchi",
        "documentation_url": "https://huggingface.co/spaces/Melinoee/jobswitch-env",
    }


@app.get("/schema")
async def schema():
    """OpenEnv required: returns JSON schemas for action, observation, and state."""
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": {
            "type": "object",
            "title": "JobSwitchState",
            "description": "Internal episode state (ground truth, not shown to agent)",
            "properties": {
                "step": {"type": "integer"},
                "phase": {"type": "string"},
                "social_capital": {"type": "number"},
                "financial_runway": {"type": "number"},
                "peak_parallel_pipelines": {"type": "integer"},
                "bridges_burned": {"type": "integer"},
                "granted_referrals": {"type": "object"},
            },
        },
    }


@app.post("/mcp")
async def mcp_endpoint(request: Request):
    """OpenEnv required: JSON-RPC 2.0 MCP endpoint for tool access."""
    try:
        body = await request.json()
    except Exception:
        return {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32700, "message": "Parse error"},
        }

    req_id = body.get("id")
    method = body.get("method", "")

    # tools/list — return the available tools (reset, step, state)
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "reset",
                        "description": "Reset the environment and start a new episode",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "task_id": {"type": "integer", "default": 1},
                                "seed": {"type": "integer", "default": 42},
                                "max_steps": {"type": "integer", "default": 30},
                            },
                        },
                    },
                    {
                        "name": "step",
                        "description": "Take an action in the environment",
                        "inputSchema": Action.model_json_schema(),
                    },
                ]
            },
        }

    # tools/call — dispatch to reset or step
    if method == "tools/call":
        params = body.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if tool_name == "reset":
            sid = str(uuid.uuid4())
            env = JobSwitchEnvironment()
            task_id = arguments.get("task_id", 1)
            task_class = TASK_REGISTRY.get(task_id)
            if task_class:
                task = task_class(env)
                obs = await task.reset()
            else:
                obs = await env.reset(**arguments)
                task = None
            sessions[sid] = {"env": env, "task": task, "created": time.time()}
            _evict_if_needed()
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": str({"observation": obs, "session_id": sid})}]},
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
        }

    # Default: method not found
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {"tools": [], "capabilities": {"simulation": True}},
    }


@app.get("/state")
async def state_global():
    """OpenEnv required: GET /state returns server-level state summary."""
    return {
        "active_sessions": len(sessions),
        "mode": "simulation",
        "environment": "jobswitch-env",
        "version": "0.1.0",
    }


# ---------------------------------------------------------------------------
# Simulation endpoints
# ---------------------------------------------------------------------------

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
async def state_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return await sessions[session_id]["env"].get_state()


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "deleted"}
