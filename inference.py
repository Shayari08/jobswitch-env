#!/usr/bin/env python3
"""
inference.py — JobSwitchEnv agent for OpenEnv hackathon evaluation.

Uses the OpenAI-compatible client (works with HuggingFace Inference Router)
to run an LLM agent across all three benchmark tasks and emits structured
stdout logs for automated scoring.

Environment variables:
    API_BASE_URL   LLM endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME     Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       HuggingFace API key
"""

import asyncio
import os
import sys
from typing import List, Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import JobSwitchEnvironment
from tasks.task1_straightforward import Task1
from tasks.task2_cold_network import Task2
from tasks.task3_competing_pressures import Task3
from baseline.prompts import SYSTEM_PROMPT, format_observation_as_prompt
from baseline.run_baseline import parse_action_from_response, _fallback_action

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "no-key")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "jobswitch-env"
SUCCESS_THRESHOLD = 0.10
TEMPERATURE = 0.7
MAX_TOKENS = 512

# Task name mapping for [START] log line
TASK_NAMES = {
    1: "straightforward_pivot",
    2: "cold_network_problem",
    3: "competing_pressures",
}

# ---------------------------------------------------------------------------
# Task-specific strategy injected into system prompt per task
# ---------------------------------------------------------------------------

TASK_STRATEGIES = {
    1: """\
## Task 1 Strategy — The Straightforward Pivot
Warm contacts: Alice Chen at TechCorp (warmth ~0.65), Carol Johnson at DataFlow (warmth ~0.65).
SEQUENCE: Step 0: ENGAGE_CONTENT Alice Chen. Step 1: ENGAGE_CONTENT Carol Johnson.
Step 2: REQUEST_REFERRAL Alice Chen company=TechCorp. Step 3: REQUEST_REFERRAL Carol Johnson company=DataFlow.
Step 4: APPLY_WITH_REFERRAL TechCorp. Step 5: APPLY_WITH_REFERRAL DataFlow.
Steps 6+: ADVANCE_ROUND on both pipelines each step. Apply cold to a 3rd company for parallel pipelines.
When offer arrives: COUNTER_OFFER at 110%, then ACCEPT_OFFER.
NEVER cold apply before getting referrals.""",

    2: """\
## Task 2 Strategy — The Cold Network Problem
All contacts cold (warmth 0.10-0.25). Social capital starts at 0.50. DO NOT apply before step 4.
Find one is_active_connector=true person at TechCorp or DataFlow.
Step 0: ENGAGE_CONTENT that person. Step 1: ENGAGE_CONTENT again. Step 2: SEND_MESSAGE.
Step 3: ENGAGE_CONTENT again. Step 4: REQUEST_REFERRAL. Step 5: APPLY_WITH_REFERRAL.
Also: APPLY_COLD to AIStartup by step 6 (it expires at step 8).
Steps 6+: ADVANCE_ROUND each step. Counter and accept first offer.""",

    3: """\
## Task 3 Strategy — The Competing Pressures
CloudBase has an EXPLODING OFFER expiring at step 4. DO NOT accept it immediately.
YOUR FIRST ACTION MUST BE: REQUEST_DELAY on CloudBase (extends deadline by 3 steps).
Step 1: ADVANCE_ROUND on MegaSoft (at FINAL stage, close to offer).
Step 2: ACCELERATE_PROCESS on TechCorp (at TECHNICAL stage).
Step 3: ENGAGE_CONTENT Jack Brown (cold contact at MegaSoft, warmth 0.30).
If MegaSoft or TechCorp reaches OFFER by step 4-6: COUNTER_OFFER at 115%, then ACCEPT_OFFER.
If forced: COUNTER_OFFER CloudBase at 110%, then ACCEPT_OFFER. Never let runway expire.""",
}

# ---------------------------------------------------------------------------
# Structured log helpers (mandatory format)
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM action selection
# ---------------------------------------------------------------------------


def get_llm_action(
    client: OpenAI,
    obs: dict,
    message_history: list,
    task_id: int,
) -> dict:
    """Call the LLM to choose the next action. Falls back to wait_for_signal on error."""
    task_strategy = TASK_STRATEGIES.get(task_id, "")
    system = SYSTEM_PROMPT + ("\n\n" + task_strategy if task_strategy else "")
    prompt = format_observation_as_prompt(obs)

    messages = [{"role": "system", "content": system}]
    messages.extend(message_history[-12:])  # last 6 exchanges
    messages.append({"role": "user", "content": prompt})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        content = (completion.choices[0].message.content or "").strip()
        # Update history
        message_history.append({"role": "user", "content": prompt})
        message_history.append({"role": "assistant", "content": content})
        return parse_action_from_response(content)
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        return _fallback_action()


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


async def run_task(client: OpenAI, task_class) -> None:
    """Run one task episode and emit [START], [STEP]*, [END] logs."""
    task_name = f"task{task_class.TASK_ID}_{TASK_NAMES.get(task_class.TASK_ID, 'unknown')}"
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    message_history: list = []

    try:
        env = JobSwitchEnvironment()
        task = task_class(env)
        obs = await task.reset()

        for step in range(1, task_class.MAX_STEPS + 1):
            action = get_llm_action(client, obs, message_history, task_class.TASK_ID)

            # Build compact action string for log (action_type + target if present)
            action_str = action.get("action_type", "unknown")
            target = action.get("target")
            if target:
                action_str = f"{action_str}({target})"

            error: Optional[str] = None
            reward = 0.0
            done = False

            try:
                result = await env.step(action)
                obs = result["observation"]
                reward = float(result.get("reward", 0.0))
                done = bool(result.get("done", False))
            except Exception as exc:
                error = str(exc)[:100]

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        score = float(task.grade())
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_class.TASK_ID} error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_class in [Task1, Task2, Task3]:
        await run_task(client, task_class)


if __name__ == "__main__":
    asyncio.run(main())
