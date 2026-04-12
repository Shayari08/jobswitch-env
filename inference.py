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
    API_KEY        Alternative API key (fallback)
"""

import asyncio
import json
import os
import re
import sys
from typing import List, Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import JobSwitchEnvironment
from tasks.task1_straightforward import Task1
from tasks.task2_cold_network import Task2
from tasks.task3_competing_pressures import Task3
from baseline.prompts import SYSTEM_PROMPT, format_observation_as_prompt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "no-key")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "jobswitch-env"
SUCCESS_THRESHOLD = 0.10
TEMPERATURE = 0.4   # lower = more deterministic, follows strategies better
MAX_TOKENS = 800

TASK_NAMES = {
    1: "straightforward_pivot",
    2: "cold_network_problem",
    3: "competing_pressures",
}

# ---------------------------------------------------------------------------
# Task-specific strategies injected into system prompt
# ---------------------------------------------------------------------------

TASK_STRATEGIES = {
    1: """\
## Task 1 Strategy — The Straightforward Pivot (EASY, 20 steps)

SITUATION:
- Alice Chen at TechCorp: warmth ~0.65 [needs +0.05 to unlock referral at 0.70]
- Carol Johnson at DataFlow: warmth ~0.65 [needs +0.05 to unlock referral at 0.70]
- Market actively hiring. Financial runway = 15 steps. Social capital = 1.0.

OPTIMAL SEQUENCE — follow this exactly:
Step 0: {"action_type":"engage_content","target":"Alice Chen","parameters":{},"reasoning":"Push Alice Chen warmth above 0.60 for safe referral"}
Step 1: {"action_type":"engage_content","target":"Carol Johnson","parameters":{},"reasoning":"Push Carol Johnson warmth above 0.60 for safe referral"}
Step 2: {"action_type":"request_referral","target":"Alice Chen","parameters":{"company":"TechCorp"},"reasoning":"Warmth now >= 0.60, safe to request referral"}
Step 3: {"action_type":"request_referral","target":"Carol Johnson","parameters":{"company":"DataFlow"},"reasoning":"Warmth now >= 0.60, safe to request referral"}
Step 4: {"action_type":"apply_with_referral","target":"TechCorp","parameters":{"company":"TechCorp","role":"SDE","referrer":"Alice Chen"},"reasoning":"Use referral for 60% pass rate"}
Step 5: {"action_type":"apply_with_referral","target":"DataFlow","parameters":{"company":"DataFlow","role":"SDE","referrer":"Carol Johnson"},"reasoning":"Use referral for 60% pass rate"}
Step 6: {"action_type":"apply_cold","target":"AIStartup","parameters":{"company":"AIStartup","role":"SDE"},"reasoning":"3rd pipeline for process_efficiency score"}
Step 7+: ADVANCE_ROUND on any active pipeline each step (rotate through companies)
When OFFER appears: counter at 110% then accept.

CRITICAL RULES:
- NEVER request_referral if warmth < 0.60 (burns bridge permanently)
- ALWAYS check granted_referrals before apply_with_referral
- Run 3 parallel pipelines for maximum process_efficiency score
- Counter EVERY offer at 110-120% before accepting""",

    2: """\
## Task 2 Strategy — The Cold Network Problem (MEDIUM, 25 steps)

SITUATION:
- ALL contacts start cold (warmth 0.10-0.25). No one is ready for referral yet.
- AIStartup: urgent deadline at step 8 — role disappears if not applied by step 6.
- Social capital starts at 0.50 (vs default 1.0). Regenerates +0.04/step.
- Henry Zhao (AIStartup): is_active_connector=True — prioritize this person.
- Alice Chen (TechCorp): is_active_connector=True — backup target.

CRITICAL INSIGHT: 4 steps of network investment on one person > 4 cold applications.
A referral gives 60% pass rate vs 10% cold. Invest FIRST, apply SECOND.

OPTIMAL SEQUENCE:
Step 0: engage_content → Henry Zhao (AIStartup, is_active_connector=True, +0.05 warmth)
Step 1: engage_content → Henry Zhao (+0.05 warmth, now ~0.35)
Step 2: send_message → Henry Zhao (+0.15 warmth, now ~0.50, costs 0.10 capital)
Step 3: engage_content → Henry Zhao (+0.05 warmth, now ~0.55-0.60)
Step 4: request_referral → Henry Zhao company=AIStartup (warmth ~0.60, SUCCESS)
Step 5: apply_with_referral → AIStartup (referral grants 60% pass rate, before deadline step 8)
Step 6: engage_content → Alice Chen (TechCorp)
Step 7: send_message → Alice Chen (push toward 0.60)
Step 8+: request_referral → Alice Chen company=TechCorp, then apply_with_referral TechCorp
Steps 8+: ADVANCE_ROUND on all active pipelines each step
When OFFER: counter at 110%, then accept.

RULES:
- DO NOT apply (cold or referral) before step 4 — invest in network first
- Always use is_active_connector=true contacts when possible (more reliable)
- Monitor social_capital: if < 0.20 use only engage_content until it regenerates""",

    3: """\
## Task 3 Strategy — The Competing Pressures (HARD, 30 steps)

SITUATION:
- CloudBase EXPLODING OFFER: $X at 70% market rate. Deadline: step 4. DO NOT accept yet.
- TechCorp: at TECHNICAL stage (expiry step 8).
- MegaSoft: at FINAL stage (expiry step 7) — one ADVANCE_ROUND away from offer.
- Jack Brown (MegaSoft): warmth 0.30 — too cold for referral, don't request it.
- Financial runway: ONLY 5 STEPS. Cannot let runway expire without accepting.

STEP-BY-STEP (follow exactly):
Step 0: request_delay → CloudBase, days=3 [EXTENDS DEADLINE from step 4 → step 7]
Step 1: advance_round → MegaSoft [FINAL→OFFER has 50% chance, creates ~$120k offer]
Step 2: accelerate_process → TechCorp [push TECHNICAL→FINAL]
Step 3: advance_round → MegaSoft [if still FINAL, try again; if OFFER already, counter it]
Step 4: engage_content → Jack Brown [re-warm to help if MegaSoft stalls]
Step 5+: advance_round on whichever pipeline is closest to OFFER
Step 5-7: if MegaSoft/TechCorp offer arrives → counter_offer at 115%, then accept_offer

FALLBACK (if no better offer by step 6):
- counter_offer → CloudBase at 110% (try to improve 70% rate)
- accept_offer → CloudBase (better than runway expiry)

PANIC RULE: If financial_runway = 1 and you have ANY offer → accept_offer immediately.
If financial_runway = 0 and no accepted offer → you lose, so accept whatever you have.

NEVER: accept CloudBase as step 0. NEVER: let runway expire with an offer in hand.""",
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
        f"[STEP] step={step} action={action} reward={reward:.4f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# JSON extraction — robust multi-strategy parser
# ---------------------------------------------------------------------------


def _extract_json_from_response(content: str) -> Optional[dict]:
    """Try multiple strategies to extract a valid action JSON from LLM output."""
    content = content.strip()

    # Strategy 1: fenced code block (```json ... ```)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 2: bare JSON object anywhere in the string
    # Find the first { and last } that form a valid JSON object
    start = content.find("{")
    if start != -1:
        # Try progressively shorter substrings to find valid JSON
        for end in range(len(content), start, -1):
            if content[end - 1] == "}":
                try:
                    return json.loads(content[start:end])
                except json.JSONDecodeError:
                    continue

    # Strategy 3: look for action_type key and build minimal action
    m = re.search(r'"action_type"\s*:\s*"([^"]+)"', content)
    if m:
        action_type = m.group(1)
        # Extract target
        target_m = re.search(r'"target"\s*:\s*"([^"]+)"', content)
        target = target_m.group(1) if target_m else None
        # Extract company from parameters
        company_m = re.search(r'"company"\s*:\s*"([^"]+)"', content)
        company = company_m.group(1) if company_m else None
        # Extract reasoning
        reason_m = re.search(r'"reasoning"\s*:\s*"([^"]*)"', content)
        reasoning = reason_m.group(1) if reason_m else "LLM response"

        params = {}
        if company:
            params["company"] = company
        return {
            "action_type": action_type,
            "target": target,
            "parameters": params,
            "reasoning": reasoning,
        }

    return None


# ---------------------------------------------------------------------------
# Context-aware fallback action
# ---------------------------------------------------------------------------


def _smart_fallback(obs: dict) -> dict:
    """Return a sensible action when LLM parsing fails."""
    # Priority 1: accept/counter expiring offers
    offers = obs.get("offers_in_hand", [])
    step = obs.get("step", 0)
    runway = obs.get("financial_runway", 15)

    for o in offers:
        deadline = o.get("deadline_step", 999)
        if deadline - step <= 1:
            return {
                "action_type": "accept_offer",
                "target": o.get("company"),
                "parameters": {"company": o.get("company")},
                "reasoning": "Offer expiring — must accept now",
            }

    if runway <= 1 and offers:
        best = max(offers, key=lambda o: o.get("base_salary", 0))
        return {
            "action_type": "accept_offer",
            "target": best.get("company"),
            "parameters": {"company": best.get("company")},
            "reasoning": "Runway critical — accepting best available offer",
        }

    # Priority 2: advance any active pipeline
    pipeline = obs.get("active_pipeline", [])
    if pipeline:
        p = pipeline[0]
        company = p.get("company", "TechCorp")
        return {
            "action_type": "advance_round",
            "target": company,
            "parameters": {"company": company},
            "reasoning": "Advancing pipeline to reach offer",
        }

    # Priority 3: engage a contact to build warmth
    nodes = obs.get("network_graph", {}).get("nodes", {})
    if nodes:
        # Find best candidate: active connector, not too cold
        best_person = None
        best_score = -1
        for name, data in nodes.items():
            warmth = data.get("warmth_signal", 0)
            connector = data.get("is_active_connector", False)
            score = warmth + (0.2 if connector else 0)
            if score > best_score:
                best_score = score
                best_person = name
        if best_person:
            return {
                "action_type": "engage_content",
                "target": best_person,
                "parameters": {},
                "reasoning": "Building warmth on best available contact",
            }

    return {
        "action_type": "wait_for_signal",
        "parameters": {},
        "reasoning": "Fallback — waiting for market signal",
    }


# ---------------------------------------------------------------------------
# LLM action selection
# ---------------------------------------------------------------------------


def get_llm_action(
    client: OpenAI,
    obs: dict,
    message_history: list,
    task_id: int,
) -> dict:
    """Call the LLM to choose the next action. Falls back smartly on error."""
    task_strategy = TASK_STRATEGIES.get(task_id, "")
    system = SYSTEM_PROMPT + ("\n\n" + task_strategy if task_strategy else "")
    prompt = format_observation_as_prompt(obs)

    messages = [{"role": "system", "content": system}]
    messages.extend(message_history[-10:])  # last 5 exchanges
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

        parsed = _extract_json_from_response(content)
        if parsed and "action_type" in parsed:
            if "parameters" not in parsed or not isinstance(parsed["parameters"], dict):
                parsed["parameters"] = {}
            if "reasoning" not in parsed:
                parsed["reasoning"] = "LLM action"
            return parsed

    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)

    return _smart_fallback(obs)


# ---------------------------------------------------------------------------
# Validate and normalize action before sending to env
# ---------------------------------------------------------------------------


def _normalize_action(action: dict, obs: dict) -> dict:
    """
    Ensure the action has required fields and sensible defaults.
    Prevents Pydantic validation errors in env.step().
    """
    valid_actions = obs.get("valid_actions", [])
    action_type = action.get("action_type", "wait_for_signal")

    # If the action type is not in the valid list, downgrade to smart fallback
    if valid_actions and action_type not in valid_actions:
        fallback = _smart_fallback(obs)
        fallback["reasoning"] = f"Original action {action_type!r} not valid — {fallback['reasoning']}"
        return fallback

    params = action.get("parameters", {}) or {}
    target = action.get("target")

    # Ensure action-specific required fields
    if action_type == "send_message":
        if "message_type" not in params:
            params["message_type"] = "professional"

    if action_type == "request_delay":
        if "days" not in params:
            params["days"] = 3

    if action_type in ("apply_cold", "apply_with_referral"):
        if "role" not in params:
            params["role"] = "SDE"

    if action_type == "tailor_resume":
        if "role_type" not in params:
            params["role_type"] = "sde"

    if action_type == "counter_offer":
        # Compute a sensible counter if amount not provided
        if "amount" not in params or not params["amount"]:
            offers = obs.get("offers_in_hand", [])
            company = params.get("company") or target
            for o in offers:
                if o.get("company") == company:
                    params["amount"] = round(o["base_salary"] * 1.12, 2)
                    break
            if "amount" not in params:
                params["amount"] = 120000

    action["parameters"] = params
    if not action.get("reasoning"):
        action["reasoning"] = "Agent action"

    return action


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
            action = _normalize_action(action, obs)

            # Build compact action string for log
            action_str = action.get("action_type", "unknown")
            target = action.get("target") or action.get("parameters", {}).get("company")
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
                error = str(exc)[:120]
                # Try smart fallback on env error
                try:
                    fallback = _smart_fallback(obs)
                    result = await env.step(fallback)
                    obs = result["observation"]
                    reward = float(result.get("reward", 0.0))
                    done = bool(result.get("done", False))
                    error = f"orig_err={error[:60]}|used_fallback"
                except Exception:
                    pass

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        score = float(task.grade())
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_class.TASK_ID} fatal error: {exc}", flush=True)

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
