"""Baseline inference script for JobSwitchEnv.

Runs one of three agents (claude, gpt4o, or random) against all 3 tasks and
records scores to baseline/results.json.

Usage:
    # Random agent (no API key needed)
    python baseline/run_baseline.py

    # Claude agent (requires ANTHROPIC_API_KEY)
    ANTHROPIC_API_KEY=sk-ant-... python baseline/run_baseline.py --agent claude

    # GPT-4o agent (requires OPENAI_API_KEY)
    OPENAI_API_KEY=sk-... python baseline/run_baseline.py --agent gpt4o

    # Run a specific task with verbose step logging
    ANTHROPIC_API_KEY=sk-ant-... python baseline/run_baseline.py --agent claude --task 3 --runs 5 --verbose
"""

import argparse
import asyncio
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import JobSwitchEnvironment
from tasks.task1_straightforward import Task1
from tasks.task2_cold_network import Task2
from tasks.task3_competing_pressures import Task3
from baseline.prompts import SYSTEM_PROMPT, format_observation_as_prompt
from env.models import ActionType

# ---------------------------------------------------------------------------
# OpenAI client (singleton, lazy-initialized)
# ---------------------------------------------------------------------------

_openai_client = None


def _get_client():
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    try:
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None
        _openai_client = OpenAI(api_key=api_key)
        return _openai_client
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# GPT-4o agent
# ---------------------------------------------------------------------------


def get_llm_action(obs: dict, message_history: list[dict]) -> dict:
    """Call GPT-4o with full conversation history."""
    client = _get_client()
    if not client:
        return get_random_action(obs)

    prompt = format_observation_as_prompt(obs)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(message_history)
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
    )

    content = response.choices[0].message.content

    message_history.append({"role": "user", "content": prompt})
    message_history.append({"role": "assistant", "content": content})

    # Trim history to last 20 exchanges to avoid token limits
    if len(message_history) > 40:
        message_history[:] = message_history[-40:]

    return parse_action_from_response(content)


# ---------------------------------------------------------------------------
# Shared response parser
# ---------------------------------------------------------------------------


def parse_action_from_response(content: str) -> dict:
    """Parse LLM response into a valid action dict."""
    content = content.strip()
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    try:
        action = json.loads(content)
        if "action_type" not in action:
            return _fallback_action()
        if "reasoning" not in action:
            action["reasoning"] = "LLM did not provide reasoning"
        if "parameters" not in action:
            action["parameters"] = {}
        return action
    except json.JSONDecodeError:
        return _fallback_action()


def _fallback_action() -> dict:
    return {
        "action_type": "wait_for_signal",
        "parameters": {},
        "reasoning": "Fallback — could not parse LLM response",
    }


# ---------------------------------------------------------------------------
# Random agent
# ---------------------------------------------------------------------------


def get_random_action(obs: dict, _rng: random.Random | None = None) -> dict:
    """Random agent for baseline comparison. Uses seeded RNG."""
    rng = _rng or random.Random(42)
    action_types = list(ActionType)
    action_type = rng.choice(action_types)

    params = {}
    target = None

    nodes = obs.get("network_graph", {}).get("nodes", {})
    people = list(nodes.keys()) if nodes else ["Alice Chen"]
    companies_list = [c.get("name", "") for c in obs.get("target_companies", [])]
    if not companies_list:
        companies_list = ["TechCorp"]

    person = rng.choice(people) if people else "Alice Chen"
    company = rng.choice(companies_list) if companies_list else "TechCorp"

    if action_type == ActionType.RESEARCH_COMPANY:
        target = company
    elif action_type in (ActionType.ADD_TO_TARGETS, ActionType.REMOVE_TARGET):
        params["company"] = company
    elif action_type in (ActionType.ENGAGE_CONTENT, ActionType.SEND_MESSAGE):
        target = person
        if action_type == ActionType.SEND_MESSAGE:
            params["message_type"] = "professional"
    elif action_type == ActionType.REQUEST_INTRO:
        target = person
        other = rng.choice([p for p in people if p != person] or [person])
        params["from_person"] = person
        params["to_person"] = other
    elif action_type == ActionType.REQUEST_REFERRAL:
        target = person
        params["person"] = person
        params["company"] = company
    elif action_type == ActionType.TAILOR_RESUME:
        params["company"] = company
        params["role_type"] = "sde"
    elif action_type == ActionType.APPLY_COLD:
        params["company"] = company
        params["role"] = "SDE"
    elif action_type == ActionType.APPLY_WITH_REFERRAL:
        params["company"] = company
        params["role"] = "SDE"
        params["referrer"] = person
    elif action_type in (
        ActionType.ADVANCE_ROUND,
        ActionType.ACCELERATE_PROCESS,
        ActionType.DROP_PROCESS,
    ):
        params["company"] = company
    elif action_type == ActionType.REQUEST_DELAY:
        params["company"] = company
        params["days"] = 2
    elif action_type == ActionType.COUNTER_OFFER:
        offers = obs.get("offers_in_hand", [])
        if offers:
            params["company"] = offers[0].get("company", company)
            params["amount"] = offers[0].get("base_salary", 100000) * 1.10
        else:
            params["company"] = company
            params["amount"] = 110000
    elif action_type == ActionType.ACCEPT_OFFER:
        offers = obs.get("offers_in_hand", [])
        if offers:
            params["company"] = offers[0].get("company", company)
        else:
            params["company"] = company

    return {
        "action_type": action_type.value,
        "target": target,
        "parameters": params,
        "reasoning": "Random agent action",
    }


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


async def run_episode(
    task_class,
    agent: str = "random",
    n_runs: int = 3,
    verbose: bool = False,
    claude_client=None,
):
    """Run n episodes on a task and return (mean_score, scores list)."""
    scores = []
    for run in range(n_runs):
        run_seed = task_class.SEED + run
        rng = random.Random(run_seed)
        message_history: list[dict] = []

        env = JobSwitchEnvironment()
        task = task_class(env)
        obs = await task.reset()

        done = False
        steps = 0
        max_steps = task.MAX_STEPS

        while not done and steps < max_steps:
            try:
                if agent == "claude":
                    from baseline.claude_agent import get_claude_action  # noqa: PLC0415
                    action = get_claude_action(
                        obs,
                        message_history,
                        task_id=task_class.TASK_ID,
                        client=claude_client,
                    )
                elif agent == "gpt4o":
                    action = get_llm_action(obs, message_history)
                else:
                    action = get_random_action(obs, rng)

                result = await env.step(action)
                obs = result["observation"]
                done = result["done"]

                if verbose:
                    reward = result.get("reward", 0.0)
                    print(
                        f"    Step {steps:2d}: {action.get('action_type', '?'):<25} "
                        f"target={str(action.get('target', '-')):<20} "
                        f"reward={reward:+.3f}"
                    )

                steps += 1
            except Exception as e:
                print(f"  Step {steps} error: {e}")
                result = await env.step(_fallback_action())
                obs = result["observation"]
                done = result["done"]
                steps += 1

        score = task.grade()
        scores.append(score)
        print(f"  Run {run + 1}: score={score:.3f}, steps={steps}")

    mean = sum(scores) / len(scores) if scores else 0.0
    return mean, scores


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(description="JobSwitchEnv baseline runner")
    parser.add_argument(
        "--agent",
        choices=["claude", "gpt4o", "random", "auto"],
        default="auto",
        help="Agent to use (default: auto-detect from environment variables)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per task (default: 3)",
    )
    parser.add_argument(
        "--task",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run a specific task only (default: all 3)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print step-by-step action log for each episode",
    )
    return parser.parse_args()


async def main():
    args = _parse_args()

    # Resolve agent
    agent = args.agent
    claude_client = None

    if agent == "auto":
        try:
            from baseline.claude_agent import _get_anthropic_client  # noqa: PLC0415
            claude_client = _get_anthropic_client()
        except ImportError:
            pass
        if claude_client:
            agent = "claude"
        elif _get_client():
            agent = "gpt4o"
        else:
            agent = "random"
    elif agent == "claude":
        try:
            from baseline.claude_agent import _get_anthropic_client  # noqa: PLC0415
            claude_client = _get_anthropic_client()
            if not claude_client:
                print("Warning: ANTHROPIC_API_KEY not set. Falling back to random agent.")
                agent = "random"
        except ImportError:
            print("Warning: anthropic package not installed. Falling back to random agent.")
            agent = "random"
    elif agent == "gpt4o":
        if not _get_client():
            print("Warning: OPENAI_API_KEY not set. Falling back to random agent.")
            agent = "random"

    print("=" * 60)
    print("JobSwitchEnv Baseline Evaluation")
    print("=" * 60)
    print(f"Agent:          {agent}")
    print(f"Runs per task:  {args.runs}")
    print(f"Verbose:        {args.verbose}")
    print()

    all_tasks = [
        ("Task 1 — Straightforward Pivot (Easy)", Task1),
        ("Task 2 — Cold Network Problem (Medium)", Task2),
        ("Task 3 — Competing Pressures (Hard)", Task3),
    ]
    if args.task:
        all_tasks = [t for t in all_tasks if t[1].TASK_ID == args.task]

    results = {}
    for name, task_class in all_tasks:
        print(f"Running: {name}")
        mean, scores = await run_episode(
            task_class,
            agent=agent,
            n_runs=args.runs,
            verbose=args.verbose,
            claude_client=claude_client,
        )
        results[name] = {"mean": mean, "scores": scores}
        print(f"  Mean score: {mean:.3f}")
        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Task':<45} {'Mean Score':>10}")
    print("-" * 60)
    for name, data in results.items():
        print(f"{name:<45} {data['mean']:>10.3f}")
    print()

    output = {
        "agent": agent,
        "n_runs": args.runs,
        "results": {k: {"mean": v["mean"], "scores": v["scores"]} for k, v in results.items()},
    }
    with open("baseline/results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Results saved to baseline/results.json")


if __name__ == "__main__":
    asyncio.run(main())
