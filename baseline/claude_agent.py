"""Claude agent for JobSwitchEnv using the Anthropic SDK with extended thinking.

Requires ANTHROPIC_API_KEY environment variable. Falls back to random agent if
the key is not set or the anthropic package is not installed.

Usage:
    ANTHROPIC_API_KEY=sk-ant-... python baseline/run_baseline.py --agent claude
"""

import os

from baseline.prompts import SYSTEM_PROMPT, format_observation_as_prompt

# ---------------------------------------------------------------------------
# Task-specific strategy blocks
# These are injected into the system prompt so the agent has explicit goal
# decomposition for each task rather than having to infer the optimal strategy
# from scratch. Each block encodes the non-obvious insight that makes the task
# non-trivial for a generic agent.
# ---------------------------------------------------------------------------

TASK_STRATEGIES = {
    1: """\
## Task 1 Strategy — The Straightforward Pivot (EASY)

You have two warm contacts already above the engagement threshold:
- Alice Chen at TechCorp (warmth ~0.65)
- Carol Johnson at DataFlow (warmth ~0.65)

The referral success threshold is warmth >= 0.60. Both contacts are close but
may need one more ENGAGE_CONTENT to safely exceed it.

OPTIMAL SEQUENCE:
  Step 0: ENGAGE_CONTENT → Alice Chen (warmth 0.65 → ~0.70, safely above 0.60)
  Step 1: ENGAGE_CONTENT → Carol Johnson (same)
  Step 2: REQUEST_REFERRAL → Alice Chen, company=TechCorp  [SUCCESS expected]
  Step 3: REQUEST_REFERRAL → Carol Johnson, company=DataFlow  [SUCCESS expected]
  Step 4: APPLY_WITH_REFERRAL → TechCorp (60% pass rate vs 10% cold)
  Step 5: APPLY_WITH_REFERRAL → DataFlow (60% pass rate). Peak pipelines = 2.
  Steps 6+: ADVANCE_ROUND on both pipelines every step to progress them.
  When a 3rd company is available or a pipeline is rejected: APPLY_COLD to
    maintain 3 parallel pipelines (needed for full process_efficiency score).
  When an OFFER arrives: COUNTER_OFFER at 110-115% then ACCEPT_OFFER.

CRITICAL RULES:
  - NEVER cold apply before getting referrals — 10% vs 60% is a 6x difference.
  - Run 3 concurrent pipelines for maximum process_efficiency reward (20% weight).
  - Counter any offer before accepting — negotiation quality is 15% of score.
  - Do NOT burn bridges (warmth < 0.40 before referral request).
""",

    2: """\
## Task 2 Strategy — The Cold Network Problem (MEDIUM)

ALL contacts start cold (warmth 0.10–0.25). Social capital starts at only 0.50
and regenerates at +0.04/step (slower than the default +0.05/step).

THE NON-OBVIOUS INSIGHT: 4 steps of targeted network investment yields higher
expected value than 4 cold applications (10% pass rate each). Invest first.

ONE urgent company: AIStartup has a role that expires at step 8. If you have
not gotten a referral there by step 6, apply cold as a backup.

OPTIMAL SEQUENCE:
  Find ONE person with is_active_connector=True at TechCorp or DataFlow (check
  the network graph). Focus all early investment on that single person.

  Step 0: ENGAGE_CONTENT → [target person]  (warmth ~0.15 → ~0.20)
  Step 1: ENGAGE_CONTENT → [target person]  (warmth ~0.20 → ~0.25)
  Step 2: SEND_MESSAGE   → [target person]  (warmth ~0.25 → ~0.40, costs 0.10 capital)
  Step 3: ENGAGE_CONTENT → [target person]  (warmth ~0.40 → ~0.45)
  Step 4: SEND_MESSAGE   → [target person]  (warmth ~0.45 → ~0.60+, unlocks referral)
  Step 5: REQUEST_REFERRAL → [target person], company=[their company]
  Step 6: APPLY_WITH_REFERRAL → [their company]
  Simultaneously (steps 5-6): APPLY_COLD → AIStartup as backup before step 8.
  Steps 7+: ADVANCE_ROUND on all active pipelines.

CAPITAL MANAGEMENT: Social capital starts at 0.50. Each SEND_MESSAGE costs
0.10 and regenerates 0.04/step. Plan your capital budget carefully.

DO NOT apply before step 4. The payoff from a referral application more than
compensates for the 4-step delay.
""",

    3: """\
## Task 3 Strategy — The Competing Pressures (HARD)

SITUATION:
  - CloudBase EXPLODING OFFER at 70% market rate, expires step 4. Do NOT accept.
  - TechCorp pipeline at TECHNICAL stage (expiry step 8).
  - MegaSoft pipeline at FINAL stage (expiry step 7) — one step from an offer.
  - Jack Brown (key contact) went cold at warmth 0.30.
  - Financial runway: only 5 steps. You MUST accept something by step 5.

THE HINGE MOVE: Your very first action MUST be REQUEST_DELAY on CloudBase.
This extends the CloudBase deadline by up to 3 steps, buying time to let the
better pipelines finish. Without this, CloudBase expires before TechCorp or
MegaSoft can produce an offer and you are forced into a bad outcome.

OPTIMAL SEQUENCE:
  Step 0: REQUEST_DELAY → CloudBase, days=3  [extends deadline from step 4 to step 7]
  Step 1: ADVANCE_ROUND → MegaSoft  [FINAL stage has ~50% chance of reaching OFFER]
  Step 2: ACCELERATE_PROCESS → TechCorp  [push TECHNICAL forward]
  Step 3: ENGAGE_CONTENT → Jack Brown  [re-warm from 0.30 toward 0.40+, avoids burn risk]
  Step 4: ADVANCE_ROUND → MegaSoft again if not yet at OFFER
           OR COUNTER_OFFER → MegaSoft/TechCorp if offer received
  Step 5+: Accept the best available offer. Prioritize MegaSoft or TechCorp offers
           (they are at better salary bands) over CloudBase.

IF NO BETTER OFFER ARRIVES BY STEP 5-6:
  COUNTER_OFFER → CloudBase at 110% (try to improve the 70% rate to ~85%),
  then ACCEPT_OFFER → CloudBase. You cannot let financial runway expire.

CRITICAL RULES:
  - First action MUST be REQUEST_DELAY on CloudBase (not ACCEPT_OFFER).
  - Do NOT accept CloudBase before step 3 unless absolutely forced.
  - Do NOT warm Jack Brown with SEND_MESSAGE early — social capital is scarce.
  - Maintain both TechCorp and MegaSoft pipeline advancement in parallel.
""",
}


# ---------------------------------------------------------------------------
# Anthropic client (singleton, lazy-initialized)
# ---------------------------------------------------------------------------

_anthropic_client = None


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is not None:
        return _anthropic_client
    try:
        import anthropic  # noqa: PLC0415

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
        return _anthropic_client
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Main agent function
# ---------------------------------------------------------------------------


def get_claude_action(
    obs: dict,
    message_history: list,
    task_id: int = 1,
    client=None,
) -> dict:
    """Call Claude claude-sonnet-4-6 with extended thinking to choose the next action.

    Extended thinking allocates a dedicated reasoning budget before the model
    commits to a JSON action. This dramatically improves performance on multi-
    constraint decisions (e.g., Task 3 where the agent must simultaneously
    manage an exploding offer, two pipelines, and a cold contact).

    Args:
        obs: Current environment observation dict.
        message_history: Mutable list of prior {role, content} messages.
            Updated in-place each call. Keep last 6 exchanges (12 messages).
        task_id: Which task (1, 2, or 3) the agent is running. Used to inject
            the task-specific strategy block into the system prompt.
        client: Optional pre-constructed Anthropic client. If None, the module-
            level singleton is used (requires ANTHROPIC_API_KEY env var).

    Returns:
        Action dict with keys: action_type, target, parameters, reasoning.
        Falls back to random agent if no client is available.
    """
    if client is None:
        client = _get_anthropic_client()

    if not client:
        # Graceful fallback — run_baseline.py imports get_random_action
        from baseline.run_baseline import _fallback_action  # noqa: PLC0415

        return _fallback_action()

    task_strategy = TASK_STRATEGIES.get(task_id, "")
    system = SYSTEM_PROMPT
    if task_strategy:
        system = system + "\n\n" + task_strategy

    prompt = format_observation_as_prompt(obs)

    # Build messages list from history + new user turn
    messages = list(message_history)
    messages.append({"role": "user", "content": prompt})

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8000,
        temperature=1,  # required when thinking is enabled
        thinking={"type": "enabled", "budget_tokens": 5000},
        system=system,
        messages=messages,
    )

    # Extended thinking responses contain both thinking blocks and text blocks.
    # Only the text block contains the JSON action we need to parse.
    content = next(
        (block.text for block in response.content if block.type == "text"),
        "",
    )

    # Update conversation history in-place; trim to last 6 exchanges (12 msgs)
    message_history.append({"role": "user", "content": prompt})
    message_history.append({"role": "assistant", "content": content})
    if len(message_history) > 12:
        message_history[:] = message_history[-12:]

    from baseline.run_baseline import parse_action_from_response  # noqa: PLC0415

    return parse_action_from_response(content)
