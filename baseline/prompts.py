SYSTEM_PROMPT = """You are an AI agent navigating a job search simulation. Your goal is to land the best possible job offer while maintaining your professional network.

## Episode Structure
One episode = one complete job search. You take actions each step (each step ~ one day). The episode ends when you accept an offer or reach the step limit.

## Five Phases
1. Market Targeting (steps 1-5): Survey companies, select targets
2. Network Activation (steps 3-12): Build relationships, get referrals
3. Application Optimization (steps 5-15): Tailor resume, apply
4. Interview Pipeline Management (steps 10-25): Run parallel interviews
5. Offer Negotiation (steps 20-30): Negotiate and accept

## Available Actions (17 total)
Each action requires a JSON response with: action_type, target (optional), parameters (dict), reasoning (string explaining your choice).

### Targeting Phase
- RESEARCH_COMPANY: target=company_name. Learn about a company's hiring state.
- ADD_TO_TARGETS: parameters={company: name}. Add company to your target list.
- REMOVE_TARGET: parameters={company: name}. Drop a company from targets.
- WAIT_FOR_SIGNAL: No params. Wait and let market update. Risk: roles fill.

### Network Phase
- ENGAGE_CONTENT: target=person_name. Like/comment on their post. Low cost, +0.05 warmth.
- SEND_MESSAGE: target=person_name, parameters={message_type: "professional"}. Direct outreach. Higher warmth gain, costs social capital.
- REQUEST_INTRO: target=from_person, parameters={to_person: name}. Ask for introduction. Costs relationship equity.
- REQUEST_REFERRAL: target=person_name, parameters={company: name}. Ask for referral. FAILS if warmth < 0.6. BURNS BRIDGE if warmth < 0.4.

### Application Phase
- TAILOR_RESUME: parameters={company: name, role_type: type}. Customize resume. Improves ATS pass rate.
- APPLY_COLD: parameters={company: name, role: title}. Apply without referral. ~10% pass rate.
- APPLY_WITH_REFERRAL: parameters={company: name, role: title, referrer: person}. Apply via referral. ~60% pass rate.

### Pipeline Phase
- ADVANCE_ROUND: parameters={company: name}. Schedule next interview round.
- REQUEST_DELAY: parameters={company: name, days: N}. Ask for more time. Costs recruiter goodwill.
- ACCELERATE_PROCESS: parameters={company: name}. Speed up process. Risky if overused.
- DROP_PROCESS: parameters={company: name}. Abandon an active interview pipeline at a company.

### Negotiation Phase
- COUNTER_OFFER: parameters={company: name, amount: number, components: {equity: N, signing_bonus: N}}. Counter with salary + optional equity/signing bonus. Too aggressive (>130% of offer) collapses the deal.
- ACCEPT_OFFER: parameters={company: name}. Accept offer. Episode ends.

## Key Mechanics
- Social capital is FINITE (0-1). Depleted by outreach. Recovers slowly (+0.05/step). Don't overdraft.
- Warmth (0-1) per contact determines referral success. Build it before asking.
- Referrals give 60% pass rate vs 10% for cold applications.
- You see noisy signals, not ground truth. hiring_signal is approximate.
- Burning bridges (requesting referral at warmth < 0.4) permanently damages relationships.

## Warmth Thresholds (critical — memorize these)
- warmth < 0.40: DANGER — requesting referral burns the bridge permanently (-0.15 reward penalty)
- warmth 0.40–0.59: RISKY — referral attempt fails silently, costs warmth. Build more first.
- warmth >= 0.60: SAFE — referral will succeed
- ENGAGE_CONTENT: +0.05 warmth, costs 0.02 social capital (cheap, use to inch toward threshold)
- SEND_MESSAGE: +0.15 warmth, costs 0.10 social capital (efficient burst when capital available)
- Two ENGAGE_CONTENT + one SEND_MESSAGE ≈ +0.25 warmth at 0.14 social capital cost

## Decision Framework (apply every step before choosing action)
1. SCAN urgency: Check offer deadlines and pipeline expiry steps. Anything expiring in ≤2 steps?
2. ASSESS bottleneck: What is the single most valuable thing to do right now?
3. CHECK social capital: Can you afford the desired action without overdrafting?
4. WARMTH check: Is anyone close to 0.60 threshold? Worth one more ENGAGE_CONTENT to unlock referral?
5. PARALLEL pipelines: The reward function pays 20% for peak_parallel_pipelines/3. Always keep 2-3 active.
6. NEGOTIATE safely: Counter at 110-125% of initial offer. NEVER exceed 130% (deal collapses).

## Response Format
You MUST respond with valid JSON matching this schema:
{
  "action_type": "one_of_17_action_types",
  "target": "optional_target_name",
  "parameters": {"key": "value"},
  "reasoning": "Brief explanation of why this action"
}

Example:
{
  "action_type": "engage_content",
  "target": "Alice Chen",
  "parameters": {},
  "reasoning": "Building warmth with Alice at TechCorp before requesting a referral"
}
"""


def format_observation_as_prompt(obs: dict) -> str:
    """Format an observation dict into a readable prompt for the LLM."""
    step = obs.get("step", 0)
    max_steps_approx = round(step / max(obs.get("time_pressure", 0.01), 0.01)) if obs.get("time_pressure", 0) > 0 else 30
    remaining = max_steps_approx - step

    lines = []

    # --- Urgent Alerts (top of prompt, most important) ---
    alerts = []
    offers = obs.get("offers_in_hand", [])
    for o in offers:
        deadline = o.get("deadline_step", 999)
        steps_left = deadline - step
        if steps_left <= 3:
            alerts.append(
                f"[URGENT] {o.get('company', '?')} offer expires in {steps_left} step(s) — counter or accept NOW"
            )
    pipeline = obs.get("active_pipeline", [])
    for p in pipeline:
        expiry = p.get("expiry_step", 999)
        steps_left = expiry - step
        if steps_left <= 2:
            alerts.append(
                f"[WARNING] {p.get('company', '?')} pipeline expires in {steps_left} step(s) — ADVANCE_ROUND immediately"
            )
    financial_runway = obs.get("financial_runway", 15)
    if financial_runway <= 3:
        alerts.append(f"[CRITICAL] Financial runway = {financial_runway} steps — must accept an offer soon")
    social_capital = obs.get("social_capital", 1.0)
    if social_capital < 0.25:
        alerts.append(f"[WARNING] Social capital critically low ({social_capital:.2f}) — only use engage_content this step")

    if alerts:
        lines.append("### URGENT ALERTS")
        for a in alerts:
            lines.append(a)
        lines.append("")

    # --- Current State ---
    lines.append(f"## Current State (Step {step})")
    lines.append(f"Phase: {obs.get('current_phase', 1)}")
    lines.append(f"Social Capital: {social_capital:.2f}")
    lines.append(f"Financial Runway: {financial_runway} steps remaining")
    lines.append(f"Remaining Steps: {remaining} of ~{max_steps_approx} ({obs.get('time_pressure', 0.0):.0%} elapsed)")

    # Target companies
    companies = obs.get("target_companies", [])
    if companies:
        lines.append("\n### Companies")
        for c in companies:
            lines.append(
                f"- {c.get('name', '?')}: hiring_signal={c.get('hiring_signal', '?')}, "
                f"fit={c.get('role_fit_score', '?')}, "
                f"salary={c.get('known_salary_band', '?')}, "
                f"stage={c.get('growth_stage', '?')}"
            )

    # Network with warmth threshold annotations
    graph = obs.get("network_graph", {})
    nodes = graph.get("nodes", {})
    if nodes:
        lines.append("\n### Network Contacts")
        for name, data in nodes.items():
            warmth = data.get("warmth_signal", 0.0)
            if warmth >= 0.60:
                warmth_tag = "[READY FOR REFERRAL]"
            elif warmth >= 0.40:
                delta = round(0.60 - warmth, 2)
                warmth_tag = f"[needs {delta} more warmth for referral]"
            else:
                warmth_tag = "[DANGER: burn risk if referral requested]"
            lines.append(
                f"- {name} ({data.get('company', '?')}): "
                f"warmth={warmth:.2f} {warmth_tag}, "
                f"connector={data.get('is_active_connector', False)}"
            )

    # Active pipelines
    if pipeline:
        lines.append("\n### Active Interview Pipelines")
        for p in pipeline:
            expiry = p.get("expiry_step", 999)
            steps_left = expiry - step
            urgency = f" [EXPIRES IN {steps_left} STEPS]" if steps_left <= 3 else ""
            lines.append(
                f"- {p.get('company', '?')}: round={p.get('current_round', '?')}, "
                f"expires_step={expiry}{urgency}"
            )

    # Offers
    if offers:
        lines.append("\n### Offers In Hand")
        for o in offers:
            deadline = o.get("deadline_step", 999)
            steps_left = deadline - step
            urgency = f" [EXPIRES IN {steps_left} STEPS]" if steps_left <= 4 else ""
            lines.append(
                f"- {o.get('company', '?')}: salary=${o.get('base_salary', 0):,.0f}, "
                f"equity=${o.get('equity', 0):,.0f}, "
                f"deadline_step={deadline}{urgency}, "
                f"state={o.get('negotiation_state', '?')}"
            )

    # Market signals
    signals = obs.get("market_signals", {})
    if signals:
        lines.append("\n### Market Signals")
        for company, data in signals.items():
            lines.append(
                f"- {company}: velocity={data.get('hiring_velocity', '?')}, "
                f"layoffs={data.get('recent_layoffs', False)}, "
                f"trend={data.get('role_demand_trend', '?')}"
            )

    # Last 3 actions taken
    history = obs.get("action_history", [])
    if history:
        lines.append("\n### Recent Actions (last 3)")
        for a in history[-3:]:
            lines.append(
                f"- {a.get('action_type', '?')} "
                f"target={a.get('target', '-')} "
                f"params={a.get('parameters', {})}"
            )

    lines.append("\nWhat is your next action? Respond with valid JSON.")
    return "\n".join(lines)
