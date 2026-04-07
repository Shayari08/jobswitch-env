---
title: JobSwitchEnv
emoji: 💼
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# JobSwitchEnv

Every year, millions of professionals navigate career transitions using intuition alone. The decisions they make — who to reach out to first, when to apply versus wait for a referral, how to run parallel interview pipelines, when to push for an offer — are sequential, irreversible, and deeply interdependent. This is exactly the problem class that reinforcement learning was designed to solve. **JobSwitchEnv is the first open RL training environment for end-to-end job search navigation.**

One episode = one complete job search. The agent navigates five interdependent phases: market targeting, network activation, application optimization, interview pipeline management, and offer negotiation. Decisions in early phases constrain what is possible in later phases. The job market responds to how you search it — applying too broadly signals desperation and changes recruiter behavior. No supervised learning dataset can capture this dependency chain. RL is the correct tool.

This environment was built because the author — a fresher SDE getting rejected before their resume was even seen — realized the entire problem was a sequential decision problem nobody had modeled. JobSwitchEnv is the training environment for agents that learn to navigate it.

---

## Action Space

| Action | Phase | Key Params | Effect |
|--------|-------|------------|--------|
| `RESEARCH_COMPANY` | Targeting | target: str | Spend a step learning hiring state. Reduces uncertainty. |
| `ADD_TO_TARGETS` | Targeting | company: str | Commit company to active pipeline. Costs attention slot. |
| `REMOVE_TARGET` | Targeting | company: str | Drop a company. Frees capacity. |
| `WAIT_FOR_SIGNAL` | Targeting | None | Hold. Market updates. Risk: roles fill. |
| `ENGAGE_CONTENT` | Network | person: str | Like/comment on post. Low cost, +0.05 warmth. |
| `SEND_MESSAGE` | Network | person, message_type | Direct outreach. High warmth, high social capital cost. |
| `REQUEST_INTRO` | Network | from_person, to_person | Ask warm contact for introduction. Spends relationship equity. |
| `REQUEST_REFERRAL` | Network | person, company | Ask for referral. Fails if warmth < 0.6. Burns bridge if < 0.4. |
| `TAILOR_RESUME` | Application | company, role_type | Customize resume. Improves ATS pass rate. |
| `APPLY_COLD` | Application | company, role | Submit without referral. ~10% pass rate. |
| `APPLY_WITH_REFERRAL` | Application | company, role, referrer | Submit via referral. ~60% pass rate. |
| `ADVANCE_ROUND` | Pipeline | company | Schedule next interview. Moves process forward. |
| `REQUEST_DELAY` | Pipeline | company, days | Ask for more time. Costs recruiter goodwill. |
| `ACCELERATE_PROCESS` | Pipeline | company | Signal urgency. Risky if overused. |
| `DROP_PROCESS` | Pipeline | company | Abandon an active interview pipeline. |
| `COUNTER_OFFER` | Negotiation | company, amount, components | Name a number. >130% collapses the deal. Supports equity/signing bonus. |
| `ACCEPT_OFFER` | Negotiation | company | Terminal action. Episode ends. |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `step` | int | Current timestep (0 to max_steps) |
| `current_phase` | int | Which phase (1-5) |
| `target_companies` | list[dict] | hiring_signal (noisy), role_fit_score, culture_match, salary_band |
| `network_graph` | dict | Nodes=people with warmth_signal (noisy, 0-1), company affiliation |
| `social_capital` | float [0,1] | Finite resource depleted by outreach. Recovers +0.05/step. |
| `active_pipeline` | list[dict] | company, current_round, expiry_step, feedback |
| `offers_in_hand` | list[dict] | base_salary, equity, signing_bonus, deadline, negotiation_state |
| `profile_strength` | dict | Per-skill demand score, seniority level |
| `financial_runway` | float | Steps before must accept any offer |
| `market_signals` | dict | Hiring velocity, layoff flags, demand trend per company |
| `time_pressure` | float [0,1] | step / max_steps |
| `action_history` | list[dict] | Recent actions taken |

**Partial observability:** The agent sees noisy estimates of `hiring_signal`, `warmth`, `role_fit_score`, and `culture_match`. Ground truth is never exposed. `RESEARCH_COMPANY` reduces hiring signal noise by 60% toward truth.

## Tasks

### Task 1 — The Straightforward Pivot (Easy)
**Expected score: 0.70–0.85 | Max steps: 20**

Two warm connections at target companies. Market actively hiring. Ample financial runway. Tests whether the agent uses referrals before cold applying and runs parallel pipelines.

### Task 2 — The Cold Network Problem (Medium)
**Expected score: 0.40–0.60 | Max steps: 25**

Zero warm connections. One urgent company (fills in 8 steps). Social capital starts at 0.50. Tests whether the agent invests in network before applying — the non-obvious insight that 4 steps of network investment yields higher expected value than 4 cold applications.

### Task 3 — The Competing Pressures (Hard)
**Expected score: 0.15–0.35 | Max steps: 30**

One exploding offer at 70% market rate (deadline in 4 steps). Two mid-stage pipelines at better companies. A key contact went cold. Financial runway: only 5 steps. Requires simultaneously delaying the exploding offer, re-warming the contact, and accelerating pipelines.

## Reward Function

| Component | Weight | Formula |
|-----------|--------|---------|
| Offer quality | 35% | final_salary / market_rate + role_growth + culture_fit |
| Process efficiency | 20% | peak_parallel_pipelines / 3 (capped at 1.0) |
| Network health | 20% | avg_warmth_end / avg_warmth_start |
| Negotiation quality | 15% | final_comp / first_offer_comp |
| Time efficiency | 10% | max(0, 1 - steps/max_steps) |
| No offer penalty | — | -0.50 if episode ends without accepted offer |
| Bridges burned | — | -0.15 per bridge burned |
| Desperation signals | — | -0.10 per desperation signal (max 3) |

Dense intermediate rewards fire each step (+0.03 warmth threshold crossed, +0.05 referral obtained, +0.04 new round scheduled, -0.05 social capital overdraft, -0.03 process expiry). Intermediate rewards are capped at 0.30 total to prevent reward hacking.

## Baseline Scores

| Agent | Task 1 (Easy) | Task 2 (Medium) | Task 3 (Hard) |
|-------|---------------|-----------------|---------------|
| Random agent | 0.033 | 0.033 | 0.100 |
| LLM agent (Claude / GPT-4o) | *run `baseline/run_baseline.py`* | | |
| Optimal (theoretical) | 0.90–0.95 | 0.80–0.88 | 0.70–0.80 |

The theoretical optimal assumes perfect warmth management, 3 parallel pipelines, and negotiating every offer. LLM baselines require an API key — see **Setup** below.

## Trajectory Examples

### Task 1 — Optimal Trajectory (15 steps)

Alice Chen (TechCorp) and Carol Johnson (DataFlow) both start at warmth ≈ 0.65.

```
Step  0: ENGAGE_CONTENT → Alice Chen       warmth: 0.65 → 0.70  [approaching 0.60 threshold]
Step  1: ENGAGE_CONTENT → Carol Johnson    warmth: 0.65 → 0.70
Step  2: REQUEST_REFERRAL → Alice Chen, TechCorp   [SUCCESS: warmth 0.70 >= 0.60]
           state.granted_referrals = {"TechCorp": "Alice Chen"}
Step  3: REQUEST_REFERRAL → Carol Johnson, DataFlow [SUCCESS]
           state.granted_referrals = {"TechCorp": "Alice Chen", "DataFlow": "Carol Johnson"}
Step  4: APPLY_WITH_REFERRAL → TechCorp, role=SDE, referrer=Alice Chen   [pipeline added, 60% pass]
Step  5: APPLY_WITH_REFERRAL → DataFlow, role=SDE, referrer=Carol Johnson [pipeline added]
           peak_parallel_pipelines = 2
Step  6: ADVANCE_ROUND → TechCorp   [→ SCREENING, step_reward +0.04]
Step  7: ADVANCE_ROUND → DataFlow   [→ SCREENING]
Step  8: APPLY_COLD → AIStartup     [3rd pipeline, peak_parallel_pipelines = 3]
Step  9: ADVANCE_ROUND → TechCorp   [→ TECHNICAL]
Step 10: ADVANCE_ROUND → DataFlow   [→ TECHNICAL]
Step 11: ADVANCE_ROUND → DataFlow   [→ FINAL]
Step 12: ADVANCE_ROUND → DataFlow   [→ OFFER: $127,000. first_offer_received = this]
Step 13: COUNTER_OFFER → DataFlow, amount=$139,700 (110%)
           Company meets 50% of gap → $133,350
Step 14: ACCEPT_OFFER → DataFlow

Grade breakdown:
  Referral obtained AND used:                     +0.30
  peak_parallel_pipelines=3, 2+ at hiring cos:   +0.25
  $133,350 / market_rate ≈ 1.11 (ratio >= 0.90): +0.25
  Network warmth ratio >= 1.0:                   +0.10
  Total:                                          0.90
```

### Task 3 — Key Hinge Move

The single most important action in Task 3 is at step 0:

```
Step 0: REQUEST_DELAY → CloudBase, days=3
  CloudBase offer deadline: step 4 → step 7
  This buys 3 extra steps for TechCorp (TECHNICAL) and MegaSoft (FINAL)
  to advance to an offer before you must decide.

Without this move: CloudBase expires at step 4 and you have no offers.
With this move: MegaSoft reaches OFFER by step 3-4 at 50% probability per
  ADVANCE_ROUND, and TechCorp is one step behind.
```

---

## Design Rationale

**Why warmth is noisy.** If agents could see true warmth, the network phase collapses into a lookup table. Noisy signals force real exploration: the agent must invest in a relationship without certainty it will pay off, mirroring how referral outcomes work in practice.

**Why social capital is finite and regenerates.** Human attention and relationship equity are both limited and self-replenishing. The finite/regenerating design creates a natural pacing constraint — you cannot spam outreach — and makes resource allocation a real skill the agent must learn.

**Why the no-offer penalty is -0.50.** The primary goal of a job search is to get an offer. A beautiful networking strategy that produces no job is a failure. The heavy penalty ensures agents don't optimize subsidiary metrics at the expense of the terminal goal.

**Why intermediate rewards are capped at 0.30.** Without a cap, a reward-hacking agent could earn more from spamming warmth-building actions than from actually completing the search. The cap keeps intermediate rewards as a learning signal without letting them dominate the objective.

**Why the referral threshold is 0.60 not 0.50.** A threshold at 0.50 would make referral-seeking almost risk-free (agents rarely start below 0.50 in warm scenarios). At 0.60, there is a meaningful gap to close — enough that the agent must allocate steps to relationship building, which is the core skill being tested.

---

## RL Training Notes

**State space:** Approximately 200 dimensions after flattening — 6 companies × 4 company features, 12 contacts × 3 contact features, 5 pipeline entries × 3 pipeline features, 3 offer entries × 5 offer features, plus scalars (social capital, financial runway, time pressure).

**Key exploration challenge:** In the referral pathway (Task 1 and 2), the agent must take 2-4 warmth-building actions before a referral attempt makes sense. A random or ε-greedy policy almost never discovers this because the referral request fires too early (warmth < 0.40), burns the bridge, and produces a permanent negative outcome. Random baseline scores reflect this: Task 1 mean = 0.033 despite the task being labeled "easy."

**Curriculum recommendation:** Train on Task 1 → Task 2 → Task 3. Task 1 teaches the referral pathway with warm starting conditions. Task 2 teaches it under resource constraint. Task 3 adds multi-objective pressure on top.

**Action masking:** 7 of 17 actions are only valid when certain state conditions hold (pipeline-required actions, offer-required actions, phase gating). Masking invalid actions dramatically improves sample efficiency for discrete-action RL algorithms.

**Interview pass rates:**
- Cold application → screening: 10%
- Referral application → screening: 60%
- Screening → technical: 55%
- Technical → final: 45%
- Final → offer: 50%

Expected steps from referral application to offer: ~6-8 `ADVANCE_ROUND` actions assuming no rejections. Probability of reaching offer from referral application in one trajectory: ≈ 0.60 × 0.55 × 0.45 × 0.50 ≈ 7.4%.

---

## Setup

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run locally
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

### Run with Docker
```bash
docker build -t jobswitch-env .
docker run -p 8000:8000 jobswitch-env
```

### Run baseline
```bash
# Random agent (no API key needed)
python baseline/run_baseline.py

# Claude agent (requires Anthropic API key)
ANTHROPIC_API_KEY=sk-ant-... python baseline/run_baseline.py --agent claude

# GPT-4o agent (requires OpenAI API key)
OPENAI_API_KEY=sk-... python baseline/run_baseline.py --agent gpt4o

# Specific task with verbose step logging
ANTHROPIC_API_KEY=sk-ant-... python baseline/run_baseline.py --agent claude --task 1 --runs 5 --verbose
```

### Verify
```bash
curl http://localhost:8000/health
```

---

*Meta x Hugging Face OpenEnv Hackathon 2026*
