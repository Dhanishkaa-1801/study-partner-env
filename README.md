<<<<<<< HEAD
---
title: Adaptive Study Partner
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

#triggering

**This project was developed for the Build at the Bleeding Edge of AI [Scaler x Meta x HuggingFace x PyTorch OpenEnv Hackathon 2026]**

## 🎯 Hackathon Objective
The goal of this submission is to demonstrate a high-fidelity **Reinforcement Learning (RL) Environment** using the `OpenEnv` framework. Our environment simulates a complex, human-centric challenge: **Personalized Educational Planning.**

> Designed for real-world adaptive learning systems and LLM-based planning agents.

# 📚 Adaptive Study Partner — OpenEnv Environment

> An RL environment where an agent acts as a personal study partner —
> recommending what to study, when to study it, and adapting the plan
> in real time based on student personality, topic memory decay, and
> mid-episode disruptions.

---

## 🧠 What It Does

The Adaptive Study Partner simulates a student preparing for an exam.
At each step, the agent observes the student's current state — weak topics,
available time slots, days until the exam, and how well they remember
previously studied material — and must decide:

- **What** to study next (which weak topic to target)
- **When** to study it (which available slot to assign)
- **How** to study it (problem / video / notes / mock test)
- **How urgently** (today / this week / optional)

The environment rewards smart prioritization and penalizes wasted time,
overloading, and ignoring the forgetting curve. Three student personalities
(Procrastinator, Anxious, Consistent) require the agent to adapt its strategy
per student, not just per topic.

---

## 🌟 What Makes It Novel

| Feature | Description |
|---|---|
| **Forgetting Curve** | Retention scores decay 10% every step for unstudied topics (Ebbinghaus model). The agent must schedule revision sessions, not just first-time study. |
| **Learning Personalities** | Three distinct student types — each requires a different scheduling strategy from the same agent. |
| **Dynamic Disruptions** | Mid-episode events (missed sessions, surprise mock tests, topic mastery updates) force the agent to adapt in real time. |
| **Partial Reward Signals** | 8 fine-grained reward components give the agent rich feedback at every step rather than a sparse end-of-episode signal. |
| **Two-domain coverage** | Academics (exam prep) and Placements (DSA + System Design) unified in one environment. |

---

## 📐 Observation Space

What the agent sees at every step (`StudyObservation`):

| Field | Type | Description |
|---|---|---|
| `task_id` | string | Which task is running (`task_1` / `task_2` / `task_3`) |
| `step` | int | Current step number |
| `max_steps` | int | Maximum steps before episode ends |
| `done` | bool | Whether the episode has ended |
| `student_id` | string | Identifier for the student profile |
| `personality` | string | `procrastinator` / `anxious` / `consistent` |
| `weak_topics` | list[string] | Topics with mastery score below 0.6 |
| `topic_scores` | dict[string, float] | Current mastery per topic (0.0–1.0) |
| `retention_scores` | dict[string, float] | Forgetting-curve retention per topic (0.0–1.0) |
| `available_slots` | list[string] | Unfilled time slots (e.g. `"Mon 9am"`) |
| `exam_date` | string | ISO date of the exam |
| `days_remaining` | int | Days left until the exam |
| `last_action_feedback` | string | Human-readable feedback from the previous step |
| `last_reward` | float | Reward received at the previous step |
| `disruption` | object \| null | Mid-episode event, if any fired this step |

---

## 🎮 Action Space

What the agent outputs each step (`StudyAction`):

| Field | Type | Allowed Values | Description |
|---|---|---|---|
| `recommended_topic` | string | Must be in `weak_topics` | Topic to study this session |
| `assigned_slot` | string | Must be in `available_slots` | Time slot to assign |
| `resource_type` | string | `problem` / `video` / `notes` / `mock_test` | How to study the topic |
| `urgency` | string | `today` / `this_week` / `optional` | How urgently to treat this session |

Invalid actions (unknown topic or busy slot) return `reward = 0.0` and a
`-0.1` penalty in `info` — the environment never crashes.

---

## 📋 Tasks

### Task 1 — Easy: Single Topic Recommendation
```
Student:   Priya (anxious) | 3 days to exam | 4 available slots
Focus:     Trees (score 0.3) — the declared critical topic
Challenge: Agent must use confidence-building resources (notes/video)
           before assigning practice problems — Priya needs to build
           confidence before tackling hard content.
Grader axes:
  - critical_topic_coverage  (0.50) — Trees scheduled and score improved
  - personality_fit          (0.30) — notes/video used appropriately
  - no_wasted_slots          (0.20) — avoided already-mastered topics
Pass threshold: 0.60
```

### Task 2 — Medium: Weekly Study Plan
```
Student:   Arjun (procrastinator) | 7 days | 9 slots
Focus:     4 critical topics: Arrays, Trees, Dynamic Programming, System Design
Disruption at step 5: "missed_session" — Wed 11am slot removed
Challenge: Cover all 4 critical topics with no day overloading,
           correct difficulty progression, and recovery from the
           missed session. Procrastinator needs urgency="today" signals.
Grader axes:
  - critical_coverage      (0.40) — all 4 critical topics scheduled
  - no_overload            (0.25) — no day has more than 2 sessions
  - difficulty_progression (0.20) — easier topics scheduled before harder ones
  - disruption_recovery    (0.15) — plan viable after missed session
Pass threshold: 0.55
```

### Task 3 — Hard: Exam Crunch Triage
```
Student:   Kiran (consistent) | 14 days | 12 slots
Focus:     3 critical topics: Dynamic Programming, Graphs, System Design
           (Arrays and Linked Lists already strong — must be deprioritized)
Disruptions:
  Step 3:  mock_test_added  — Tue 9am blocked
  Step 8:  topic_mastered   — Trees no longer weak
  Step 12: missed_session   — Thu 9am removed
Challenge: Protect all 3 critical topics across 3 disruptions, keep
           retention scores above 0.5, and avoid over-scheduling
           already-mastered topics.
Grader axes:
  - critical_topic_protection (0.35) — DP, Graphs, System Design covered + improved
  - correct_deprioritization  (0.25) — Arrays/Linked Lists not over-represented
  - schedule_feasibility      (0.20) — no overloading, no blocked slots used
  - retention_health          (0.20) — critical topics retain ≥ 0.5 at episode end
Pass threshold: 0.45
```

---

## 🏆 Reward Function

Each step returns a detailed `RewardBreakdown` in `StepResult.info`:

| Signal | Value | Condition |
|---|---|---|
| `topic_match` | +0.40 | Recommended topic is genuinely weak (score < 0.6) |
| `difficulty_match` | +0.10 to +0.30 | Resource type fits the student's current skill level |
| `slot_valid` | +0.20 | Slot is free and before the exam date |
| `pacing_score` | +0.10 | Coverage % is on track relative to days remaining |
| `personality_bonus` | +0.10 | Action matches the student's personality type |
| `mastered_penalty` | -0.20 | Topic is already mastered (score ≥ 0.8) |
| `overload_penalty` | -0.20 | More than 2 sessions booked on the same day |
| `repetition_penalty` | -0.10 | Same topic recommended twice in a row |
| `retention_penalty` | -0.10 | Topic retention is below 0.5 (should have been revised earlier) |

Reward is clipped to `[0.0, 1.0]`.

---

## 🔁 Forgetting Curve

The environment implements the **Ebbinghaus forgetting curve**:

- Every `step()` call simulates one study day
- All topics **not** studied that step lose **10% retention**
- The topic that **was** studied gains **+0.15 retention** (capped at 1.0)
- `retention_scores` are separate from `topic_scores` — a topic can be
  "mastered" (high score) but forgotten (low retention) if not revised

This means the agent must balance learning new topics vs. revising old ones.

---

## ⚡ Dynamic Disruptions

Mid-episode events are fired at specific steps per task:

| Type | Effect |
|---|---|
| `missed_session` | Removes one or more slots from `available_slots` |
| `mock_test_added` | Blocks a slot (reserved for mock test) |
| `topic_mastered` | Boosts a topic's score to ≥ 0.6 (no longer weak) |

The agent sees the disruption in `observation.disruption` and must adapt
its next action accordingly.

---

## 🏗️ Architecture

```
study-partner-env/
├── environment.py      ← StudyEnv: reset(), step(), state(), grade()
├── models.py           ← Shared Pydantic contract (observation, action, reward)
├── server/app.py       ← FastAPI wrapper (OpenEnv endpoints)
├── inference.py        ← Baseline LLM agent
├── tasks/
│   ├── task_1.py       ← Easy grader
│   ├── task_2.py       ← Medium grader
│   └── task_3.py       ← Hard grader
├── data/
│   ├── students.json   ← 3 synthetic student profiles
│   ├── problems.json   ← 28 problems across 8 topics
│   └── resources.json  ← 30 curated resources (GFG, LeetCode, Striver, etc.)
├── Dockerfile
├── openenv.yaml
└── requirements.txt
```

---

## 🚀 Setup

### Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/study-partner-env
cd study-partner-env

# Install dependencies
pip install -r requirements.txt

# Start the environment server
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run with Docker

```bash
docker build -t study-partner-env .

docker run -p 8000:8000 \
  -e API_BASE_URL=https://api.groq.com/openai/v1 \
  -e MODEL_NAME=llama-3.1-8b-instant \
  -e GROQ_API_KEY=your_token_here \
  study-partner-env
```

### Run the Baseline Agent

```bash
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.1-8b-instant
export GROQ_API_KEY=your_api_key
export ENV_URL=https://dhani1801-study-partner-env.hf.space

python inference.py
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/reset?task_id=task_1` | Start a new episode |
| POST | `/step` | Submit an action, get next observation |
| GET | `/state` | Get current observation without advancing |
| GET | `/tasks` | List all available tasks |

---

## 📊 Baseline Scores

Evaluated using `llama-3.1-8b-instant` via Groq (Compatible API)

> The agent is provider-agnostic and supports any OpenAI-compatible API 
> (e.g., HuggingFace Router, Groq) via environment variables.

| Task | Difficulty | Score | Passed |
|---|---|---|---|
| task_1 | Easy | **0.8667** | ✅ (threshold: 0.60) |
| task_2 | Medium | **0.8** | ✅ (threshold: 0.55) |
| task_3 | Hard | **0.9273** | ✅ (threshold: 0.45) |
| **Average** | | **0.8646** | ✅ |

---

## 👥 Team

| Name | Role | GitHub |
|---|---|---|
| Dhanishkaa D | Agent & Infra  | https://github.com/Dhanishkaa-1801 |
| Ilamadhi J | Environment | https://github.com/finac-ilamadhi |

HuggingFace Space: (https://huggingface.co/spaces/dhani1801/study-partner-env)

---

## 🏷️ Tags

`openenv` `reinforcement-learning` `education` `adaptive-learning` `spaced-repetition` `llm-agent`




=======
---
title: Study Partner Env
emoji: 🦀
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> 41162b46b5a8dba6ad063ab447498a4ec47a5613
