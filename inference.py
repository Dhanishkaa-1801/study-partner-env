"""
inference.py — Baseline LLM agent for the Adaptive Study Partner environment.

Usage:
    python inference.py

Required environment variables:
    API_BASE_URL   LLM endpoint  e.g. https://router.huggingface.co/v1
    MODEL_NAME     Model id      e.g. MiniMaxAI/MiniMax-M2.1
    API_KEY       Your Hugging Face / API key
    ENV_URL = os.getenv("ENV_URL", "https://dhani1801-study-partner-env.hf.space")
"""

import os
import json
import time
import sys
from openai import OpenAI
from dotenv import load_dotenv
import httpx

load_dotenv()

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────


TEMPERATURE  = 0.2
MAX_TOKENS   = 300
MAX_RETRIES  = 3

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "MiniMaxAI/MiniMax-M2.1")
API_KEY      = os.environ.get("API_KEY")
ENV_URL = os.environ.get(
    "ENV_URL",
    "https://dhani1801-study-partner-env.hf.space"
)

if not API_KEY:
    print("WARNING: No API key found, using fallback policy")
    client = None
else:
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception:
        client = None

def wait_for_env():
    for _ in range(30):  # ~60 sec
        try:
            r = httpx.get(f"{ENV_URL}/health", timeout=5)
            if r.status_code == 200:
                return
        except:
            pass
        time.sleep(2)

    print("WARNING: Env not ready, continuing anyway...", flush=True)

# ─────────────────────────────────────────────
# Structured logging — mandatory format
# ─────────────────────────────────────────────

def log_start(task_id: str):
    print(f"[START] task={task_id} env=adaptive-study-partner model={MODEL_NAME}", flush=True)


def log_step(step: int, action: dict, reward: float, done: bool, error=None):
    action_str = json.dumps(action, separators=(",", ":"))
    error_val = error if error else "null"

    print(
    f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_val}",
    flush=True,
    )


def log_end(task_id: str, score: float, total_steps: int, rewards: list):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success = "true" if score >= 0.4 else "false"
    print(
    f"[END] success={success} steps={total_steps} score={score:.2f} rewards={rewards_str}",
    flush=True,
    )

# ─────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert study planning agent for a student preparing for technical exams and placement interviews.

Given a student's current learning state, your job is to recommend:
1. Which topic to study next (from their weak topics)
2. Which time slot to use (from their available slots)
3. What type of study resource to use
4. How urgently they should do it

Personality guidance:
- procrastinator: Push hard topics first. Use "today" urgency. Avoid "optional".
- anxious: Build confidence. Start with easier topics. Mix practice problems with notes.
- consistent: Optimize for coverage. Balance all weak topics evenly.

Resource guidance:
- score < 0.3: Use "video" or "notes" to build foundation first
- score 0.3-0.6: Use "problem" to reinforce understanding
- score > 0.6 but still weak: Use "mock_test" to solidify

Urgency guidance:
- days_remaining < 3: Almost everything is "today"
- days_remaining 3-7: Mix "today" and "this_week"
- days_remaining > 7: Use "this_week" and "optional" strategically

CRITICAL: You must respond with ONLY a valid JSON object. No explanation. No markdown. No extra text.
Exact format:
{
  "recommended_topic": "<topic from weak_topics list>",
  "assigned_slot": "<slot from available_slots list>",
  "resource_type": "<problem|video|notes|mock_test>",
  "urgency": "<today|this_week|optional>"
}"""


def build_user_prompt(obs: dict) -> str:
    disruption_text = ""
    if obs.get("disruption"):
        d = obs["disruption"]
        disruption_text = f"\nDISRUPTION: {d.get('description', '')} (affected slots: {d.get('affected_slots', [])})"

    return f"""Student state:
- Personality: {obs['personality']}
- Weak topics: {obs['weak_topics']}
- Topic scores: {obs['topic_scores']}
- Retention scores: {obs.get('retention_scores', {})}
- Available slots: {obs['available_slots']}
- Days remaining: {obs['days_remaining']}
- Exam date: {obs['exam_date']}
- Step: {obs['step']} of {obs['max_steps']}
- Last feedback: {obs.get('last_action_feedback', 'none')}{disruption_text}

Recommend the best study action right now."""


# ─────────────────────────────────────────────
# Agent logic
# ─────────────────────────────────────────────

def get_agent_action(obs: dict) -> dict:
    """Call the LLM and parse its action. Returns a valid action dict."""

    if client is None:
        return get_fallback_action(obs)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_user_prompt(obs)},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )

            raw = response.choices[0].message.content.strip()

            raw = raw.replace("```json", "").replace("```", "").strip()

            action = json.loads(raw)

            required = ["recommended_topic", "assigned_slot", "resource_type", "urgency"]
            for field in required:
                if field not in action:
                    raise ValueError(f"Missing field: {field}")

            if action["recommended_topic"] not in obs["weak_topics"]:
                action["recommended_topic"] = obs["weak_topics"][0]

            if action["assigned_slot"] not in obs["available_slots"]:
                action["assigned_slot"] = obs["available_slots"][0]

            if action["resource_type"] not in ("problem", "video", "notes", "mock_test"):
                action["resource_type"] = "problem"

            if action["urgency"] not in ("today", "this_week", "optional"):
                action["urgency"] = "today"

            return action

        except json.JSONDecodeError as e:
            if attempt == MAX_RETRIES - 1:
                return get_fallback_action(obs)
            time.sleep(1)

        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return get_fallback_action(obs)
            time.sleep(2)

    return get_fallback_action(obs)


def get_fallback_action(obs: dict) -> dict:
    """Safe fallback if LLM fails — picks lowest score topic in first slot."""
    weak   = obs.get("weak_topics", [])
    slots  = obs.get("available_slots", [])
    scores = obs.get("topic_scores", {})

    topic    = min(weak, key=lambda t: scores.get(t, 0.0)) if weak else "Arrays"
    slot     = slots[0] if slots else "Mon 9am"
    score    = scores.get(topic, 0.5)
    resource = "video" if score < 0.3 else "problem"
    urgency  = "today" if obs.get("days_remaining", 7) < 3 else "this_week"

    return {
        "recommended_topic": topic,
        "assigned_slot":     slot,
        "resource_type":     resource,
        "urgency":           urgency,
    }


# ─────────────────────────────────────────────
# Environment interaction
# ─────────────────────────────────────────────

def call_env(method: str, path: str, **kwargs) -> dict:
    url = f"{ENV_URL}{path}"
    try:
        if method == "GET":
            r = httpx.get(url, timeout=30)
        elif method == "POST":
            r = httpx.post(url, timeout=30, **kwargs)

        r.raise_for_status()
        return r.json()

    except Exception as e:
        print(f"ENV CALL FAILED: {e}", flush=True)
        raise


def run_task(task_id: str) -> float:
    """Run one full episode for a task. Returns final score."""

    try:
        obs = call_env("POST", f"/reset?task_id={task_id}")
    except Exception as e:      
        log_end(task_id, 0.0, 0, [])
        return 0.0
    
    if not obs:
        log_end(task_id, 0.0, 0, [])
        return 0.0

    log_start(task_id)

    episode_rewards = []
    step_num        = 0

    while isinstance(obs, dict) and not obs.get("done", False) and obs.get("available_slots"):
        step_num += 1

        if not obs:
            break

        action = get_agent_action(obs)

        try:
            result = call_env("POST", "/step", json=action)
        except Exception as e:
            log_step(step_num, action, 0.0, True, str(e))
            break

        if not result:
            log_step(step_num, action, 0.0, True, "empty_response")
            break

        obs    = result.get("observation", {})
        reward = result.get("reward", 0.0)
        done   = result.get("done", True)

        episode_rewards.append(reward)

        error = result.get("info", {}).get("error")
        log_step(step_num, action, reward, done, error)

    final_score = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0
    final_score = max(0.0, min(final_score, 1.0))

    log_end(task_id, final_score, step_num, episode_rewards)

    return final_score


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    try:
        wait_for_env()

        for task_id in ["task_1", "task_2", "task_3"]:
            try:
                run_task(task_id)
            except Exception:
                log_end(task_id, 0.0, 0, [])

    except Exception as e:
        print(f"FATAL ERROR: {e}", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"UNCAUGHT ERROR: {e}", flush=True)