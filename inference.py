"""
inference.py — Baseline LLM agent for the Adaptive Study Partner environment.

Usage:
    python inference.py

Required environment variables:
    API_BASE_URL   LLM endpoint  e.g. https://router.huggingface.co/v1
    MODEL_NAME     Model id      e.g. meta-llama/Llama-3.3-70B-Instruct
    HF_TOKEN       HF API key
    ENV_URL        Environment server (default: http://localhost:8000)
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

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:8000")

TEMPERATURE  = 0.2
MAX_TOKENS   = 300
MAX_RETRIES  = 3

if not HF_TOKEN:
    print("ERROR: HF_TOKEN environment variable not set")
    sys.exit(1)

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

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
- score 0.3–0.6: Use "problem" to reinforce understanding
- score > 0.6 but still weak: Use "mock_test" to solidify

Urgency guidance:
- days_remaining < 3: Almost everything is "today"
- days_remaining 3–7: Mix "today" and "this_week"
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

            # Strip markdown code fences if model adds them
            raw = raw.replace("```json", "").replace("```", "").strip()

            action = json.loads(raw)

            # Validate required fields exist
            required = ["recommended_topic", "assigned_slot", "resource_type", "urgency"]
            for field in required:
                if field not in action:
                    raise ValueError(f"Missing field: {field}")

            # Validate values are legal
            if action["recommended_topic"] not in obs["weak_topics"]:
                print(f"  [warn] LLM chose topic not in weak_topics, correcting...")
                action["recommended_topic"] = obs["weak_topics"][0]

            if action["assigned_slot"] not in obs["available_slots"]:
                print(f"  [warn] LLM chose invalid slot, correcting...")
                action["assigned_slot"] = obs["available_slots"][0]

            if action["resource_type"] not in ("problem", "video", "notes", "mock_test"):
                action["resource_type"] = "problem"

            if action["urgency"] not in ("today", "this_week", "optional"):
                action["urgency"] = "today"

            return action

        except json.JSONDecodeError as e:
            print(f"  [attempt {attempt+1}] JSON parse error: {e}")
            if attempt == MAX_RETRIES - 1:
                return get_fallback_action(obs)
            time.sleep(1)

        except Exception as e:
            print(f"  [attempt {attempt+1}] LLM error: {e}")
            if attempt == MAX_RETRIES - 1:
                return get_fallback_action(obs)
            time.sleep(2)

    return get_fallback_action(obs)


def get_fallback_action(obs: dict) -> dict:
    """Safe fallback if LLM fails — picks lowest score topic in first slot."""
    weak = obs.get("weak_topics", [])
    slots = obs.get("available_slots", [])
    scores = obs.get("topic_scores", {})

    # Pick the weakest topic
    topic = min(weak, key=lambda t: scores.get(t, 0.0)) if weak else "Arrays"
    slot  = slots[0] if slots else "Mon 9am"
    score = scores.get(topic, 0.5)

    resource = "video" if score < 0.3 else "problem"
    urgency  = "today" if obs.get("days_remaining", 7) < 3 else "this_week"

    print(f"  [fallback] using: {topic} / {slot} / {resource} / {urgency}")
    return {
        "recommended_topic": topic,
        "assigned_slot": slot,
        "resource_type": resource,
        "urgency": urgency,
    }


# ─────────────────────────────────────────────
# Environment interaction
# ─────────────────────────────────────────────

def call_env(method: str, path: str, **kwargs) -> dict:
    """Make an HTTP call to the environment server."""
    url = f"{ENV_URL}{path}"
    try:
        if method == "GET":
            r = httpx.get(url, timeout=30)
        elif method == "POST":
            r = httpx.post(url, timeout=30, **kwargs)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        print(f"  [env error] {e.response.status_code}: {e.response.text}")
        raise
    except httpx.ConnectError:
        print(f"  [env error] Cannot connect to {ENV_URL}. Is the server running?")
        raise


def run_task(task_id: str) -> float:
    """Run one full episode for a task. Returns final score."""
    print(f"\n{'='*55}")
    print(f"  TASK: {task_id}")
    print(f"{'='*55}")

    obs = call_env("POST", f"/reset?task_id={task_id}")
    print(f"  Student: {obs['student_id']} | Personality: {obs['personality']}")
    print(f"  Weak topics: {obs['weak_topics']}")
    print(f"  Days remaining: {obs['days_remaining']}")

    episode_rewards = []

    while not obs.get("done", False) and obs.get("available_slots"):
        step_num = obs["step"] + 1
        print(f"\n  Step {step_num}/{obs['max_steps']}")

        action = get_agent_action(obs)
        print(f"  → {action['recommended_topic']} | {action['assigned_slot']} | {action['resource_type']} | {action['urgency']}")

        result = call_env("POST", "/step", json=action)
        obs    = result["observation"]
        reward = result["reward"]
        info   = result.get("info", {})

        episode_rewards.append(reward)
        print(f"  ← reward: {reward:.3f} | {obs.get('last_action_feedback', '')}")

        breakdown = info.get("reward_breakdown", {})
        if breakdown:
            print(f"     topic={breakdown.get('topic_match', 0):.2f} "
                  f"diff={breakdown.get('difficulty_match', 0):.2f} "
                  f"slot={breakdown.get('slot_valid', 0):.2f} "
                  f"pace={breakdown.get('pacing_score', 0):.2f}")

    final_score = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0
    print(f"\n  Final score: {final_score:.4f}")
    return final_score


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print(f"\nAdaptive Study Partner — Baseline Agent")
    print(f"Model:  {MODEL_NAME}")
    print(f"Server: {ENV_URL}")

    # Quick health check
    try:
        health = call_env("GET", "/health")
        print(f"Server: {health}")
    except Exception:
        print("ERROR: Cannot reach environment server.")
        print(f"Make sure it is running at {ENV_URL}")
        sys.exit(1)

    scores = {}
    for task_id in ["task_1", "task_2", "task_3"]:
        try:
            scores[task_id] = run_task(task_id)
        except Exception as e:
            print(f"\nERROR running {task_id}: {e}")
            scores[task_id] = 0.0

    # Summary
    print(f"\n{'='*55}")
    print("  BASELINE SCORES")
    print(f"{'='*55}")
    for task_id, score in scores.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {task_id}: {score:.4f}  {bar}")
    avg = sum(scores.values()) / len(scores)
    print(f"\n  Average: {avg:.4f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()