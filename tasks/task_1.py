"""
tasks/task_1.py — Easy Task Grader
Adaptive Study Partner | OpenEnv Hackathon

Scenario:
    Student: Priya (anxious personality)
    Weak focus: Trees (score 0.3), Dynamic Programming (0.2), Graphs (0.2), System Design (0.1)
    Days: 3
    Slots: 4 available (Mon–Thu 10am)
    Critical topic: Trees

What the agent must do:
    - Target Trees (the declared critical topic)
    - Use confidence-building resources first (notes/video) — Priya is anxious
    - Schedule in a valid slot
    - Not overload any single day
    - Not waste slots on already-strong topics (Arrays 0.8, Linked Lists 0.7)

Grader axes (all 0.0–1.0):
    1. critical_topic_coverage  (weight 0.50) — was Trees scheduled and improved?
    2. personality_fit          (weight 0.30) — did agent use notes/video before problems?
    3. no_wasted_slots          (weight 0.20) — did agent avoid already-mastered topics?

Final score = weighted average of axes.
Pass threshold: 0.60
"""

from typing import Dict, List
from models import GradeResult

TASK_ID         = "task_1"
PASS_THRESHOLD  = 0.60
CRITICAL_TOPICS = ["Trees"]
MASTERY_THRESHOLD = 0.6


def grade(
    topic_scores     : Dict[str, float],
    retention_scores : Dict[str, float],
    scheduled        : Dict[str, str],    # slot → topic
    history          : List[str],
    available_slots  : List[str],
    days_remaining   : int,
    episode_rewards  : List[float],
    critical_topics  : List[str],
) -> GradeResult:
    """
    Grade a completed Task 1 episode.

    Args:
        topic_scores:     final topic mastery scores
        retention_scores: final retention scores after forgetting curve
        scheduled:        dict of slot → topic (what was actually scheduled)
        history:          ordered list of recommended topics across all steps
        available_slots:  slots still unused at episode end
        days_remaining:   days left when episode ended
        episode_rewards:  per-step reward values
        critical_topics:  from task config (should be ["Trees"])

    Returns:
        GradeResult with per-axis breakdown and final score.
    """

    scheduled_topics = list(scheduled.values())

    # ── Axis 1: Critical topic coverage (weight 0.50) ────────────────────
    # Did the agent actually schedule Trees AND did its score improve?
    trees_scheduled = "Trees" in scheduled_topics
    trees_score_end = topic_scores.get("Trees", 0.0)
    trees_improved  = trees_score_end > 0.3   # started at 0.3

    if trees_scheduled and trees_improved:
        critical_topic_coverage = 1.0
    elif trees_scheduled and not trees_improved:
        critical_topic_coverage = 0.6   # scheduled but no improvement (odd)
    elif not trees_scheduled and trees_score_end >= MASTERY_THRESHOLD:
        critical_topic_coverage = 0.4   # somehow mastered without scheduling
    else:
        critical_topic_coverage = 0.0   # completely missed

    # ── Axis 2: Personality fit — anxious needs notes/video first (weight 0.30) ─
    # Check history: did agent ever recommend Trees with notes or video?
    # We infer this from episode_rewards — high rewards imply good resource choice.
    # Direct resource tracking would need environment internals, so we use
    # avg reward as a proxy (environment already penalises wrong resource type).
    avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0

    if avg_reward >= 0.70:
        personality_fit = 1.0
    elif avg_reward >= 0.50:
        personality_fit = 0.6
    elif avg_reward >= 0.30:
        personality_fit = 0.3
    else:
        personality_fit = 0.0

    # ── Axis 3: No wasted slots (weight 0.20) ────────────────────────────
    # Did agent recommend already-strong topics (Arrays ≥ 0.8, Linked Lists ≥ 0.7)?
    # Starting scores: Arrays=0.8, Linked Lists=0.7, DBMS=0.6, OS=0.5
    already_strong = {"Arrays", "Linked Lists", "DBMS"}
    wasted = [t for t in scheduled_topics if t in already_strong]

    if len(wasted) == 0:
        no_wasted_slots = 1.0
    elif len(wasted) == 1:
        no_wasted_slots = 0.5
    else:
        no_wasted_slots = 0.0

    # ── Weighted final score ─────────────────────────────────────────────
    axes = {
        "critical_topic_coverage": round(critical_topic_coverage, 4),
        "personality_fit"        : round(personality_fit, 4),
        "no_wasted_slots"        : round(no_wasted_slots, 4),
    }
    weights = {
        "critical_topic_coverage": 0.50,
        "personality_fit"        : 0.30,
        "no_wasted_slots"        : 0.20,
    }
    final_score = round(
        sum(axes[k] * weights[k] for k in axes), 4
    )

    feedback_parts = []
    if critical_topic_coverage < 1.0:
        feedback_parts.append("Trees was not properly scheduled or improved.")
    if personality_fit < 0.6:
        feedback_parts.append("Agent did not use confidence-building resources for anxious student.")
    if no_wasted_slots < 1.0:
        feedback_parts.append(f"Agent wasted slots on already-mastered topics: {wasted}.")
    if not feedback_parts:
        feedback_parts.append("Great job — Trees covered, personality respected, no wasted slots.")

    return GradeResult(
        task_id     = TASK_ID,
        score       = final_score,
        passed      = final_score >= PASS_THRESHOLD,
        breakdown   = axes,
        feedback    = " ".join(feedback_parts),
    )