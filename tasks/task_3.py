"""
tasks/task_3.py — Hard Task Grader
Adaptive Study Partner | OpenEnv Hackathon

Scenario:
    Student: Kiran (consistent personality)
    All topics weak except Arrays (0.9) and Linked Lists (0.8)
    Days: 14, but 3 disruptions will eat slots and change priorities
    Critical topics: Dynamic Programming, Graphs, System Design
    Disruptions:
        Step 3:  mock_test_added   → "Tue 9am" blocked
        Step 8:  topic_mastered    → Trees bumped up (no longer weak)
        Step 12: missed_session    → "Thu 9am" removed

What the agent must do:
    - Protect critical topics (DP, Graphs, System Design) above all else
    - Deprioritize already-strong topics (Arrays, Linked Lists)
    - Maintain schedule feasibility through 3 disruptions
    - Keep retention scores above 0.5 for critical topics (forgetting curve)
    - Not overload any single day

Grader axes (all 0.0–1.0):
    1. critical_topic_protection  (weight 0.35) — all 3 critical topics scheduled + scores improved
    2. correct_deprioritization   (weight 0.25) — strong topics NOT overrepresented in schedule
    3. schedule_feasibility       (weight 0.20) — no overload + no invalid slots used
    4. retention_health           (weight 0.20) — critical topics retain ≥ 0.5 at episode end

Final score = weighted average of axes.
Pass threshold: 0.45  (hard task — lower bar is intentional)
"""

from typing import Dict, List
from models import GradeResult

TASK_ID          = "task_3"
PASS_THRESHOLD   = 0.45
MASTERY_THRESHOLD = 0.6
RETENTION_FLOOR  = 0.5

KIRAN_INITIAL_SCORES = {
    "Arrays"             : 0.9,
    "Linked Lists"       : 0.8,
    "Trees"              : 0.5,
    "Dynamic Programming": 0.3,
    "Graphs"             : 0.3,
    "OS"                 : 0.4,
    "DBMS"               : 0.5,
    "System Design"      : 0.2,
}

STRONG_TOPICS   = {"Arrays", "Linked Lists"}      # should be deprioritized
CRITICAL_TOPICS = {"Dynamic Programming", "Graphs", "System Design"}


def grade(
    topic_scores     : Dict[str, float],
    retention_scores : Dict[str, float],
    scheduled        : Dict[str, str],
    history          : List[str],
    available_slots  : List[str],
    days_remaining   : int,
    episode_rewards  : List[float],
    critical_topics  : List[str],
) -> GradeResult:
    """
    Grade a completed Task 3 episode.
    """

    scheduled_topics = list(scheduled.values())
    total_scheduled  = len(scheduled_topics)

    # ── Axis 1: Critical topic protection (weight 0.35) ──────────────────
    # All 3 critical topics must be scheduled AND their scores must improve
    # from starting values.
    critical_hits = 0
    for topic in CRITICAL_TOPICS:
        scheduled_flag = topic in scheduled_topics
        improved_flag  = topic_scores.get(topic, 0.0) > KIRAN_INITIAL_SCORES.get(topic, 0.0)
        if scheduled_flag and improved_flag:
            critical_hits += 1
        elif scheduled_flag:
            critical_hits += 0.5   # partial credit: scheduled but no improvement

    critical_topic_protection = critical_hits / len(CRITICAL_TOPICS)

    # ── Axis 2: Correct deprioritization (weight 0.25) ───────────────────
    # Strong topics (Arrays, Linked Lists) should take up ≤ 20% of schedule.
    # Topic_mastered disruption at step 8 means Trees should also be
    # deprioritized from that point — we check Trees is not over-recommended.
    if total_scheduled == 0:
        correct_deprioritization = 0.0
    else:
        strong_count = sum(1 for t in scheduled_topics if t in STRONG_TOPICS)
        strong_ratio = strong_count / total_scheduled

        # Trees score at end should be ≥ 0.6 (mastered via disruption at step 8)
        trees_over_studied = scheduled_topics.count("Trees") > 2

        if strong_ratio <= 0.20 and not trees_over_studied:
            correct_deprioritization = 1.0
        elif strong_ratio <= 0.30 and not trees_over_studied:
            correct_deprioritization = 0.7
        elif strong_ratio <= 0.40:
            correct_deprioritization = 0.4
        else:
            correct_deprioritization = 0.0

    # ── Axis 3: Schedule feasibility (weight 0.20) ───────────────────────
    # No day should have > 2 sessions.
    # Disrupted slots (Tue 9am, Thu 9am) should NOT appear in schedule.
    DISRUPTED_SLOTS = {"Tue 9am", "Thu 9am"}

    day_counts: Dict[str, int] = {}
    for slot in scheduled:
        day = slot.split(" ")[0]
        day_counts[day] = day_counts.get(day, 0) + 1

    overloaded_days   = [d for d, c in day_counts.items() if c > 2]
    used_blocked_slots = [s for s in scheduled if s in DISRUPTED_SLOTS]

    if len(overloaded_days) == 0 and len(used_blocked_slots) == 0:
        schedule_feasibility = 1.0
    elif len(overloaded_days) == 0 and len(used_blocked_slots) == 1:
        schedule_feasibility = 0.6   # missed one disruption
    elif len(overloaded_days) == 1 and len(used_blocked_slots) == 0:
        schedule_feasibility = 0.5
    else:
        schedule_feasibility = max(
            0.0,
            1.0 - (len(overloaded_days) * 0.2) - (len(used_blocked_slots) * 0.3)
        )

    # ── Axis 4: Retention health (weight 0.20) ────────────────────────────
    # All 3 critical topics should have retention ≥ 0.5 at episode end.
    # This tests whether the agent revised topics over time vs studied once
    # and forgot (forgetting curve).
    healthy_retention = 0
    for topic in CRITICAL_TOPICS:
        if retention_scores.get(topic, 0.0) >= RETENTION_FLOOR:
            healthy_retention += 1

    retention_health = healthy_retention / len(CRITICAL_TOPICS)

    # ── Weighted final score ─────────────────────────────────────────────
    axes = {
        "critical_topic_protection" : round(critical_topic_protection, 4),
        "correct_deprioritization"  : round(correct_deprioritization, 4),
        "schedule_feasibility"      : round(schedule_feasibility, 4),
        "retention_health"          : round(retention_health, 4),
    }
    weights = {
        "critical_topic_protection" : 0.35,
        "correct_deprioritization"  : 0.25,
        "schedule_feasibility"      : 0.20,
        "retention_health"          : 0.20,
    }
    final_score = round(
        sum(axes[k] * weights[k] for k in axes), 4
    )

    feedback_parts = []
    missed_critical = [
        t for t in CRITICAL_TOPICS
        if t not in scheduled_topics
        or topic_scores.get(t, 0.0) <= KIRAN_INITIAL_SCORES.get(t, 0.0)
    ]
    if missed_critical:
        feedback_parts.append(f"Critical topics not adequately covered: {missed_critical}.")
    if correct_deprioritization < 0.7:
        feedback_parts.append("Agent over-scheduled already-strong topics (Arrays/Linked Lists).")
    if overloaded_days:
        feedback_parts.append(f"Overloaded days: {overloaded_days}.")
    if used_blocked_slots:
        feedback_parts.append(f"Agent used disrupted/blocked slots: {used_blocked_slots}.")
    if retention_health < 0.67:
        low_retention = [
            t for t in CRITICAL_TOPICS
            if retention_scores.get(t, 0.0) < RETENTION_FLOOR
        ]
        feedback_parts.append(f"Low retention on critical topics: {low_retention}. Needs revision sessions.")
    if not feedback_parts:
        feedback_parts.append(
            "Excellent triage — critical topics protected, strong topics deprioritized, "
            "schedule feasible through disruptions, retention healthy."
        )

    return GradeResult(
        task_id   = TASK_ID,
        score     = final_score,
        passed    = final_score >= PASS_THRESHOLD,
        breakdown = axes,
        feedback  = " ".join(feedback_parts),
    )