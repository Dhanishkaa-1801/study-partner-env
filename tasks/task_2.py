"""
tasks/task_2.py — Medium Task Grader
Adaptive Study Partner | OpenEnv Hackathon

Scenario:
    Student: Arjun (procrastinator personality)
    Weak topics: Arrays (0.3), Trees (0.2), Dynamic Programming (0.1), System Design (0.2)
    Also weak: Graphs (0.4), Linked Lists (0.5)
    Days: 7
    Slots: 9 available across the week
    Critical topics: Arrays, Trees, Dynamic Programming, System Design
    Disruption at step 5: missed_session → "Wed 11am" slot removed

What the agent must do:
    - Cover ALL 4 critical topics before exam
    - Schedule sessions with urgency "today" (procrastinator needs pushing)
    - No day overloading (max 2 sessions/day)
    - Handle the missed session disruption gracefully (reschedule if needed)
    - Maintain difficulty progression (don't assign hard problems to 0.1 topics)

Grader axes (all 0.0–1.0):
    1. critical_coverage     (weight 0.40) — all 4 critical topics scheduled
    2. no_overload           (weight 0.25) — no day has >2 sessions
    3. difficulty_progression(weight 0.20) — harder topics scheduled later
    4. disruption_recovery   (weight 0.15) — schedule still viable after disruption

Final score = weighted average of axes.
Pass threshold: 0.55
"""

from typing import Dict, List
from models import GradeResult

TASK_ID          = "task_2"
PASS_THRESHOLD   = 0.55
MASTERY_THRESHOLD = 0.6

ARJUN_INITIAL_SCORES = {
    "Arrays"            : 0.3,
    "Trees"             : 0.2,
    "Dynamic Programming": 0.1,
    "System Design"     : 0.2,
    "Graphs"            : 0.4,
    "Linked Lists"      : 0.5,
    "OS"                : 0.6,
    "DBMS"              : 0.7,
}


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
    Grade a completed Task 2 episode.
    """

    scheduled_topics = list(scheduled.values())

    # ── Axis 1: Critical topic coverage (weight 0.40) ────────────────────
    # All 4 critical topics must appear in the schedule
    covered_critical = [t for t in critical_topics if t in scheduled_topics]
    critical_coverage = len(covered_critical) / len(critical_topics) if critical_topics else 0.0

    # ── Axis 2: No overload (weight 0.25) ────────────────────────────────
    # Count sessions per day. Max 2 per day allowed.
    day_counts: Dict[str, int] = {}
    for slot in scheduled:
        day = slot.split(" ")[0]    # "Mon" from "Mon 9am"
        day_counts[day] = day_counts.get(day, 0) + 1

    overloaded_days = [d for d, count in day_counts.items() if count > 2]

    if len(overloaded_days) == 0:
        no_overload = 1.0
    elif len(overloaded_days) == 1:
        no_overload = 0.5
    else:
        no_overload = 0.0

    # ── Axis 3: Difficulty progression (weight 0.20) ─────────────────────
    # Easier topics (lower initial score) should appear earlier in history.
    # We check if the first half of history has lower avg initial score
    # than the second half.
    if len(history) >= 4:
        mid = len(history) // 2
        first_half_avg = sum(
            ARJUN_INITIAL_SCORES.get(t, 0.5) for t in history[:mid]
        ) / mid
        second_half_avg = sum(
            ARJUN_INITIAL_SCORES.get(t, 0.5) for t in history[mid:]
        ) / (len(history) - mid)

        if first_half_avg <= second_half_avg:
            difficulty_progression = 1.0   # good: harder stuff pushed later
        elif second_half_avg >= first_half_avg * 0.85:
            difficulty_progression = 0.5   # roughly okay
        else:
            difficulty_progression = 0.0   # agent scheduled hardest things first

    elif len(history) >= 2:
        difficulty_progression = 0.5       # partial credit — not enough steps to judge
    else:
        difficulty_progression = 0.0

    # ── Axis 4: Disruption recovery (weight 0.15) ────────────────────────
    # The disruption at step 5 removes "Wed 11am".
    # Recovery is good if:
    #   (a) critical topics were still all scheduled despite the lost slot, OR
    #   (b) agent managed to cover ≥ 3/4 critical topics
    disrupted_slot  = "Wed 11am"
    slot_was_lost   = disrupted_slot not in available_slots and disrupted_slot not in scheduled

    if critical_coverage >= 1.0:
        disruption_recovery = 1.0   # covered everything despite disruption
    elif critical_coverage >= 0.75:
        disruption_recovery = 0.6   # covered most — good enough
    elif slot_was_lost and critical_coverage >= 0.5:
        disruption_recovery = 0.4   # disruption hurt but agent partially adapted
    else:
        disruption_recovery = 0.0   # agent failed to recover

    # ── Weighted final score ─────────────────────────────────────────────
    axes = {
        "critical_coverage"     : round(critical_coverage, 4),
        "no_overload"           : round(no_overload, 4),
        "difficulty_progression": round(difficulty_progression, 4),
        "disruption_recovery"   : round(disruption_recovery, 4),
    }
    weights = {
        "critical_coverage"     : 0.40,
        "no_overload"           : 0.25,
        "difficulty_progression": 0.20,
        "disruption_recovery"   : 0.15,
    }
    final_score = round(
        sum(axes[k] * weights[k] for k in axes), 4
    )

    feedback_parts = []
    missing = [t for t in critical_topics if t not in scheduled_topics]
    if missing:
        feedback_parts.append(f"Critical topics not covered: {missing}.")
    if overloaded_days:
        feedback_parts.append(f"Overloaded days: {overloaded_days}.")
    if difficulty_progression < 0.5:
        feedback_parts.append("Agent scheduled harder topics too early.")
    if disruption_recovery < 0.6:
        feedback_parts.append("Agent did not recover well from the missed session disruption.")
    if not feedback_parts:
        feedback_parts.append("All critical topics covered, no overload, good progression, disruption handled.")

    return GradeResult(
        task_id   = TASK_ID,
        score     = final_score,
        passed    = final_score >= PASS_THRESHOLD,
        breakdown = axes,
        feedback  = " ".join(feedback_parts),
    )