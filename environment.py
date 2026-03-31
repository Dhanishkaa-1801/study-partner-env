"""
environment.py — Adaptive Study Partner RL Environment
OpenEnv Hackathon

Implements reset(), step(), and state() against the shared contract in models.py.

Episode lifecycle:
  - Each step() simulates one study session (one day advances).
  - retention_scores decay 10% every step for topics NOT studied that step.
  - Episode ends when days_remaining hits 0 (exam day) or max_steps is reached.
  - Invalid actions return reward 0.0 with a -0.1 penalty in info — no crash.
"""

from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from models import (
    Difficulty,
    Disruption,
    GradeResult,
    Personality,
    Problem,
    Resource,
    ResourceType,
    RewardBreakdown,
    StudyAction,
    StudyObservation,
    StepResult,
    Urgency,
)

# ---------------------------------------------------------------------------
# Paths to data files
# ---------------------------------------------------------------------------

DATA_DIR      = os.path.join(os.path.dirname(__file__), "data")
STUDENTS_PATH = os.path.join(DATA_DIR, "students.json")
PROBLEMS_PATH = os.path.join(DATA_DIR, "problems.json")
RESOURCES_PATH= os.path.join(DATA_DIR, "resources.json")

# ---------------------------------------------------------------------------
# Task configs — maps task_id → student + episode settings
# ---------------------------------------------------------------------------

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "task_1": {
        "student_id"         : "s002",          # Priya — anxious, 3 days
        "max_steps"          : 5,
        "critical_topics"    : ["Trees"],
        "disruption_schedule": {},               # no disruptions on easy task
        "description"        : "Single topic recommendation. Student has 1 weak focus topic (Trees), 3 days to exam. Agent must pick right problem + slot.",
    },
    "task_2": {
        "student_id"         : "s001",           # Arjun — procrastinator, 7 days
        "max_steps"          : 10,
        "critical_topics"    : ["Arrays", "Trees", "Dynamic Programming", "System Design"],
        "disruption_schedule": {
            5: {
                "type"           : "missed_session",
                "description"    : "Arjun skipped yesterday's session.",
                "affected_slots" : ["Wed 11am"],
            }
        },
        "description"        : "Weekly study plan. 4 weak topics, 7 days, 2 hrs/day. Agent must cover all weak topics with no overloading.",
    },
    "task_3": {
        "student_id"         : "s003",           # Kiran — consistent, 14 days
        "max_steps"          : 15,
        "critical_topics"    : ["Dynamic Programming", "Graphs", "System Design"],
        "disruption_schedule": {
            3: {
                "type"           : "mock_test_added",
                "description"    : "A mock test has been added for tomorrow. Adjust the plan.",
                "affected_slots" : ["Tue 9am"],
            },
            8: {
                "type"           : "topic_mastered",
                "description"    : "Kiran says Trees feel okay now. Update weak topic list.",
                "affected_slots" : [],
            },
            12: {
                "type"           : "missed_session",
                "description"    : "Kiran missed Thursday morning due to a college event.",
                "affected_slots" : ["Thu 9am"],
            },
        },
        "description"        : "Exam crunch triage. 8 topics, 3 critical, blocked slots, 3 disruptions. Agent must prioritize ruthlessly.",
    },
}

# ---------------------------------------------------------------------------
# Forgetting curve
# ---------------------------------------------------------------------------

DECAY_PER_STEP = 0.10   # 10% retention loss per step for unstudied topics
MASTERY_THRESHOLD = 0.6  # topics below this are "weak"


def apply_forgetting_curve(
    retention_scores: Dict[str, float],
    studied_topic: Optional[str],
) -> Dict[str, float]:
    """
    Decay all topics by DECAY_PER_STEP except the one studied this step.
    Studied topic gets a +0.15 retention boost (capped at 1.0).
    """
    updated = {}
    for topic, retention in retention_scores.items():
        if topic == studied_topic:
            updated[topic] = round(min(1.0, retention + 0.15), 4)
        else:
            updated[topic] = round(max(0.0, retention * (1 - DECAY_PER_STEP)), 4)
    return updated


# ---------------------------------------------------------------------------
# Helper — pick appropriate problem difficulty based on topic score
# ---------------------------------------------------------------------------

def pick_difficulty(topic_score: float) -> Difficulty:
    if topic_score < 0.3:
        return Difficulty.EASY
    elif topic_score < 0.6:
        return Difficulty.MEDIUM
    else:
        return Difficulty.HARD


def difficulty_to_float(d: Difficulty) -> float:
    return {"easy": 0.2, "medium": 0.5, "hard": 0.9}[d.value]


# ---------------------------------------------------------------------------
# Main Environment
# ---------------------------------------------------------------------------

class StudyEnv:
    """
    Adaptive Study Partner — OpenEnv RL Environment.

    Usage:
        env = StudyEnv(task_id="task_1")
        obs = env.reset()
        while not obs.done:
            action = agent.act(obs)
            result = env.step(action)
            obs = result.observation
        grade = env.grade()
    """

    def __init__(self, task_id: str = "task_1"):
        if task_id not in TASK_CONFIGS:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from: {list(TASK_CONFIGS.keys())}")

        self.task_id    = task_id
        self.task_cfg   = TASK_CONFIGS[task_id]

        # Load static data once
        self._all_students  = self._load_json(STUDENTS_PATH)["students"]
        self._all_problems  = [Problem(**p) for p in self._load_json(PROBLEMS_PATH)["problems"]]
        self._all_resources = [Resource(**r) for r in self._load_json(RESOURCES_PATH)["resources"]]

        # Runtime state (populated by reset())
        self._obs: Optional[StudyObservation] = None
        self._step_count    = 0
        self._scheduled: Dict[str, str] = {}          # slot → topic
        self._topic_studied_steps: Dict[str, int] = {} # topic → last step studied
        self._history: List[str] = []                  # recommended topics log
        self._episode_rewards: List[float] = []
        self._done = False

    # -----------------------------------------------------------------------
    # reset()
    # -----------------------------------------------------------------------

    def reset(self) -> StudyObservation:
        """Start a fresh episode. Returns the initial observation."""
        cfg = self.task_cfg
        student_data = self._get_student(cfg["student_id"])

        # Deep copy scores so mutations don't bleed between episodes
        topic_scores     = deepcopy(student_data["topic_scores"])
        retention_scores = deepcopy(student_data["topic_scores"])  # starts equal to mastery
        available_slots  = deepcopy(student_data["available_slots"])

        self._step_count    = 0
        self._scheduled     = {}
        self._topic_studied_steps = {}
        self._history       = []
        self._episode_rewards = []
        self._done          = False

        # Internal mutable state (not exposed directly — only via observation)
        self._topic_scores     = topic_scores
        self._retention_scores = retention_scores
        self._available_slots  = available_slots
        self._days_remaining   = student_data["days_remaining"]
        self._personality      = Personality(student_data["personality"])
        self._student_id       = student_data["student_id"]
        self._exam_date        = student_data["exam_date"]

        weak_topics = [t for t, s in topic_scores.items() if s < MASTERY_THRESHOLD]

        self._obs = StudyObservation(
            task_id              = self.task_id,
            step                 = 0,
            max_steps            = cfg["max_steps"],
            done                 = False,
            student_id           = student_data["student_id"],
            personality          = self._personality,
            weak_topics          = weak_topics,
            topic_scores         = deepcopy(topic_scores),
            retention_scores     = deepcopy(retention_scores),
            available_slots      = deepcopy(available_slots),
            exam_date            = student_data["exam_date"],
            days_remaining       = student_data["days_remaining"],
            last_action_feedback = "",
            last_reward          = 0.0,
            disruption           = None,
        )
        return self._obs

    # -----------------------------------------------------------------------
    # step()
    # -----------------------------------------------------------------------

    def step(self, action: StudyAction) -> StepResult:
        """
        Advance the environment by one step.

        Args:
            action: StudyAction from the agent.

        Returns:
            StepResult with updated observation, reward, done flag, and info.
        """
        if self._obs is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step_count += 1
        reward_bd = RewardBreakdown()
        feedback_parts: List[str] = []
        invalid = False

        # ── 1. Validate action ──────────────────────────────────────────────
        validation_error = self._validate_action(action)
        if validation_error:
            # Invalid action: 0.0 reward + -0.1 penalty, no state update
            reward_bd.total = 0.0
            info = {
                "reward_breakdown": reward_bd.model_dump(),
                "invalid_action"  : True,
                "invalid_reason"  : validation_error,
                "penalty"         : -0.1,
                "task_id"         : self.task_id,
                "step"            : self._step_count,
            }
            feedback = f"Invalid action: {validation_error}"
            invalid = True
        else:
            # ── 2. Compute reward ───────────────────────────────────────────
            reward_bd = self._compute_reward(action)
            feedback_parts = self._build_feedback(action, reward_bd)
            feedback = " | ".join(feedback_parts)

            # ── 3. Update state ─────────────────────────────────────────────
            self._apply_action(action)
            info = {
                "reward_breakdown": reward_bd.model_dump(),
                "invalid_action"  : False,
                "task_id"         : self.task_id,
                "step"            : self._step_count,
            }

        # ── 4. Advance forgetting curve (every step regardless of validity) ─
        studied = action.recommended_topic if not invalid else None
        self._retention_scores = apply_forgetting_curve(self._retention_scores, studied)

        # ── 5. Advance simulated day ────────────────────────────────────────
        self._days_remaining = max(0, self._days_remaining - 1)

        # ── 6. Fire disruption if scheduled for this step ───────────────────
        disruption = self._get_disruption(self._step_count)
        if disruption:
            self._apply_disruption(disruption)

        # ── 7. Check done ───────────────────────────────────────────────────
        max_steps_hit  = self._step_count >= self.task_cfg["max_steps"]
        exam_day_hit   = self._days_remaining <= 0
        self._done     = max_steps_hit or exam_day_hit

        # ── 8. Track reward ─────────────────────────────────────────────────
        final_reward = reward_bd.total if not invalid else 0.0
        self._episode_rewards.append(final_reward)

        # ── 9. Build next observation ────────────────────────────────────────
        weak_topics = [t for t, s in self._topic_scores.items() if s < MASTERY_THRESHOLD]

        next_obs = StudyObservation(
            task_id              = self.task_id,
            step                 = self._step_count,
            max_steps            = self.task_cfg["max_steps"],
            done                 = self._done,
            student_id           = self._student_id,
            personality          = self._personality,
            weak_topics          = weak_topics,
            topic_scores         = deepcopy(self._topic_scores),
            retention_scores     = deepcopy(self._retention_scores),
            available_slots      = deepcopy(self._available_slots),
            exam_date            = self._exam_date,
            days_remaining       = self._days_remaining,
            last_action_feedback = feedback,
            last_reward          = final_reward,
            disruption           = disruption,
        )
        self._obs = next_obs

        return StepResult(
            observation = next_obs,
            reward      = final_reward,
            done        = self._done,
            info        = info,
        )

    # -----------------------------------------------------------------------
    # state()
    # -----------------------------------------------------------------------

    def state(self) -> StudyObservation:
        """Return the current observation without advancing the episode."""
        if self._obs is None:
            raise RuntimeError("Call reset() before state().")
        return self._obs

    # -----------------------------------------------------------------------
    # grade() — call after episode ends
    # -----------------------------------------------------------------------

    def grade(self) -> GradeResult:
        """
        Compute the final episode grade.
        Delegates to the per-task grader in tasks/.
        """
        from tasks.task_1 import grade as grade_task_1
        from tasks.task_2 import grade as grade_task_2
        from tasks.task_3 import grade as grade_task_3

        graders = {
            "task_1": grade_task_1,
            "task_2": grade_task_2,
            "task_3": grade_task_3,
        }
        return graders[self.task_id](
            topic_scores      = self._topic_scores,
            retention_scores  = self._retention_scores,
            scheduled         = self._scheduled,
            history           = self._history,
            available_slots   = self._available_slots,
            days_remaining    = self._days_remaining,
            episode_rewards   = self._episode_rewards,
            critical_topics   = self.task_cfg["critical_topics"],
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _validate_action(self, action: StudyAction) -> Optional[str]:
        """Return error string if action is invalid, else None."""
        if action.recommended_topic not in self._topic_scores:
            return f"Topic '{action.recommended_topic}' does not exist in student profile."
        if action.assigned_slot not in self._available_slots:
            return f"Slot '{action.assigned_slot}' is not available (already used or doesn't exist)."
        return None

    def _compute_reward(self, action: StudyAction) -> RewardBreakdown:
        """Compute the full reward breakdown for a valid action."""
        bd = RewardBreakdown()
        topic = action.recommended_topic
        slot  = action.assigned_slot
        score = self._topic_scores.get(topic, 0.5)

        # ── Positive signals ─────────────────────────────────────────────
        # +0.4 weak topic targeted
        if score < MASTERY_THRESHOLD:
            bd.topic_match = 0.4
        else:
            bd.mastered_penalty = -0.2   # already mastered

        # +0.3 difficulty appropriate
        ideal_diff   = pick_difficulty(score)
        # We can't know exactly which problem will be picked yet,
        # so we reward if the resource_type makes sense for difficulty
        if ideal_diff == Difficulty.EASY and action.resource_type in (ResourceType.NOTES, ResourceType.VIDEO):
            bd.difficulty_match = 0.3
        elif ideal_diff == Difficulty.MEDIUM and action.resource_type in (ResourceType.PROBLEM, ResourceType.NOTES):
            bd.difficulty_match = 0.3
        elif ideal_diff == Difficulty.HARD and action.resource_type in (ResourceType.PROBLEM, ResourceType.MOCK_TEST):
            bd.difficulty_match = 0.3
        else:
            bd.difficulty_match = 0.1   # partial credit — not ideal but not zero

        # +0.2 slot valid (free + before exam) — slot already confirmed free by validation
        bd.slot_valid = 0.2

        # +0.1 pacing on track
        total_topics  = len(self._topic_scores)
        covered       = len(set(self._history))
        expected_pct  = 1.0 - (self._days_remaining / max(1, self._days_remaining + self._step_count))
        actual_pct    = covered / total_topics if total_topics > 0 else 0.0
        if actual_pct >= expected_pct * 0.8:
            bd.pacing_score = 0.1

        # ── Penalties ────────────────────────────────────────────────────
        # -0.2 overload: more than 2 sessions already booked today
        day_of_slot   = slot.split(" ")[0]   # e.g. "Mon" from "Mon 9am"
        sessions_today = sum(
            1 for s in self._scheduled if s.split(" ")[0] == day_of_slot
        )
        if sessions_today >= 2:
            bd.overload_penalty = -0.2

        # -0.1 repetition: same topic recommended twice in a row
        if self._history and self._history[-1] == topic:
            bd.repetition_penalty = -0.1

        # ── Personality bonus (not a RewardBreakdown field — tracked separately) ──
        # +0.1 if action fits personality
        personality_bonus = 0.0
        if self._personality == Personality.PROCRASTINATOR and action.urgency == Urgency.TODAY:
            personality_bonus = 0.1
        elif self._personality == Personality.ANXIOUS and action.resource_type in (ResourceType.NOTES, ResourceType.VIDEO):
            personality_bonus = 0.1
        elif self._personality == Personality.CONSISTENT and action.urgency == Urgency.THIS_WEEK:
            personality_bonus = 0.1

        # ── Forgetting curve penalty (not a RewardBreakdown field — tracked separately) ──
        retention = self._retention_scores.get(topic, 1.0)
        retention_penalty = -0.1 if retention < 0.5 else 0.0

        # Compute total using only fields that exist in teammate's RewardBreakdown
        raw = (
            bd.topic_match
            + bd.difficulty_match
            + bd.slot_valid
            + bd.pacing_score
            + bd.mastered_penalty
            + bd.overload_penalty
            + bd.repetition_penalty
            + personality_bonus
            + retention_penalty
        )
        bd.total = round(max(0.0, min(1.0, raw)), 4)
        return bd

    def _apply_action(self, action: StudyAction) -> None:
        """Mutate internal state to reflect the action taken."""
        topic = action.recommended_topic
        slot  = action.assigned_slot

        # Remove slot from available pool
        self._available_slots.remove(slot)

        # Record schedule
        self._scheduled[slot] = topic

        # Improve topic score (studying helps, caps at 1.0)
        self._topic_scores[topic] = round(
            min(1.0, self._topic_scores.get(topic, 0.0) + 0.12), 4
        )

        # Track topic studied this step
        self._topic_studied_steps[topic] = self._step_count

        # Add to history
        self._history.append(topic)

    def _get_disruption(self, step: int) -> Optional[Disruption]:
        """Return a Disruption if one is scheduled for this step, else None."""
        raw = self.task_cfg["disruption_schedule"].get(step)
        if raw is None:
            return None
        return Disruption(**raw)

    def _apply_disruption(self, disruption: Disruption) -> None:
        """Mutate state based on disruption type."""
        if disruption.type == "missed_session":
            # Remove affected slots from available pool
            for slot in disruption.affected_slots:
                if slot in self._available_slots:
                    self._available_slots.remove(slot)

        elif disruption.type == "mock_test_added":
            # Block the affected slot (reserved for mock test)
            for slot in disruption.affected_slots:
                if slot in self._available_slots:
                    self._available_slots.remove(slot)

        elif disruption.type == "topic_mastered":
            # Student reports a topic feels okay — boost its score
            for topic in self._topic_scores:
                if topic in disruption.description:
                    self._topic_scores[topic] = max(
                        self._topic_scores[topic], MASTERY_THRESHOLD
                    )

    def _build_feedback(self, action: StudyAction, bd: RewardBreakdown) -> List[str]:
        """Build human-readable feedback strings for last_action_feedback."""
        parts = []
        if bd.topic_match > 0:
            parts.append(f"✅ Good: '{action.recommended_topic}' is a weak topic")
        if bd.mastered_penalty < 0:
            parts.append(f"⚠️ '{action.recommended_topic}' is already mastered")
        if bd.difficulty_match >= 0.3:
            parts.append("✅ Resource type fits current skill level")
        if bd.slot_valid > 0:
            parts.append(f"✅ Slot '{action.assigned_slot}' is valid")
        if bd.overload_penalty < 0:
            parts.append("⚠️ Day overloaded — too many sessions")
        if bd.repetition_penalty < 0:
            parts.append("⚠️ Same topic recommended twice in a row")
        retention = self._retention_scores.get(action.recommended_topic, 1.0)
        if retention < 0.5:
            parts.append(f"⚠️ '{action.recommended_topic}' retention is low — good to revise")
        parts.append(f"Reward: {bd.total:.3f}")
        return parts

    def _get_student(self, student_id: str) -> Dict[str, Any]:
        for s in self._all_students:
            if s["student_id"] == student_id:
                return s
        raise ValueError(f"Student '{student_id}' not found in students.json")

    @staticmethod
    def _load_json(path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return json.load(f)