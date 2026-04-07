"""
models.py — shared contract between environment and (Person B)

"""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────
# Enums — fixed value sets both sides must use exactly
# ─────────────────────────────────────────────────────────────

class ResourceType(str, Enum):
    PROBLEM   = "problem"
    VIDEO     = "video"
    NOTES     = "notes"
    MOCK_TEST = "mock_test"


class Urgency(str, Enum):
    TODAY     = "today"
    THIS_WEEK = "this_week"
    OPTIONAL  = "optional"


class Personality(str, Enum):
    PROCRASTINATOR = "procrastinator"   # avoids hard topics, needs push
    CONSISTENT     = "consistent"       # steady pace, reliable
    ANXIOUS        = "anxious"          # needs confidence before hard topics


class Difficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


# ─────────────────────────────────────────────────────────────
# Core data types
# ─────────────────────────────────────────────────────────────

class Problem(BaseModel):
    id: str
    topic: str
    difficulty: Difficulty
    estimated_minutes: int
    description: str = ""


class Resource(BaseModel):
    topic: str
    title: str
    url: str
    resource_type: ResourceType


class Disruption(BaseModel):
    """A mid-episode event that changes the student's situation."""
    type: str           # "missed_session" | "mock_test_added" | "topic_mastered"
    description: str
    affected_slots: List[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# Action — what Person B's agent sends each step
# ─────────────────────────────────────────────────────────────

class StudyAction(BaseModel):
    """
    The action the agent takes each step.

    ALL fields are required. Person B must ensure the LLM
    always returns all four fields in valid JSON.

    Validation rules (enforced by Person A's environment):
    - recommended_topic must be in observation.weak_topics
    - assigned_slot must be in observation.available_slots
    - resource_type must be a valid ResourceType enum value
    - urgency must be a valid Urgency enum value
    """
    recommended_topic: str
    assigned_slot: str
    resource_type: ResourceType
    urgency: Urgency


# ─────────────────────────────────────────────────────────────
# Observation — what Person A's environment returns each step
# ─────────────────────────────────────────────────────────────

class StudyObservation(BaseModel):
    """
    What the agent sees at every step.

    Person B's inference.py reads these fields to build the
    prompt sent to the LLM.
    """
    # Task info
    task_id: str
    step: int
    max_steps: int
    done: bool = False

    # Student profile
    student_id: str
    personality: Personality

    # Academic state
    weak_topics: List[str]              # topics with score < 0.6
    topic_scores: Dict[str, float]      # topic → 0.0 to 1.0
    retention_scores: Dict[str, float]  # topic → decayed retention 0.0–1.0

    # Schedule
    available_slots: List[str]          # e.g. ["Mon 9am", "Tue 2pm"]
    exam_date: str                      # ISO date string "2024-03-15"
    days_remaining: int

    # Feedback from previous step
    last_action_feedback: str = ""
    last_reward: float = 0.0

    # Mid-episode disruption (None if no disruption this step)
    disruption: Optional[Disruption] = None


# ─────────────────────────────────────────────────────────────
# Reward breakdown — returned in info dict each step
# ─────────────────────────────────────────────────────────────

class RewardBreakdown(BaseModel):
    total: float = Field(0.0, ge=0.0, le=1.0)

    # Positive signals (each 0.0–1.0)
    topic_match:      float = 0.0
    difficulty_match: float = 0.0
    slot_valid:       float = 0.0
    pacing_score:     float = 0.0

    # Bonuses
    personality_bonus:       float = 0.0

    # Penalties (subtracted, each 0.0–1.0)
    overload_penalty:        float = 0.0
    past_exam_penalty:       float = 0.0
    repetition_penalty:      float = 0.0
    mastered_penalty:        float = 0.0
    retention_decay_penalty: float = 0.0
# ─────────────────────────────────────────────────────────────
# Step result — what step() returns
# ─────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    """
    Returned by environment.step(action).
    Person B reads .observation for next step,
    .reward for logging, .done to know when episode ends.
    """
    observation: StudyObservation
    reward: float = Field(0.0, ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
    # info always contains:
    #   "reward_breakdown": RewardBreakdown.model_dump()
    #   "task_id": str
    #   "step": int


# ─────────────────────────────────────────────────────────────
# Task grade result — returned by grader at episode end
# ─────────────────────────────────────────────────────────────

class GradeResult(BaseModel):
    """
    Final score for a completed episode.
    Returned by each task's grade() function.
    """
    task_id: str
    score: float = Field(0.0, ge=0.0, le=1.0)    # overall 0.0–1.0
    passed: bool                                   # score >= pass_threshold
    breakdown: Dict[str, float] = Field(default_factory=dict)
    feedback: str = ""