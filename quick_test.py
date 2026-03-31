"""
quick_test.py — Person A Integration Test
Run this before telling Person B your code is ready.

Usage:
    python quick_test.py

All 3 tasks should pass with no errors.
"""

from environment import StudyEnv
from models import StudyAction, ResourceType, Urgency

PASS = "✅"
FAIL = "❌"

def test_task(task_id: str, action_overrides: dict = {}):
    print(f"\n{'='*50}")
    print(f"Testing {task_id}")
    print(f"{'='*50}")

    try:
        env = StudyEnv(task_id)
        obs = env.reset()
        print(f"{PASS} reset() works | student: {obs.student_id} | weak topics: {obs.weak_topics}")
    except Exception as e:
        print(f"{FAIL} reset() crashed: {e}")
        return

    # Run a few valid steps
    step = 0
    while not obs.done and step < 4:
        # Pick valid topic + slot from observation
        topic = action_overrides.get("topic") or obs.weak_topics[0]
        slot  = action_overrides.get("slot")  or obs.available_slots[0]

        action = StudyAction(
            recommended_topic = topic,
            assigned_slot     = slot,
            resource_type     = ResourceType.NOTES,
            urgency           = Urgency.TODAY,
        )

        try:
            result = env.step(action)
            obs    = result.observation
            print(f"{PASS} step {step+1} | reward: {result.reward:.3f} | days left: {obs.days_remaining} | done: {obs.done}")
            if result.info.get("invalid_action"):
                print(f"     ⚠️  Invalid action reason: {result.info.get('invalid_reason')}")
        except Exception as e:
            print(f"{FAIL} step {step+1} crashed: {e}")
            return

        step += 1

    # Test invalid action handling
    print(f"\n-- Testing invalid action handling --")
    try:
        bad_action = StudyAction(
            recommended_topic = "FakeTopic",
            assigned_slot     = obs.available_slots[0] if obs.available_slots else "Mon 9am",
            resource_type     = ResourceType.PROBLEM,
            urgency           = Urgency.TODAY,
        )
        result = env.step(bad_action)
        if result.reward == 0.0 and result.info.get("invalid_action"):
            print(f"{PASS} Invalid action handled gracefully | reward: {result.reward} | reason: {result.info.get('invalid_reason')}")
        else:
            print(f"{FAIL} Invalid action should return reward=0.0 and invalid_action=True")
    except Exception as e:
        print(f"{FAIL} Invalid action crashed instead of being handled: {e}")
        return

    # Test grade()
    print(f"\n-- Testing grade() --")
    try:
        # Force episode to end
        env._done = True
        grade = env.grade()
        print(f"{PASS} grade() works | score: {grade.score:.3f} | passed: {grade.passed}")
        print(f"     breakdown: {grade.breakdown}")
        print(f"     feedback:  {grade.feedback}")
    except Exception as e:
        print(f"{FAIL} grade() crashed: {e}")

    # Test state()
    print(f"\n-- Testing state() --")
    try:
        s = env.state()
        print(f"{PASS} state() works | step: {s.step} | done: {s.done}")
    except Exception as e:
        print(f"{FAIL} state() crashed: {e}")


if __name__ == "__main__":
    print("🚀 Running Person A integration tests...\n")

    test_task("task_1")
    test_task("task_2")
    test_task("task_3")

    print(f"\n{'='*50}")
    print("All tests done.")
    print("If all lines show ✅ — push to GitHub and tell Person B.")
    print("If any show ❌ — fix before pushing.")
    print(f"{'='*50}\n")