from environment import StudyEnv

try:
    # Create the environment for Task 1
    env = StudyEnv("task_1")

    # Try to reset it
    obs = env.reset()

    # If this prints, you're a genius and Person A's job is 50% done!
    print("✅ reset OK:")
    print(f"   Task ID: {obs.task_id}")
    print(f"   Student: {obs.student_id}")
    print(f"   Weak Topics: {obs.weak_topics}")

except Exception as e:
    print("❌ Error during sanity check:")
    print(e)