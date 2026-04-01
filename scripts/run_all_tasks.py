import json
import subprocess

tasks = ["easy", "medium", "hard"]

results = {}

for task in tasks:
    print(f"\nRunning task: {task}")

    process = subprocess.run(
        [
            "python",
            "scripts/run_baseline.py",
            "--task",
            task,
            "--agent",
            "random",
            "--seed",
            "0",
        ],
        capture_output=True,
        text=True,
    )

    try:
        output = json.loads(process.stdout)
        score = output["grader"]["score"]

        results[task] = score

        print(f"Score: {score}")

    except Exception:
        print("Failed to parse output")
        print(process.stdout)

print("\nFINAL SCOREBOARD")

for task, score in results.items():
    print(f"{task.upper()} : {score}")