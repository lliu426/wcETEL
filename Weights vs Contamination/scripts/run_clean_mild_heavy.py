import os

for mode in ["clean", "mild", "heavy"]:
    print(f"\nRunning mode: {mode}")
    os.system(f"python parallel_runner.py {mode}")
