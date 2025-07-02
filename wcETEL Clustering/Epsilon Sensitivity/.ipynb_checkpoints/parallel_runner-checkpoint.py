import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

epsilon_grid = [2.25, 3, 4, 8, 16]

def run_single_epsilon(epsilon):
    result = subprocess.run(
        [sys.executable, "epsilon_worker.py", str(epsilon)],
        capture_output=True, text=True
    )
    return (epsilon, result.stdout, result.stderr)

def main():
    with ProcessPoolExecutor(max_workers=len(epsilon_grid)) as executor:
        futures = [executor.submit(run_single_epsilon, eps) for eps in epsilon_grid]
        for future in futures:
            eps, stdout, stderr = future.result()
            print(f"\n--- Epsilon {eps} STDOUT ---\n{stdout}")
            print(f"\n--- Epsilon {eps} STDERR ---\n{stderr}")

if __name__ == "__main__":
    main()
