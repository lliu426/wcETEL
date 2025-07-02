import os
import glob
import pickle
import matplotlib.pyplot as plt

modes = ['clean', 'mild', 'heavy']
N = 300

for mode in modes:
    print(f"Plotting weights for {mode} contamination")

    result_files = sorted(glob.glob(f"wcETEL_results/{mode}/dataset_*_weights.pkl"))

    plt.figure(figsize=(10, 6))

    for file in result_files:
        with open(file, 'rb') as f:
            result = pickle.load(f)
            masses1 = result['masses1']
            plt.plot(masses1, alpha=0.3)

    plt.axhline(1/N, color='red', linestyle='--', label='1/300')

    plt.title(f"Overlay of Actual Weights for 50 {mode.capitalize()} Contaminated Datasets")
    plt.xlabel("Index")
    plt.ylabel("Probability Weight")
    plt.legend()
    plt.tight_layout()

    # Save figure
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{mode}_weights_overlay.png")
    plt.close()

    print(f"Saved plot to plots/{mode}_weights_overlay.png")