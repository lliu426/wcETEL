import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from data_generator import generate_datasets
from run_wcETEL_analysis_module import run_wcETEL_analysis
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import pickle
import sys

# read contamination mode from command line
if len(sys.argv) != 2:
    print("Usage: python parallel_runner.py [clean|mild|heavy]")
    sys.exit(1)

mode = sys.argv[1]
datasets = generate_datasets(mode)

output_dir = f"wcETEL_results/{mode}"
os.makedirs(output_dir, exist_ok=True)



def run_and_save(index_X):
    index, X = index_X
    print(f"Running dataset: {index}")
    weights, masses1 = run_wcETEL_analysis(X)
    
    with open(f"{output_dir}/dataset_{index}_weights.pkl", "wb") as f:
        pickle.dump({"weights": weights, "masses1": masses1}, f)



if __name__ == "__main__":
    data_list = list(enumerate(datasets))
    
    max_workers = min(4, multiprocessing.cpu_count())
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(run_and_save, data_list)