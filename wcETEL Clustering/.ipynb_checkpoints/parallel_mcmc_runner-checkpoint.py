import numpy as np
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from mcmc_wcetel import single_replication
from data_generator_GM import generate_GM_data
from wcetel_config import nrep, N, nIter, d, burn_in, lamda, X_data, Alpha



os.makedirs("wcETEL_MCMC_results", exist_ok=True)

def run_and_save(m):
    print(f"Running replication: {m}")
    result = single_replication(m, lamda, N, nIter, d, burn_in, X_data)
    
    with open(f"wcETEL_MCMC_results/replication_{m}.pkl", "wb") as f:
        pickle.dump(result, f)

if __name__ == "__main__":
    max_workers = min(4, multiprocessing.cpu_count())
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(run_and_save, range(nrep))