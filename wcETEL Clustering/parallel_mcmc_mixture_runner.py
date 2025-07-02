import numpy as np
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from data_generator_GM import generate_GM_data
from mcmc_wcetel_mixture import single_replication_mixture
from wcetel_config import nrep, N, nIter, d, burn_in, lamda, X_data, Alpha



os.makedirs("wcETEL_MCMC_results_M2", exist_ok=True)

def run_and_save(m):
    print(f"Running replication: {m}")
    result = single_replication_mixture(m, lamda, N, nIter, burn_in, X_data)

    with open(f"wcETEL_MCMC_results_M2/replication_{m}.pkl", "wb") as f:
        pickle.dump(result, f)

if __name__ == "__main__":
    max_workers = min(4, multiprocessing.cpu_count())
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(run_and_save, range(nrep))
