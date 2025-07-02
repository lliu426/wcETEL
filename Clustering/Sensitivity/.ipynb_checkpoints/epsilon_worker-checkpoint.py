import numpy as np
from scipy.stats import skewnorm
import arviz as az
import sys
import time
import traceback
import datetime
from mcmc_wcetel import single_replication
from wcetel_config import nrep, N, nIter, d, burn_in, lamda, X_data, Alpha

def run_for_epsilon(epsilon):
    start_time = time.time()
    print(f"[{datetime.datetime.now()}] Starting epsilon: {epsilon}")
    sys.stdout.flush()

    try:
        M1_Results = np.full((nrep, 3), np.nan)
        for m in range(nrep):
            np.random.seed(123 + m)

            x = X_data
            x = np.sort(x)

            result = single_replication(m, lamda= 1 / epsilon, N=N, nIter=nIter, d=1, burn_in=burn_in, X_data=x)
            MH_storage = result['MH_storage']
            loglik_mat = result['loglik_mat']
            mu_Storage = result['mu_Storage']

            post_prob = np.exp(MH_storage - np.max(MH_storage))
            post_prob /= np.sum(post_prob)
            M1_Marglik = np.max(MH_storage) - np.log(np.sum(post_prob))

            posterior_samples = mu_Storage[burn_in:, :].reshape((1, -1, 1))
            loglik_post_burnin = loglik_mat[burn_in:, :].reshape((1, -1, N))

            idata = az.from_dict(
                posterior={"mu": posterior_samples},
                log_likelihood={"likelihood": loglik_post_burnin},
                coords={"obs_id": np.arange(N)},
                dims={"likelihood": ["obs_id"]}
            )

            loo_result = az.loo(idata, pointwise=True)
            MS1_elpd = loo_result.elpd_loo
            MS1_se = loo_result.se

            M1_Results[m, :] = [M1_Marglik, MS1_elpd, MS1_se]

        mean_result = np.nanmean(M1_Results, axis=0)
        elapsed_time = time.time() - start_time
        print(f"[{datetime.datetime.now()}] Finished epsilon: {epsilon} | Time: {elapsed_time:.2f} sec")
        sys.stdout.flush()

        # Save result for later aggregation
        np.save(f"result_eps_{epsilon}.npy", np.array([epsilon, *mean_result]))

    except Exception as e:
        print(f"Error for epsilon={epsilon}: {e}")
        traceback.print_exc()
        sys.stdout.flush()

if __name__ == "__main__":
    epsilon = float(sys.argv[1])
    run_for_epsilon(epsilon)
