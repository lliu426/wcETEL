# wcETEL Clustering Experiments

This folder implements clustering experiments under the Wasserstein-constrained exponentially tilted empirical likelihood (wcETEL) framework. You’ll find two pipelines:

1. **Single Gaussian** – fit a univariate Normal model via wcETEL  
2. **Two-component Gaussian Mixture** – fit a 2-Gaussian mixture via wcETEL  

---

## Repository Structure

- data_generator_GM.py # generate synthetic 2-Gaussian mixture data
- optimal_transport_1d.py # 1D power-diagram & semi-discrete OT solver
- wc_ETEL.py # wcETEL methods for single-Gaussian target
- wc_ETEL_GM.py # wcETEL methods for Gaussian mixture target
- lel_ws.py / lel_ws_d2.py # general EL & Wasserstein helper routines
- mcmc_wcetel.py # single-Gaussian MCMC replication
- mcmc_wcetel_mixture.py # mixture MCMC replication
- wcetel_config.py # shared settings (N, λ, MCMC iters, etc.)
- parallel_mcmc_runner.py # parallel launcher for single-Gaussian runs
- parallel_mcmc_mixture_runner.py # parallel launcher for mixture runs
- wcETEL_clustering.ipynb # notebook: run both pipelines & visualize

---

## Configuration

Edit wcetel_config.py to control:

- N – number of observations per replicate
- nrep – number of dataset replicates
- nIter, burn – MCMC iterations and burn-in
- lambda_w – Wasserstein penalty strength
- Data settings – mixture weights & component parameters

---

## Inspecting & Visualizing Results

Open wcETEL_clustering.ipynb to:

- Load both result directories
- Compare posterior parameter estimates

---

## Output Format

Each pickle file (.pkl) is a dict containing:

- trace – sampled parameter chains
- weights – final empirical weights
- assignments – cluster labels (mixture only)
- log_lik – per-iteration log-likelihoods

---

## ε-Sensitivity Analysis

We also assess how the choice of the Wasserstein constraint level ε affects predictive performance.

### Purpose

By varying ε over a grid, we compute leave-one-out predictive scores (elpdₗₒₒ) to see how ε affects of out-of-sample fit. 

### Key Files

- **`wcetel_config.py`**  
  Defines the grid of ε values to explore.
- **`epsilon_worker.py`**  
  Launches a single ε job: runs MCMC, computes elpdₗₒₒ via Pareto-smoothed importance sampling.
- **`parallel_runner.py`**  
  Dispatches all ε jobs in parallel and collates the results.
- **`epsilon_sensitivity.ipynb`**  
  (Optional) notebook that
  1. Executes the full ε sweep,
  2. Aggregates `elpd_loo` and its standard error,
  3. Plots `elpd_loo` vs. ε,
  4. Highlights the optimal ε.

### Inspect results

Raw outputs are saved under
sensitivity_results/epsilon_<ε>.pkl

A summary CSV is written in epsilon_worker.ipynb:

Epsilon                    MargLik                       ELPD                         SE
------------------------------------------------------------------------------------------
      2.25            -460.5176062628            -460.5177456985               0.0381631886
      3.00            -460.5173495935            -460.5174280950               0.0286307892
      4.00            -460.5172049666            -460.5172491528               0.0214778204
      8.00            -460.5170652596            -460.5170763171               0.0107424603
     16.00            -460.5170302726            -460.5170330384               0.0053721185

Visual diagnostics and the final LOO-score vs. ε plot:
![download](https://github.com/user-attachments/assets/2b3f3999-90f4-4edc-b1a4-09807284a458)
