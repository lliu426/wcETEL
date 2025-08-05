#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from lel_ws import LEL_WS
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from joblib import Parallel, delayed
import os
from datetime import datetime
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from IPython.display import display


# In[2]:


def generate_data(N, mu, sigma2, pi, contam_mean=10):
    n_contam = int(N * pi)
    n_clean = N - n_contam
    data_clean = np.random.normal(loc=mu, scale=np.sqrt(sigma2), size=n_clean)
    data_contam = np.random.normal(loc=contam_mean, scale=1.0, size=n_contam)
    return np.sort(np.concatenate([data_clean, data_contam]))


# In[3]:


def neg_log_lel_ws(params, x, lambda_param):
    mu, log_sigma2 = params
    sigma2 = np.exp(log_sigma2)  # ensures sigma² > 0
    out = LEL_WS(x, mu, sigma2, lambda_param)
    print(f"Trying mu={mu:.4f}, sigma²={sigma2:.4f}, LEL_WS={out['Optimal_Value']:.4f}")
    return -out["Optimal_Value"]  # negate for minimization


# In[4]:


def eval_lel(mu, sigma2):
    try:
        result = LEL_WS(x, mu, sigma2, lambda_param)
        return (mu, sigma2, result["Optimal_Value"])
    except:
        return (mu, sigma2, -np.inf)


# In[5]:


mu_grid     = np.linspace(-0.5, 1.5, 10)
sigma2_grid = np.linspace(0.5, 3.0, 10)
epsilon_list = [0.01, 0.1, 0.25, 0.5]
pi_list = [0.0, 0.01, 0.1]
n_rep = 5  # Change to 50 when needed
true_mu = 0

# Pre-generate datasets using fixed seeds
datasets = {pi: [] for pi in pi_list}
for pi in pi_list:
    for rep in range(n_rep):
        seed = 1000 * rep + int(pi * 1000)  # unique seed per (π, rep)
        rng = np.random.default_rng(seed)
        x = generate_data(N=100, mu=true_mu, sigma2=1, pi=pi, contam_mean=5)
        datasets[pi].append(x)

# Storage
error_results = {pi: {} for pi in pi_list}
sample_stats = {pi: {} for pi in pi_list}

for epsilon in epsilon_list:
    lambda_param = 1 / epsilon
    print(f"\n========================")
    print(f"Running for ε = {epsilon:.4f} (λ = {lambda_param:.2f})")

    for pi in pi_list:
        print(f"\n--- π = {pi} ---")

        mu_errors = []
        sample_means = []
        sample_vars = []

        for rep in range(n_rep):
            x = datasets[pi][rep]

            # Save sample stats
            sample_means.append(np.mean(x))
            sample_vars.append(np.var(x))

            param_grid = [(mu, sigma2) for mu in mu_grid for sigma2 in sigma2_grid]

            raw_results = Parallel(n_jobs=-1, verbose=0)(
                delayed(eval_lel)(mu, sigma2) for (mu, sigma2) in param_grid
            )

            filtered = [(mu, sigma2, val) for (mu, sigma2, val) in raw_results if np.isfinite(val)]
            if filtered:
                best_mu, _, _ = max(filtered, key=lambda x: x[2])
                mu_errors.append(abs(best_mu - true_mu))
            else:
                mu_errors.append(np.nan)

        error_results[pi][epsilon] = mu_errors
        sample_stats[pi][epsilon] = {"mean": sample_means, "var": sample_vars}

        print(f"Finished π = {pi:.2f}, ε = {epsilon:.2f} with {sum(np.isfinite(mu_errors))}/{n_rep} successes.")


# In[6]:


records = []
for pi in pi_list:
    for epsilon in epsilon_list:
        for val in error_results[pi][epsilon]:
            if np.isfinite(val):
                records.append({
                    "π": pi,
                    "ε": epsilon,
                    "|μ̂ - μ|": val
                })

df_errors = pd.DataFrame(records)

# Plot boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(data=df_errors, x="ε", y="|μ̂ - μ|", hue="π")
plt.title("Boxplot of |μ̂ − μ| across 50 Replications")
plt.xlabel("ε (epsilon)")
plt.ylabel("Absolute Error in μ̂")
plt.legend(title="π (contamination)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

