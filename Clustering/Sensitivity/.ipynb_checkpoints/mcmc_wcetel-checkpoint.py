# ==========================
# mcmc_wcetel.py
# Fully modular MCMC for wc-ETEL
# ==========================

import numpy as np
from scipy.stats import norm, gamma
import arviz as az
from lel_ws import LEL_WS
import matplotlib.pyplot as plt

def single_replication(m, lamda, N, nIter, d, burn_in, X_data):
    np.random.seed(123 + m)

    x = X_data

    # MLE starting point
    mu_MLE = np.mean(x)
    var_MLE = np.var(x, ddof=1)

    # Storage for MCMC
    loglik_mat = np.full((nIter, N), np.nan)
    MH_storage = np.full(nIter, np.nan)
    mu_Storage = np.full((nIter, 1), np.nan)
    Sigma_Storage = np.full((nIter, 1), np.nan)

    # Initialize chain at MLE
    mu_Init = mu_MLE
    Sigma_Init = var_MLE
    D = 1

    for iter in range(nIter):
        # Propose new parameters
        mu_prop = np.random.normal(mu_Init, D * (Sigma_Init / N)**0.5)
        Sigma_prop = np.random.gamma(50, 1 / (50 / var_MLE))

        # Likelihood calculations via wc_ETEL
        llTemp_top = LEL_WS(X_data=x, mu1=mu_prop, sigma1=Sigma_prop, lamda=lamda)
        llTemp_bot = LEL_WS(X_data=x, mu1=mu_Init, sigma1=Sigma_Init, lamda=lamda)

        loglik_mat[iter, :] = np.log(llTemp_bot["P"])

        ll_top = llTemp_top["Optimal_Value"]
        ll_bot = llTemp_bot["Optimal_Value"]

        # Priors
        lPrior_top = norm.logpdf(mu_prop, loc=0, scale=1000) + gamma.logpdf(Sigma_prop, 1, scale=2000)
        lPrior_bot = norm.logpdf(mu_Init, loc=0, scale=1000) + gamma.logpdf(Sigma_Init, 1, scale=2000)

        # Proposals
        lProp_top = norm.logpdf(mu_Init, loc=mu_prop, scale=D * (Sigma_prop / N)**0.5) + \
                    gamma.logpdf(Sigma_Init, 50, scale=1 / (50 / Sigma_prop))
        lProp_bot = norm.logpdf(mu_prop, loc=mu_Init, scale=D * (Sigma_Init / N)**0.5) + \
                    gamma.logpdf(Sigma_prop, 50, scale=1 / (50 / Sigma_Init))

        # MH ratio
        top = ll_top + lPrior_top + lProp_top
        bot = ll_bot + lPrior_bot + lProp_bot
        p_accept = min(1, np.exp(top - bot))

        if np.random.uniform() < p_accept:
            mu_Init = mu_prop
            Sigma_Init = Sigma_prop
            loglik_mat[iter, :] = np.log(llTemp_top["P"])

        mu_Storage[iter] = mu_Init  # Store as scalar
        Sigma_Storage[iter] = Sigma_Init

        if np.any(np.isnan(loglik_mat[iter, :])):
            print(f"NaN detected in loglik_mat at iteration {iter}")
        
        # MH_storage[iter] = np.sum(np.log(loglik_mat[iter, :]))
        MH_storage[iter] = np.sum(loglik_mat[iter, :])

    # Marginal likelihood
    max_log_like = np.max(MH_storage)
    post_prob = np.exp(MH_storage - max_log_like)
    post_prob /= np.sum(post_prob)
    M1_Marglik = max_log_like - np.log(np.sum(post_prob))

    # LOO calculation
    # loglik_post_burnin = loglik_mat[burn_in:, :]
    loglik_post_burnin = loglik_mat[burn_in:, :].reshape((1, -1, N))
    idata = az.from_dict(
        # posterior={"mu": mu_Storage[burn_in:]},
        posterior={"mu": mu_Storage[burn_in:, :].reshape((1, -1, 1))},
        log_likelihood={"likelihood": loglik_post_burnin},
        coords={"obs_id": np.arange(N)},
        dims={"likelihood": ["obs_id"]}
    )
    # loo_result = az.loo(idata)
    loo_result = az.loo(idata, pointwise=True)
    pareto_k = loo_result.pareto_k
    MS1_elpd = loo_result.elpd_loo
    MS1_se = loo_result.se
    
    # Quick diagnostics:
    # print("Max Pareto k:", np.max(pareto_k))
    # print("Num of points with k > 0.7:", np.sum(pareto_k > 0.7))
    # print("Num of points with k > 1.0:", np.sum(pareto_k > 1.0))

    # Plot Pareto-k after diagnostics:
    # az.plot_khat(loo_result)
    # plt.show()

    return {
        'M1_Marglik': M1_Marglik,
        'MS1_elpd': MS1_elpd,
        'MS1_se': MS1_se,
        'MH_storage': MH_storage,
        'mu_Storage': mu_Storage,
        'Sigma_Storage': Sigma_Storage,
        'loglik_mat': loglik_mat
    }