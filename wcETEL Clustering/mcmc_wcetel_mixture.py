# ==========================
# mcmc_wcetel_mixture.py
# Fully modular MCMC for wc-ETEL Mixture Model (M2)
# ==========================

import numpy as np
from scipy.stats import norm, gamma
import arviz as az
from lel_ws_d2 import LEL_WS_d2

def single_replication_mixture(m, lamda, N, nIter, burn_in, X_data):
    np.random.seed(123 + m)

    x = X_data

    # Initialize at MLE
    mu1_Init = mu2_Init = np.mean(x)
    Sigma1_Init = Sigma2_Init = np.var(x, ddof=1)
    wt_Init = 0.5  # unused for now

    # Storage
    MH_storage = np.zeros(nIter)
    loglik_mat = np.full((nIter, N), np.nan)
    mu_Storage = np.full((nIter, 2), np.nan)

    D = 1

    for iter in range(nIter):
        # Propose new parameters
        mu1_prop = np.random.normal(mu1_Init, D * (Sigma1_Init / N)**0.5)
        mu2_prop = np.random.normal(mu2_Init, D * (Sigma2_Init / N)**0.5)

        Sigma1_prop = np.random.gamma(100 / 2, 2 * Sigma1_Init / 100)
        Sigma2_prop = np.random.gamma(100 / 2, 2 * Sigma2_Init / 100)

        wt_prop = np.random.uniform()  # unused

        # Compute likelihood
        llTemp_top = LEL_WS_d2(x, mu1_prop, Sigma1_prop, mu2_prop, Sigma2_prop, lamda)
        llTemp_bot = LEL_WS_d2(x, mu1_Init, Sigma1_Init, mu2_Init, Sigma2_Init, lamda)

        loglik_mat[iter, :] = np.log(llTemp_bot["P"])

        ll_top = llTemp_top["Optimal_Value"]
        ll_bot = llTemp_bot["Optimal_Value"]

        # Priors
        lPrior_top = (
            norm.logpdf(mu1_prop, 0, 1000)
            + gamma.logpdf(Sigma1_prop, 2, scale=1000 / 2)
            + norm.logpdf(mu2_prop, 0, 1000)
            + gamma.logpdf(Sigma2_prop, 2, scale=1000 / 2)
        )
        lPrior_bot = (
            norm.logpdf(mu1_Init, 0, 1000)
            + gamma.logpdf(Sigma1_Init, 2, scale=1000 / 2)
            + norm.logpdf(mu2_Init, 0, 1000)
            + gamma.logpdf(Sigma2_Init, 2, scale=1000 / 2)
        )

        # Proposals
        lProp_top = (
            norm.logpdf(mu1_Init, mu1_prop, D * (Sigma1_prop / N)**0.5)
            + gamma.logpdf(Sigma1_Init, 100 / 2, scale=2 * Sigma1_prop / 100)
            + norm.logpdf(mu2_Init, mu2_prop, D * (Sigma2_prop / N)**0.5)
            + gamma.logpdf(Sigma2_Init, 100 / 2, scale=2 * Sigma2_prop / 100)
        )
        lProp_bot = (
            norm.logpdf(mu1_prop, mu1_Init, D * (Sigma1_Init / N)**0.5)
            + gamma.logpdf(Sigma1_prop, 100 / 2, scale=2 * Sigma1_Init / 100)
            + norm.logpdf(mu2_prop, mu2_Init, D * (Sigma2_Init / N)**0.5)
            + gamma.logpdf(Sigma2_prop, 100 / 2, scale=2 * Sigma2_Init / 100)
        )

        # MH ratio
        top = ll_top + lPrior_top + lProp_top
        bot = ll_bot + lPrior_bot + lProp_bot
        p_accept = min(1, np.exp(top - bot))

        if np.random.uniform() < p_accept:
            mu1_Init = mu1_prop
            Sigma1_Init = Sigma1_prop
            mu2_Init = mu2_prop
            Sigma2_Init = Sigma2_prop
            wt_Init = wt_prop
            loglik_mat[iter, :] = np.log(llTemp_top["P"])

        mu_Storage[iter, :] = [mu1_Init, mu2_Init]
        MH_storage[iter] = np.sum(loglik_mat[iter, :])

    # Marginal likelihood
    max_log_like = np.max(MH_storage)
    post_prob = np.exp(MH_storage - max_log_like)
    post_prob /= np.sum(post_prob)
    M2_Marglik = max_log_like - np.log(np.sum(post_prob))

    # LOO calculation
    # loglik_post_burnin = loglik_mat[burn_in:, :]
    loglik_post_burnin = loglik_mat[burn_in:, :].reshape((1, -1, N))
    idata = az.from_dict(
        # posterior={"mu": mu_Storage[burn_in:, :]},
        posterior={"mu": mu_Storage[burn_in:, :].reshape((1, -1, 2))},
        log_likelihood={"likelihood": loglik_post_burnin},
        coords={"obs_id": np.arange(N)},
        dims={"likelihood": ["obs_id"]}
    )
    # loo_result = az.loo(idata)
    loo_result = az.loo(idata, pointwise=True)
    pareto_k = loo_result.pareto_k
    MS2_elpd = loo_result.elpd_loo
    MS2_se = loo_result.se

    return {
        'M2_Marglik': M2_Marglik,
        'MS2_elpd': MS2_elpd,
        'MS2_se': MS2_se,
        'MH_storage': MH_storage,
        'mu_Storage': mu_Storage,
        'loglik_mat': loglik_mat
    }
