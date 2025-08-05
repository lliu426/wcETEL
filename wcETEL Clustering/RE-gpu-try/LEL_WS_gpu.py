import numpy as np
from wc_ETEL_gpu import wc_ETEL_gpu

def LEL_WS_gpu(X_data, mu, sigma2, lamda, device='cuda'):
    sigma = np.sqrt(sigma2)
    try:
        masses, logsum = wc_ETEL_gpu(
            theta0=mu,
            sigma0=sigma,
            lamda=lamda,
            data=X_data,
            device=device,
            verbose=False
        )
        return {
            "Optimal_Value": logsum,
            "P": masses
        }
    except Exception as e:
        return {
            "Optimal_Value": -1e10,
            "P": np.full(len(X_data), np.nan)
        }
