import numpy as np
from wc_ETEL import wc_ETEL



def LEL_WS(X_data, mu1, sigma1, lamda):
    try:
        masses, logsum = wc_ETEL(theta0=mu1, sigma0=sigma1, lamda=lamda, data=X_data)
        return {"Optimal_Value": logsum, "P": masses}
    except Exception:
        return {"Optimal_Value": -1e10, "P": np.full(len(X_data), np.nan)}
