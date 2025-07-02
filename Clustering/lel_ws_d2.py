import numpy as np
from wc_ETEL_GM import wc_ETEL_GM



def LEL_WS_d2(X_data, mu1, sigma1, mu2, sigma2, lamda):
    try:
        masses, logsum = wc_ETEL_GM(theta1=mu1, sigma1=sigma1,
                                     theta2=mu2, sigma2=sigma2,
                                     lamda=lamda, data=X_data)
        if not np.isfinite(logsum):
            raise ValueError("logsum returned Inf or NaN")
        return {"Optimal_Value": logsum, "P": masses}
    except Exception:
        return {"Optimal_Value": -1e10, "P": np.full(len(X_data), np.nan)}
