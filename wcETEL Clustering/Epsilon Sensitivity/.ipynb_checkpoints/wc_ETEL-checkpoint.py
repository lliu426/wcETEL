import numpy as np
import matplotlib.pyplot as plt
from optimal_transport_1d import OptimalTransport1D
import scipy.stats as sps



def wc_ETEL(theta0, sigma0, lamda=5, verbose=False, data=None): 
    if data is None:
        raise ValueError("Data must be provided.")
    
    N = len(data)
    masses = np.ones(N) / N

    rho0 = lambda x: sps.norm.pdf(x, loc=theta0, scale=sigma0)

    eta = 1e-9
    maxiter = 100

    for i in range(maxiter): 
        ot = OptimalTransport1D(data, masses, rho0, L=float("inf"), S=float("-inf"))
        ot.update_weights(maxIter=1, verbose=False)
        ww = ot.weights

        masses1 = masses + eta * (-1 - np.log(N * masses) - 2 * lamda * ww)

        temp = np.max(np.where((masses1 + 1 / (np.arange(1, N + 1)) * (1 - np.cumsum(masses1))) > 0)) + 1
        right_shift = (1 / temp) * (1 - np.cumsum(masses1)[temp - 1])
        masses1 = masses1 + right_shift
        masses1[masses1 < 0] = 0

        err = np.sum((masses - masses1) ** 2)
        if err <= 1e-15:
            break

        if verbose:
            print(f"Iteration {i}: Error = {err}")

        masses = masses1

    return masses, np.sum(np.log(masses))