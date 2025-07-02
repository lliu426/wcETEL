import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import beta
from optimal_transport_1d import OptimalTransport1D



def run_wcETEL_analysis(X):
    L = 1.
    N = len(X)
    masses = np.ones(N) / N
    rho0 = lambda x: 6 * x * (1 - x)

    mass = 1.
    masses = masses * mass

    eta = 1e-3
    maxiter = 3000

    for i in range(maxiter):
        ot = OptimalTransport1D(X, masses, rho0, L=L)
        ot.update_weights(maxIter=1, verbose=False)
        ww = ot.weights

        masses1 = masses + eta * (-1 - np.log(N * masses) - 2 * 5 * ww)
        temp = np.max(np.where((masses1 + 1 / (np.arange(1, N + 1)) * (1 - np.cumsum(masses1)) > 0))) + 1
        right_shift = (1 / temp) * (1 - np.cumsum(masses1)[temp - 1])
        masses1 = masses1 + right_shift
        masses1[masses1 < 0] = 0

        err = np.sum((masses - masses1) ** 2)
        if err <= 1e-20:
            break

        masses = masses1

    print(f"Final error after loop: {err:.2e}")

    return ot.weights, masses1