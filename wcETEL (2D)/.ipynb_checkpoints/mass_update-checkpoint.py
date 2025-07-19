from optimal_transport import OptimalTransport2D
import numpy as np



def run_mass_update_for_lambda(lam, X_data):
    N = len(X_data)
    masses = np.ones(N) / N
    rho = lambda x, y: 1.0
    L = 1.0
    mass = 1.0
    masses *= mass
    eta = 1e-3
    tol = 1e-20
    maxiter = 3000

    for i in range(maxiter):
        ot = OptimalTransport2D(X_data, masses, rho, L=L)
        ot.update_weights(maxIter=1, verbose=False)
        ww = ot.weights

        masses1 = masses + eta * (-1 - np.log(N * masses) - 2 * lam * ww)

        temp = np.max(np.where((masses1 + 1 / (np.arange(1, N + 1)) * (1 - np.cumsum(masses1)) > 0))) + 1
        right_shift = (1 / temp) * (1 - np.cumsum(masses1)[temp - 1])
        masses1 = masses1 + right_shift
        masses1[masses1 < 0] = 0

        err = np.sum((masses - masses1) ** 2)
        if err <= tol:
            break

        masses = masses1

    return lam, masses, ot.weights, err, i+1