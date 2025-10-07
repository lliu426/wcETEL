import numpy as np
from scipy.integrate import quad
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import time
from scipy.stats import beta
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import sys
import subprocess
import time
import os
import pickle
from IPython.display import Image, display
import glob
###########################################################

###########################################################
N = 300
noise_factor = 0
n = int(N*noise_factor)
n_datasets = 50

datasets = []
np.random.seed(0)
for i in range(n_datasets):
    sig = np.random.normal(0, 1, size = N - n)
    noi_1 = np.random.normal(-10,1, size = n)
#     noi_2 = np.random.normal(-5,1, size = n)
    X = np.sort(np.concatenate([sig, noi_1]))
    datasets.append(X)
    

###########################################################
class PowerDiagram1D:
    def __init__(self, X, weights=None, L_lb = None, L_ub = None):
        """
        Parameters
        ----------
        X : ndarray,
            positions of particles (assumed ordered with no repeats)
        weights : ndarray, optional,
            weights of Laguerre cells
        L_lb: Domain lower bound (float or -np.inf)
        L_ub: Domain upper bound (float or +np.inf)
        L_lb and L_ub needs to be provided and must satisfy L_lb < L_ub
        """
        self.X = np.array(X, copy=True)
        self.n = len(X)
        self.weights = np.zeros(self.n) if weights is None else np.array(weights, copy=True)
        self.L_lb = L_lb
        self.L_ub = L_ub
        self.Bounds = np.empty(self.n)
        self.indices = []
        self.updated_flag = False

    def set_positions(self, X):
        self.X = np.array(X, copy=True)

    def set_weights(self, weights):
        self.weights = np.array(weights, copy=True)
        self.updated_flag = False

    def update_boundaries(self):
        u = (self.X**2 - self.weights) / 2
        def slope(i, j):
            return (u[j] - u[i]) / (self.X[j] - self.X[i])
        indices = []
        indices.append(self.L_lb)
        for i in range((self.n - 1)):
            boundary = slope(i, i+1)
            indices.append(boundary)
        indices.append(self.L_ub)
        for i in indices:
            if (i < self.L_lb) or (i > self.L_ub):
                indices.remove(i)
        self.indices = np.array(range(self.n))
        self.Bounds = np.array(sorted(indices)) 
        self.updated_flag = True

    def compute_integrals(self, fun):
        if not self.updated_flag:
            self.update_boundaries()
        N = len(self.Bounds) - 1
        integrals = np.zeros(N)
        for i in range(N):
            integrals[i], _ = quad(fun, self.Bounds[i], self.Bounds[i + 1])
        return integrals

    def compute_integrals_ipp(self, intp_fun, p=None):
        if p is None:
            p = len(intp_fun) - 1
        else:
            assert len(intp_fun) >= p + 1
        if p == 0:
            return intp_fun[0](self.Bounds[1:]) - intp_fun[0](self.Bounds[:-1])
        integrals_p = ((self.Bounds[1:] - self.X[self.indices])**p * intp_fun[0](self.Bounds[1:]) -
                       (self.Bounds[:-1] - self.X[self.indices])**p * intp_fun[0](self.Bounds[:-1]))
        integrals = -p * self.compute_integrals_ipp(intp_fun[1:], p=p - 1) + integrals_p
        return integrals

    def compute_integrals_gradient(self, fun):
        if not self.updated_flag:
            self.update_boundaries()
        Xact = self.X[self.indices]
        feval = fun(self.Bounds[1:-1])
        feval = feval * (1 - 1e-2) + 1e-2
        vect = 0.5 * feval / np.abs(Xact[1:] - Xact[:-1])
        vect0 = vect.copy()
        vect0[0] = 0
        costhess = diags(vect, -1) + diags(vect0, 1)
        vect1 = np.array(costhess.sum(axis=1)).flatten()
        vect1[0] = -1
        costhess -= diags(vect1)
        return -costhess

################################################################    
class OptimalTransport1D(PowerDiagram1D):
    def __init__(self, X, masses, rho, intp_rho=None, L_lb = None, L_ub = None):
        super().__init__(X, L_lb=L_lb, L_ub=L_ub)
        self.masses = np.array(masses, copy=True)
        self.rho = rho
        self.intp_rho = intp_rho

    def compute_ot_cost(self):
        N = len(self.Bounds) - 1
        integrals = np.zeros(N)
        for i in range(N):
            fun = lambda x: ((x - self.X[self.indices][i])**2 - self.weights[self.indices][i]) * self.rho(x)
            integrals[i], _ = quad(fun, self.Bounds[i], self.Bounds[i + 1])
        return np.sum(integrals) + np.sum(self.masses * self.weights)

    def compute_ot_cost_ipp(self):
        term1 = np.sum(self.compute_integrals_ipp(self.intp_rho, p=2))
        term2 = np.sum(self.compute_integrals_ipp(self.intp_rho, p=0) * self.weights[self.indices])
        return np.sum(self.masses * self.weights) + (term1 - term2)

    def update_weights(self, tol=1e-6, maxIter=500, verbose=False):
        alphaA = 0.01
        tau_init = 0.5
        max_line_search_trials = 3
        self.update_boundaries()
        F = -self.masses.copy()
        if self.intp_rho is None:
            F[self.indices] += self.compute_integrals(self.rho)
            cost_old = self.compute_ot_cost()
        else:
            F[self.indices] += self.compute_integrals_ipp(self.intp_rho, p=0)
            cost_old = self.compute_ot_cost_ipp()
        error = np.linalg.norm(F)
        i = 0
        while error > tol and i < maxIter:
            Hess = self.compute_integrals_gradient(self.rho)
            theta = 0.0
            deltaw = -theta * F
            deltaw[self.indices] -= (1 - theta) * spsolve(Hess, F[self.indices])
            weights_old = self.weights.copy()
            tau = tau_init
            trial = 0
            while trial < max_line_search_trials:
                self.weights = weights_old + tau * deltaw
                self.update_boundaries()
                if self.intp_rho is None:
                    cost = self.compute_ot_cost()
                else:
                    cost = self.compute_ot_cost_ipp()

                if (cost >= cost_old + tau * alphaA * np.dot(F, deltaw)
                        and len(self.indices) == len(self.X)):
                    break
                else:
                    tau *= 0.8
                    trial += 1
            cost_old = cost
            i += 1
            F = -self.masses.copy()
            if self.intp_rho is None:
                F[self.indices] += self.compute_integrals(self.rho)
            else:
                F[self.indices] += self.compute_integrals_ipp(self.intp_rho, p=0)
            error = np.linalg.norm(F)
            if verbose:
                print(f"Newton step {i}, cost: {cost:.6f}, tau: {tau:.2e}, error: {error:.2e}")
            tau_init = min(tau * 1.1, 1.0)
        if i < maxIter and verbose:
            print("Optimization success!")
            
##############################################################

def run_wcETEL_analysis(X):
    L_lb = -np.inf
    L_ub = np.inf
    N = len(X)
    masses = np.ones(N) / N
    rho0 = lambda x: (1/(2*np.pi)**0.5)*(np.exp(-(x**2)/2))
    mass = 1.0
    masses = masses * mass

    eta = 1e-3
    maxiter = 1
    ot = OptimalTransport1D(X, masses, rho0, L_lb=L_lb, L_ub=L_ub)
    ot.update_weights(maxIter=1, verbose=False)
    ww = ot.weights
#     print("weights after OT:", ww)
    for i in range(maxiter):
        print("starting iter:", i)
        
#         print("intial wi's:", masses)
        masses1 = masses + eta * (-1 - np.log(N * masses) - 2 * 5 * ww)
        temp = np.max(np.where((masses1 + 1 / (np.arange(1, N + 1)) * (1 - np.cumsum(masses1)) > 0))) + 1
        right_shift = (1 / temp) * (1 - np.cumsum(masses1)[temp - 1])
#         print("right shift =", right_shift)
        masses1 = masses1 + right_shift
        masses1[masses1 < 0] = 0

        err = np.sum((masses - masses1) ** 2)
        if err <= 1e-20:
            break

        masses = masses1
#         print("final wi's:", masses)
#         print("------------------------------------------------------------------------------")

    print(f"Final error after loop: {err:.2e}")

    return ot.weights, masses1

###################################################################

os.makedirs("wcETEL_results_gen", exist_ok=True)

def run_and_save(index_X):
    index, X = index_X
    print(f"Running dataset: {index}")
    weights, masses1 = run_wcETEL_analysis(X)

    with open(f"wcETEL_results_gen/dataset_{index}_weights.pkl", "wb") as f:
        pickle.dump({"weights": weights, "masses1": masses1}, f)

if __name__ == "__main__":
    data_list = list(enumerate(datasets))
    
    max_workers = min(4, multiprocessing.cpu_count())
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(run_and_save, data_list)