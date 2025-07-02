import numpy as np
from scipy.integrate import quad
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


class PowerDiagram1D:
    def __init__(self, X, weights=None, L=1.0, S=0.0):
        self.X = np.array(X, copy=True)
        self.n = len(X)
        self.weights = np.zeros(self.n) if weights is None else np.array(weights, copy=True)
        self.L = L
        self.S = S
        self.Bounds = np.empty(self.n)
        self.indices = []
        self.updated_flag = False

    def set_positions(self, X):
        self.X = np.array(X, copy=True)

    def set_weights(self, weights):
        self.weights = np.array(weights, copy=True)
        self.updated_flag = False

    def update_boundaries(self):
        indices = [0, 1]
        u = (self.X**2 - self.weights) / 2

        def slope(i, j):
            return (u[j] - u[i]) / (self.X[j] - self.X[i])

        for i in range(2, self.n):
            while len(indices) >= 2 and slope(i, indices[-1]) <= slope(indices[-1], indices[-2]):
                indices.pop()
            indices.append(i)

        Bounds_noconstr = (u[indices][1:] - u[indices][:-1]) / (self.X[indices][1:] - self.X[indices][:-1])
        i0 = np.sum(Bounds_noconstr <= self.S)
        iend = np.sum(Bounds_noconstr >= (self.L + self.S))
        indices = indices[i0:len(indices) - iend]

        self.indices = indices
        self.Bounds = np.zeros(len(indices) + 1)
        self.Bounds[-1] = self.L + self.S
        self.Bounds[1:-1] = Bounds_noconstr[i0:len(Bounds_noconstr) - iend]
        self.updated_flag = True

    def compute_energy(self, fun=None):
        fplus = (self.Bounds[1:] - self.X[self.indices])**3 / 3
        fminus = (self.Bounds[:-1] - self.X[self.indices])**3 / 3
        return np.sum(fplus - fminus)

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


class OptimalTransport1D(PowerDiagram1D):
    def __init__(self, X, masses, rho, intp_rho=None, L=1.0, S=0.0):
        super().__init__(X, L=L, S=S)
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
