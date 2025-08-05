import torch

class OptimalTransport1D_GPU:
    def __init__(self, X, masses, weights=None, device='cuda'):
        self.device = device
        self.X = torch.tensor(X, dtype=torch.float64, device=self.device)
        self.n = len(X)
        self.masses = torch.tensor(masses, dtype=torch.float64, device=self.device)
        self.weights = torch.zeros(self.n, dtype=torch.float64, device=self.device) if weights is None \
            else torch.tensor(weights, dtype=torch.float64, device=self.device)
        self.Bounds = None
        self.indices = None

    def update_boundaries(self):
        u = (self.X**2 - self.weights) / 2
        indices = [0, 1]

        def slope(i, j):
            return (u[j] - u[i]) / (self.X[j] - self.X[i])

        for i in range(2, self.n):
            while len(indices) >= 2 and slope(i, indices[-1]) <= slope(indices[-1], indices[-2]):
                indices.pop()
            indices.append(i)

        self.indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        self.cell_indices = self.indices[:-1]  # this matches number of integrals


        u_sel = u[self.indices]
        X_sel = self.X[self.indices]
        Bounds = (u_sel[1:] - u_sel[:-1]) / (X_sel[1:] - X_sel[:-1])

        self.Bounds = torch.cat([
            torch.tensor([-10.0], device=self.device),
            Bounds,
            torch.tensor([10.0], device=self.device)
        ])

    

    def compute_integrals(self, rho, num_points=100):
        if self.Bounds is None or self.indices is None:
            self.update_boundaries()

        N = len(self.cell_indices)
        integrals = torch.zeros(N, dtype=torch.float64, device=self.device)

        for i in range(N):
            a = self.Bounds[i]
            b = self.Bounds[i + 1]

            grid = torch.linspace(a, b, num_points, device=self.device)
            fx = rho(grid)

            integrals[i] = torch.trapz(fx, grid)

        return integrals




    def compute_ot_cost(self, rho, num_points=100):
        """
        GPU version of OT cost:
        ∑ mass_i * weight_i + ∑ ∫ [(x - x_i)^2 - w_i] * rho(x) dx over Laguerre cells
        """
        if self.Bounds is None or self.indices is None:
            self.update_boundaries()

        total_cost = torch.sum(self.masses * self.weights)

        for i in range(len(self.Bounds) - 1):
            a = self.Bounds[i]
            b = self.Bounds[i + 1]
            xi = self.X[self.indices][i]
            wi = self.weights[self.indices][i]

            grid = torch.linspace(a, b, num_points, device=self.device)
            integrand = ((grid - xi)**2 - wi) * rho(grid)
            integral = torch.trapz(integrand, grid)
            total_cost += integral

        return total_cost

    

    def update_weights(self, rho, tol=1e-6, max_iter=100, verbose=False):
        alphaA = 0.01
        tau_init = 0.5
        max_line_search_trials = 5

        self.update_boundaries()
        indices = self.indices
        masses = self.masses
        X = self.X
        n = len(indices)

        for it in range(max_iter):
            self.update_boundaries()
            integrals = self.compute_integrals(rho)

            # Gradient: F = -masses; then add ∫ rho over each cell
            F = -masses.clone()
            # print(f"indices len: {len(self.indices)}")
            # print(f"cell_indices len: {len(self.cell_indices)}")
            # print(f"integrals len: {len(integrals)}")
            F = F.index_add(0, self.cell_indices, integrals)

            error = torch.norm(F).item()
            if verbose:
                print(f"[{it}] Error = {error:.2e}")

            if error < tol:
                if verbose:
                    print("Converged.")
                break

            # Approximate Hessian as identity for now (simplified Newton)
            delta_w = torch.zeros_like(self.weights)
            delta_w[self.cell_indices] = -F[self.cell_indices]

            # Line search
            tau = tau_init
            weights_old = self.weights.clone()
            cost_old = self.compute_ot_cost(rho)

            for trial in range(max_line_search_trials):
                self.weights = weights_old + tau * delta_w
                self.update_boundaries()
                cost_new = self.compute_ot_cost(rho)
                lhs = cost_new
                rhs = cost_old + alphaA * tau * torch.dot(F, delta_w)

                if lhs >= rhs and len(self.indices) == len(self.X):
                    break
                tau *= 0.8

            if verbose:
                print(f"Step size: {tau:.2e}, New cost: {cost_new.item():.6f}")
