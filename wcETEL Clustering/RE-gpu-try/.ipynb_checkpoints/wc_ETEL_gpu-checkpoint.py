import torch
import numpy as np
from optimal_transport_1d_gpu import OptimalTransport1D_GPU

def wc_ETEL_gpu(theta0, sigma0, lamda=5.0, data=None, verbose=False, device='cuda'):
    if data is None:
        raise ValueError("Data must be provided.")
    
    N = len(data)
    X = data
    masses = np.ones(N) / N
    masses = torch.tensor(masses, dtype=torch.float64, device=device)

    # Define ρ₀(x) = N(theta0, sigma0^2)
    def rho(x):
        return torch.exp(-0.5 * ((x - theta0) / sigma0)**2) / (
            sigma0 * torch.sqrt(torch.tensor(2 * torch.pi, dtype=torch.float64, device=x.device))
        )

    eta = 1e-9
    max_iter = 100

    for i in range(max_iter):
        ot = OptimalTransport1D_GPU(X, masses, device=device)
        ot.update_weights(rho, tol=1e-6, max_iter=20, verbose=False)
        weights = ot.weights

        # Update masses (same logic as original)
        masses1 = masses + eta * (-1 - torch.log(N * masses) - 2 * lamda * weights)

        # Projection back to simplex
        temp = torch.arange(1, N + 1, device=device, dtype=torch.float64)
        mask = (masses1 + (1 / temp) * (1 - torch.cumsum(masses1, dim=0))) > 0
        if torch.any(mask):
            max_idx = torch.max(torch.nonzero(mask)).item() + 1
        else:
            max_idx = N

        right_shift = (1 / max_idx) * (1 - torch.cumsum(masses1, dim=0)[max_idx - 1])
        masses1 = masses1 + right_shift
        masses1 = torch.clamp(masses1, min=0.0)

        err = torch.sum((masses - masses1)**2).item()
        if err <= 1e-15:
            break

        if verbose:
            print(f"Iteration {i}, Error = {err:.2e}")

        masses = masses1

    logsum = torch.sum(torch.log(masses)).item()
    return masses, logsum
