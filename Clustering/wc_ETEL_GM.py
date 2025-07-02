import numpy as np
import scipy.stats as sps
from optimal_transport_1d import OptimalTransport1D



def wc_ETEL_GM(theta1, sigma1, theta2, sigma2, lamda = 5, verbose = False, data = None): 

    if data is None:
        raise ValueError("Data must be provided.")
    
    a = 0.5
    b = 0.5
    N = len(data)
    #L = 1.
    masses = np.ones(N)
    masses = masses/np.sum(masses)
    #rho0 = lambda x: (gamma(alpha+beta)/(gamma(alpha)*gamma(beta)))*(x**(alpha-1))*((1-x)**(beta-1))
    rho0 = lambda x: a*sps.norm.pdf(x, loc = theta1, scale = sigma1) + b*sps.norm.pdf(x, loc = theta2, scale = sigma2)

    mass = 1.
    masses = masses*mass
  
    X = data
    eta = 1e-9
    n = len(X)
    maxiter = 100
    for i in range(maxiter): 
        ot = OptimalTransport1D(X,masses,rho0, L=float("inf"), S=float("-inf"))
        ot.update_weights(maxIter=1,verbose =False)
        ww=ot.weights
        masses1 = masses + eta*(-1-np.log(n*masses) - 2*lamda*ww)
        temp = np.max(np.where((masses1+1/(np.arange(1, len(masses1)+1))*(1-np.cumsum(masses1)) > 0))) + 1
        right_shift = (1/temp)*(1-np.cumsum(masses1)[temp-1])
        masses1 = masses1 + right_shift
        masses1[masses1 < 0] = 0
        err = np.sum((masses-masses1)**2)
        if err <= 1e-15: 
            break
        ## print iteration and error
        if verbose == True: 
            print(i)
            print(err)
        masses = masses1
    return masses, np.sum(np.log(masses))