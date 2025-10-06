from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.domain_types import ScaledImage
import numpy as np
from pysdot import PowerDiagram
from scipy.sparse import csr_matrix
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from numpy.linalg import cond
from scipy.stats import beta
from scipy.stats import norm
from functools import partial
import matplotlib.pyplot as plt
from IPython.display import display
from new_display_jupyter import new_display_jupyter
PowerDiagram.display_jupyter = new_display_jupyter



# constructs a uniform probability density on [0,1] x [0,1]
#ConvexPolyhedra is a density object.
def make_square(box=[0,0,1,1]):
    domain = ConvexPolyhedraAssembly()
    domain.add_box([box[0], box[1]], [box[2], box[3]])
    return domain



# constructs a square domain, to be passed to the laguerre_* functions
#Note: if you pass in a proper histogram already, division by the mean won't do anything.
#division by the mean just converts improper histgorams to probability measures
def make_image_CPA(img,box=[0,0,1,1],display=False):
    img = img / ((box[2]-box[0])*(box[3]-box[1])*np.mean(img))
    if display:
        plt.imshow(img,cmap='gray',extent=[box[0],box[2],box[1],box[3]],origin='lower')
        plt.colorbar(label='Intensity')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    domain = ConvexPolyhedraAssembly()
    domain.add_img([box[0], box[1]], [box[2], box[3]], img)
    return domain



#Note: if you pass in a proper histogram already, division by the sample mean won't do anything.
#division by the  mean just converts improper histgorams to probability measures
def make_image_SI(img, box=[0, 0, 1, 1],display = False):
    img = img / ((box[2] - box[0]) * (box[3] - box[1]) * np.mean(img))
    if display:
        plt.imshow(img,cmap='gray',extent=[box[0],box[2],box[1],box[3]],origin='lower')
        plt.colorbar(label='Intensity')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    return ScaledImage([box[0], box[1]], [box[2], box[3]], img)



#frequency determines the number of bins per axis.
#alpha1,beta1 : the alpha, beta params for the beta density on the x-axis
#alpha2,beta2 : along the y-axis
def make_product_beta_density(alpha1,beta1,alpha2,beta2,frequency,display=False):
    t = np.linspace(0,1,frequency)
    x,y = np.meshgrid(t,t)
    img = np.zeros(shape = (frequency-1,frequency-1))
    for i in np.arange(frequency-1):
        for j in np.arange(frequency-1):
                xLow = x[i,j]
                xHigh = x[i+1,j+1]
                yLow = y[i,j]
                yHigh = y[i+1,j+1]
                img[i,j] = (beta.cdf(xHigh,alpha1,beta1)-beta.cdf(xLow,alpha1,beta1))*(beta.cdf(yHigh,alpha2,beta2)-beta.cdf(yLow,alpha2,beta2))
    img = ((frequency-1)**2)*img
    return make_image_SI(img,display=display)



# computes the areas of the Laguerre cells intersected with the domain, and returns it as an array
# if der = True, also returns a sparse matrix representing the Hessian of the kantorovich potential function H(psi) = int psi^c + inner(g,w)
#domain : the density (a convexPolyhedra or ScaledImage)
#Y: the data. (should be in sorted order already if d=1)
#psi: initial starting position
def laguerre_areas(domain, Y, psi, der=False):
    pd = PowerDiagram(Y, -psi, domain)
    if der:
        N = len(psi)
        mvs = pd.der_integrals_wrt_weights()
        return mvs.v_values, csr_matrix((-mvs.m_values, mvs.m_columns, mvs.m_offsets), shape=(N,N))
    else:
        return pd.integrals()



#domain: this is the density (convexPolyhedra or scaledImage).
#Y: the data
#nu: The masses
#verbose: Make true to get iteration print outs
#maxerr: When the gradient norm falls below maxerr, the second order ascent stops
#maxiter: After this many iterations of ascent, stop. 
#learningRate: Prior to backtracking on a given step, this is the initial learning rate
#illConditionThresh: Backtracking is done until the hessian is sufficiently far from being non-invertible. This number is threshold on the ratio of max singular value to min singular value to make sure this is the case.
#method: currently only second order (hessian based), is implemented
#maxBacktracks: Backtracking will stop after this number of iterations. If this happens, its generally bad ( and a warning will be issued)
#beta: The multiplier during backtracking
def optimal_transport(domain, Y, nu, psi0=None, verbose=False, maxerr=1e-6, maxiter=1000,learningRate = 1.0,illConditionThresh = 10e5,method="secondOrder",maxBacktracks = 100,beta=3/4):
    if psi0 is None:
        psi0 = np.zeros(len(nu))
        
    def F(psip):
        g,h = laguerre_areas(domain, Y, np.hstack((psip,0)), der=True)
        return g[0:-1], h[0:-1,0:-1]
    
    psip = psi0[0:-1] - psi0[-1]
    nup = nu[0:-1]
    g,h = F(psip)
    for it in range(maxiter):
        err = np.linalg.norm(nup - g)
        # if verbose:
            # print("it %d: |err| = %g" % (it, err))
            # print("The condition number of the matrix is ")
            # print(cond(h.toarray()))
        if err <= maxerr:
            if verbose: print("        finished OT at it = "+str(it))
            break
        if method == "secondOrder":
            d = spsolve(h, nup - g)
        else:
            d = nup - g
        t = learningRate
        psip0 = psip.copy()
        j=0
        while j < maxBacktracks:
            psip = psip0 + t*d
            try:
                g,h = F(psip)
                #compute the condition number of h
                condVal = cond(h.toarray())
                #print("The condition number is "+str(condVal))
            except ValueError:
                t = beta*t
            if np.min(g) > 0 and condVal < illConditionThresh:
                if verbose: print("            Finished line search at j = "+str(j))
                break
            else:
                t = beta*t
                j=j+1
        if j == maxBacktracks:
            print("Optimal transport backtracking is stuck on an ill conditioned location")
            return ["NA","NA"]
    return np.hstack((psip,0))



#Self-explanatory.
def computeW2_squared(Y,masses,density,plotPowerDiagram = False):
    g = optimal_transport(density,Y,masses)
    if g[0] == "NA":
        print("WARNING: optimal tranport calculation became stuck on an ill conditioned location")
        return np.inf
    pd = PowerDiagram(Y,-g,density)
    if plotPowerDiagram:
        toDisplay = pd.display_jupyter(disp_centroids = True,disp_positions = True,disp_ids = False,disp_arrows = True,title="Power Diagram at Solution")
        display(toDisplay)
    return (np.sum(pd.second_order_moments()))


