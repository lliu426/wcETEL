import numpy as np
from functools import partial
from scipy.stats import norm
from scipy.stats import beta
from scipy.stats import multivariate_normal

#For each parametric model that you consider, you will have to supply a function that collects simulations. For standard distributions like the normal
#scipy.stats.norm is sufficient. For more complex parametric models, you might need to write your own sampler. An toy example is given
#below: writing a simulator for a parametric model that is a snapshot of an AR1 of length d.

def AR1(phi,d,size):
    eps = np.random.randn(size,d)
    X = np.zeros((size, d))
    for j in range(0, d):
        if j == 0:
            X[:,j] = eps[:,j]
        else:
            X[:, j] = phi * X[:, j-1] + eps[:, j]
    return X

#likelihoodCalculator: Stochastic Gradient algorithm to approx solve argmin_{w} sum w_i log(w_i)+lambda * W_2^2( P(X,w), F_{theta})
#where P(X,w) is the discrete distribution with sample locations X = (X_1,...,X_n) and where F_{theta} is some param model.
#F_{theta}, the source probability distribution, is assumed to be accessible via simulation.
#This function returns the vector of weights. Arguments are given below.

#X: This is an n x d matrix of samples
#distr: A string indicating the source probability distribution. For given distr, you need to fill in the sampler.
#g0: The initial starting location for the gradient ascent
#eps: parameter of entropic OT 
#C: the initial learning rate.
#optional_arguments: The parameters of the source probability distribution
#max_iter: The number of iterations before returning.
#a: The learning rate dampening exponent of Genans (To user: For now Ignore and leave as None)
#b: The epsilon dampening exponent of Genans (To user: For now Ignore and leave as None)
def likelihoodCalculator(X,distr,eps,C,optional_arguments,num_iter = 1000,g0 = None, lam = 0,a = None,b = None):
    N = np.shape(X)[0]
    if g0 is None:
        g0 = np.zeros(N-1)
    
    if distr == "normal_1d":
        mn = optional_arguments[0]
        var = optional_arguments[1]
        sampler = partial(norm.rvs,loc=mn,scale=np.sqrt(var))
    elif distr == "beta":
        alph = optional_arguments[0]
        bet = optional_arguments[1]
        sampler = partial(beta.rvs,a=alph,b=bet)
    elif distr == "normal_highD":
        mn = optional_arguments[0]
        cov = optional_arguments[1]
        sampler = partial(multivariate_normal.rvs,mean = mn,cov=cov)
    elif distr == "AR1":
        phi = optional_arguments[0]
        d = optional_arguments[1]
        sampler = partial(AR1,phi=phi,d=d)



    #Draw the samples used during the burn-in period
    gTilde = g0
    g = gTilde
    Xprime = X[0:-1,]
    Xlast = X[-1,]

    sampleBatch = sampler(size = num_iter)
    k=1

    #If method of Genans, set starting epsilon.
    if a:
        if lam < 1:
            coeff = 1
        else:
            coeff = (1/lam)*(1/2)
        eps = coeff

    for i in np.arange(1,num_iter+1):

        #If method of Genans,
        if b:
            learningRate = C*(i**(-b))
        else:
            learningRate = C/np.sqrt(i)

        samp = sampleBatch[i-1,]
        logExponents = (-np.sum((Xprime-samp)**2,axis=1)+gTilde)/eps
        largest = max(logExponents)
        lastLogExponent = (-np.sum((Xlast-samp)**2))/eps
        if lastLogExponent > largest:
            largest = lastLogExponent
        logExponents = logExponents - largest
        lastLogExponent = lastLogExponent - largest
        exponents = np.exp(logExponents)
        lastExponent = np.exp(lastLogExponent)
        total = np.sum(exponents)+lastExponent
        Z = np.exp(-(lam/(1-lam*eps))*gTilde)
        Z = Z/(1+np.sum(Z))
        grad = lam*(Z - exponents / total)

        # newReweight = (np.log(i+1))**2
        # reWeightTotal = reWeightTotal+newReweight
        # newProp = newReweight/reWeightTotal
        # oldProp = 1-newProp

        gTilde = gTilde+(learningRate)*grad
        g = (1/(i+1))*gTilde+(i/(i+1))*g
        #g = newProp*gTilde+oldProp*g

        #If the method of Genans, then
        if a:
            eps = coeff*(i**(-a))

    lastWeight = 1-np.sum(Z)
    finalWeights = np.concatenate((Z,np.array([lastWeight])))

    return g,finalWeights,k
