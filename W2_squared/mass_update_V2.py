import optimal_transport_V2
import numpy as np
import matplotlib.pyplot as plt


#Note this assumes masses is just the first n-1 masses
def evalDescentFunction(masses,g,lam,L,lamMethod):
    curTot = np.sum(masses)
    leftover = 1-curTot
    if lamMethod == "no_KL":
        return np.sum(masses*g)
    elif lamMethod == "absolute":
        return ((np.sum(masses*np.log(masses))+leftover*np.log(leftover))+lam*(np.sum(masses*g)))

#This is just checking that the first n-1 components of the masses can be prob simplex satisfactory
def notInSimplex(masses,L):
    return (np.max(masses) > 1-L or np.min(masses) < L or np.sum(masses) > 1)


#g: the current potential
#masses is in the probability simplex
#lam: lambda
#beta: backtracking multiplier
#eta: initial learning rate before backtracking
#lamMethod:
    #absolute: Performs the optimization corresponding to H(w) = F(w)+lambda ( \int g^c + inner(g,w)))
    #no_KL: Performs the optimization corersponding to H(w) = ( \int g^c + inner(g,w)))
#IMPORTANT note: no_KL is currently not working.
def backtrackingDescent(g,masses,lam,beta,L,maxIter,N,eta,lamMethod):
    gamma = eta
    curPos = masses[0:-1]
    curG = g[0:-1]
    curTot = np.sum(curPos)
    if lamMethod == "no_KL":
        grad = curG
    elif lamMethod == "absolute":
        grad = np.log(N*(curPos+L))-np.log(N*(1-curTot+L))+lam*curG
    nextPos = curPos - gamma*grad
    i = 0
    while notInSimplex(nextPos,L) and i < maxIter:
        gamma = gamma*beta
        nextPos = curPos-gamma*grad
        i = i+1
    if i == maxIter:
        print("WARNING: The gradient descent step can't move to a point in the probability simplex")
        print(curPos)
        print(grad)
        return 0
    i = 0
    gradNorm = np.sum((grad) ** 2)
    curVal = evalDescentFunction(curPos,curG,lam,L,lamMethod)
    nextVal = evalDescentFunction(nextPos,curG,lam,L,lamMethod)
    while ( ((gamma/2)*gradNorm > curVal - nextVal) and i < maxIter ):
        gamma = gamma*beta
        nextPos = curPos - gamma*grad
        nextVal = evalDescentFunction(nextPos,curG,lam,L,lamMethod)
        i=i+1
    if i == maxIter:
        print("Error: The gradient descent step can't identify a lower cost next move")
    totMass = np.sum(nextPos)
    lastVal = 1-totMass
    return np.append(nextPos,lastVal)

#X_data: the data
#massInit: initial probability masses for each location
#twoDimDensity: (If d=1, set to None). If d=2, Either a Scaled_Image or Convex_Polyhedra object. See class make_square, make_image_CPA, and make_image_SI for ways to initalize a density
#eta: This is the initial learning rate used for a step of descent, and a step of ascent.
#lam: The value of lambda (not relevant when lamMethod = no_KL)
#verbose: set to true for debugging
#gMethod: ****Currently only "secondOrder" is implemented.**** The method, "secondOrder", or "firstOrder", determines whether ascent steps are based on the hessian or not. 
#numGstepsPerIter: How many iterations of gradient ascent should be run during a single ascent step. Defaults to 1, correspond to true 1 step at a time.
#maxiter: How many total alternating iterations of ascent/descent to run IF THE STOPPING CRITERIA IS NEVER MET
#tol: The stopping tolerance. After each full iteration, check whether squared norm difference between probability masses is below tol. If so, stop.
#lamMethod: 
    #absolute: Performs the optimization corresponding to H(w) = F(w)+lambda ( \int g^c + inner(g,w)))
    #no_KL: Performs the optimization corersponding to H(w) = ( \int g^c + inner(g,w)))
        #Important note: no_KL not working now.
#illConditionThresh: Compare ratio of maximum over minimum singular values of Hessian to determine closeness to non-invertability. Backstepping of ascent will
    #ensure that this ratio is sufficiently low before choosing a new value for g.
#distr, oneDimBounds, optional_arguments: These are your arguments for 1 dimensional density initlaization.
def run_mass_update(X_data,massInit,twoDimDensity,eta,lam = 1,verbose = False,gMethod = "secondOrder", numGstepsPerIter = 1,maxiter = 10000,tol = 1e-10, lamMethod = "absolute",illConditionThresh = 10e5,dimension=2,distr=None,oneDimBounds=None,optional_arguments=None):
    masses = massInit
    if dimension == 1:
            #optimal_transport expects that Y will be in sorted order when d=1
            sortedOrder = np.argsort(X_data)
            X_data = X_data[sortedOrder]
            masses = massInit[sortedOrder]
    #Smallest number in python
    L = np.finfo(np.float64).tiny

    N = len(X_data)
    beta = 3/4 #backtracking multiplier for the descent and ascent backtracking.
    maxIterBacktracking = 5000 #max number of backtracks

    errors = np.zeros(maxiter)

    #An initial potential vector
    negativePotential = np.zeros(N)

    for i in range(maxiter):
        negativePotential = optimal_transport_V2.optimal_transport(twoDimDensity,X_data,masses,negativePotential,maxiter = numGstepsPerIter,learningRate = eta,illConditionThresh = illConditionThresh,method=gMethod,verbose=False,beta=beta,distr=distr,dimension=dimension,oneDimBounds = oneDimBounds,optional_arguments=optional_arguments)
        if negativePotential[0] == "NA":
            return "NA","NA","NA","NA","NA"
        g = -negativePotential
        if verbose:
            #print out the vitals
            print("The iteration is "+str(i))
            print("The masses are currently")
            print(masses)
            print("The g is ")
            print(g)
            print("the integrals from the g are ")
            if dimension == 2:
                pd = optimal_transport_V2.PowerDiagram(X_data,g,twoDimDensity)
                print(pd.integrals())
                print("The approximate Wasserstein-2 Squared is ")
                print(np.sum(pd.second_order_moments()))
            elif dimension == 1:
                integs = optimal_transport_V2.laguerre_areas(None,X_data,negativePotential,der=False,dimension=dimension,distr=distr,oneDimBounds=oneDimBounds,optional_arguments=optional_arguments)
                print(integs)
        if dimension == 1:
            masses1 = backtrackingDescent(negativePotential,masses,lam,beta,L,maxIterBacktracking,N,eta,lamMethod)
        elif dimension == 2:
            masses1 = backtrackingDescent(-negativePotential,masses,lam,beta,L,maxIterBacktracking,N,eta,lamMethod)


        err = np.sum((masses1 - masses) ** 2)
        errors[i] = err
        if err <= tol:
            break

        masses = masses1
    #Non algorithmic display consideration: Recall for d=1, ot expects the data in sorted order. To make d=1 plots 
    #we report back the masses associated with the data in by-index order so that the plots are comparable to the 2d plots.
    if dimension == 1:
        massesForDisplay= np.empty_like(masses)
        massesForDisplay[sortedOrder] = masses
        gForDisplay = np.empty_like(g)
        gForDisplay[sortedOrder] = g
        masses = massesForDisplay
        g = gForDisplay
    return masses, g, err, i+1, errors


#X_data: the data
#numWeightInits: How many repeated runs of the random weight initialization to do
#concentrationInit: What is the concentration vector of the dirichlet to be used
#density: This is a convexPolyhedra or ScaledImage. It is the density.
#The rest of the parameters are mass_update paramters for repeated runs. Each repeated run has identical settings.
#traceCanvas: An Axes Subplot object. This will plot the traces of the final weights in increasing order
def mass_update_repeated_runner(X_data,numWeightInits,concentrationInit,twoDimDensity,lamMethod,eta,lam,numGstepsPerIter = 1,maxiter = 2000,tol = 1e-7,verbose=False,traceCanvas=None,upperLimOnTracePlot=None,dimension=2,distr=None,oneDimBounds = None,optional_arguments=None,illConditionThresh=10e5):
    n = len(X_data)
    bestWeights = np.zeros(n)
    bestObjective = np.inf
    objVals = np.zeros(numWeightInits)
    wassers = np.zeros(numWeightInits)
    klCosts = np.zeros(numWeightInits)
    for j in np.arange(numWeightInits):
            massInit = np.random.dirichlet(concentrationInit)
            #print("The ill condition threshold is "+str(illConditionThresh))
            masses, g, err, numIter, errors = run_mass_update(X_data,massInit,twoDimDensity,eta,lam=lam,verbose=False,gMethod = "secondOrder",numGstepsPerIter = numGstepsPerIter,maxiter = maxiter,tol=tol,lamMethod = lamMethod,dimension=dimension,distr=distr,oneDimBounds = oneDimBounds,optional_arguments=optional_arguments,
                                                              illConditionThresh = illConditionThresh)
            if verbose:
                print("The number of iterations is "+str(numIter))
            if numIter == "NA":
                objVals[j] = np.inf
                wassers[j] = np.inf
                klCosts[j] = np.inf
            else:
                wasserCost = optimal_transport_V2.computeW2_squared(X_data,masses,twoDimDensity,dimension=dimension,distr=distr,oneDimBounds=oneDimBounds,optional_arguments=optional_arguments,illConditionThresh=illConditionThresh)
                KLcost = np.sum(masses*np.log(masses))
                if lamMethod == "no_KL":
                    objVal = wasserCost
                elif lamMethod == "absolute":
                    objVal = np.sum(masses*np.log(masses))+lam*wasserCost
                objVals[j] = objVal
                wassers[j] = wasserCost
                klCosts[j] = KLcost
                if objVal < bestObjective:
                    bestWeights = masses
                    bestObjective = objVal
                if traceCanvas:
                    if objVal < np.inf:
                        #print(objVal)
                        traceCanvas.plot(np.sort(masses),alpha=0.3)
                        traceCanvas.set_ylim((0,upperLimOnTracePlot))
                        traceCanvas.set_title("Lambda = "+str(lam)+": weights in increasing order \n (all successful runs plotted)")
    return bestWeights,objVals,wassers,klCosts


#Visual tool. Run this with your data, a number of repetitions (numWeightsInits), and a concentration vector concentrationInit, and bunch of lambdas to figure out max reasonable lambda.
#X_data: the data
#numWeightInits: number of times you'd like to initiate random weights
#concentrationInit: the dirichlet concentration for the random weight generation for each run
#densityString: This is a string that provides a title description of the density
#dataString: This is a string that provides a title description of the data generation process.
#concForNoKL: unused parameter. ignore.
def max_lambda_finder(X_data,numWeightInits,concentrationInit,twoDimDensity,lambdas,concForNoKL =1,visualize = True,densityString = "NA",dataString = "NA",dimension=2,distr=None,oneDimBounds = None,optional_arguments=None):
    #Set a learning rate
    eta = .5

    N = len(X_data)

    wassers = np.zeros(len(lambdas))
    j=0
    epsMinCandidates = np.zeros(len(lambdas))
    objectiveValues = []
    wasserValues = []
    klValues = []
    for lam in lambdas:
        print("Now working on lambda = "+str(lam))
        #print(numWeightInits)
        #print(concentrationInit)
        #print(density)
        bestMasses,objVals,wasList,klCosts = mass_update_repeated_runner(X_data,numWeightInits,concentrationInit,twoDimDensity,"absolute",eta,lam,dimension=dimension,distr=None,oneDimBounds = None,optional_arguments=None)
        wasserAtBest = optimal_transport_V2.computeW2_squared(X_data,bestMasses,twoDimDensity,dimension=dimension,distr=distr,oneDimBounds=oneDimBounds,optional_arguments=optional_arguments)
        wassers[j] = wasserAtBest
        objectiveValues.append(objVals)
        wasserValues.append(wasList)
        klValues.append(klCosts)
        j=j+1

    epsMin = min(epsMinCandidates)

    if visualize:
        fig,axs = plt.subplots(2,2,figsize=(10,8))
        axs[0,0].boxplot(objectiveValues,positions=lambdas,widths=0.5)
        #axs[0].grid(True)
        axs[0,0].set_xticks(lambdas)
        axs[0,0].set_ylabel("Objective Values at solution")
        axs[0,0].set_xlabel("lambda")

        
        axs[0,1].boxplot(wasserValues,positions=lambdas,widths=0.5)
        axs[0,1].set_xticks(lambdas)
        axs[0,1].set_ylabel("W2_squared at solution")
        axs[0,1].set_xlabel("lambda")

        axs[1,0].boxplot(klValues,positions=lambdas,widths=0.5)
        axs[1,0].set_xticks(lambdas)
        axs[1,0].set_ylabel("KL at solution")
        axs[1,0].set_xlabel("lambda")

        axs[1,1].scatter(lambdas,wassers,color='blue',label='BETEL_opt')
        axs[1,1].set_xlabel("Lambda")
        axs[1,1].set_ylabel("Value of W2^2 at BEST solution")
        plt.suptitle(str(numWeightInits)+" reps per Lambda. Density: "+densityString+". Data: "+dataString)
    
    return lambdas,wassers,objectiveValues,wasserValues,klValues