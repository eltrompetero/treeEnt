# EntropyEstimates.py
#
# Bryan Daniels
# Modified by Eddie Lee
#
# from Wolpert, Wolpert 1995 and Nemenman, Shafee, Bialek, 2008 

from scipy.special import gamma,gammaln,polygamma
from scipy.integrate import quad
import numpy as np
from numpy.lib.scimath import sqrt
from scipy.optimize import fmin, brentq

Phi = lambda n,z: polygamma(n-1,z)
deltaPhi = lambda n,z1,z2: Phi(n,z1) - Phi(n,z2)



def meanEntropy(nVec, beta=1, m=None):
    """
    Measured in nats (I think)
    
    m (None)            : If m=None, assume nVec lists all 
                          possibilities ( len(nVec) = m ).
                          If a number is given, it's used 
                          and assumed that all other bins
                          have zero counts. 
    """
    nVec = np.array(nVec)
    N = sum(nVec)
    numBins = float( len(nVec) )
    if m is None:
        m = len(nVec) # aka K
    # assuming we can use result from WolWol95 and replace
    # n_i + 1 --> n_i + beta
    # m --> beta*m
    # (I think this is correct, based on the definition of beta
    #  and p.33 of WolWol94.)
    return Phi(1,N+beta*m+1) +                                  \
           sum( - (nVec+beta)/(N+beta*m) * Phi(1,nVec+beta+1) ) \
           - (m-numBins) * beta/(N+beta*m) * Phi(1,beta+1)
    
    return sum( - (nVec+beta)/(N+beta*m)                        \
                * deltaPhi(1,nVec+beta+1,N+beta*m+1) )          \
                - (m-numBins) * beta/(N+beta*m)                 \
                * deltaPhi(1,beta+1,N+beta*m+1)
    
# 10.25.2010
def s2s0Slow(nVec,beta=1):
    """
    Assumes nVec lists all possibilities ( len(nVec) = m )
    
    Measured in nats (I think)
    
    Can be done much faster (see s2s0)
    """
    nVec = np.array(nVec)
    N = sum(nVec)
    m = len(nVec) # aka K
    total = 0.
    for i in range(m):
      for j in range(m):
        if i == j:
          total += (nVec[i]+beta)*(nVec[i]+beta+1)              \
            *(deltaPhi(1,nVec[i]+beta+2,N+beta*m+2)**2          \
            + deltaPhi(2,nVec[i]+beta+2,N+beta*m+2))
        else:
          total += (nVec[i]+beta)*(nVec[j]+beta)                \
            *(deltaPhi(1,nVec[i]+beta+1,N+beta*m+2)             \
             *deltaPhi(1,nVec[j]+beta+1,N+beta*m+2)             \
             -Phi(2,N+beta*m+2))
    return total / ( (N+beta*m)*(N+beta*m+1) )

# 10.25.2010 version made to match with NemShaBia08
# (turns out to be identical)
def s2s0NemBetaOld(nVec,beta=1):
    """
    Assumes nVec lists all possibilities ( len(nVec) = m )
    
    Measured in nats (I think)
    
    Can be done much faster (see s2s0)
    """
    nVec = np.array(nVec)
    N = sum(nVec)
    m = len(nVec) # aka K
    totalOffDiag,totalDiag = 0.,0.
    for i in range(m):
      for j in range(m):
        if i == j:
          totalDiag += (nVec[i]+beta)*(nVec[i]+beta+1)              \
            *(deltaPhi(1,nVec[i]+beta+1,N+beta*m+1)**2          \
            + deltaPhi(2,nVec[i]+beta+1,N+beta*m+1))
        else:
          totalOffDiag += (nVec[i]+beta)*(nVec[j]+beta)                \
            *(deltaPhi(1,nVec[i]+beta+1,N+beta*m+1)             \
             *deltaPhi(1,nVec[j]+beta+1,N+beta*m+1)             \
             -Phi(2,N+beta*m+1))
    print("Phi(2,N+beta*m+1) =",Phi(2,N+beta*m+1))
    print("totalOffDiag =",totalOffDiag)
    print("totalDiag =",totalDiag)
    total = totalDiag + totalOffDiag
    return total / ( (N+beta*m)*(N+beta*m+1) )
    
# 10.25.2010 version made to match with NemShaBia08
# (turns out to be identical)
# 6.21.2011 (turns out to be identical?)
def s2s0NemBetaNew(nVec,beta=1):
    """
    Assumes nVec lists all possibilities ( len(nVec) = m )
    
    Measured in nats (I think)
    
    Can be done much faster (see s2s0)
    """
    nVec = np.array(nVec)
    N = sum(nVec)
    m = len(nVec) # aka K
    totalOffDiag,totalDiag = 0.,0.
    for i in range(m):
      for j in range(m):
        if i == j:
          totalDiag += (nVec[i]+beta)*(nVec[i]+beta+1)              \
            *(deltaPhi(1,nVec[i]+beta+2,N+beta*m+2)**2          \
            + deltaPhi(2,nVec[i]+beta+2,N+beta*m+2))
        else:
          totalOffDiag += (nVec[i]+beta)*(nVec[j]+beta)                \
            *(deltaPhi(1,nVec[i]+beta+1,N+beta*m+2)             \
             *deltaPhi(1,nVec[j]+beta+1,N+beta*m+2)             \
             -Phi(2,N+beta*m+2))
    print("Phi(2,N+beta*m+1) =",Phi(2,N+beta*m+2))
    print("totalOffDiag =",totalOffDiag)
    print("totalDiag =",totalDiag)
    total = totalDiag + totalOffDiag
    return total / ( (N+beta*m)*(N+beta*m+1) )

# 10.25.2010
# 6.21.2011 fixed bugs
def s2s0(nVec,beta=1,m=None,useLessMemory=False):
    """
    Assumes nVec lists all possibilities ( len(nVec) = m )
    
    Measured in nats (I think)
    """
    nVec = np.array(nVec)
    N = sum(nVec)
    numBins = len(nVec)
    if m is None:
        m = numBins # aka K
    factor1 = nVec+beta
    factor2 = deltaPhi(1,nVec+beta+1,N+beta*m+2)
    factor1zero = beta
    factor2zero = deltaPhi(1,beta+1,N+beta*m+2)
    if (m > 1e4) or useLessMemory or m!=numBins:
        # use different (slower?) method that uses
        # less memory
        total = 0
        p2 = Phi(2,N+beta*m+2)
        for i in range(numBins):
            sumVec = np.dot(factor1[i],factor1)*             \
                   ( np.dot(factor2[i],factor2) - p2 )
            # remove diagonal
            sumVec[i] = 0.
            total += np.sum(sumVec)
            # 7.19.2011 take into account extra zeros
            sumVecZeros = 2.*(m-numBins)*factor1[i]*factor1zero*\
                          (factor2[i]*factor2zero - p2)
            total += sumVecZeros
        # add zero-zero elements (except diagonal)
        total += (m-numBins)*((m-numBins)-1)*                   \
                 factor1zero*factor1zero*                       \
                (factor2zero*factor2zero - p2)
        # add zero-zero diagonal elements
        total += (m-numBins)*factor1zero*(factor1zero+1)*       \
                 ( deltaPhi(1,beta+2,N+beta*m+2)**2             \
                + deltaPhi(2,beta+2,N+beta*m+2) )
                
    else:
        sumMatrix = np.outer(factor1,factor1)*               \
            ( np.outer(factor2,factor2)-Phi(2,N+beta*m+2) )
        # remove diagonal
        sumMatrix = sumMatrix - np.diag(np.diag(sumMatrix))
        total = np.sum(sumMatrix)
    
    diagonal = factor1*(factor1+1)*                             \
        ( deltaPhi(1,nVec+beta+2,N+beta*m+2)**2                 \
        + deltaPhi(2,nVec+beta+2,N+beta*m+2) )
    #sumMatrix += np.diag(diagonal)
    total += np.sum(diagonal)
        
    return total / ( (N+beta*m)*(N+beta*m+1) )

# 10.25.2010
def varianceEntropy(nVec,beta=1):
    """
    Assumes nVec lists all possibilities ( len(nVec) = m )
    
    Measured in nats (I think)
    """
    return s2s0(nVec,beta) - meanEntropy(nVec,beta)**2
    
# 10.25.2010
def varianceEntropyNemZero(beta,K):
    beta,K = float(beta),float(K)
    return (beta+1)/(beta*K+1)*Phi(2,beta+1) - Phi(2,beta*K+1)
    
def xiFromBeta(beta, K):
    """Modified to be an odd function, such that finding the inverse is easier.

    Parameters
    ----------
    beta : float
    K : int

    Returns
    -------
    float
    """
    oddMultiplier = (beta > 0.)*2 - 1
    beta = abs(beta)
    kappa = K * beta
    xi = polygamma(0, kappa+1) - polygamma(0, beta+1)
    return oddMultiplier * xi
    
def xiFromBetaPrime(beta, K):
    """(Using the modified xi -- see xiFromBeta)
    """
    beta = abs(beta)
    kappa = K * beta
    xiDeriv = K * polygamma(1, kappa+1) - polygamma(1, beta+1)
    return xiDeriv
    
def betaFromXi(xi, K, xtol=1e-20, maxiter=10_000, beta_mn=0., beta_mx=100.):
    while xiFromBeta(beta_mx, K) < xi:
        beta_mx *= 10
    
    #betaStart = 0.
    #beta = np.optimize.newton(                               \
    #    lambda beta:(xiFromBeta(beta,K) - xi)/xi, betaStart,    \
    #    fprime=lambda beta:xiFromBetaPrime(beta,K)/xi,             \
    #    maxiter=maxiter,tol=tol)
    if xi==0:
        beta = brentq(lambda beta:xiFromBeta(beta, K), beta_mn, beta_mx,
                      maxiter=maxiter, xtol=xtol)
    else:
        beta = brentq(lambda beta:(xiFromBeta(beta, K) - xi)/xi, beta_mn, beta_mx,
                      maxiter=maxiter, xtol=xtol)
    if abs((xiFromBeta(beta,K) - xi)/xi) > 0.01:
        print("found beta =", beta)
        print("xi desired =", xi)
        print("xi found =", xiFromBeta(beta, K))
        raise Exception("Loss of precision in betaFromXi.")
    return beta
        
def lnXiDistrib(xi, nVec, K=None):
    """Assumes nVec lists all possibilities ( len(nVec) = m )
    """
    nVec = np.array(nVec)
    N = nVec.sum()
    if K is None:
        K = nVec.size  # aka m
    beta = betaFromXi(xi, K)
    kappa = K*beta
    #return gamma(kappa)/gamma(N+kappa)                         \
    #     * np.prod( gamma(nVec+beta)/gamma(beta) )
    #return exp( gammaln(kappa) - gammaln(N+kappa) )            \
    #    * prod( exp( gammaln(nVec+beta) - gammaln(beta) ) )
    return gammaln(kappa) - gammaln(N+kappa) + sum(gammaln(nVec+beta) - gammaln(beta))
         
def integrateOverXi(func, nVec,
                    maxScaledXi=0.9999,
                    constMult=True,
                    range=None,
                    K=None,
                    verbose=False):
    """
    TODO: Potential source of numerical error in the integration procedure. Is there
    a good way to keep track of the error?

    maxScaledXi used to stay away from the problematic endpoint at log(K)...
    
    constMult       : Multiply integrand by a constant that 
                      depends only on nVec [currently 
                      e^(-max(lnXiDistrib)); 
                      used to be e^(-<lnXiDistrib>)].  Designed to remove 
                      difficulties with extremely small xiDistrib.
    """
    if K is None:
        K = nVec.size
    if range is None:
        mn, mx = 0., maxScaledXi * np.log(K)
    else:
        mn, mx = range
    if constMult:
        # use average LnXi
        #fn = lambda xi: lnXiDistrib(xi,nVec)
        #avgLnXi = quad(fn,mn,mx)[0]/(mx-mn)
        #lnConst = -avgLnXi
        
        # 3.30.2011 use mx LnXi
        def fn(xi): 
            #if xi < mx: return -lnXiDistrib(xi,nVec,K=K)
            #else: return -lnXiDistrib(mx,nVec,K=K)
            if xi >= mx: return -lnXiDistrib(mx, nVec, K=K)
            elif xi <= mn: return -lnXiDistrib(mn, nVec, K=K) 
            else: return -lnXiDistrib(xi, nVec, K=K)
        xiMax = fmin(fn, (mx+mn)/2., maxiter=100, disp=verbose)[0]
        if xiMax > mx: xiMax = mx
        if xiMax < mn: xiMax = mn
        lnConst = fn(xiMax)
       
        if verbose:
            print("lnConst =", lnConst)
    else:
        lnConst = 0.

    if verbose:
        print("maxBeta = ", betaFromXi(mx, K))
        print("integrating over", mn, "< xi <", mx)

    # a little trick to make this integral easier numerically by factoring out the largest
    # value from the exponential
    integrand = lambda xi, exp_const=0.: np.exp(lnConst + lnXiDistrib(xi, nVec, K=K) +  exp_const) * func(xi)
    exp_const = np.nanmax([(lnConst + lnXiDistrib(xi, nVec, K=K)) for xi in np.linspace(mn, mx, 100)])
    result = quad(lambda xi: integrand(xi, -exp_const), mn, mx,
                  limit=100,
                  epsabs=1e-12)
    # re-introduce missing factor
    result = result[0]*np.exp(exp_const), result[1]*np.exp(exp_const)
    return result
    
def meanEntropyNem(nVec, K=None, verbose=False, **kwargs):
    """Flat prior on beta.
    """
    nVec = np.array(nVec)
    N = nVec.sum()
    if K is None:
        K = nVec.size  # aka m
    entropyFunc = lambda xi: meanEntropy(nVec, beta=betaFromXi(xi, K), m=K)
    numInt = integrateOverXi(entropyFunc, nVec, K=K, verbose=verbose, **kwargs)
    denInt = integrateOverXi(lambda x: 1, nVec, K=K, verbose=verbose, **kwargs)
    num, den = numInt[0], denInt[0]
    if verbose:
        print("num =", num, ", den =", den)
        print("numAbsErr =", numInt[1], ", denAbsErr =", denInt[1])
        print("s1s0 =", num/den)
    return num/den
    
def s2s0Nem(nVec,K=None,verbose=False,**kwargs):
    """
    Flat prior on beta.
    """
    nVec = np.array(nVec)
    N = sum(nVec)
    if K is None:
        K = len(nVec) # aka m
    s2s0Func =                                                  \
        lambda xi: s2s0(nVec,beta=betaFromXi(xi,K),m=K)
    numInt = integrateOverXi(s2s0Func,nVec,K=K,                 \
        verbose=verbose,**kwargs)
    denInt = integrateOverXi(lambda x:1,nVec,K=K,               \
        verbose=verbose,**kwargs)
    num,den = numInt[0],denInt[0]
    if verbose:
        print("num =",num,", den =",den)
        print("numAbsErr =",numInt[1],", denAbsErr =",denInt[1])
        print("s2s0 =", num/den)
    return num/den
    
    
# 3.29.2011
def meanAndStdevEntropyNem(freqData,bits=True,**kwargs):
    mean = meanEntropyNem(freqData,**kwargs)
    s2s0 = s2s0Nem(freqData,**kwargs)
    stdev = sqrt(s2s0-mean*mean)

    # this is indicative of precision errors in the calculation of the mean or var
    if bits:
        mean = nats2bits( mean )
        stdev = nats2bits( stdev )
    return mean, stdev
    
def varianceEntropyNem(nVec,**kwargs):
    """Flat prior on beta.
    """
    print("10.26.2010 Not sure if this is correct.")
    # do we have to integrate s2s0 separately?
    nVec = np.array(nVec)
    N = sum(nVec)
    K = len(nVec) # aka m
    varianceFunc =                                              \
        lambda xi: varianceEntropy(nVec,beta=betaFromXi(xi,K))
    num = integrateOverXi(varianceFunc,nVec,**kwargs)[0]
    den = integrateOverXi(lambda x:1,nVec,**kwargs)[0]
    return num/den

# 3.29.2011
def nats2bits(nats):
    return nats / np.log(2)




if False:
    # 6.20.2011 TEST
    def sigmaNemBeta(nVec,beta=1):
        s2 = s2s0(nVec,beta=beta)
        s1 = meanEntropy(nVec,beta=beta)
        return sqrt(s2-s1*s1)

    # 6.20.2011 TEST
    def sigmaNem(nVec,**kwargs):
        """
        Flat prior on beta.
        """
        nVec = np.array(nVec)
        N = sum(nVec)
        K = len(nVec) # aka m
        sigmaFunc =                                                  \
            lambda xi: sigmaNemBeta(nVec,beta=betaFromXi(xi,K))
        num = integrateOverXi(sigmaFunc,nVec,**kwargs)[0]
        den = integrateOverXi(lambda x:1,nVec,**kwargs)[0]
        return num/den
