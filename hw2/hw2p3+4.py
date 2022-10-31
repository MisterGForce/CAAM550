# -*- coding: utf-8 -*-

# Michael Goforth
# CAAM 550 HW 2 Problem 3 and 4
# Due 9/8/2021


import math
import numpy as np
import matplotlib.pyplot as plt


def plotphi(K, g, gtrue):
    '''Plots function phi
    
    Parameters
    ----------
    K : np.matrix
        kernel
    g : np.vector
        blurred image with noise
    gtrue : np.vector
            blurred image with no noise
           
    Returns
    -------
    none
    
    Michael Goforth
    CAAM 550
    Rice University
    September 8, 2021
    '''
    lam = np.logspace(-10, -2, num=500)
    y1 = np.zeros(500)
    for i in range(500):
        y1[i] = phi1(lam[i], K, g)
    const = [.5 * np.linalg.norm(g - gtrue)**2 for i in lam]
    plt.xlabel("Lambda")
    plt.loglog(lam, y1, lam, const)
    plt.legend(['1/2 * ||Kf(lambda)-g||_2^2', 'Constant 1/2||g - gtrue||_2^2'])
    plt.title('1/2 * ||Kf(lambda)-g||_2^2 vs Constant 1/2||g - gtrue||_2^2')
    

def phi1(lam, K, g):
    '''Function of residual of Morozov discrepency, 1/2 ||K f(lambda) - g||_2^2.'
    
    Parameters
    ----------
    lam : value
          lambda
    K : np.matrix
        kernel
    g : np.vector
        blurred image with noise
           
    Returns
    -------
    phi1(lambda) : value
                   1/2 ||K f(lambda) - g||_2^2
           
    Michael Goforth
    CAAM 550
    Rice University
    September 8, 2021
    '''
    KK = np.vstack( (K, np.sqrt(lam)*np.identity(n)) )
    gg = np.hstack( (g, np.zeros(n)) )
    flam = np.linalg.lstsq(KK,gg,rcond=None)[0]
    return .5 * np.linalg.norm(np.dot(K, flam) - g, 2)**2


def phi(lam, K, g):
    '''Function of residual of Morozov discrepency minus the error,
    1/2 ||K f(lambda) - g||_2^2. - 1/2 ||g - gtrue||_2^2
    
    Parameters
    ----------
    lam : value
          lambda
    K : np.matrix
        kernel
    g : np.vector
        blurred image with noise
           
    Returns
    -------
    phi(lambda) : value
                  1/2 ||K f(lambda) - g||_2^2 - g||_2^2. - 1/2 ||g - gtrue||_2^2
           
    Michael Goforth
    CAAM 550
    Rice University
    September 8, 2021
    '''
    KK = np.vstack( (K, np.sqrt(lam)*np.identity(n)) )
    gg = np.hstack( (g, np.zeros(n)) )
    flam = np.linalg.lstsq(KK,gg,rcond=None)[0]
    return .5 * np.linalg.norm(K @ flam - g, 2)**2 \
            - .5 * np.linalg.norm(g - gtrue, 2)**2


def phiprime(lam, K, g):
    '''Derivative of phi function in Morozov descrepency principle.  See pdf
    with calculations for details.
    
    Parameters
    ----------
    lam : value
          lambda
    K : np.matrix
        kernel
    g : np.vector
        blurred image with noise
           
    Returns
    -------
    phi'(lambda) : value
                   derivative of phi at lambda
           
    Michael Goforth
    CAAM 550
    Rice University
    September 8, 2021
    '''
    KK = np.vstack( (K, np.sqrt(lam)*np.identity(n)) )
    gg = np.hstack( (g, np.zeros(n)) )
    flam = np.linalg.lstsq(KK,gg,rcond=None)[0]
    JF = np.transpose(K @ flam - g)
    JG = K
    JH = -1 * np.linalg.inv(np.transpose(K) @ K + lam * np.identity(n)) @ flam
    return JF @ JG @ JH


def phiprimeest(lam, K, g):
    '''Estimate of derivative of phi function in Morozov descrepency principle.  
    Used to test Newton's method before explicit derivative calculation was
    correct.'
    
    Parameters
    ----------
    lam : value
          lambda
    K : np.matrix
        kernel
    g : np.vector
        blurred image with noise
           
    Returns
    -------
    phi'(lambda) : value
                   derivative of phi at lambda
           
    Michael Goforth
    CAAM 550
    Rice University
    September 8, 2021
    '''
    delta = 1e-8
    return(phi(lam + delta, K, g) - phi(lam, K, g)) / delta

def bisection(func, a=0, b=100, tol=10**-7, maxiter=100, fargs=None):
    '''Utilize bisection method to approximate solution of func(x)=0
    
    Parameters
    ----------
    func : function
           Function for which a root will be found
           The call func(x) should return a value
    a,b : values
          Interval in which a root will be searched.  Note that if 
          func(a) * func(b) >= 0 no root will be found
    tol : value (optional, default = 1e-7)
          Algorithm stops when |a-b| < tol
    maxiter : value (optional, default = 100)
              maximum number of iterations of bisection method that will be
              performed
    fargs : tuple, optional
            parameters to be passed to func
           
    Returns
    -------
    x : value
        approximation of root
    a, b : values
           a root of func is in the interval (a,b)
    ithist : np.array
             iteration history; i-th row of ithist contains [it, a, b, c, fc]
    iflag : integer
            return flag
            iflag = 0 if |b-a| <= tol
            iflag = 1 iteration terminated due to max number of iterations
            iflag = 2 no root found because func(a)*func(b)>=0
    
    Michael Goforth
    CAAM 550
    Rice University
    September 8, 2021
    '''

    i = 0 # iteration counter
    ithist = None
    
    if func(a, *fargs) * func(b, *fargs) >= 0:
        raise Exception("The scalars a and b do not bound a root")

    while abs(b - a) > tol and i <= maxiter:
        a, b, ithist = bisectioniter(func, a, b, i, ithist, fargs)
        i = i + 1
    if i < maxiter:
        iflag = 0
    else:
        iflag = 1
    return (a + b)/2, a, b, ithist, iflag


def bisectioniter(func, a, b, i, ithist, fargs=None):
    '''Run one iteration of bisection method
    
    Parameters
    ----------
    func : function
           Function for which a root will be found
           The call func(x) should return a value
    a,b : values
          Interval in which a root will be searched.  Note that if 
          func(a) * func(b) >= 0 no root will be found
    i : value
        current count of number of iterations performed
    ithist : None or np.array
             array keeping history of each iteration.  will be None for first 
             iteration
    fargs : tuple, optional
            parameters to be passed to func
           
    Returns
    -------
    a, b : values
           a root of func is in the interval (a,b)
    ithist : np.array
             iteration history; i-th row of ithist contains [it, a, b, c, fc]

    Michael Goforth
    CAAM 550
    Rice University
    September 1, 2021
    '''
    
    fa = func(a, *fargs)
    c = .5 * (a + b)
    fc = func(c, *fargs)
    if ithist is None:
        ithist = np.array([i, a, b, c, fc])
    else:
        ithist = np.vstack((ithist, np.array([i, a, b, c, fc])))
    if fa * fc < 0:
        return a, c, ithist
    else:
        return b, c, ithist


# Newton's method
def newt1d(f, fprime, x, tolx = 1.e-7, tolf = 1.e-7, maxit = 100, fargs=None): 
    '''Approximate solution of f(x)=0 using Newton's method.
    
    Parameters
    ----------
    f : function
        Function for which we want to to find a root.
        The call f(x)  should return return the value f(x)
    fprime : function
             Derivative of f
             The call fprime(x) should return the value f'(x)
    x : number
        Initial approximation of root
    tolx : number (optional, default = 1.e-7)
        Algorithm stops when | s | < tolx, where s = Newton step
    tolf : number (optional, default = 1.e-7)
        Algorithm stops when | f(x | < tolf
    maxit: integer (optional, default = 100)
        maximum number of iterations
    fargs : tuple, optional
            parameters to be passed to f
        

    Returns
    -------
    x : numbers
        approximation of the root.
    ithist : np.array 
        iteration history; i-th row of ithist contains [it, x, fx, s]
    ifag : integer
        return flag
        iflag = 0   if |b-a| <= tolx, or   | f(x) | <= tolf
        iflag =  1  iteration terminated because maximum number of
                    iterations was reached. |b-a| > tolx, |f(x)| > tolf


    Matthias Heinkenschloss
    Department of Computational and Applied Mathematics
    Rice University
    August 29, 2021
    Slight modifications by Michael Goforth
    September 8, 2021
    ''' 
   
    it    = 0
    iflag = 0
    
    fx = f(x, *fargs)
    fpx = fprime(x, *fargs)
    s  = - fx/fpx
    ithist = np.array([it, x, fx, s])
    
    while( it < maxit and abs(s) > tolx and abs(fx) > tolf ):
        x = x+s
        fx, fpx = f(x, *fargs), fprime(x, *fargs)
        s  = - fx/fpx
        it = it+1;
        
            
        ithist = np.vstack((ithist, np.array([it, x, fx, s]) ))

    # check why the bisection method truncated and set iflag
    if abs(s) > tolx and abs(fx) > tolf :
        # bisection method truncated because maximum number of iterations reached
        iflag = 1
    
    return x, ithist, iflag

    
if __name__ == "__main__":
    # From assignments page:
    # specify problem
    n = 100
    h = 1/n
    gamma = 0.05
    xi = np.arange(1/2, n, dtype=float)/n
    # true image
    ftrue = np.exp( -(xi-0.75)**2 *70 )
    ind = np.all([0.1<=xi, xi<=0.25], axis =0)  # indices for which xi in [0.1,0.25]
    ftrue[ind] = 0.8
    ind = np.all([0.3<=xi, xi<=0.35], axis =0)  # indices for which xi in [0.3,0.35]
    ftrue[ind] = 0.3
    # matrix K
    C = 1/(gamma*np.sqrt(2*np.pi))
    K = np.zeros((n,n))
    for i in np.arange(n):
        for j in np.arange(n):
            K[i,j] = C*h* np.exp( -(xi[i]-xi[j])**2 / (2*gamma**2) )
    
    gtrue = np.dot(K, ftrue)
                             
    fig, ax = plt.subplots()
    ax.plot(xi, ftrue, '-k')
    ax.plot(xi, gtrue, '-.r')
    ax.legend(['true image','blurred image'])
    ax.set(xlabel='xi')
    
    #fig.savefig("denoise_1d_ftrue")
    plt.title('True and Blurred Image')
    plt.show()
    
    # add error to true image
    gerr = 0.001*np.multiply( 0.5 - np.random.uniform(0.05,0.5,n) , gtrue ) 
    g = gtrue + gerr
    gerror = 0.5*np.linalg.norm(gerr, 2)**2
    
    # naive reconstruction by solving linear system
    f = np.linalg.solve(K, g)
    
    fig, ax = plt.subplots()
    ax.plot(xi, ftrue, '-k')
    ax.plot(xi, f, ':b')
    ax.plot(xi, gtrue, '-.r')
    ax.legend(['true image','recovered image','blurred image'])
    ax.set_ylim(-0.2, 1.2)
    plt.title('Naive Reconstruction')
    
    #fig.savefig("denoise_1d_frecovered2.png")
    plt.show()
    

    # reconstruction by solving linear least squares with fixed lambda
    lam = 1.e-2
    KK = np.vstack( (K, np.sqrt(lam)*np.identity(n)) )
    gg = np.hstack( (g, np.zeros(n)) )
    flam = np.linalg.lstsq(KK,gg,rcond=None)[0]
    
    fig, ax = plt.subplots()
    ax.plot(xi, ftrue, '-k')
    ax.plot(xi, flam, ':b')
    ax.plot(xi, gtrue, '-.r')
    ax.legend(['true image','recovered image','blurred image'])
    ax.set(xlabel='xi')
    ax.set_ylim(-0.2, 1.2)
    plt.title('Reconstruction with Lambda = .01')
    #fig.savefig("denoise_1d_frecovered2.png")
    plt.show()
    
    plotphi(K, g, gtrue)
    c, a, b, ithist, iflag = bisection(phi, a=0, b=.01, tol=1e-7, 
                                       fargs=(K, g))
    print("Bisection Method returned with iflag = " + str(iflag))
    print("Final root approximation: " + str(c))
    print(' iter       a             b             c           f(c) ')
    for i in np.arange(ithist.shape[0]):
        print(f' {ithist[i,0]:3.0f}  {ithist[i,1]:13.6e}  {ithist[i,2]:13.6e}  {ithist[i,3]:13.6e}  {ithist[i,4]:13.6e}')
    print()
    
    # reconstruction by solving linear least squares with lambda found with bisection method
    lam = c
    KK = np.vstack( (K, np.sqrt(lam)*np.identity(n)) )
    gg = np.hstack( (g, np.zeros(n)) )
    flam = np.linalg.lstsq(KK,gg,rcond=None)[0]
    
    fig, ax = plt.subplots()
    ax.plot(xi, ftrue, '-k')
    ax.plot(xi, flam, ':b')
    ax.plot(xi, gtrue, '-.r')
    ax.legend(['true image','recovered image','blurred image'])
    ax.set(xlabel='xi')
    ax.set_ylim(-0.2, 1.2)
    plt.title('Reconstruction with Lambda found with Bisection Method')
    plt.show()
    
    # Problem 4
    lam1 = phiprime(.001, K, g)
    print("For lambda = .001, phi'(lambda) = " + str(lam1))
    print()
    delta = np.logspace(-15, -1, 10)
    fd = np.zeros(10)
    for i in range(10):
        fd[i] = (phi(.001 + delta[i], K, g)-phi(.001, K, g)) / delta[i]
    lam1vec = [lam1 for i in fd]
    error = abs(lam1vec - fd)
    plt.xlabel("delta")
    plt.loglog(delta[:-2], error[:-2])
    plt.legend(['Error'])
    plt.title('Error of Finite Difference method compared to phi\' for different deltas')
    plt.show()
     # run Newton's method
    x, ithist, iflag = newt1d(phi, phiprime, .01, tolf = 1.e-7, fargs=(K, g))
    print(f" Newton's method returned with iflag = {iflag:1d}")
    print("Final root approximation: " + str(x))
    print(' iter        x           f(x)           step')
    for i in np.arange(ithist.shape[0]):
        print(f' {ithist[i,0]:3.0f}  {ithist[i,1]:13.6e}  {ithist[i,2]:13.6e}  {ithist[i,3]:13.6e}')
    
    # reconstruction by solving linear least squares with lambda found by Newton's method
    lam = c
    KK = np.vstack( (K, np.sqrt(lam)*np.identity(n)) )
    gg = np.hstack( (g, np.zeros(n)) )
    flam = np.linalg.lstsq(KK,gg,rcond=None)[0]
    
    fig, ax = plt.subplots()
    ax.plot(xi, ftrue, '-k')
    ax.plot(xi, flam, ':b')
    ax.plot(xi, gtrue, '-.r')
    ax.legend(['true image','recovered image','blurred image'])
    ax.set(xlabel='xi')
    ax.set_ylim(-0.2, 1.2)
    plt.title('Reconstructed Image using Lambda found from Newton\'s Method')
    plt.show()
    