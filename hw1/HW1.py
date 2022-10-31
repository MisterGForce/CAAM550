# Michael Goforth
# CAAM 550 HW 1
# Due 9/4/2021



import math
import numpy as np
import matplotlib.pyplot as plt


def freezedepth(x, t=5184000, Ti=20, Ts=-15, alpha=.18*10**-6):
    '''Function to determine soil temperature T(x) of a pipe buried x meters
       underground after time t if it starts at temperature Ti, the soil is 
       temperature Ts, and thermal diffusivity of the soil is alpha.
    
    Parameters
    ----------
    x : value
        distance pipe is buried
    t : value
        time elapsed
    Ti : value
         initial temperature of the pipe
    Ts : value
         temperature of the soil surrounding the pipe
    Tf : value
         desired final temperature of pipe
    alpha : value
            Thermal diffusivity of soil
           
    Returns
    -------
    f(x) : Difference between desired temperature and actual temperature of a 
           pipe buried x meters underground.
    
    Michael Goforth
    CAAM 550
    Rice University
    September 1, 2021
    '''
    return (Ti - Ts)*math.erf(x / (2 * math.sqrt(alpha * t))) + Ts
    # See text hw assignment for work to get here and derivative

def plotfunc(func, xmin=0, xmax=5):
    '''Plot function over a given range
    
    Parameters
    ----------
    func : function
           Funciton which will be plotted
           The call func(x) should return a value
    xmin : value
           Minimum X value to be plotted
    xmax : value
           Maximum X value to be plotted
           
    Returns
    -------
    none
    
    Michael Goforth
    CAAM 550
    Rice University
    September 1, 2021
    '''
    
    x = np.linspace(xmin, xmax, 1000)
    fx = [ func(i) for i in x]
    zero = [ 0 for i in x]
    plt.plot(x, fx)
    plt.plot(x, zero)
    plt.xlabel("Depth (m)")
    plt.ylabel("temperature after 60 days (degrees C)")
    plt.show()



def bisection(func, a=0, b=100, tol=10**-7, maxiter=100):
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
    September 1, 2021
    '''


    i = 0 # iteration counter
    ithist = None
    
    if func(a) * func(b) >= 0:
        raise Exception("The scalars a and b do not bound a root")
        return(None, None, None, None, 2)

    while abs(b - a) > tol and i <= maxiter:
        a, b, ithist = bisectioniter(func, a, b, i, ithist)
        i = i + 1
    if i < maxiter:
        iflag = 1
    else:
        iflag = 0
    return (a + b)/2, a, b, ithist, iflag


def bisectioniter(func, a, b, i, ithist):
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

    fa = func(a)
    c = .5 * (a + b)
    fc = func(c)
    if ithist is None:
        ithist = np.array([i, a, b, c, fc])
    else:
        ithist = np.vstack((ithist, np.array([i, a, b, c, fc])))
    if fa * fc < 0:
        return a, c, ithist
    else:
        return b, c, ithist


def regulafalsi(func, a=0, b=100, tolx=10**-7, tolf=10**-7, maxiter=100):
    '''Utilize regula falsi to approximate solution of func(x)=0
    
    Parameters
    ----------
    func : function
           Function for which a root will be found
           The call func(x) should return a value
    a,b : values
          Interval in which a root will be searched.  Note that if 
          func(a) * func(b) >= 0 no root will be found
    tolx : value (optional, default = 1e-7)
           Algorithm stops when |a-b| < tolx
    tolf : value (optional, default = 1e-7)
           Algorithm stops when |func(.5*(a+b))| < tolf
    maxiter : value (optional, default = 100)
              maximum number of iterations of bisection method that will be
              performed
           
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
            iflag = 0 if |b-a| <= tolx or |f(x)| <= tolf
            iflag = 1 iteration terminated due to max number of iterations
            iflag = 2 no root found because func(a)*func(b)>=0
    
    Michael Goforth
    CAAM 550
    Rice University
    September 1, 2021
    '''

    if func(a) * func(b) >= 0:
        raise Exception("The scalars a and b do not bound a root")
        return(None, None, None, None, 2)
    i = 0  # iteration counter
    ithist = None
    c = .5 * (a + b)

    while abs(b - a) > tolx and abs(func(c)) > tolf and i <= maxiter:
        c, a, b, ithist = rfiter(func, a, b, i, ithist)
        i = i + 1
    if abs(b - a) < tolx or abs(func(c)) < tolf:
        iflag = 1
    else:
        iflag = 2
    return c, a, b, ithist, iflag


def rfiter(func, a, b, i, ithist):
    '''Run one iteration of regula falsi
    
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
           
    Returns
    -------
    c : value
        current best guess of root of func
    a, b : values
           a root of func is in the interval (a,b)
    ithist : np.array
             iteration history; i-th row of ithist contains [it, a, b, c, fc]

    Michael Goforth
    CAAM 550
    Rice University
    September 1, 2021
    '''

    fa = func(a)
    fb = func(b)
    c = a - (b - a) / (fb - fa) * fa
    fc = func(c)
    if ithist is None:
        ithist = np.array([i, a, b, c, fc])
    else:
        ithist = np.vstack((ithist, np.array([i, a, b, c, fc])))
    if fa * fc < 0:
        # print(a, c)
        return c, a, c, ithist
    else:
        # print(b, c)
        return c, b, c, ithist


if __name__ == "__main__":
    # Plot function
    plotfunc(freezedepth, xmin=0, xmax=2)
    # Root located between 0 and 2
    
    # Bisection method with initial values 0 and 2
    c, a, b, ithist, iflag = bisection(freezedepth, a=0, b=2)
    print("Bisection Method returned with iflag = " + str(iflag))
    print("Final root approximation: " + str(c))
    print(' iter       a             b             c           f(c) ')
    for i in np.arange(ithist.shape[0]):
        print(f' {ithist[i,0]:3.0f}  {ithist[i,1]:13.6e}  {ithist[i,2]:13.6e}  {ithist[i,3]:13.6e}  {ithist[i,4]:13.6e}')
    print()
    # Regula Falsi with initial values 0 and 2
    c, a, b, ithist, iflag = regulafalsi(freezedepth, a=0, b=2)
    print("Regula Falsi returned with iflag = " + str(iflag))
    print("Final root approximation: " + str(c))
    print(' iter       a             b             c           f(c) ')
    for i in np.arange(ithist.shape[0]):
        print(f' {ithist[i,0]:3.0f}  {ithist[i,1]:13.6e}  {ithist[i,2]:13.6e}  {ithist[i,3]:13.6e}  {ithist[i,4]:13.6e}')

