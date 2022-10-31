# -*- coding: utf-8 -*-

# Michael Goforth
# CAAM 550 HW 2 Problem 1
# Due 9/8/2021



import math
import numpy as np
import matplotlib.pyplot as plt


def f(T, D, Ts, Tinf, I2R, sigma=5.67e-8, eps=.8, h=20):
    '''Function to determine temperature of conducting rod with diameter, D,
    electrical resistance per unit, R,  in enclosure whose walls are kept
    at temperature, Ts, with air flow across the road at temperature, Tinf,
    and current squared times resistance, I2R.  Rod is at steady-state temp, T,
    if f(T) = 0.
    
    Parameters
    ----------
    T : value
        temperature of rod (Kelvin)
    D : value
        diameter of rod (meters)
    Ts : value
         temperature of enclosure walls (degrees Celsius)
    Tinf : value
           temperature of air blowing across the pipe (degrees Celsius)
    I2R : value
          resistance of rod squared multiplied by radius of rod
    sigma : value (optional, default = 5.67e-8)
            Stefan-Boltzman constant, units Watts/meter^2Kelvin^4
    eps : value (optional, default = .8)
          rod surface emissivity
    h : value (optional, default = 20)
        heat transfer coefficient of air flow, units Watts/meter^2Kelvin
           
    Returns
    -------
    f(T) : tuple
           f(T), f'(T).  (Rod is at steady-state temp if f(T) = 0)
           
    Michael Goforth
    CAAM 550
    Rice University
    September 8, 2021
    '''
    
    Tinf = Tinf + 273.15 # convert Celsius to Kelvin
    Ts = Ts + 273.15 # convert Celsius to Kelvin
    fval = math.pi * D * h * (T - Tinf) \
            + math.pi * D * eps * sigma * (T**4 - Ts**4) \
            - I2R
    fderiv = math.pi * D * h + 4 * math.pi * D * eps * sigma * T**3
    return(fval, fderiv)


def plotfunc(func, xmin=200, xmax=400, fargs=None):
    '''Plot function over a given range
    
    Parameters
    ----------
    func : function
           Funciton which will be plotted
           The call func(x) should return 2 values, f(x) and f'(x)
    xmin : value
           Minimum X value to be plotted
    xmax : value
           Maximum X value to be plotted
    fargs : tuple, optional
            parameters to be passed to func
           
    Returns
    -------
    none
    
    Michael Goforth
    CAAM 550
    Rice University
    September 8, 2021
    '''
    
    x = np.linspace(xmin, xmax, 1000)
    fx = [ func(i, *fargs)[0] for i in x]
    zero = [ 0 for i in x]
    plt.plot(x, fx, x, zero)
    plt.xlabel("Rod Temperature, T, Kelvin")
    plt.ylabel("F(T)")
    plt.legend(('F(T)', 'F(T)=0'))
    plt.show()


def bisection(func, a=0, b=100, tol=10**-7, maxiter=100, fargs=None):
    '''Utilize bisection method to approximate solution of func(x)=0
    
    Parameters
    ----------
    func : function
           Function for which a root will be found
           The call func(x) should return 2 values, f(x) and f'(x)
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
    
    if func(a, *fargs)[0] * func(b, *fargs)[0] >= 0:
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
           The call func(x) should return 2 values, f(x) and f'(x)
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

    fa = func(a, *fargs)[0]
    c = .5 * (a + b)
    fc = func(c, *fargs)[0]
    if ithist is None:
        ithist = np.array([i, a, b, c, fc])
    else:
        ithist = np.vstack((ithist, np.array([i, a, b, c, fc])))
    if fa * fc < 0:
        return a, c, ithist
    else:
        return b, c, ithist
    

# Newton's method
def newt1d(f, x, tolx = 1.e-7, tolf = 1.e-7, maxit = 100, fargs=None): 
    '''Approximate solution of f(x)=0 using Newton's method.
    
    Parameters
    ----------
    f : function
        Function for which we wan to to find a root.
        The call f(x)  should return  return 2 values, f(x) and f'(x)
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
    
    fx, fpx = f(x, *fargs)
    s  = - fx/fpx
    ithist = np.array([it, x, fx, s])
    
    while( it < maxit and abs(s) > tolx and abs(fx) > tolf ):
        x = x+s
        fx, fpx = f(x, *fargs)
        s  = - fx/fpx
        it = it+1;
        
            
        ithist = np.vstack((ithist, np.array([it, x, fx, s]) ))

    # check why the bisection method truncated and set iflag
    if abs(s) > tolx and abs(fx) > tolf :
        # bisection method truncated because maximum number of iterations reached
        iflag = 1
    
    return x, ithist, iflag


if __name__ == "__main__":
    # Plot function
    plotfunc(f, xmin=200, xmax=400, fargs=(.1, 25, 25, 100))
    # Root located between 200 and 400
    c, a, b, ithist, iflag = bisection(
        f, a=200, b=400, tol=1e-7, maxiter=100, fargs=(.1, 25, 25, 100))
    print("Bisection Method returned with iflag = " + str(iflag))
    print("Final root approximation: " + str(c))
    print(' iter       a             b             c           f(c) ')
    for i in np.arange(ithist.shape[0]):
        print(f' {ithist[i,0]:3.0f}  {ithist[i,1]:13.6e}  {ithist[i,2]:13.6e}  {ithist[i,3]:13.6e}  {ithist[i,4]:13.6e}')
    print()
    # run Newton's method
    x, ithist, iflag = newt1d(f, 2.0, tolf = 1.e-7, fargs=(.1, 25, 25, 100))
    print(f" Newton's method returned with iflag = {iflag:1d}")
    print("Final root approximation: " + str(x))
    print(' iter        x           f(x)           step')
    for i in np.arange(ithist.shape[0]):
        print(f' {ithist[i,0]:3.0f}  {ithist[i,1]:13.6e}  {ithist[i,2]:13.6e}  {ithist[i,3]:13.6e}')