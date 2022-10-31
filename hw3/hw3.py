# -*- coding: utf-8 -*-

# Michael Goforth
# CAAM 550 HW 2 Problem 1
# Due 9/8/2021



import math
import numpy as np
import matplotlib.pyplot as plt

def multiQR(A, j=100):
    '''Function as described in problem 4 of HW3.  
    
    Parameters
    ----------
    A : np.array
        matrix to be decomposed
    j : value (optional, default = 10)
        number of iterations of QR decomposition
           
    Returns
    -------
    A_j : np.array
          resulting matrix.  A_j = (Q_0 Q_1 ... Q_j-1)^T A_0 (Q_0 Q_1 ... Q_j-1)
           
    Michael Goforth
    CAAM 550
    Rice University
    September 15, 2021
    '''
    
    for i in range(j):
        Q, R = np.linalg.qr(A)
        A = R @ Q
    return A

if __name__ == "__main__":
    # Confirm QR factorization
    A = np.array([[1, 2, 3], [-1, 2, 1], [0, 1, 1]])
    Q, R = np.linalg.qr(A)
    print("Q = " + str(Q))
    print("R = " + str(R))
    A = np.array([[1, 3], [-7, 1]])
    print(str(multiQR(A)))
    print(np.linalg.eig(A))