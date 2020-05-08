import numpy as np
from pywt import threshold

def SVT(X, tau):
    U, S, V = np.linalg.svd(X,full_matrices=False)
    T = np.maximum(0, S - tau)
    T = np.diagflat(T)
    A = np.matmul(U,T)
    return np.matmul(A,V)


def ST(X,tau):
    return threshold(X,tau,'soft')

def objective(L,S,lambd):
    return np.linalg.norm(L,ord='nuc')+lambd*np.linalg.norm(S,ord=1)


if __name__ == '__main__':
    A = np.linspace(-4, 4, 7)
    print(A)
    print(ST(A, 2.5))
