import autograd.numpy as np
from SymplecticStiefel import SymplecticStiefel
from pymanopt import Problem
import matplotlib.pyplot as plt
from checkgradient import checkgradient
from pymanopt.solvers import SteepestDescent

n = 3
p = 2

# symplectic matrices
J2n = np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])
J2p = np.block([[np.zeros((p, p)), np.eye(p)], [-np.eye(p), np.zeros((p,p))]])

# define the symplectic manifold using the factory SymplecticStiefel
M = SymplecticStiefel(2*n,2*p,retraction='geodesic')

# Fixed starting point for debugging
# X = np.block([[np.eye(p),np.zeros((p,p))],
#               [np.zeros((n-p,2*p))],
#               [np.zeros((p,p)),np.eye(p)],
#               [np.zeros((n-p,2*p))]])

# Random starting point
X = M.rand()

# cost is Frobenius norm ||X-A||^2
def cost(X) : return np.einsum('ij,ij->',X-A,X-A)
# def cost(X) : return np.linalg.norm(X-A)


# Fixed matrix from objective for debugging purpose
A = np.array([[-11., -10.,  11., -20.],
       [-18.,  11.,   0.,   0.],
       [ -0.,   5.,   4.,   6.],
       [  9., -21.,   6.,  15.],
       [-24., -11.,  12.,  -1.],
       [ -4., -15.,  -4.,   6.]])

# Random matrix from objective
#A = np.random.randn(2*n,2*p)


# xi is a small perturbation matrix
# xi = 0.01*np.random.randn(2*n,2*p)
# A = X+xi

def eegrad(X):
    return 2*(X-A)
egrad = M.eegrad2egrad(eegrad)

problem = Problem(manifold = M, cost = cost, egrad=egrad)


solver = SteepestDescent(logverbosity=2,maxiter=500)
Xopt, optlog = solver.solve(problem,x=X)

t = optlog['iterations']['time']
gradnorm = optlog['iterations']['gradnorm']

plt.plot(t,gradnorm,label=r'$ \mid gradf(x)\mid$')
plt.legend()
plt.yscale('log')
plt.show()

