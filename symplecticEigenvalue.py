import autograd.numpy as np
# from SymplecticStiefel import SymplecticStiefel
from SymplecticStiefel2 import SymplecticStiefel
from pymanopt import Problem
import matplotlib.pyplot as plt
from checkgradient import checkgradient
from pymanopt.solvers import SteepestDescent
from pymanopt.solvers import ConjugateGradient
from pymanopt.solvers import TrustRegions
from NonMonotoneLineSearch import NonMonotoneLineSearch
import scipy.io
from RiemannianGradient import eegrad2grad
from scipy.sparse import diags
from scipy.linalg import companion

# fix the seed for debugging purpouses
np.random.seed(1)

n = 75
p = 1

# symplectic matrices
J2n = np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])
J2p = np.block([[np.zeros((p, p)), np.eye(p)], [-np.eye(p), np.zeros((p,p))]])

# define the symplectic manifold using the factory SymplecticStiefel
M = SymplecticStiefel(2*n,2*p,retraction='geodesic')

# Fixed starting point for debugging
X = np.block([[np.eye(p),np.zeros((p,p))],
              [np.zeros((n-p,2*p))],
              [np.zeros((p,p)),np.eye(p)],
              [np.zeros((n-p,2*p))]])

# Random starting point
# X = M.rand()

# create lehmer matrix
# def lehmer(k,l):
#     L = np.zeros ((k,l))
#     for i in range ( 0, k ):
#         for j in range ( 0, l ):
#             L[i,j] = float ( min ( i + 1, j + 1 ) ) / float ( max ( i + 1, j + 1 ) )
#     return L
# C = lehmer(2*n,2*n)
# scipy.io.savemat('C_matrix.mat', mdict={'C_matrix': C})

# Create Wilkinson matrix
def wilkinson(n):
    half_diag = np.arange((n-1)/2,0,-1)
    if n%2==0:
        mid_diag = np.hstack((half_diag,np.flip(half_diag)))
    else:
        mid_diag = np.hstack((half_diag,0,np.flip(half_diag)))
    diagonals = [mid_diag,np.ones(n-1),np.ones(n-1)]
    return diags(diagonals, [0, -1, 1]).toarray()
C = wilkinson(2*n)
C = C.T@C
scipy.io.savemat('C_matrix.mat', mdict={'C_matrix': C})

# C = companion(np.arange(1,2*n+2))
# C = C.T@C
# scipy.io.savemat('C_matrix.mat', mdict={'C_matrix': C})

# C = C/np.linalg.norm(C)
# cost is tr(X^TCX) = <X,CX>
def cost(X) : return np.einsum('ji,jk,ki->',X,C,X)
# def cost(X) : return np.linalg.norm(X-A)

def eegrad(X):
    return 2*C@X
egrad = M.eegrad2egrad(eegrad)

rho = 0.5
grad = eegrad2grad(eegrad,rho)


problem = Problem(manifold = M, cost = cost, grad = grad)


solver = SteepestDescent(logverbosity=2, maxiter=4000, linesearch=NonMonotoneLineSearch(alpha=0.85))
# solver = SteepestDescent(logverbosity=2, maxiter=4000)
# solver = ConjugateGradient(logverbosity=2,maxiter=100)
# solver = TrustRegions(logverbosity=2,maxiter=100)

Xopt, optlog = solver.solve(problem,x=X)

t = optlog['iterations']['time']
gradnorm = optlog['iterations']['gradnorm']
fx = optlog['iterations']['f(x)']
iter = optlog['iterations']['iteration']

# plt.plot(t,gradnorm,label=r'$ \mid gradf(x)\mid$')
# plt.legend()
# plt.yscale('log')
# plt.show()

fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(12, 4))
axes[0].set_yscale('log')
axes[0].plot(iter, gradnorm, label='Quasi-geodesic')
axes[1].plot(iter, fx, label='Quasi-geodesic')

axes[0].set_xlabel('iteration')
axes[0].set_ylabel(r'$\mid \operatorname{grad}f\mid_F$')

axes[1].set_xlabel('iteration')
axes[1].set_ylabel('fval')

axes[0].legend()
axes[1].legend()

fig.suptitle('2*(largest symplectic) for Lehmer matrix with n = '+str(n))

plt.show()
