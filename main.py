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

# fix the seed for debugging purpouses
np.random.seed(1)

n = 300
p = 30

# symplectic matrices
J2n = np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])
J2p = np.block([[np.zeros((p, p)), np.eye(p)], [-np.eye(p), np.zeros((p,p))]])

# define the symplectic manifold using the factory SymplecticStiefel
M_geo = SymplecticStiefel(2 * n, 2 * p, retraction='geodesic', rho=0.5)
M_cay = SymplecticStiefel(2 * n, 2 * p, retraction='cayley', rho=0.5)

# Fixed starting point for debugging
X = np.block([[np.eye(p),np.zeros((p,p))],
              [np.zeros((n-p,2*p))],
              [np.zeros((p,p)),np.eye(p)],
              [np.zeros((n-p,2*p))]])

# Random starting point
# X = M.rand()

# Random matrix from objective
A = np.random.randn(2*n,2*p)
A = A/np.linalg.norm(A)
scipy.io.savemat('A_matrix.mat', mdict={'A_matrix': A})

# cost is Frobenius norm ||X-A||^2
def cost(X) : return np.einsum('ij,ij->',X-A,X-A)

# xi is a small perturbation matrix
# xi = 0.01*np.random.randn(2*n,2*p)
# A = X+xi

def eegrad(X):
    return 2*(X-A)
egrad = M_cay.eegrad2egrad(eegrad)

rho = 0.5
grad = eegrad2grad(eegrad,rho)

# I am not sure about how to compute the hessian
def ehess(x,v):
       return 2*v # donc la Hessienne vaut 2*I

problem_geo = Problem(manifold= M_geo, cost = cost, grad = grad, ehess = ehess)
problem_cay = Problem(manifold = M_cay, cost = cost, grad = grad, ehess = ehess)


solver = SteepestDescent(logverbosity=2, maxiter=500, linesearch=NonMonotoneLineSearch(alpha=0.85, initial_step=1e-3))
# solver = SteepestDescent(logverbosity=2, maxiter=300)
# solver = ConjugateGradient(logverbosity=2,maxiter=100)
# solver = TrustRegions(logverbosity=2,maxiter=100)

X_geo, optlog_geo = solver.solve(problem_geo, x=X)

t_geo = np.array(optlog_geo['iterations']['time'])
t_geo = t_geo - t_geo[0]
gradnorm_geo = optlog_geo['iterations']['gradnorm']
fx_geo = optlog_geo['iterations']['f(x)']
iter_geo = optlog_geo['iterations']['iteration']

X_cay, optlog_cay = solver.solve(problem_cay, x=X)

t_cay = np.array(optlog_cay['iterations']['time'])
t_cay = t_cay - t_cay[0] # relative time from absolute time
gradnorm_cay = optlog_cay['iterations']['gradnorm']
fx_cay = optlog_cay['iterations']['f(x)']
iter_cay = optlog_cay['iterations']['iteration']

fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(12, 4))
axes[0].set_yscale('log')

# axes[0].plot(t_geo, gradnorm_geo, label='Quasi-geodesic')
axes[0].plot(iter_geo, gradnorm_geo, label='Quasi-geodesic')
axes[1].plot(iter_geo,fx_geo,label='Quasi-geodesic')


# axes[0].plot(t_cay, gradnorm_cay, label='Cayley')
axes[0].plot(iter_cay, gradnorm_cay, label='Cayley')
axes[1].plot(iter_cay,fx_cay,label='Cayley')

axes[0].set_xlabel('iteration')
axes[0].set_ylabel(r'$\mid \operatorname{grad}f\mid_F$')

axes[1].set_xlabel('iteration')
axes[1].set_ylabel('fval')

axes[0].legend()
axes[1].legend()

plt.show()

pass
