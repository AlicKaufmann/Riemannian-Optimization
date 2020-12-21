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

n = 1000
p = 5

# symplectic matrices
J2n = np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])
J2p = np.block([[np.zeros((p, p)), np.eye(p)], [-np.eye(p), np.zeros((p,p))]])

# define the symplectic manifold using the factory SymplecticStiefel
M = SymplecticStiefel(2*n,2*p,retraction='cayley', rho=0.5)

# Fixed starting point for debugging
X = np.block([[np.eye(p),np.zeros((p,p))],
              [np.zeros((n-p,2*p))],
              [np.zeros((p,p)),np.eye(p)],
              [np.zeros((n-p,2*p))]])

# Random starting point
# X = M.rand()

lam = 1.01
L = np.diag([lam**-i for i in range(2*n)])
Q = np.linalg.qr(np.random.normal(size=(2*n,2*n)))[0]
A = Q@L@Q.T
scipy.io.savemat('A_brockett.mat', mdict={'A_brockett': A})

def cost(X) : return np.einsum('ji,jk,ki->',X,A,X)

def eegrad(X):
    return 2*A@X

rho = 0.5
grad = eegrad2grad(eegrad,rho)

problem = Problem(manifold = M, cost = cost, grad = grad)


solver = SteepestDescent(logverbosity=2, maxiter=500, linesearch=NonMonotoneLineSearch(alpha=0.85, initial_step=1e-3))
# solver = SteepestDescent(logverbosity=2, maxiter=300)
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

data = np.array([fx[-1],gradnorm[-1], iter[-1], t[-1]-t[1]])
np.set_printoptions(precision=3)
print(data)

fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(12, 4))
axes[0].set_yscale('log')
axes[0].plot(iter, gradnorm, label=r'$ \mid gradf(x)\mid$')
axes[1].plot(fx)
plt.show()


