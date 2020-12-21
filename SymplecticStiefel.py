from __future__ import division

import numpy as np

from pymanopt.manifolds.manifold import Manifold
from scipy.linalg import expm

class SymplecticStiefel(Manifold):
    """
    norm_cond : normalisation condition needs to be one of the two possible normalisation condition, i.e. 1 or 2
    norm_cond=1 means X_perp is orthogonal
    norm_cond=2 means X_perp(X_perp.T*J*X_perp)^-1 is orthogonal
    rho is used in the multiplicative factor 1/rho in front of the metric
    """
    def __init__(self,two_height,two_width,norm_cond=1,rho=1,k=1,retraction='geodesic'):
        #check that is greater or equal to p
        if two_height<two_width:
            raise ValueError("Need 2n >= 2p. Values supplied were 2n = %d ""and 2p = %d." %(two_height,two_width))
        #check that k is greater or equal to 1
        if k<1:
            raise ValueError("Need k>=1. Value supplied was %k=d" %k)
        #check that the height and the width are a multiple of 2
        if two_height%2==1 or two_width%2==1:
            raise ValueError("Need 2n and 2p to be multiples of 2. Values supplied were 2n = %d" "and 2p=%d" %(two_height,two_width))

        #set dimensions of the symplectic Stiefel manifold
        self._n = two_height//2
        self._p = two_width//2
        self._k = k
        #set dimension

        #set normalization conditions
        self._norm_cond = norm_cond
        self._rho = rho

        # set type of retraction
        self._retraction = retraction

        #set the canonical symplectic matrices
        self.J2n = np.block([[np.zeros((self._n,self._n)),np.eye(self._n)],[-np.eye(self._n),np.zeros((self._n,self._n))]])
        self.J2p = np.block([[np.zeros((self._p,self._p)),np.eye(self._p)],[-np.eye(self._p),np.zeros((self._p,self._p))]])

    def sym(self,A):
        return (A+A.T)/2

    def skew(self,A):
        return (A-A.T)/2




    def rand(self):
        if self._k==1:
            Jp = np.block([[np.zeros((self._p,self._p)),np.eye(self._p)],[-np.eye(self._p),np.zeros((self._p,self._p))]])
            W = np.random.randn(2*self._p,2*self._p)
            symplectic = expm(Jp@(W+W.T))
            X0 = np.block([[symplectic[0:self._p,:]],[np.zeros((self._n-self._p,2*self._p))],[symplectic[self._p:,:]],[np.zeros((self._n-self._p,2*self._p))]])
            return X0

    # I am not so sure about this procedure but it is coded in the same way for the Stiefel manifold
    def randvec(self, X):
        S = np.random.randn(*X.shape)
        S = self.proj(X,S)
        return S/self.norm(X,S)


    # For tangent vectors, we just store the "S part" so we need a way to construct the full tangent vector
    # tangent vector -> matrix
    def toMat(self, X, S):
        return S@self.J2n@X

    def toVec(self, X, Y):
        G_X = np.eye(2*self._n)-0.5*X@self.J2p@X.T@self.J2n.T
        return G_X@Y@(X@self.J2p).T+X@self.J2p@(G_X@Y).T

    def inner(self, X, S1, S2):
        # Inner product (Riemannian metric) on the tangent space
        # S1, S2 are elements of the tangent space at X, so we just store their "S" matrix

        # compute B_X
            if self._norm_cond == 1:
                X_perp_term = -np.linalg.matrix_power((self.J2n@X@self.J2p@X.T@self.J2n.T-self.J2n),2)
            else:
                X_perp_term = np.eye(2*self._n)-X@np.linalg.solve(X.T@X,X.T)

            B_X = 1/self._rho*self.J2n@X@X.T@self.J2n.T + X_perp_term

            # two different ways of computing tr(Z1^T(B_X)Z2)
            #return np.tensordot(self.toMat(X, S1), B_X @ self.toMat(X, S2))
            return np.einsum('ki,kj,ji->',self.toMat(X,S1),B_X,self.toMat(X,S2))

    def eegrad2egrad(self, eegrad):
       # takes the gradiant in ambiant R^{mxn} manifold endowed with standart Frobenius scalar product and returns
       # the the gradiant again in ambiant R^{mxn} manifold but with the special metric (which is the wrapper function)
        def wrapper(X):
            if self._norm_cond == 1:
                X_perp_term = self.J2n@(np.eye(2*self._n)-X@np.linalg.solve(X.T@X,X.T))@self.J2n.T
            else:
                dummy = np.eye(2*self._n)-X@self.J2p@X.T@J.T
                X_perp_term = dummy@dummy.T
            return (self._rho*X@X.T+X_perp_term)@eegrad(X) # return the S part of the gradient
            # B_X = 1/self._rho*self.J2n@X@X.T@self.J2n.T - np.linalg.matrix_power((self.J2n@X@self.J2p@X.T@self.J2n.T-self.J2n),2)
            # return np.linalg.solve(B_X,eegrad(X))
        return wrapper



    def proj(self, X, Y):
        G = np.eye(2*self._n)-0.5*X@self.J2p@X.T@self.J2n.T
        S = G@Y@(X@self.J2p).T + X@self.J2p@(G@Y).T
        return S #@self.J2n@X



    def norm(self, X, S):
        # We use the norm induced by the special metric induced by B_X
        SVec = S
        return np.sqrt(self.inner(X, SVec, SVec))

    # Retract to the symplectic-Stiefel manifold using a quasi-geodesic
    def retr(self, X, S):
        if self._retraction == 'geodesic':
            Z = self.toMat(X,S)
            W = X.T@self.J2n@Z
            JW = np.block([[W[self._p:,:]],[-W[:self._p,:]]])
            exp = expm(np.block([[-JW, self.J2p@Z.T@self.J2n@Z], [np.eye(2*self._p), -JW]]))
            XNew = np.block([X,Z])@exp[:,:2*self._p]@expm(JW)
            return XNew
        elif self._retraction =='cayley':
            t = 1
            return np.linalg.solve(np.eye(2*self._n)-t/2*S@self.J2n, np.eye(2*self._n)+t/2*S@self.J2n)@X
