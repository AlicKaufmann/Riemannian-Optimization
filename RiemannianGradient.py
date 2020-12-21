import numpy as np

def eegrad2grad(eegrad,rho):
    def grad(X):
        n = X.shape[0]//2
        p = X.shape[1]//2
        G = eegrad(X)
        JX = np.block([[X[n:,:]],[-X[:n,:]]])
        XJ = np.block([-X[:,p:],X[:,:p]])
        GX = 0.5*rho*G.T@X
        XX = X.T@X
        invXXXJG = np.linalg.solve(XX, JX.T@G)
        HG = G - JX@invXXXJG + X@GX.T
        return HG@(XJ.T@JX) + XJ@((HG.T@JX))
    return grad
