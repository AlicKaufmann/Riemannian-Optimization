import numpy as np

class NonMonotoneLineSearch(object):
    """
    Non monotone line search based on the paper of Gao, Thnh and Absil
    """

    def __init__(self, delta = 0.1, alpha = 0.85, beta = 1e-4, initial_step = 1e-3):
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.initial_step = initial_step

        self._gamma_min = 1e-15
        self._gamma_max = 1e15

        self._oldq = 1.0
        self._oldc = None

        self._oldf0 = None

        # things we need for ABB
        self._oldS = None
        self._oldg = None
        self._ABB_idx = 1

    def search(self,objective, manifold, x, d, f0, df0):

        # trick to get ||X'JX-J||_F
        #if self._oldf0 is None:... to be continued

        norm_d = manifold.norm(x,d)

        # ABB initial step
        if self._oldf0 is not None:
            S = self._oldS
            Y = -d-self._oldg
            SY = np.abs(np.einsum('ji,ji->',S,Y))
            if(self._ABB_idx):
                gamma = np.linalg.norm(S,'fro')**2/SY
            else:
                gamma = SY/np.linalg.norm(Y,'fro')**2



        # initial step of manopt
        # if self._oldf0 is not None:
        #     gamma = 2 * np.abs((f0 - self._oldf0)/df0)
        else:
            # gamma = self.initial_step/norm_d
            gamma = self.initial_step

        t = gamma
        newx = manifold.retr(x,t*d)
        newf = objective(newx)

        # initialize c (q is already initialized in constructor)
        if self._oldc == None:
            self._oldc = f0

        q = self._oldq
        c = self._oldc

        while(newf > c + self.beta*t*df0):
            t = self.delta*t
            newx = manifold.retr(x,t*d)
            newf = objective(newx)

        # update q and c
        newq = self.alpha*q + 1
        newc = self.alpha*q*c/newq + 1/newq*newf
        self._oldq = newq
        self._oldc = newc

        # update position and gradient
        self._oldS = newx - x
        self._oldg = -d
        self._ABB_idx = not self._ABB_idx

        self._oldf0 = f0

        stepsize = t*norm_d

        return stepsize, newx

