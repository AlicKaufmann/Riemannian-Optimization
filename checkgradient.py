import numpy as np
import matplotlib.pyplot as plt

def checkgradient(problem,x,v):
    """
    Written by Alic Kaufmann by copying the matlab manopt function to check the gradient
    """

    cost = problem.cost
    retr = problem.manifold.retr
    inner = problem.manifold.inner
    grad = problem.grad
    norm = problem.manifold.norm

    # normalise the direction v
    v = v/norm(x,v)

    # compute the error
    log_time = np.logspace(-15, 1, 100)
    E = np.array([])
    ref = np.array([])

    # compute Et = |f(R_x(tv))−f(x)−t<gradf(x),v>_x| which is to be compared to t->t^2
    for t in log_time:
        Et = abs(cost(retr(x,t*v))-cost(x)-t*inner(x,grad(x),v))
        E = np.append(E,Et)
        ref = np.append(ref,t**2)

    # compute the quantity <grad(x),tv> and compare it to the quantity f(R_x(tv))-f(x)
    time = np.linspace(0,1,100)
    q1 = np.array([])
    q2 = np.array([])
    suff_decr = 1e-4 # same parameter as in linesearch.py
    for t in time:
        q1_t = problem.manifold.inner(x,grad(x),t*v)
        q1 = np.append(q1,q1_t)
        q2_t = cost(retr(x,t*v))-cost(x)
        q2 = np.append(q2,q2_t)

    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (12,4))
    axes[0].loglog(log_time,E, label=r'$E(t)=\mid f(R_x(tv))-(f(x)+\langle gradf(x),v\rangle \mid$')
    axes[0].loglog(log_time,ref, label=r'$t\mapsto t^2$')

    axes[1].plot(time, q1, label=r'$\langle grad(x),tv \rangle$')
    axes[1].plot(time, q2, label=r'$f(R_x(tv))-f(x)$')
    axes[1].plot(time, suff_decr*time*inner(x,grad(x),v),label=r'suff_decr*$\langle gradf(x),v\rangle$')


    axes[0].legend()
    axes[1].legend()
    plt.show()


