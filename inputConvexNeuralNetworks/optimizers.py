# optimizers for the convex bundle entropy problem
import torch
from torch.autograd import grad


def bundleIteration(f, x, G=None, h=None, maxSteps=10):
    """Iterations for the bundle entropy method
    """
    if G is None:
        # initial guess
        y = torch.rand(2, 1, requires_grad=True) 
        t = 0.1

        G = torch.empty(0, 2)
        h = torch.empty(0, 1)

    else:  # solve the optimization
        y, t = PrimalDualOptimizer(G, h)
        y.requires_grad = True

        if G.shape[0] == maxSteps:
            # final step
            return y

    # update G and H and run again.
    f_vals = f(x, y)
    Entropy = 0.25*(y*torch.log(y) + (1-y)*torch.log(1-y)).sum()

    #print(G)
    slopes = grad(f_vals, y, grad_outputs=torch.ones(f_vals.shape))[0]

    #print(slopes - 2*(y-x))
    #breakpoint()


    intercepts = torch.tensor([[f_vals - slopes.T @ y]])

    G = torch.cat((G, slopes[None, :, 0]), dim=0)
    h = torch.cat((h, intercepts), dim=0)

    #print('GHGHGHG',G,h)
    #breakpoint()

    return bundleIteration(f, x, G, h, maxSteps=maxSteps)


def PrimalDualOptimizer(G, h):
    """
        Following the updated ICNN code on github
        https://github.com/locuslab/icnn/blob/master/lib/bundle_entropy.py,
        I'm going to try using a primal-dual optimizer to solve the convex 
        Bundle entropy problem
    """

    # !!! to do: add stop criteria

    # not batched for a moment.

    dimDual, dimPrimal = G.shape

    def negHGrad(y): return 0.2*(torch.log(y) - torch.log(1.-y))
    def negHHessian(y): return 0.2*(torch.diag_embed((1. / y + 1./(1-y)).squeeze()))

    def residuals(y, t, L, u):
        ry = negHGrad(y) + G.T @ L
        rt = (1. - L.sum())*torch.ones(1, 1)
        rcentral = torch.diag_embed(L[:, 0]) @ (G @ y + h - t) + u

        return torch.cat([ry, rt, rcentral])

    def newtonJacobian(y, t, L, u):
        return torch.cat([
            torch.cat([negHHessian(y), torch.zeros(dimPrimal, 1), G.T], dim=1),
            torch.cat([torch.zeros(1, dimPrimal+1), 
                      -torch.ones(1, dimDual)], dim=1),
            torch.cat([torch.diag_embed(L[:, 0]) @ G, -torch.ones(dimDual, 1),
                       torch.diag_embed((G @ y + h - t)[:, 0])], dim=1)
            ], dim=0)

    def updateVars(y, t, L, deltas):
        dy, dt, dL = torch.split(deltas, (dimPrimal, 1, dimDual))
        return y + dy, t + dt, L + dL

    def linearInterp(y, t, L, residuals):
        return (torch.cat([y, t, L]).T @ residuals).item()

    alpha = 0.05
    beta = 0.9

    def findStepsSize(y, t, L, u, deltas):
        """Step size determination for Newton's method"""
        currentState = torch.cat([y, t, L])

        # ensure that no variables are taken below 0
        size = 1
        while (y + size*deltas[0:2]).max() > 1.:
            size = beta *size

        if (deltas < 0).any():
            maxSize = (-currentState / deltas)[deltas < 0].min()
            size = min(size, 0.99*maxSize)


        #while (G @ y + h - t).max() > 0:
        #    size = beta * size

        oldResidualNorm = residuals(y, t, L, u).norm()
        while (residuals(*updateVars(y, t, L, size*deltas), u).norm() > 
                (1 - size*alpha) * oldResidualNorm):
            size = beta * size

        return size

    steps = 30

    # initial guesses
    L = torch.rand(dimDual, 1) / dimDual
    y = 0.25*(torch.ones(dimPrimal, 1) + torch.rand(dimPrimal, 1))
    t = torch.max(G @ y + h) * torch.ones(1, 1)
    u = 1.

    for step in range(steps):
        # compute residuals
        resid = residuals(y, t, L, u)

        # compute deltas
        jacobian = newtonJacobian(y, t, L, u)
        deltas = torch.linalg.solve(jacobian, -resid)

        #if step == 9:
        #    print(deltas)
        #    breakpoint()

        # determine step size and update
        stepSize = findStepsSize(y, t, L, u, deltas)
        y, t, L = updateVars(y, t, L, stepSize*deltas)

        u = -(L.T @ (G @ y + h - t)) / dimDual

    print('opt out', (y - 1./(1+torch.exp(G.T @ L))).norm(), L.sum(), u )
    return y, t
