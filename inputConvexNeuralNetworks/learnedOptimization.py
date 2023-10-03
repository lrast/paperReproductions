# A variety of different optimizers for input convex neural networks

import torch
import pytorch_lightning as pl

from torch import nn
from torch.autograd import grad

from icnn import PartialICNN
from optimizers import PrimalDualOptimizer


class GradientDescentOptimizer(pl.LightningModule):
    """
        Learns an optimization problem linking inputs with optimal outputs.
        Optimization is performend through gradient descent
    """
    def __init__(self, **hyperparameterValues):
        super(GradientDescentOptimizer, self).__init__()

        hyperparameters = {
            'convexInputDim': 2,
            'optimizationSteps': 10,
            'stepSize': 0.01
        }
        hyperparameters.update(hyperparameterValues)

        self.save_hyperparameters(hyperparameters)

        self.partialConvexNN = PartialICNN(
                                    convexInputDim=self.hparams.convexInputDim
                                    )

        self.trainLoss = nn.MSELoss()

    def forward(self, x):
        y = torch.randn(x.shape[0], self.hparams.convexInputDim, 
                        requires_grad=True)

        for i in range(self.hparams.optimizationSteps):
            objective = self.partialConvexNN(x, y)

            g = grad(objective, y, grad_outputs=torch.ones(objective.shape),
                     create_graph=True)[0]
            y = y - self.hparams.stepSize * g
        return y

    def training_step(self, batch, batchidx):
        xs, ys = batch
        loss = self.trainLoss(self.forward(xs), ys)
        self.log('Train Loss', loss)
        return loss

    def validation_step(self, batch, batchidx):
        torch.set_grad_enabled(True)
        xs, ys = batch
        loss = self.trainLoss(self.forward(xs), ys)

        self.log('Val Loss', loss)
        self.zero_grad()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1E-3)


class BundleEntropyOptimizer(pl.LightningModule):
    """
        Learns an optimization problem linking inputs with optimal outputs.
        Optimization is performend through gradient descent
    """
    def __init__(self):
        super(BundleEntropyOptimizer, self).__init__()
        self.partialConvexNN = PartialICNN()

        self.convexDim = self.partialConvexNN.hparams.convexInputDim

    def bundleIteration(self, x, G=None, h=None, maxSteps=10):
        """Iterations for the bundle entropy method
            x - batch, dim
            y - batch, dim
            G - batch, steps, dim
            h - batch, steps, 1
        """

        if G is None:
            # initial guess
            y = 0.5*torch.ones(x.shape[0], self.convexDim, requires_grad=True)

        else:  # solve the optimization
            y, t = PrimalDualOptimizer(G, h)

        if G.shape[1] == maxSteps:
            # final step
            return y
        else:
            # update G and H and run again.
            f_vals = self.partialConvexNN(x, y)
            slopes = grad(f_vals, y, grad_outputs=torch.ones(f_vals.shape))[0]
            intercepts = f_vals - torch.einsum('nd, nd ->n', slopes, y)[:, None]

            G = torch.cat((G, slopes[:, None, :]), dim=1)
            h = torch.cat((h, intercepts[:, None, :]), dim=1)

            return self.bundleIteration(x, G, h, maxSteps=maxSteps)

    def gradientThroughBundle(self, x, ySteps):
        pass

    def forward(self, x, y):
        pass

    def training_step(self, batch, batchidx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1E-3)



