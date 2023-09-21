import torch
import pytorch_lightning as pl

from torch import nn
from torch.nn.parameter import Parameter


class BasicICNN(pl.LightningModule):
    """BasicICNN: Input convex neural network implementation"""
    def __init__(self, **hyperparameterValues):
        super(BasicICNN, self).__init__()
        
        hyperparameters = {
            'hiddenDim': 6
        }
        hyperparameters.update(hyperparameterValues)

        self.save_hyperparameters(hyperparameters)

        # layers from the inputs
        self.initialLayer = nn.Linear(2, self.hparams.hiddenDim)
        self.shortcutLayer0 = nn.Linear(2, self.hparams.hiddenDim)
        self.shortcutLayer1 = nn.Linear(2, 1)

        # internal layers
        self.innerLayer0 = PositiveLinear(self.hparams.hiddenDim,
                                          self.hparams.hiddenDim)
        self.innerLayer1 = PositiveLinear(self.hparams.hiddenDim, 1)

        self.nlin = nn.LeakyReLU()

        self.trainLoss = nn.MSELoss()

    def forward(self, y):
        z = self.initialLayer(y)
        z = self.nlin(z)
        z = self.innerLayer0(z) + self.shortcutLayer0(y)
        z = self.nlin(z)
        z = self.innerLayer1(z) + self.shortcutLayer1(y)

        return z

    def training_step(self, batch, batchidx):
        xs, ys = batch

        loss = self.trainLoss(self.forward(xs), ys)
        self.log('Train Loss', loss)
        return loss

    def validation_step(self, batch, batchidx):
        xs, ys = batch
        loss = self.trainLoss(self.forward(xs), ys)
        self.log('Val Loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1E-3)


class PartialICNN(pl.LightningModule):
    """PartialICNN: a neural network that is convex in some of its inputs"""
    def __init__(self, **hyperparameterValues):
        super(PartialICNN, self).__init__()
        
        hyperparameters = {
            'convexInputDim': 2,
            'convexHiddenDim': 20,

            'nonconvexInputDim': 2, 
            'nonconvexHiddenDim': 20,

            'lr': 1E-3
        }
        hyperparameters.update(hyperparameterValues)

        self.save_hyperparameters(hyperparameters)

        # layers from the inputs
        self.inputToInternal = nn.ModuleList([
                nn.Linear(self.hparams.convexInputDim,
                          self.hparams.convexHiddenDim),
                nn.Linear(self.hparams.convexInputDim,
                          self.hparams.convexHiddenDim),
                nn.Linear(self.hparams.convexInputDim, 1)
        ])

        # internal layers
        self.convexInternal = nn.ModuleList([
                None,
                PositiveLinear(self.hparams.convexHiddenDim,
                               self.hparams.convexHiddenDim),
                PositiveLinear(self.hparams.convexHiddenDim, 1)
        ])

        # non-convex auxiliary network
        self.nonConvexInternal = nn.ModuleList([
                nn.Linear(self.hparams.nonconvexInputDim,
                          self.hparams.nonconvexHiddenDim),
                nn.Linear(self.hparams.nonconvexHiddenDim,
                          self.hparams.nonconvexHiddenDim),
                nn.Linear(self.hparams.nonconvexHiddenDim,
                          self.hparams.nonconvexHiddenDim)
        ])

        self.auxBatchNorm = nn.ModuleList([
                nn.BatchNorm1d(self.hparams.nonconvexHiddenDim),
                nn.BatchNorm1d(self.hparams.nonconvexHiddenDim),
                nn.BatchNorm1d(self.hparams.nonconvexHiddenDim)
        ])

        # auxiliary to input
        self.auxToInput = nn.ModuleList([
                nn.Linear(self.hparams.nonconvexInputDim,
                          self.hparams.convexInputDim),
                nn.Linear(self.hparams.nonconvexHiddenDim,
                          self.hparams.convexInputDim),
                nn.Linear(self.hparams.nonconvexHiddenDim,
                          self.hparams.convexInputDim)
        ])

        # auxiliary to internal
        self.auxToInternal = nn.ModuleList([
                None,
                nn.Linear(self.hparams.nonconvexHiddenDim,
                          self.hparams.convexHiddenDim),
                nn.Linear(self.hparams.nonconvexHiddenDim, 1)
        ])

        self.auxToAdditional = nn.ModuleList([
                nn.Linear(self.hparams.nonconvexInputDim,
                          self.hparams.convexHiddenDim),
                nn.Linear(self.hparams.nonconvexHiddenDim,
                          self.hparams.convexHiddenDim),
                nn.Linear(self.hparams.nonconvexHiddenDim, 1)
        ])

        self.nlin = nn.LeakyReLU()
        self.ReLU = nn.ReLU()

        self.trainLoss = nn.MSELoss()

    def forward(self, x, y):
        # u and z represent internal layers of the 
        # non-convex and convex networks, respectively

        for lInd in range(3):
            if lInd == 0:
                z = self.nlin(
                        self.inputToInternal[0](y * self.auxToInput[0](x))
                        + self.auxToAdditional[0](x)
                )
                u = self.nlin(self.auxBatchNorm[0](
                             self.nonConvexInternal[0](x)))
            else:
                z = self.nlin(
                    self.convexInternal[lInd](
                        z * self.ReLU(self.auxToInternal[lInd](u))
                    )
                    + self.inputToInternal[lInd](y * self.auxToInput[lInd](u))
                    + self.auxToAdditional[lInd](u)
                )
                u = self.nlin(self.auxBatchNorm[lInd](
                            self.nonConvexInternal[lInd](u)))
        return z

    def training_step(self, batch, batchidx):
        xs, ys, targets = batch

        loss = self.trainLoss(self.forward(xs, ys), targets)
        self.log('Train Loss', loss)
        return loss

    def validation_step(self, batch, batchidx):
        xs, ys, targets = batch
        loss = self.trainLoss(self.forward(xs, ys), targets)
        self.log('Val Loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class PositiveLinear(nn.Module):
    """Positive linear layer for the internal pass of the ICNN"""
    def __init__(self, in_features, out_features):
        super(PositiveLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.nonLin = nn.Softplus()

        self.weight = Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        """ Some fiddling with the initialization might be required
            because these are positive matrices that are applied repeatedly
        """
        self.weight.data.normal_(0., 1./(self.in_features))

    def forward(self, x):
        return nn.functional.linear(x, self.nonLin(self.weight))
