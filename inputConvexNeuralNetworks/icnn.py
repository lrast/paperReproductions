import torch
import pytorch_lightning as pl

from torch import nn
from torch.nn.parameter import Parameter
from torch.utils.data import TensorDataset, DataLoader

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


class BasicICNN(pl.LightningModule):
    """docstring for BasicICNN"""
    def __init__(self, **hyperparameterValues):
        super(BasicICNN, self).__init__()
        
        hyperparameters = {
            'latentDim': 6
        }
        hyperparameters.update(hyperparameterValues)

        self.save_hyperparameters(hyperparameters)

        # layers from the inputs
        self.initialLayer = nn.Linear(2, self.hparams.latentDim)
        self.shortcutLayer0 = nn.Linear(2, self.hparams.latentDim)
        self.shortcutLayer1 = nn.Linear(2, 1)

        # internal layers
        self.innerLayer0 = PositiveLinear(self.hparams.latentDim,
                                          self.hparams.latentDim)
        self.innerLayer1 = PositiveLinear(self.hparams.latentDim, 1)

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


def curveFit(model, targetFunction, dirname, M=128):
    """ Trains the model with early-stopping on validation loss """
    xtrain = 5*torch.randn(8*M, 2)
    ytrain = targetFunction(xtrain)

    xval = 5*torch.randn(2*M, 2)
    yval = targetFunction(xval)

    trainDL = DataLoader(TensorDataset(xtrain, ytrain), batch_size=32)
    valDL = DataLoader(TensorDataset(xval, yval), batch_size=2*M)

    earlystopping = EarlyStopping(monitor='Val Loss', mode='min', 
                                  patience=200
                                  )

    checkpoint = ModelCheckpoint(dirpath=f'lightning_logs/{dirname}',
                                 every_n_epochs=1, 
                                 save_top_k=1,
                                 monitor='Val Loss'
                                 )

    trainer = pl.Trainer(logger=WandbLogger(project='InputConvexNN'),
                         max_epochs=3000,
                         callbacks=[checkpoint, earlystopping]
                         )

    trainer.fit(model, trainDL, valDL)
