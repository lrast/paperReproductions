import torch
import wandb
import pytorch_lightning as pl

from torch.utils.data import TensorDataset, DataLoader

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


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


def curveFit_partial(model, targetFunction, dirname, M=512):
    """Partial input convex network trained with early-stoping on validation"""
    wandb.init(project='InputConvexNN')
    xtrain = torch.randn(8*M, 2)
    ytrain = torch.randn(8*M, 2)

    xval = torch.randn(2*M, 2)
    yval = torch.randn(2*M, 2)

    # noisy training targets
    target_train = targetFunction(xtrain, ytrain) + 0.1*torch.randn(8*M)[:, None]
    target_val = targetFunction(xval, yval) + 0.1*torch.randn(2*M)[:, None]

    trainDL = DataLoader(TensorDataset(xtrain, ytrain, target_train),
                         batch_size=32, shuffle=True)
    valDL = DataLoader(TensorDataset(xval, yval, target_val),
                       batch_size=2*M)

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
    wandb.finish()
