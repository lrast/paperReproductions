# Trainers for self-training
import shutil
import pytorch_lightning as pl

from pytorch_lightning.callbacks import LearningRateMonitor, Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


def get_trainer(datamodule, num_train_epochs=1, 
                output_dir=None, run_name='debug', 
                **kwargs):
    """
    Get PyTorch Lightning trainer
    
    Args:
        output_dir: Directory to save checkpoints
        run_name: Name for wandb run
        num_train_epochs: Number of training epochs
        datamodule: Optional DataModule instance for epoch-end updates
        **kwargs: Additional arguments for Lightning Trainer
    """

    # Dataset update callback: saves checkpoints and updates the dataset

    dataset_update_callback = DatasetUpdateCallback(datamodule, output_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='best-checkpoint',
        save_top_k=1,
        monitor='val/loss',
        mode='min'
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    callbacks = [lr_monitor, dataset_update_callback, checkpoint_callback]
    
    # Setup wandb logger
    wandb_logger = WandbLogger(
        project='huggingface',
        name=run_name + '_manual'
    )

    # Create Lightning trainer
    trainer = pl.Trainer(
        max_epochs=num_train_epochs,
        callbacks=callbacks,
        logger=wandb_logger,
        enable_progress_bar=True,
        log_every_n_steps=1,
        reload_dataloaders_every_n_epochs=1,
        **kwargs
    )
    
    return trainer


class DatasetUpdateCallback(Callback):
    """
    Callback for generating new datasets from the current model
    1. save current model to disk 
    2. invoke the datamodule's update
    3. log the metrics
    """
    
    def __init__(self, datamodule, dirpath=None,
                 filename='checkpoint-epoch={epoch}', **kwargs):
        super().__init__()
        self.datamodule = datamodule
        self.dirpath = dirpath
        self.filename = filename

    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of each training epoch"""
        model_path = self.dirpath / self.filename.format(epoch=trainer.current_epoch)
        pl_module.save_pretrained(model_path)

        train_metrics, val_metrics = self.datamodule.update_model(model_path)

        shutil.rmtree(model_path)

        train_metrics = {f'train/{key}': value for key, value in train_metrics.items()}
        val_metrics = {f'eval/{key}': value for key, value in val_metrics.items()}

        pl_module.log_dict({**train_metrics, **val_metrics})
