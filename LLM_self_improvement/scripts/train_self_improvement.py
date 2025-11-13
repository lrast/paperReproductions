# Runs the self-improvement loop
import hydra

from hydra.core.hydra_config import HydraConfig
from pathlib import Path

from src.models import LlamaLightningModule
from src.data import SelfImprovementDataModule
from src.trainer import get_trainer


@hydra.main(config_path="../configs", config_name="base", version_base="1.2")
def main(config):
    """ The core self-improvement iteration: alternate between model updates
    and dataset updates. """

    # Current objective: simple fintuning for ground truth answers.
    model = LlamaLightningModule(**config.model)
    dataset = SelfImprovementDataModule(model, **config.data)

    output_dir = Path(HydraConfig.get().runtime.output_dir) 

    trainer = get_trainer(dataset, **config.train,
                          output_dir=output_dir / 'checkpoints',
                          )

    # initial answers setup
    


    trainer.fit(model, datamodule=dataset)


if __name__ == '__main__':
    main()
