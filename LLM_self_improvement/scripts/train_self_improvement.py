# Runs the self-improvement loop
import hydra
import shutil

from hydra.core.hydra_config import HydraConfig
from pathlib import Path

from src.models import LlamaLightningModule
from src.data import SelfImprovementDataModule
from src.trainer import get_trainer


@hydra.main(config_path="../configs", config_name="base", version_base="1.2")
def main(config):
    """ The core self-improvement iteration: alternate between model updates
    and dataset updates. """
    output_dir = Path(HydraConfig.get().runtime.output_dir) 

    model = LlamaLightningModule(**config.model)
    initial_ckpt = output_dir / 'checkpoints/initial'
    model.save_pretrained(initial_ckpt)

    dataset = SelfImprovementDataModule(initial_ckpt, model.tokenizer, **config.data)

    trainer = get_trainer(dataset, **config.train,
                          output_dir=output_dir / 'checkpoints',
                          )

    trainer.fit(model, datamodule=dataset)
    shutil.rmtree(initial_ckpt)


if __name__ == '__main__':
    main()
