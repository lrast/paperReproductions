# Train a model for a single epoch
import wandb
import argparse

from omegaconf import OmegaConf
from datasets import load_from_disk

from src.models import load_model_and_tokenizer
from src.trainer import get_trainer
from src.data import chat_formatted_QA


def train():
    """ Run a single step of training """
    # Arguments parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_in', required=True)
    parser.add_argument('--resume_from', required=True)
    parser.add_argument('--ckpt_out', required=True)

    parser.add_argument('--config_path', required=True)
    parser.add_argument('--logger_id', default=None)
    parser.add_argument('--epoch_num', default=0)

    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)

    run = wandb.init(project='self-improvement',
                     group=args.logger_id,
                     job_type='child',
                     name=f'train_step_{args.epoch_num}'
                     )

    model, tokenizer = load_model_and_tokenizer(args.resume_from)
    dataset = load_from_disk(args.data_in)
    # make the dataset chat formatted
    dataset = dataset.map(chat_formatted_QA, remove_columns=dataset.column_names)

    # Train
    trainer = get_trainer(model, tokenizer, dataset,
                          output_dir=args.ckpt_out,
                          **config.train)

    trainer.train()
    trainer.save_model(args.ckpt_out)
    run.finish()


if __name__ == "__main__":
    train()
