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
    wandb.init(id=args.logger_id)

    model, tokenizer = load_model_and_tokenizer(args.resume_from)
    dataset = load_from_disk(args.data_in)
    # make the dataset chat formatted
    dataset = dataset.map(chat_formatted_QA, remove_columns=dataset.column_names)

    # Train
    if int(args.epoch_num) == 0:
        resume_from = None
    else:
        resume_from = args.resume_from

    # override resumption logic
    resume_from = None

    trainer = get_trainer(model, tokenizer, dataset,
                          resume_from=resume_from, current_epoch=args.epoch_num,
                          num_train_epochs=1, output_dir=args.ckpt_out,
                          **config.train)

    print('global step before:', trainer.state.global_step)
    trainer.train()
    print('global step after:', trainer.state.global_step)


if __name__ == "__main__":
    train()
