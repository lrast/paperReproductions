# runs inference with a trained model

import wandb
import argparse
from omegaconf import OmegaConf
# import vllm # If using vLLM
# import torch

from src.data import get_raw_dataset, model_answers


def generate_data():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--data_out', required=True)

    parser.add_argument('--config_path', required=True)
    parser.add_argument('--logger_id', default=None)
    parser.add_argument('--epoch_num', default=0)

    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)
    run = wandb.init(id=args.logger_id)

    train_raw = get_raw_dataset(**cfg.data, split='train')
    val_raw = get_raw_dataset(**cfg.data, split='eval')

    next_dataset, train_metrics = model_answers(args.model_path, train_raw, **cfg.data)
    next_dataset.save_to_disk(args.data_out)
    _, val_metrics = model_answers(args.model_path, val_raw, **cfg.data)

    # Log the metrics to wandb 

    train_metrics = {**{'train/'+k: v for k, v in train_metrics.items()},
                     **{'train/epoch': float(args.epoch_num)}}
    val_metrics = {**{'eval/'+k: v for k, v in val_metrics.items()},
                   **{'train/epoch': float(args.epoch_num)}}

    print('metrics: ', train_metrics)
    run.log(train_metrics)
    run.log(val_metrics)


if __name__ == "__main__":
    generate_data()
