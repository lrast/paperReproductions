# runs inference with a trained model

import wandb
import argparse
import torch
import gc

from omegaconf import OmegaConf

from src.data import get_raw_dataset, model_answers


def generate_data():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--data_out', required=True)

    parser.add_argument('--config_path', required=True)
    parser.add_argument('--logger_id', default=None)
    parser.add_argument('--epoch_num', default=0)

    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)

    run = wandb.init(project='self-improvement',
                     group=args.logger_id,
                     job_type='child',
                     name=f'inference_step_{args.epoch_num}'
                     )

    train_raw = get_raw_dataset(**config.data, split='train')
    val_raw = get_raw_dataset(**config.data, split='eval')

    next_dataset, train_metrics = model_answers(args.model_path, train_raw, **config.data)
    next_dataset.save_to_disk(args.data_out)
    _, val_metrics = model_answers(args.model_path, val_raw, **config.data)

    # Log the metrics to wandb 
    train_metrics = {**{'train/'+k: v for k, v in train_metrics.items()},
                     **{'train/epoch': float(args.epoch_num)}}
    val_metrics = {**{'eval/'+k: v for k, v in val_metrics.items()},
                   **{'train/epoch': float(args.epoch_num)}}

    run.log(train_metrics, step=int(args.epoch_num))
    run.log(val_metrics, step=int(args.epoch_num))
    run.finish()

    # cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    generate_data()
