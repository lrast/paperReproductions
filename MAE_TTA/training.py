# Files for efficient training
import torch
import evaluate
import inspect
import numpy as np

from transformers import Trainer, TrainingArguments


def raw_batch_collator(data):
    return {'pixel_values': torch.stack([ele[0] for ele in data]),
            'labels': torch.stack([ele[1] for ele in data])
            }


def compute_metrics(eval_preds):
    metric = evaluate.load('accuracy')
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def make_cosine_annealing_scheduler(**kwargs):
    """ Learning rate scheduler that anneals from warmup"""
    pass


def probing_trainer(model, train_dataset, val_dataset, **kwargs):
    """ Trainer for probing models, that only train the classifier.
        Goal: _good enough_ performance across a range of ViT probing
    """
    model.freeze_embedding()

    kwarg_defaults = {
        'learning_rate': 5E-5,
        'num_train_epochs': 3,

        'weight_decay': 0.01,

        'lr_scheduler_type': 'cosine',
        'warmup_ratio': 0.05,

        'logging_steps': 50,
        'logging_strategy': "steps",

        'eval_strategy': "epoch",
        'metric_for_best_model': 'eval_accuracy',
        'save_strategy': "best",
        'save_total_limit': 1,

        'output_dir': 'initial_train',
        'dataloader_num_workers': 8
    }

    all_args = {**kwarg_defaults, **kwargs}

    training_args = TrainingArguments(**args_filter(all_args, TrainingArguments))

    # potential future implementation
    #optimizer = AdamW(model.parameters(), **args_filter(all_args, AdamW))

    #lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup(
    #    optimizer=optimizer,
    #    **args_filter(all_args, get_cosine_with_min_lr_schedule_with_warmup)
    #)

    trainer = Trainer(model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=val_dataset,
                      data_collator=raw_batch_collator,
                      compute_metrics=compute_metrics,
                      )
    return trainer


def args_filter(args, func):
    possible_args = inspect.signature(func).parameters
    return {k: v for k, v in args.items() if k in possible_args}
