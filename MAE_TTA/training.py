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


def full_trainer_classification(model, train_dataset, val_dataset, **kwargs):
    """ Trainer for full models to perform classification 
    """
    model.unfreeze_all()

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


def make_hf_optimizer(model, **kwargs):
    training_args = TrainingArguments(**kwargs)

    optimizer = Trainer(model,
                        args=training_args,
                        ).create_optimizer()
    return optimizer


def test_time_adaptation(model, inputs, labels=None, 
                         repeats=128, steps=20, evaluate_freq=20,
                         device='cuda:0', mask_ratio=0.75,
                         **kwargs):
    """
        Trainer for TTA: low level trainer optimized for speed.
    """
    # setup
    if f'{model.embedding.device}' != device:
        model = model.to(device)
    embedding_model = model.embedding

    embedding_model.enable_masking(mask_ratio)

    train_data = model.preprocess(inputs)
    inputs = inputs.to(device)
    train_data = train_data.to(device)

    if train_data.shape[0] == 1:
        train_data = train_data.expand([repeats, 3, 224, 224])
    else:
        # multiple input points
        train_data = train_data.expand([repeats, train_data.shape[0], 3, 224, 224])
        train_data = train_data.reshape(-1, 3, 224, 224)

    def run_eval():
        """ Model accuracy evaluation  """
        model.disable_masking()

        preds = model.classify(inputs)
        accuracy = (preds == labels).sum().item() / inputs.shape[0]

        model.enable_masking(mask_ratio)
        return accuracy

    if labels is not None:
        # setup our evaluations 
        labels = labels.to(device)

        num_evals = steps // evaluate_freq + 1
        results = np.zeros(num_evals)
        results[0] = run_eval()

    # train loop
    opt = make_hf_optimizer(embedding_model, **kwargs)
    for i in range(1, steps+1):
        # train
        opt.zero_grad()
        loss = embedding_model(train_data).loss
        loss.backward()
        opt.step()

        # evaluate
        if labels is not None and i % evaluate_freq == 0:
            results[i // evaluate_freq] = run_eval()

    if labels is not None:
        return results


def args_filter(args, func):
    possible_args = inspect.signature(func).parameters
    return {k: v for k, v in args.items() if k in possible_args}
