"""Minimal neural network training loop with configurable hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass
import simple_parsing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
    DataCollatorWithPadding
)

from datasets import load_dataset
from prompts import prompts

from tqdm import tqdm
import wandb
import bitsandbytes as bnb


@dataclass(frozen=True)
class Hyperparameters:
    # model parameters
    base_model: str = 'google/gemma-2-2b'
    temperature: float = 0.7  # sampling temperature

    # data parameters
    query_dataset: str = 'trl-lib/tldr'
    response_length: int = 100
    max_length: int = 700

    # training parameters
    batch_size: int = 1
    gradient_accumulation_steps: int = 32

    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    epochs: int = 5
    num_warmup_steps: int = 500

    # IRL objective
    use_IRL: bool = False
    reg_lambda: float = 0.1
    train_temp: float = 1.
    discount: float = 0.9

    # seed
    seed: int = 137

    # development
    local_files_only: bool = False

    # wandb
    use_wandb: bool = True
    wandb_project: str = "scalable_IRL"
    wandb_run_name: str | None = None
    wandb_log_every: int = 10


def model_setup(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        local_files_only=args.local_files_only,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16
    )

    return model


class CustomCollator():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.base_collator = DataCollatorWithPadding(tokenizer)
        self.sep_token = tokenizer.convert_tokens_to_ids(tokenizer.all_special_tokens[-1])
        self.pad_token = tokenizer.pad_token_id

    def mask_question(self, input_ids, ignore_index=-100):
        labels = input_ids.clone()
        
        # Create a mask: True for all tokens after the first occurrence of sep_token
        sep_mask = (input_ids == self.sep_token).cumsum(dim=-1) > 0
        
        # Set labels to -100 where the mask is False (the question and the SEP token itself)
        labels[~sep_mask] = ignore_index
        labels[labels == self.sep_token] = ignore_index
        labels[labels == self.pad_token] = ignore_index

        return labels

    def __call__(self, batch):
        """ data collator """
        batch = self.base_collator(batch)
        labels = self.mask_question(batch['input_ids'])

        batch['labels'] = labels

        return batch


def make_dataloaders(args: Hyperparameters) -> tuple[DataLoader, DataLoader]:
    g = torch.Generator().manual_seed(args.seed)

    dataset = load_dataset(args.query_dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model,
                                              local_files_only=args.local_files_only
                                              )

    prompt = prompts[args.query_dataset]['prompt']
    input_col = prompts[args.query_dataset]['input']
    target_col = prompts[args.query_dataset]['target']

    sep_token = tokenizer.all_special_tokens[-1]

    def preprocess(rows):
        """
        Apply the prompt and tokenizes the question answer pair.
        """
        questions = [prompt.format(input=a) for a in rows[input_col]]
        answers = rows[target_col]

        full_texts = [question + sep_token + answer + tokenizer.eos_token
                      for question, answer in zip(questions, answers)]

        # Tokenize full sequence
        return tokenizer(full_texts, truncation=True, max_length=args.max_length)

    dataset = dataset.map(preprocess, batched=True,
                          remove_columns=dataset['train'].column_names
                          )

    train_dl = DataLoader(
        dataset['train'],
        collate_fn=CustomCollator(tokenizer),
        batch_size=args.batch_size,
        shuffle=True,
        generator=g,
        #num_workers=3,
        #persistent_workers=True,
        #prefetch_factor=2,
    )

    val_dl = DataLoader(
        dataset['validation'],
        collate_fn=CustomCollator(tokenizer),
        batch_size=args.batch_size,
        shuffle=False,
        #num_workers=3,
        #persistent_workers=True,
        #prefetch_factor=2,
    )
    return train_dl, val_dl


def train_one_epoch(
                    model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    scheduler,
                    accelerator: Accelerator,
                    args,
                    *,
                    global_step: int = 0,
                    wandb_run=None,
                    log_every: int = 10,
                    ) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader):
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                if wandb_run is not None and (global_step % log_every == 0):
                    current_lr = scheduler.get_last_lr()[0]

                    wandb_run.log(
                        {
                            "train/loss": float(loss.detach()),
                            "train/lr": float(current_lr),
                        },
                        step=global_step,
                    )

            total_loss += float(loss.detach())
            n_batches += 1

    return total_loss / max(n_batches, 1), global_step


def train_one_epoch_IRL(
                    model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    scheduler,
                    accelerator: Accelerator,
                    args,
                    *,
                    global_step: int = 0,
                    wandb_run=None,
                    log_every: int = 10,
                    ) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader):
        with accelerator.accumulate(model):
            outputs = model(**batch)

            # IRL loss function 
            # treat logits as Q values, policies are softmaxes,
            # values are log sum exp

            shift_logits = outputs.logits[..., :-1, :].reshape(-1, outputs.logits.shape[-1])
            shift_labels = batch['labels'][..., 1:].reshape(-1)
            valid = shift_labels != -100

            neg_log_policy = F.cross_entropy(shift_logits[valid] / args.train_temp,
                                             shift_labels[valid],
                                             reduction='none'
                                             )

            values = torch.logsumexp(outputs.logits / args.train_temp, -1)
            values_curr = values[..., :-1]
            values_next = values[..., 1:]
            value_diffs = (values_curr - args.discount * values_next).reshape(-1)[valid]

            loss = (args.reg_lambda*(value_diffs - neg_log_policy)**2
                    + neg_log_policy 
                    ).mean()

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                if wandb_run is not None and (global_step % log_every == 0):
                    current_lr = scheduler.get_last_lr()[0]

                    wandb_run.log(
                        {
                            "train/loss": float(loss.detach()),
                            "train/lr": float(current_lr),
                        },
                        step=global_step,
                    )

            total_loss += float(loss.detach())
            n_batches += 1

    return total_loss / max(n_batches, 1), global_step


@torch.no_grad()
def evaluate(
             model: nn.Module,
             loader: DataLoader,
             ) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(loader):
        outputs = model(**batch)

        total_loss += outputs.loss.item()
        preds = outputs.logits.argmax(dim=-1)

        labels = batch['labels']
        scored = labels != -100

        correct += int((preds[scored] == labels[scored]).sum())
        total += scored.sum().item()

    n_batches = max(len(loader), 1)
    return total_loss / n_batches, correct / max(total, 1)


def main(args: Hyperparameters | None = None) -> None:
    args = args or Hyperparameters()
    print(args.use_IRL)
    torch.manual_seed(args.seed)

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)

    wandb_run = None
    if args.use_wandb:
        if accelerator.is_main_process:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={
                    "base_model": args.base_model,
                    "temperature": args.temperature,
                    "query_dataset": args.query_dataset,
                    "response_length": args.response_length,
                    "max_length": args.max_length,
                    "batch_size": args.batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "epochs": args.epochs,
                    "seed": args.seed,
                    "local_files_only": args.local_files_only,
                },
            )

    model = model_setup(args)

    train_dl, val_dl = make_dataloaders(args)

    optimizer = bnb.optim.AdamW8bit(model.parameters(),
                                    lr=args.learning_rate,
                                    weight_decay=args.weight_decay
                                    )

    scheduler = get_scheduler(
                                "cosine",
                                optimizer=optimizer,
                                num_warmup_steps=args.num_warmup_steps,
                                num_training_steps=args.epochs * len(train_dl),
                            )

    model, train_dl, val_dl, optimizer, scheduler = accelerator.prepare(model, train_dl, val_dl, optimizer, scheduler)
    #model = torch.compile(model, mode="max-autotune")

    if args.use_IRL:
        train_epoch = train_one_epoch_IRL
    else:
        train_epoch = train_one_epoch

    global_step = 0
    for epoch in range(args.epochs):
        train_loss, global_step = train_epoch(
            model,
            train_dl,
            optimizer,
            scheduler,
            accelerator,
            args,
            global_step=global_step,
            wandb_run=wandb_run,
            log_every=args.wandb_log_every,
        )

        val_loss, val_acc = evaluate(model, val_dl)
        print(
            f"epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.3f}"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/epoch_loss": float(train_loss),
                    "val/loss": float(val_loss),
                    "val/acc": float(val_acc),
                },
                step=global_step,
            )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    args = simple_parsing.parse(Hyperparameters)
    main(args)
