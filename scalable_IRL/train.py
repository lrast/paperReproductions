"""Minimal neural network training loop with configurable hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass
import simple_parsing

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
)

from datasets import load_dataset
from prompts import prompts

from tqdm import tqdm


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
    batch_size: int = 4
    gradient_accumulation_steps: int = 8

    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 5
    train_samples: int = 2048
    val_samples: int = 512

    # seed
    seed: int = 137

    # development
    local_files_only: bool = False


def model_setup(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        local_files_only=args.local_files_only
    )

    return model


def make_dataloaders(args: Hyperparameters) -> tuple[DataLoader, DataLoader]:
    g = torch.Generator().manual_seed(args.seed)

    dataset = load_dataset(args.query_dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model,
                                              local_files_only=args.local_files_only
                                              )

    prompt = prompts[args.query_dataset]['prompt']
    input_col = prompts[args.query_dataset]['input']
    target_col = prompts[args.query_dataset]['target']

    def preprocess(rows):
        """
        Apply the prompt and tokenizes the question answer pair.
        """
        questions = [prompt.format(input=a) for a in rows[input_col]]
        answers = rows[target_col]

        full_texts = [question + answer + tokenizer.eos_token
                      for question, answer in zip(questions, answers)]

        # Tokenize full sequence
        inputs = tokenizer(
                            full_texts, 
                            truncation=True, 
                            max_length=args.max_length, 
                            padding="max_length", 
                            return_tensors="pt"
                        )

        # 3. Copy input_ids to labels
        labels = inputs["input_ids"].clone()

        # 4. Mask the question item by item
        answer_encodings = tokenizer(
            answers, 
            truncation=True, 
            max_length=args.max_length, 
            add_special_tokens=False
        )

        for i, tokens in enumerate(answer_encodings["input_ids"]):
            answer_len = len(tokens)
            # Set all tokens outside of the answer
            labels[i, :-(answer_len+1)] = -100

        # 5. Mask the padding tokens so they don't contribute to loss
        labels[labels == tokenizer.pad_token_id] = -100

        return {
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "labels": labels.squeeze(0)
            }

    dataset = dataset.map(preprocess, batched=True,
                          remove_columns=dataset['train'].column_names
                          )

    dataset.set_format('pt')

    train_dl = DataLoader(
        dataset['train'],
        batch_size=args.batch_size,
        shuffle=True,
        generator=g,
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=2,  
    )
    val_dl = DataLoader(
        dataset['validation'],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=2,
    )
    return train_dl, val_dl


def train_one_epoch(
                    model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    scheduler,
                    accelerator: Accelerator
                    ) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader):
        with accelerator.accumulate(model):
            outputs = model(**batch)
            print('outputs:', outputs.logits.shape)
            loss = outputs.loss
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += float(loss.detach())
            n_batches += 1

    return total_loss / max(n_batches, 1)


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
    torch.manual_seed(args.seed)

    model = model_setup(args)
    model = model.to(torch.bfloat16)

    train_dl, val_dl = make_dataloaders(args)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay)

    scheduler = get_scheduler(
                                "cosine",
                                optimizer=optimizer,
                                num_warmup_steps=50,
                                num_training_steps=args.epochs * len(train_dl),
                            )

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps
                              )

    model, train_dl, val_dl, optimizer, scheduler = accelerator.prepare(model, train_dl, val_dl, optimizer, scheduler)
    model = torch.compile(model, backend="aot_eager")


    #for epoch in range(args.epochs:
    #    train_loss = train_one_epoch(model, train_dl, loss_fn, optimizer)
    #    val_loss, val_acc = evaluate(model, val_dl, loss_fn)
    #    print(
    #        f"epoch {epoch:03d} | train_loss={train_loss:.4f} | "
    #        f"val_loss={val_loss:.4f} | val_acc={val_acc:.3f}"
    #    )


if __name__ == "__main__":
    args = simple_parsing.parse(Hyperparameters)
    main(args)
