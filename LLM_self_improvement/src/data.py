# Datasets for self-training
import torch
import gc
import pytorch_lightning as pl

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import pipeline, DataCollatorForLanguageModeling

from src.models import LlamaLightningModule
from src.answer_generation import generate_QA_pairs
#from vllm import LLM


class SelfImprovementDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for self-improvement training
    Re-generates answers between epochs.
    """
    
    def __init__(self, checkpoint, tokenizer, prompt='{question}',
                 split='train', generation_mode='gt_answers', chat_format=False,
                 use_vllm=False, seed=42,
                 batch_size=4, num_workers=0, 
                 max_length=512, **kwargs):
        super().__init__()

        # Set up the primary dataset
        def extract_answer(row):
            row['gt'] = int(row['answer'].split('####')[-1].strip().replace(',', ''))
            return row

        def add_prompt(row):
            row['question'] = prompt.format(question=row['question'])
            return row

        problems = load_dataset("openai/gsm8k", "main", split=split)
        self.problems = problems.map(extract_answer, batched=False,
                               ).map(add_prompt, batched=False
                               ).train_test_split(test_size=0.1, seed=seed)
        # rename the validation set
        val_set = self.problems.pop('test')
        self.problems['eval'] = val_set

        # generation parameters
        self.use_vllm = use_vllm
        self.generation_mode = generation_mode
        self.chat_format = chat_format

        # train parameters that live in the dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        
        self.tokenizer = tokenizer
        # Create data collator
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            return_tensors='pt',
            mlm=False  # Causal LM, not masked LM,
        )

        # initialize the dataset
        self.train_dataset = None
        self.val_dataset = None
        self.checkpoint = checkpoint
    
    def setup(self, stage=None):
        """Setup datasets for training/validation"""

        with TemporaryModel(self.checkpoint, self.use_vllm) as generation_pipe:
            def generate_tokenized_answers(dataset):
                # generate new question-answer pairs
                QA_pairs, metrics = generate_QA_pairs(dataset, generation_pipe,
                                                      self.generation_mode)
                # Tokenize the dataset
                tokenized_QA = QA_pairs.map(self.tokenized_chat, batched=False,
                                            remove_columns=QA_pairs.column_names)

                return tokenized_QA, metrics

            self.train_dataset, train_metrics = generate_tokenized_answers(self.problems['train'])
            self.val_dataset, val_metrics = generate_tokenized_answers(self.problems['eval'])

            self.train_dataset.set_format(type='torch')
            self.val_dataset.set_format(type='torch')

        return train_metrics, val_metrics

    def update_model(self, model_checkpoint):
        """
        Reload the model that is used for generating data
        """
        self.checkpoint = model_checkpoint
        # delete old 
        del self.train_dataset
        del self.val_dataset

        train_metrics, val_metrics = self.setup()
        return train_metrics, val_metrics

    def tokenized_chat(self, row, include_answer=True):
        """ Tokenization for question-answer pairs """
        if self.chat_format:
            messages = [
                {"role": "user", "content": row['question']}
            ]
            if include_answer:
                messages.append({"role": "assistant", "content": row['answer']})

            return self.tokenizer.apply_chat_template(messages, tokenize=True,
                                                      return_dict=True,
                                                      add_generation_prompt=(not include_answer),
                                                      )
        else:
            if include_answer:
                output = self.tokenizer(row['question'] + 'Answer:' + row['answer'])
            else:
                output = self.tokenizer(row['question'])
            return output

    def train_dataloader(self):
        """Return training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    def val_dataloader(self):
        """Return validation dataloader"""
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
            pin_memory=True if torch.cuda.is_available() else False
        )


class TemporaryModel(object):
    def __init__(self, model_checkpoint, use_vllm):
        self.use_vllm = use_vllm
        self.raw_model = LlamaLightningModule.from_pretrained(model_checkpoint)

        if self.use_vllm:
            # apply vllm model
            raise NotImplementedError('vllm later')
        else:
            self.model = pipeline("text-generation",
                                  model=self.raw_model.model,
                                  tokenizer=self.raw_model.tokenizer,
                                  return_full_text=False,
                                  do_sample=True, temperature=1.,
                                  num_return_sequences=8)

    def __enter__(self):
        return self.model

    def __exit__(self, type, value, traceback):
        self.raw_model.to('cpu')
        del self.raw_model
        del self.model
        gc.collect()
        torch.mps.empty_cache()
