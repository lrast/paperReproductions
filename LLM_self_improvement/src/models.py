# Models for self-training
import torch
import pytorch_lightning as pl

from torch.optim.lr_scheduler import LambdaLR
from transformers import LlamaForCausalLM, AutoTokenizer

from huggingface_hub import PyTorchModelHubMixin


class LlamaLightningModule(pl.LightningModule, PyTorchModelHubMixin):
    """PyTorch Lightning module for LLaMA fine-tuning"""
    
    def __init__(self, base_model='meta-llama/Llama-3.2-1B-Instruct', 
                 learning_rate=2e-5, weight_decay=0.01, max_steps=None,
                 warmup_steps=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # Load model and tokenizer
        self.model = LlamaForCausalLM.from_pretrained(base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.tokenizer.add_special_tokens({'pad_token': '<|finetune_right_pad_id|>'})
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # training parameters that live in the model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps

        # huggingface compatibility
        self.can_generate = True

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch.get('attention_mask'),
            labels=batch['labels']
        )
        loss = outputs.loss
        
        self.log('train/loss', loss, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch.get('attention_mask'),
            labels=batch['labels']
        )
        loss = outputs.loss
        
        self.log('val/loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        if self.max_steps is not None and self.warmup_steps is not None:
            min_lr = getattr(self, 'min_learning_rate', 1e-6)  # Default min LR if not set

            def lr_lambda(current_step):
                if current_step < self.warmup_steps:
                    return float(current_step) / float(max(1, self.warmup_steps))
                progress = float(current_step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
                # Linear decay after warmup
                lr = max(0.0, 1.0 - progress)
                # Scale base LR down to lower bound
                base_lr = self.learning_rate
                scaled_lr = lr * (base_lr - min_lr) + min_lr
                return scaled_lr / base_lr

            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }
        else:
            return optimizer
