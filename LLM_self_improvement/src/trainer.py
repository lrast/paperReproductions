# Trainers for self-training
import os
import shutil
import json

from pathlib import Path
from transformers import TrainerCallback
from trl import SFTTrainer, SFTConfig


def get_trainer(model, tokenizer, dataset, resume_from=None,
                current_epoch=0,
                num_train_epochs=1, output_dir='debug',
                **kwargs):
    """Make single epoch trainer for model"""
    training_args = SFTConfig(assistant_only_loss=True,
                              num_train_epochs=num_train_epochs,
                              output_dir=output_dir,
                              max_steps=-1,
                              report_to="wandb",
                              save_strategy="epoch",
                              **kwargs)

    trainer = SFTTrainer(model=model,
                         processing_class=tokenizer,
                         train_dataset=dataset,
                         args=training_args,
                         callbacks=[RenameCheckpointToBest()]
                         )

    # manually reload trainer state to prevent step based re-calculations
    if resume_from is not None:
        # load the previous trainer state and global step count
        with open(Path(resume_from) / 'trainer_state.json', "r") as f:
            state_dict = json.load(f)
            trainer.state.global_step = state_dict['global_step']

        trainer.state.epoch = current_epoch

        # Force creation of optimizer and scheduler
        trainer.create_optimizer()
        trainer.create_scheduler(num_training_steps=0)

        trainer._load_optimizer_and_scheduler(resume_from)
        trainer.lr_scheduler.last_epoch = trainer.state.global_step - 1

    return trainer


class RenameCheckpointToBest(TrainerCallback):
    """
    Saves a copy of the full, resumable checkpoint to a fixed name
    after the Trainer's internal checkpointing process completes.
    """
    def on_save(self, args, state, control, **kwargs):
        # 1. Find the path to the most recent checkpoint folder
        latest_checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        
        # 2. Define the target directory with your known, stable name
        custom_save_name = "post_train"
        custom_save_path = os.path.join(args.output_dir, custom_save_name)

        # Ensure the Trainer successfully created the checkpoint folder
        if os.path.isdir(latest_checkpoint_dir):
            print(f"Copying full checkpoint from {latest_checkpoint_dir} to {custom_save_path}")
            
            # 3. Handle existing custom folder (delete/clean before copying)
            if os.path.exists(custom_save_path):
                shutil.rmtree(custom_save_path)
            
            # 4. remave the checkpoint folder
            shutil.move(latest_checkpoint_dir, custom_save_path)
        else:
            # This should generally not happen if save_strategy is set
            print(f"Warning: Checkpoint folder {latest_checkpoint_dir} not found.")
