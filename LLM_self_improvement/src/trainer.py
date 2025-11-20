# Trainers for self-training
from trl import SFTTrainer, SFTConfig


def get_trainer(model, tokenizer, dataset,
                num_train_epochs=1, output_dir='debug',
                **kwargs):
    """ SFT Trainer for the model
    Doesn't load the trainer state.
    """
    training_args = SFTConfig(assistant_only_loss=True,
                              num_train_epochs=num_train_epochs,
                              output_dir=output_dir,
                              max_steps=-1,
                              report_to="wandb",
                              save_strategy="no",
                              **kwargs)

    trainer = SFTTrainer(model=model,
                         processing_class=tokenizer,
                         train_dataset=dataset,
                         args=training_args,
                         )

    return trainer
