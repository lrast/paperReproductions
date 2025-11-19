# Runs the self-improvement loop
import hydra
import shutil
import subprocess
import wandb

from hydra.core.hydra_config import HydraConfig
from pathlib import Path

from src.models import initialize_model_and_tokenizer


@hydra.main(config_path="../configs", config_name="base", version_base="1.2")
def main(config):
    """ The core self-improvement iteration: alternate between model updates
    and dataset updates. """
    output_dir = Path(HydraConfig.get().runtime.output_dir) 
    run_config = output_dir / '.hydra/config.yaml'

    ckpt_name = 'checkpoints/epoch_{epoch}'
    data_name = 'data/epoch_{epoch}'

    clear_old = True
    # initialization steps: initialize the dataset

    # initialize the wandb run, send forward run id
    run = wandb.init(project='self-improvement', name=config.train.run_name)
    run_id = run.id

    # initialization step
    curr_ckpt = output_dir / ckpt_name.format(epoch=-1) / 'post_train'
    curr_data = output_dir / data_name.format(epoch=-1)

    initialize_model_and_tokenizer(curr_ckpt, **config.model)

    run_process('scripts.inference_step',
                ['--config_path', run_config,
                 '--data_out', curr_data,
                 '--model_path', curr_ckpt,
                 '--logger_id', run_id,
                 '--epoch_num', str(-1)
                 ])

    for epoch in range(config.self_improvement.num_loops):
        next_ckpt = output_dir / ckpt_name.format(epoch=epoch)
        next_data = output_dir / data_name.format(epoch=epoch)

        general_args = ['--config_path', run_config,
                        '--logger_id', run_id,
                        '--epoch_num', str(epoch)
                        ]

        # train model and update the model checkpoint pointer
        run_process('scripts.train_step', 
                    ['--resume_from', curr_ckpt,
                     '--data_in', curr_data,
                     '--ckpt_out', next_ckpt,
                     ] + general_args)

        if clear_old and (epoch < config.self_improvement.num_loops - 1):
            shutil.rmtree(curr_ckpt.parent)

        curr_ckpt = next_ckpt / 'post_train'

        # make a new dataset and update the data checkpoint pointer
        run_process('scripts.inference_step',
                    ['--model_path', curr_ckpt,
                     '--data_out', next_data,
                     ] + general_args)

        if clear_old:
            shutil.rmtree(curr_data)

        curr_data = next_data

    wandb.finish()


def run_process(script_name, args):
    """Executes a script as a separate system process and waits for it to complete.
    Forwards errors to the main process to facilitate debugging.
    """
    print(f"\n===== 🚀 Starting {script_name} =====")
    command = ["python", '-m', script_name] + args
    
    # Use Popen to run the script and wait for its completion
    process = subprocess.Popen(command, stdout=None, stderr=None)
    process.wait()
    #stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print(f"!!! Error in {script_name} !!!")
        raise RuntimeError(f"{script_name} failed with exit code {process.returncode}")


if __name__ == '__main__':
    main()
