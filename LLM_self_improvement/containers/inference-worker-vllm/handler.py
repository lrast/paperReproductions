"""Example handler file."""

import runpod
import torch

from vllm import LLM, SamplingParams
from datasets import load_from_disk
from answer_checker import make_new_dataset


loaded_model = {
    'name': 'llama_pretrain',
    'model': LLM('workspace/llama_pretrain')
}

sampling_params = SamplingParams(n=8, temperature=1., max_tokens=512)


def filter_ground_truth(dataset):
    dataset = dataset[dataset['answer'] == dataset['gt']]
    return dataset[['question', 'answer', 'gt']]


def filter_majority(dataset):
    dataset = dataset[dataset['answer'] == dataset['majority_vote']]
    return dataset[['question', 'output', 'answer', 'gt']]


filter_methods = {
                    'ground truth': filter_ground_truth,
                    'majority': filter_majority,
                    None: lambda x: x
                }


def handler(job):
    """Handler function that will be used to process jobs."""
    job_input = job["input"]

    # model
    model_path = job_input["model"]
    if model_path != loaded_model['name']:
        del loaded_model['model']
        torch.cuda.empty_cache()

        loaded_model['name'] = model_path
        loaded_model['model'] = LLM(f'workspace/{model_path}')

    # data
    data_path = job_input["data"]
    data_split = job_input.get("split", "train")

    dataset = load_from_disk(data_path)[data_split]

    # run
    samples = loaded_model['model'].generate(dataset['question'],
                                             sampling_params=sampling_params
                                             )

    filter_method = filter_methods[job_input['filter_by']]
    filename = job_input.get('data_out', None)

    new_dataset, score = make_new_dataset(dataset, samples, filter_method)

    if filename is not None:
        new_dataset.save_to_disk(f'workspace/{filename}')

    return score


runpod.serverless.start({"handler": handler})
