# Datasets for self-training
import pandas as pd

from datasets import load_dataset, Dataset
from transformers import pipeline

from src.answer_generation import check_format_and_get_answer, score_results
from vllm import LLM, SamplingParams


def get_raw_dataset(prompt, split='train', seed=42, **kwargs):
    """ Set up the primary dataset """
    def extract_answer(row):
        row['gt'] = int(row['answer'].split('####')[-1].strip().replace(',', ''))
        return row

    def add_prompt(row):
        row['question'] = prompt.format(question=row['question'])
        return row

    if split == 'test':
        problems = load_dataset("openai/gsm8k", "main", split='test')
    else:
        problems = load_dataset("openai/gsm8k", "main", split='train[0:30]')

    problems = problems.map(extract_answer, batched=False,
                            ).map(add_prompt, batched=False)

    if split == 'test':
        return problems

    problems = problems.train_test_split(test_size=0.1, seed=seed)
    val_set = problems.pop('test')
    problems['eval'] = val_set

    if split == 'train':
        return problems['train']
    elif split == 'val' or split == 'valid' or split =='eval':
        return problems['eval']
    else:
        raise KeyError('unknown dataset split')


def model_answers(model_dir, raw_dataset, generation_mode,
                  use_vllm=False,
                  **kwargs):
    """Use the model to generate a question / answer dataset"""
    # question: does this use the tokenizers chat format?

    if use_vllm:
        raise NotImplementedError('working on vllm compatibility')
        sampling_params = SamplingParams(n=8, temperature=1.0)
        model = LLM(model_dir)

    else:
        model = pipeline("text-generation",
                         model=model_dir,
                         tokenizer=model_dir,
                         return_full_text=False,
                         do_sample=True, temperature=1.,
                         num_return_sequences=8)

        QA_pairs, metrics = generate_QA_pairs(raw_dataset, model, generation_mode)

    return QA_pairs, metrics


def generate_QA_pairs(dataset, pipeline, generation_mode):
    """ Primary interface for answer generation"""
    metrics = []
    outputs = []

    for batch in dataset.iter(batch_size=8):
        # make sure that the inputs are chat formatted
        chat_formatted = list(map(chat_formatted_problems, batch['question']))

        # Question: do we need to add an assistant prompt to start generation?
        # Empirically, we're fine without it.

        answers = pipeline(chat_formatted)
        rows, batch_metrics = make_new_rows(batch, answers)
        metrics.extend(batch_metrics)

        # filter the answers according to the generation mode.
        match generation_mode:
            case 'gt_answers':
                return batch
            case 'gt_targets':
                rows = rows[rows['result'] == rows['gt']]
            case 'majority_vote':
                rows = rows[rows['result'] == rows['majority_vote']]
            case 'perfect_formatting':
                rows = rows[rows['perfect_formatting'] == 1]
            case 'good_formatting':
                rows = rows[rows['good_formatting'] == 1]
            case _:
                raise Exception('Unkown generation mode')

        outputs.append(rows.drop(columns=['good_formatting', 'perfect_formatting', 'result',
                                 'majority_vote']))

    data_with_answers = Dataset.from_pandas(pd.concat(outputs))

    return data_with_answers, pd.DataFrame(metrics).mean().to_dict()


def make_new_rows(inputs, outputs):
    """ Parse the pipeline inputs and outputs to a new dataset, with 
    answer evaluations
    """
    full_outputs = []
    metrics = []
    
    for ind in range(len(outputs)):
        question = inputs['question'][ind]
        gt = inputs['gt'][ind]

        results = pd.DataFrame(map(check_format_and_get_answer, outputs[ind]))

        majority_value = results['result'].mode()
        if len(majority_value) > 1:
            results['majority_vote'] = None
        else:
            try:
                results = results.assign(majority_vote=majority_value.item())
            except:
                print('!!!!!!!!!!!!! error: majority of ', majority_value)

        results = results.assign(gt=gt, question=question)
        full_outputs.append(results)

        # update the metric
        metrics.append(score_results(results))

    return pd.concat(full_outputs, ignore_index=True), metrics


def chat_formatted_problems(question):
    return [{"role": "user", "content": question}]


def chat_formatted_QA(row):
    return {'messages': [{"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row['answer']}]}
