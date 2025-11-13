# basic solution checkers

import re
import pandas as pd

from datasets import Dataset, DatasetDict


def make_new_dataset(inputs, outputs, subset_selection=lambda x: x):
    """ generates a new dataset based on outputs """
    data = map(new_rows, inputs, outputs)
    dataset, scores = zip(*data)

    dataset = pd.concat(dataset).reset_index(drop=True)
    dataset['majority_vote'] = dataset['majority_vote'].astype('Int64')
    dataset['answer'] = dataset['answer'].astype('Int64')

    dataset = subset_selection(dataset)

    dataset = DatasetDict({'train': Dataset.from_pandas(dataset, preserve_index=False)})
    scores = pd.DataFrame(scores)

    return dataset, scores.mean()


def new_rows(inputs, outputs):
    """ Parse the results to a new input / output set """
    results = map(check_format_and_get_answer, outputs.outputs)

    def add_input_and_gt(row):
        row['gt'] = inputs['gt']
        row['question'] = inputs['question']
        return row

    results = pd.DataFrame(map(add_input_and_gt, results))

    majority_value = results['answer'].mode()
    if len(majority_value) > 1:
        results['majority_vote'] = None
    else:
        results['majority_vote'] = majority_value

    # score the results
    is_gt = (results['answer'] == results['gt'])
    score = {
              'pass@8': is_gt.any(), 
              'maj@8': (is_gt.sum() >= 4),
              'good_formatting': results['good_formatting'].sum() / 8,
              'perfect_formatting': results['perfect_formatting'].sum() / 8,
            }

    return results, score


def check_format_and_get_answer(output):
    """ check whether the answer ends with
        'The final answer is: __'
        and returns the imputed answer
    """
    generated_text = output.text 
    result = {'output': generated_text, 'good_formatting': 0, 'perfect_formatting': 0, 'answer': None}

    last_line = generated_text.split('\n')[-1]

    last_line_numbers = re.findall(r'[0-9,]*\.?[0-9]+', last_line)

    if len(last_line_numbers) > 0:
        result['answer'] = int(float(last_line_numbers[-1].replace(',', '')))

        if last_line.startswith('The final answer is:'):
            result['perfect_formatting'] = 1
            result['good_formatting'] = 1
        elif 'answer' in last_line.lower():
            result['good_formatting'] = 1

    return result
