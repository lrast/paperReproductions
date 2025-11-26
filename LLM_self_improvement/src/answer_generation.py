# Functions for answer extraction and format checking
import re
import numpy as np


def check_format_and_get_answer(generated_text):
    """ check whether the answer ends with
        'The final answer is: __'
        and returns the imputed answer
    """
    result = {'answer': generated_text, 'good_formatting': 0, 'perfect_formatting': 0, 'result': np.nan}

    last_line = generated_text.split('\n')[-1]

    last_line_numbers = re.findall(r'[+-]?[0-9,]*\.?[0-9]+', last_line)

    if len(last_line_numbers) > 0:
        # Handle run-on string outputs 
        str_answer = last_line_numbers[-1].replace(',', '')[0:300]
        result['result'] = int(float(str_answer))

        if last_line.startswith('The final answer is:'):
            result['perfect_formatting'] = 1
            result['good_formatting'] = 1
        elif 'answer' in last_line.lower():
            result['good_formatting'] = 1

    return result


def score_results(batch_results):
    """Score the batch results"""
    is_gt = (batch_results['result'] == batch_results['gt'])
    score = {
              'pass@8': is_gt.any(), 
              'maj@8': (is_gt.sum() >= 4),
              'good_formatting': batch_results['good_formatting'].sum() / 8,
              'perfect_formatting': batch_results['perfect_formatting'].sum() / 8,
            }
    return score
