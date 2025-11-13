# Functions for answer generation and checking
import re
import pandas as pd


def generate_QA_pairs(dataset, pipeline, generation_mode):
    """ Primary interface for answer generation"""
    metrics = []

    def apply_pipe(batch):
        answers = pipeline(batch['question'])
        rows, batch_metrics = make_new_rows(batch, answers)
        metrics.extend(batch_metrics)

        print('answer generation, metrics: ', len(metrics))

        # filter the answers according to the generation mode.
        match generation_mode:
            case 'gt_answers':
                return batch
            case 'gt_targets':
                rows = rows[rows['result'] == rows['gt']]
            case 'majority_vote':
                rows = rows[rows['result'] == rows['majority_vote']]
            case 'perfect formatting':
                rows = rows[rows['perfect_formatting'] == 1]
            case 'perfect formatting':
                rows = rows[rows['good_formatting'] == 1]
            case _:
                raise Exception('Unkown generation mode')

        # Drop extraneous columns
        return rows.drop(columns=['good_formatting', 'perfect_formatting', 'result',
                                  'majority_vote'])

    data_with_answers = dataset.map(apply_pipe, batched=True, batch_size=8)

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
            results = results.assign(majority_vote=majority_value[0])

        results = results.assign(gt=gt, question=question)
        full_outputs.append(results)

        # update the metric
        metrics.append(score_results(results))

    return pd.concat(full_outputs, ignore_index=True), metrics


def check_format_and_get_answer(output):
    """ check whether the answer ends with
        'The final answer is: __'
        and returns the imputed answer
    """
    generated_text = output['generated_text'] 
    result = {'answer': generated_text, 'good_formatting': 0, 'perfect_formatting': 0, 'result': None}

    last_line = generated_text.split('\n')[-1]

    last_line_numbers = re.findall(r'[0-9,]*\.?[0-9]+', last_line)

    if len(last_line_numbers) > 0:
        result['result'] = int(float(last_line_numbers[-1].replace(',', '')))

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
