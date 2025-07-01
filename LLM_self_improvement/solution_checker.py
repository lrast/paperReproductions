# basic solution checkers

import re

import multiprocessing as mp


def final_answer_formatting(generated_text):
    """ check whether the answer ends with
        'The final answer is: __'
        and returns the imputed answer

        formatting codes: 0 - no good,
                          1 - contains answer,
                          2 - ends with 'The final answer is: __'
    """
    output = {'good_formatting': 0, 'perfect_formatting': 0, 'answer': ''}

    last_line = generated_text.split('\n')[-1]

    last_line_numbers = re.findall('[0-9]+', last_line)

    if len(last_line_numbers) > 0:
        output['answer'] = last_line_numbers[-1]

        if last_line.startswith('The final answer is:'):
            output['perfect_formatting'] = 1
            output['good_formatting'] = 1
        elif 'answer' in last_line.lower():
            output['good_formatting'] = 1

    return output


def COT_formatting(generated_text):
    pass


def results_worker(queue, output):
    """ worker for answer checker """
    while True:
        results = queue.get()
        if results is None:  # signal that the testing is done
            break

        samples, ground_truth = results
        output['num_inputs'] += 1

        correct_count = 0
        for sample in samples:
            answer_check = final_answer_formatting(sample['generated_text'])
            for key in ['good_formatting', 'perfect_formatting']:
                output[key] += answer_check[key]

            if answer_check['answer'] == ground_truth:
                correct_count += 1

        if correct_count > 0:
            output['passOf8'] += 1
        if correct_count > 4:
            output['majorityOf8'] += 1


class AnswerChecker(object):
    """ Multiprocessing answer checker
    """
    def __init__(self):
        self.queue = mp.Queue()
        self.results = mp.Manager().dict()
        self.results['good_formatting'] = 0
        self.results['perfect_formatting'] = 0
        self.results['majorityOf8'] = 0
        self.results['passOf8'] = 0
        self.results['num_inputs'] = 0

        self.worker = mp.Process(target=results_worker, args=(self.queue, self.results))
        self.worker.start()

    def add_result(self, results):
        self.queue.put(results)

    def shutdown(self):
        self.queue.put(None)

    def summarize(self):
        results = {key: self.results[key] / self.results['num_inputs']
                   for key in self.results.keys()}
        results.pop('num_inputs')
        results['num_samples'] = self.results['num_inputs']

        results['good_formatting'] = results['good_formatting'] / 8
        results['perfect_formatting'] = results['perfect_formatting'] / 8

        return results
