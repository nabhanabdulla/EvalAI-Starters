'''
evaluate() code is modified version from https://github.com/live-wire/EvalAI-Examples.git mnist-challenge branch
'''


import random
import sys
import os
import configparser
import importlib

from helpers import *

def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    print("Submission related metadata:")
    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            'execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """
    print(kwargs["submission_metadata"])
    
    result = {}
    result['result'] = []
    
    ## 1. Add submission file path to system path and import submission file as module
    ## 'input_file': 'https://abc.xyz/path/to/submission/main.py'

    # input file name - 'main.py'
    input_script = user_submission_file.split('/')[-1]
    print(input_script)

    # input file path - 'https://abc.xyz/path/to/submission'
    input_script_path = user_submission_file[:-len(input_script)] # or '/'.join(user_submission_file.split('/')[:-1])
    print(input_script_path)

    # input file name - 'main'
    input_script_name = input_script.split('.')[0]
    print(input_script_name)

    # add file path to system path
    sys.path.insert(0, input_script_path)

    # import python script submitted as module
    submission_script = importlib.import_module(input_script_name)


    ## 2. Read question metadata from config and save

    # read the config file
    config = configparser.ConfigParser()
    config.read('config.ini')

    # variable to store question data
    qn_map = {}

    # add question test file name
    qn_map['test_file'] = config[phase_codename]['TestFile']


    ## 3. Get features and labels
    qn_map['features'], qn_map['label'] = get_test_data(qn_map['test_file'])

    
    ## 4. Get prediction
    y_pred = submission_script.main(qn_map['features'])


    ## 5. Find accuracy
    y = qn_map['label']

    acc = get_accuracy(y_pred, y)

    ## 6. Store data for leaderboard
    res = {}

    if phase_codename == 'phase-q1':
        split_name = 'data-q1'
    elif phase_codename == 'phase-q2':
        split_name = 'data-q2'
    elif phase_codename == 'phase-q3':
        split_name = 'data-q3'

    res[split_name] = {}

    res[phase_codename]['Accuracy'] = acc 
    result['result'].append(res)    

    submission_result = "Evaluated scores for the phase '" + str(phase_codename) + "' - Accuracy=" + str(acc) +'.'
    result['submission_result'] = submission_result

    print(result)

    return result 

def main():
    evaluate()

if __name__ == "__main__":
    evaluate()