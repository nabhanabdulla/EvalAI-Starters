'''
evaluate() code is modified version from https://github.com/live-wire/EvalAI-Examples.git mnist-challenge branch
'''


import random


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
    # output = {}
    # if phase_codename == "phase1":
    #     print("Evaluating for Phase 1")
    #     output["result"] = [
    #         {
    #             "train_split":{
    #                 "test_score": random.randint(0, 99),
    #             }
    #         },
    #         {
    #             "test_split": {
    #                 "test_score": random.randint(0, 99),
    #             }
    #         },
    #     ]
    #     # To display the results in the result file
    #     output["submission_result"] = output["result"][0]
    #     print("Completed evaluation for Test Phase")
    result = {}
    
    if phase_codename == "phase1":
        test_file = "data/answers.csv"
        
        answers = pd.read_csv(test_file)
        user = pd.read_csv(user_submission_file)
        
        submission_result = ""
        
        result['result'] = []
        result["submission_result"] = submission_result
        
        if len(user) != len(answers):
            submission_result = "Number of rows in the training data ("+str(len(answers))+") and the submission file ("+str(len(user))+") don't match."
            result["submission_result"] = submission_result
            return result
        
        temp = {}
        temp[phase_codename] = {}
        matches = 0
        
        for i in range(len(user)):
            if user.iloc[i]['label'] == answers.iloc[i]['label']:
                matches = matches+1
                
        accuracy = (matches/len(user))*100
        print("Accuracy:",accuracy)
        
        temp[phase_codename]['accuracy'] = accuracy
        result['result'].append(temp)
        submission_result = "Evaluated accuracy for "+str(phase_codename)+". Accuracy="+str(accuracy)
        result["submission_result"] = submission_result
        
    return result 
