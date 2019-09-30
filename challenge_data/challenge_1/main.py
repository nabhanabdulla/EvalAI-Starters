'''
evaluate() code is modified version from https://github.com/live-wire/EvalAI-Examples.git mnist-challenge branch
'''


import random


def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    print("Submission related metadata:")
    
    print(kwargs["submission_metadata"])

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
