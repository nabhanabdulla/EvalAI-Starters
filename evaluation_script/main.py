'''
evaluate() code is modified version from https://github.com/live-wire/EvalAI-Examples.git mnist-challenge branch
'''


import random
import sys
import os
import configparser
import importlib

import numpy as np
import time

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
    # print(kwargs["submission_metadata"])
    
    result = {}
    result['result'] = []
    
    ## 1. Add submission file path to system path and import submission file as module
    ## 'input_file': 'https://abc.xyz/path/to/submission/main.py'

    # input file name - 'main.py'
    if '/' in user_submission_file:
        input_script = user_submission_file.split('/')[-1]
    else:
        input_script = user_submission_file.split('\\')[-1]

    # print(input_script)

    # input file path - 'https://abc.xyz/path/to/submission'
    input_script_path = user_submission_file[:-len(input_script)] # or '/'.join(user_submission_file.split('/')[:-1])
    # print(input_script_path)

    # input file name - 'main'
    input_script_name = input_script.split('.')[0]
    # print(input_script_name)

    # add file path to system path
    sys.path.insert(0, input_script_path)

    # import python script submitted as module
    submission_script = importlib.import_module(input_script_name)


    ## 2. Read question metadata from config and save

    # read the config file
    config = configparser.ConfigParser()
    config.read('config.ini')

    print(config.sections())
    # variable to store question data
    # qn_map = {}

    # add question test file name
    data_dir = config[phase_codename]['TestFile']


    # ## 3. Get features and labels
    # qn_map['features'], qn_map['label'] = get_test_data(qn_map['test_file'])


    if input_script_name == "gradient_descent":
        optimizer = submission_script.optimizer

        # load data
        x_, y_ = submission_script.load_data(data_dir)

        # time stamp
        start = time.time()

        try:
            # gradient descent
            params = optimizer['init_params']
            old_cost = 1e10
            for iter_ in range(optimizer['max_iterations']):
                # evaluate cost and gradient
                cost = submission_script.evaluate_cost(x_,y_,params)
                grad = submission_script.evaluate_gradient(x_,y_,params)
                # display
                if(iter_ % 10 == 0):
                    print('iter: {} cost: {} params: {}'.format(iter_, cost, params))
                # check convergence
                if(abs(old_cost - cost) < optimizer['eps']):
                    break
                # udpate parameters
                params = submission_script.update_params(params,grad,optimizer['alpha'])
                old_cost = cost
        except:
            cost = optimizer['inf']
        
        res = {}
        split_name = 'q1'
        res[split_name] = {}

        res[split_name]['rmse'] = cost 
        result['result'].append(res)

        submission_result = "Evaluated scores for the phase '" + str(phase_codename) + "' - split '" + str(split_name) + "': " + str(cost) +'.'

    elif input_script_name == "eigenfaces":
        opts = submission_script.opts 

        # time stamp
        start = time.time()

        try:
            # extract features of all faces
            featFaces, featTest = submission_script.readImages(opts['dirName'],opts['refSize'],opts['fExt'])
            print("featFaces: {}, featTest {}".format(featFaces.shape, featTest.shape))
            
            
            # extract mean face
            meanFaces, stddFaces = submission_script.extract_mean_stdd_faces(featFaces)
            print("meanFaces: {}, stddFaces: {}".format(meanFaces.shape, stddFaces.shape))
            
            # normalize faces
            # ref: https://stats.stackexchange.com/questions/69157/why-do-we-need-to-normalize-data-before-principal-component-analysis-pca
            # ref: https://stackoverflow.com/questions/23047235/matlab-how-to-normalize-image-to-zero-and-unit-variance
            normFaces = submission_script.normalize_faces(featFaces, meanFaces, stddFaces)
            print("normFaces: {}".format(normFaces.shape))
                
            # covariance matrix
            covrFaces = submission_script.compute_covariance_matrix(normFaces) + opts['eps']
            print("covrFaces: {}".format(covrFaces.shape))
            
            # eigenvalues and eigenvectors
            eigval, eigvec = submission_script.compute_eigval_eigvec(covrFaces)
            print("eigval: {} eigvec: {}".format(eigval.shape, eigvec.shape))
            
            # find number of eigvenvalues cumulatively smaller than energhTh
            cumEigval = np.cumsum(eigval / sum(eigval))
            numSignificantEigval = next(i for i,v in enumerate(cumEigval) if v > opts['energyTh'])
            
            # show top 90% eigenvectors
            # call this function to visualize eigenvectors
            submission_script.show_eigvec(eigvec, cumEigval, opts['refSize'],opts['energyTh'])
            
            # reconstruct test image
            rmse = submission_script.reconstruct_test(featTest, meanFaces, stddFaces, eigvec, numSignificantEigval)
            print('#eigval preserving {}% of energy: {}'.format(100*opts['energyTh'],numSignificantEigval))
        except:
            rmse = opts['inf']

        # final output
        print('time elapsed: {}'.format(time.time() - start))
        print('rmse on compressed test image: {} (lower the better)'.format(rmse))


        res = {}
        split_name = 'q2'
        res[split_name] = {}

        res[split_name]['rmse'] = rmse 
        result['result'].append(res)

        submission_result = "Evaluated scores for the phase '" + str(phase_codename) + "' - split '" + str(split_name) + "': " + str(rmse) +'.'


    elif input_script_name == "classification":
        opts = submission_script.opts 

        np.random.seed(opts['seed'])

        # time stamp
        start = time.time()

        # data_dir = r"D:\\Projects\\MLabs\ML18\\ML19 Recruitment\\mlabs-2018-problem-set\\data\\ETHZShapeClasses-V1.2\\"
        # print(data_dir)

        # read the data
        feat,label = submission_script.read_data(data_dir,
                            opts['classNames'],
                            opts['fExt'],
                            opts['refSize'])

        # train test split
        ftrain,ftest,ltrain,ltest = submission_script.train_test_split(feat,label,opts['trainSplit'])

        try:
            classifier_svm = submission_script.train_svm(ftrain, ltrain)
            predicted = submission_script.test_classifier(ftest, classifier_svm)
            f1ScoreSVC = submission_script.eval_performance(predicted,ltest,classifier_svm)

            classifier_rf = submission_script.train_random_forest(ftrain, ltrain)
            predicted = submission_script.test_classifier(ftest, classifier_rf)
            f1ScoreRF = submission_script.eval_performance(predicted,ltest,classifier_rf)

            f1ScoreBest = f1ScoreSVC if(f1ScoreSVC>f1ScoreRF) else f1ScoreRF
            f1ScoreReport = 1-f1ScoreBest
        except:
            f1ScoreReport = opts['inf']
    
        res = {}
        split_name = 'q3'
        res[split_name] = {}

        res[split_name]['f1-score'] = f1ScoreReport 
        result['result'].append(res)

        submission_result = "Evaluated scores for the phase '" + str(phase_codename) + "' - split '" + str(split_name) + "': " + str(f1ScoreReport) +'.'

    elif input_script_name == "disparity":
        opts = submission_script.opts 

        # time stamp
        start = time.time()

        # read the data
        view1,view2,gth12,gth21 = submission_script.read_data(data_dir,opts['downsample'])

        try:
            disp12 = submission_script.compute_disparity(view1, view2, opts['halfPatchSize'])
            disp21 = submission_script.compute_disparity(view2, view1, opts['halfPatchSize'])
            sse = np.sum(np.power((disp12-gth12),2)) + np.sum(np.power((disp21-gth21),2))
            sse /= (disp12.shape[0]*disp12.shape[1])
            
            # display results
            plt.subplot(321)
            plt.imshow(view1)
            plt.title('view1')
            plt.subplot(322)
            plt.imshow(view2)
            plt.title('view2')
            plt.subplot(323)
            plt.imshow(gth12)
            plt.title('gth12')
            plt.subplot(324)
            plt.imshow(gth21)
            plt.title('gth21')
            plt.subplot(325)
            plt.imshow(disp12)
            plt.title('computed12')
            plt.subplot(326)
            plt.imshow(disp21)
            plt.title('computed21')
        except:
            sse = opts['inf']
            
        # final output
        print('time elapsed: {}'.format(time.time() - start))
        print("total sum of squared error: {} (lower the better)".format(sse))

        res = {}
        split_name = 'q4'
        res[split_name] = {}

        res[split_name]['sse'] = sse 
        result['result'].append(res)

        submission_result = "Evaluated scores for the phase '" + str(phase_codename) + "' - split '" + str(split_name) + "': " + str(sse) +'.'


    # ## 4. Get prediction
    # y_pred = submission_script.main(qn_map['features'])


    # ## 5. Find accuracy
    # y = qn_map['label']

    # acc = get_accuracy(y_pred, y)

    ## 6. Store data for leaderboard
    # res = {}

    # if phase_codename == 'phase-q1':
    #     split_name = 'data-q1'
    # elif phase_codename == 'phase-q2':
    #     split_name = 'data-q2'
    # elif phase_codename == 'phase-q3':
    #     split_name = 'data-q3'

    # res[split_name] = {}

    # res[split_name]['accuracy'] = acc 
    # result['result'].append(res)    

    result['submission_result'] = submission_result

    print(result)

    return result 


if __name__ == "__main__":
    evaluate()