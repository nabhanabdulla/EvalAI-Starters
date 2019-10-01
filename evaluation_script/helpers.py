import pandas as pd
import numpy as np

def get_test_data(path):
    '''
        Input: path to test file with features and labels
        Output: features, labels
    '''
    test = pd.read_csv(path)
    
    test_labels = np.array(test['label'])
    test_features = test.drop('label', axis='columns')

    return test_features, test_labels

def get_accuracy(pred, y_pred):
    # print('Pred: ', pred, type(pred))
    # print('y_pred:', y_pred, type(y_pred))

    acc = np.mean(np.array(pred) == y_pred)

    return acc 