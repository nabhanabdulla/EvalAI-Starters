import sys
import pandas as pd
import numpy as np

def get_accuracy(n):
    return [n/100]

def get_prediction(features):
    #print('From python_from_python: ', features, type(features))
    pred = [np.sum(feature) for feature in features.values]
    return pred

def main(features):
    return get_prediction(features)

        
