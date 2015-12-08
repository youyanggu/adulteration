import pickle

import numpy as np
import pandas as pd

def save_regr(regr, fname):
    with open(fname, 'wb') as f:
        pickle.dump(regr, f)

def load_regr(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print x
    pd.reset_option('display.max_rows')

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def scramble_data(inputs, outputs):
    assert(len(inputs)==len(outputs))
    indices = np.random.permutation(len(inputs))
    return inputs[indices], outputs[indices]
