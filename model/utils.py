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

def flip_arr(arr, num_flips=1):
    assert(arr.sum() >= num_flips)
    if num_flips == 0:
        return arr
    one_indices = np.where(arr==1)[0]
    new_one_indices = []
    while len(new_one_indices)<num_flips:
        idx = np.random.choice(len(arr))
        if idx not in one_indices and idx not in new_one_indices:
            new_one_indices.append(idx)

    z = np.zeros(len(arr))
    all_one_indices = np.hstack((
            np.random.choice(one_indices, len(one_indices)-num_flips, replace=False),
            np.array(new_one_indices)))
    z[all_one_indices] = 1
    assert(arr.sum()==z.sum())
    return z

def flip_inputs(inputs, num_flips=1):
    return np.array([flip_arr(i, num_flips) for i in inputs])
