import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import *

rasff_dir = '../rasff_data'

def plot_matrix(Y):
    plt.imshow(Y, interpolation='none')
    plt.colorbar()
    plt.show(block=False)

def f_score(precision, recall, beta=1):
    return (1+beta**2)*(precision*recall)/(beta**2*precision+recall)

def calc_score(Y, Y_predict, print_stats=True):
    products, chemicals = get_prod_chem()
    count = true_no_guess = false_no_guess = 0
    true_true = true_false = false_true = false_false = 0
    for (x,y), value in np.ndenumerate(Y):
        if value != 0:
            count += 1
            if print_stats: print "\n{}, {}".format(products[x], chemicals[y])
            if print_stats: print "Actual  : {}".format(value>0)
            predict_value = Y_predict[x,y]
            if predict_value == 0:
                if print_stats: print "Predict: No guess"
                if value > 0:
                    true_no_guess += 1
                else:
                    false_no_guess += 1
            else:
                if print_stats: print "Predict: {}".format(predict_value>0)
                if predict_value == value:
                    if value > 0:
                        true_true += 1
                    else:
                        false_false += 1
                else:
                    if value > 0:
                        true_false += 1
                    else:
                        false_true += 1
            if print_stats: 
                print "When True:  Correct: {}, Incorrect: {}, No guess: {}".format(
                    true_true, true_false, true_no_guess)
                print "When False: Correct: {}, Incorrect: {}, No guess: {}".format(
                    false_false, false_true, false_no_guess)
                print "Total:      Correct: {}, Incorrect: {}, No guess: {}".format(
                    true_true+false_false, true_false+false_true, true_no_guess+false_no_guess)
    correct = true_true + false_false
    incorrect = true_false + false_true
    no_guess = true_no_guess + false_no_guess
    trues = true_true + true_false + true_no_guess
    falses = false_false + false_true + false_no_guess
    #print "\nCount:", count
    recall = true_true*1.0/(trues)
    precision = true_true*1.0/(true_true+false_true)
    accuracy = true_true*1.0/(true_true+true_false)
    #accuracy = (true_true+false_false)*1.0/count
    #print "F-score:", f_score(precision, recall, 0.25)
    return recall, precision, accuracy


def classify(Y, threshold=0):
    pos = (Y>threshold).astype('int')
    neg = -(Y<-threshold).astype('int')
    return pos+neg

def get_naive_error(Y, valid):
    n = valid.sum()
    mean = Y[Y>0].mean()
    return np.sqrt(np.sum((valid * (Y - mean))**2) / n)

def get_error_rmse(Y, U, V, valid):
    n = valid.sum()
    return np.sqrt(np.sum((valid * (Y - np.dot(U, V.T)))**2) / n)

def split_data(ratings):
    """60/20/20 split."""
    size = len(ratings)
    training_size = int(0.6 * size)
    validation_size = int(0.2 * size)
    test_size = size - training_size - validation_size

    training_indices = np.random.choice(range(size), training_size, replace=False)
    remaining = np.setdiff1d(range(size), training_indices)
    validation_indices = np.random.choice(remaining, validation_size, replace=False)
    test_indices = np.setdiff1d(remaining, validation_indices)

    return ratings[training_indices], ratings[validation_indices], ratings[test_indices]

def solve_iter(U, V, Y, valid, lambda_, rank_):
    for row_idx, row in enumerate(valid):
        U[row_idx] = np.linalg.solve(np.dot(V.T, np.dot(np.diag(row), V)) + lambda_ * np.eye(rank_),
                               np.dot(V.T, np.dot(np.diag(row), Y[row_idx].T))).T
    for col_idx, col in enumerate(valid.T):
        V[col_idx] = np.linalg.solve(np.dot(U.T, np.dot(np.diag(col), U)) + lambda_ * np.eye(rank_),
                                 np.dot(U.T, np.dot(np.diag(col), Y[:, col_idx])))
    return U, V

def validate(matrix):
    # 0 is nan, 1 otherwise.
    return np.isfinite(matrix).astype(np.float64)

def collab_filter(Y, valid, Y_validate, valid_validate, lambda_, rank_, iterations):
    m, n = Y.shape
    U = 5 * np.random.rand(m, rank_)
    V = 5 * np.random.rand(n, rank_)

    rmse = [get_naive_error(Y, valid)]
    rmse_validate = [get_naive_error(Y_validate, valid_validate)]
    for i in range(iterations):
        U, V = solve_iter(U, V, Y, valid, lambda_, rank_)
        rmse_error = get_error_rmse(Y, U, V, valid)
        rmse_validate_error = get_error_rmse(Y_validate, U, V, valid_validate)
        #print 'Iteration #{}'.format(i+1), rmse_error, rmse_validate_error
        #calc_score(Y_validate, classify(np.dot(U, V.T), 0.02), False)
        rmse.append(rmse_error)
        rmse_validate.append(rmse_validate_error)
    Y_hat = np.dot(U, V.T)
    #print rmse_error, rmse_validate_error
    #calc_score(Y_validate, classify(np.dot(U, V.T), 0.02), False)
    return U, V

def cf_test(U, V, Y_test, valid_test):
    return get_error_rmse(Y_test, U, V, valid_test)


LAMBDAS = np.arange(0,0.07,0.01)
RANKS = [1,2,3,4,5]
ITERATIONS = 10
THRESHOLD = 0
NEG_FILE = '{}/negative_pairs.csv'.format(rasff_dir)

def filter_by_indices(Y, indices):
    matrix = np.zeros(Y.shape)
    for x,y in indices:
        matrix[x,y] = Y[x,y]
    return matrix

def split_matrix(Y):
    x_, y_ = np.where(Y != 0)
    pairs = np.array([(i,j) for i,j in zip(x_, y_)])
    training, validation, testing = split_data(pairs)
    Y_train = filter_by_indices(Y, training)
    Y_validate = filter_by_indices(Y, validation)
    Y_test = filter_by_indices(Y, testing)
    return Y_train, Y_validate, Y_test

def run_cf_rasff(lambda_, rank_, neg_file=None):
    Y, rows, columns = np.load('{}/matrix.npy'.format(rasff_dir))
    Y = gen_neg(Y, csv_file=neg_file, num=int(Y.sum()))
    Y_train, Y_validate, Y_test = split_matrix(Y)
    valid_train, valid_validate, valid_test = [validate(Y_train), validate(Y_validate), validate(Y_test)]
    U, V = collab_filter(Y_train, valid_train, Y_validate, valid_validate, lambda_, rank_, ITERATIONS)
    return U, V, Y_train, Y_validate, Y_test

def cross_validate(n=1, lambda_=0.06, rank_=3, threshold=THRESHOLD, test=True, neg_file=None):
    # Do repeated random sub-sampling validation
    print "\nLambda: {}, rank: {}".format(lambda_, rank_)
    recalls, precisions, accuracies = [], [], []
    for i in range(n):
        print "Iteration:", i
        U, V, Y_train, Y_validate, Y_test = run_cf_rasff(lambda_, rank_, neg_file)
        Y_ = np.dot(U, V.T)
        Y_predict = classify(Y_, threshold)
        Y = Y_test if test else Y_validate
        if n==1:
            r, p, a = calc_score(Y, Y_predict)
        else:
            r, p, a = calc_score(Y, Y_predict, False)
        recalls.append(r)
        precisions.append(p)
        accuracies.append(a)
    print (Y_predict>0).sum(), (Y_predict==0).sum(), (Y_predict<0).sum()
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    accuracies = np.array(accuracies)
    print "\n    Recall\t  Precision\tAccuracy"
    print recalls.mean(), precisions.mean(), accuracies.mean()

def optimize(lambdas, ranks):
    for lambda_, rank_ in itertools.product(lambdas, ranks):
        cross_validate(5, lambda_, rank_, THRESHOLD, False, None)

#cross_validate(5)


