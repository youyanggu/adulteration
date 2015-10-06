import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import *

rasff_dir = '../rasff_data'

def print_predictions(Y, Y_predict, print_stats=True):
    #total = (Y*Y_predict!=0).sum()
    #correct = (Y*Y_predict>0).sum()
    #print "Total values to predict:      ", int((Y!=0).sum())
    #print "Total values predicted:       ", abs(Y_predict).sum()
    #print "Total wanted values predicted:", int(total)
    #print "% predicted:", total*1.0/(Y!=0).sum()
    #print "Accuracy:", correct*1.0/total
    #print "Recall:", correct*1.0/(Y>0).sum()

    products, chemicals = get_prod_chem()
    count = correct = incorrect = no_guess = 0
    for (x,y), value in np.ndenumerate(Y):
        if value != 0:
            count += 1
            if print_stats: print "\n{}, {}".format(products[x], chemicals[y])
            if print_stats: print "Guess:  {}".format(value>0)
            predict_value = Y_predict[x,y]
            if predict_value == 0:
                if print_stats: print "Actual: No guess"
                no_guess += 1
            else:
                if print_stats: print "Actual: {}".format(predict_value>0)
                if predict_value == value:
                    correct += 1
                else:
                    incorrect += 1
            if print_stats: print "Correct: {}, Incorrect: {}, No guess: {}".format(correct, incorrect, no_guess)
    print "\nCount:", count
    print "% predicted:", 1-no_guess*1.0/count
    print "Accuracy:", correct*1.0/(correct+incorrect)
    print "Recall:", correct*1.0/count


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
    size = len(ratings)
    training_size = int(0.5 * size)
    validation_size = int(0.3 * size)
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
    return np.isfinite(matrix).astype(np.float64)

def collab_filter(Y, valid, Y_validate, valid_validate, lambda_, rank_, iterations):
    d = {}
    m, n = Y.shape
    print "Lambda: {}, rank: {}".format(lambda_, rank_)
    U = 5 * np.random.rand(m, rank_)
    V = 5 * np.random.rand(n, rank_)

    rmse = [get_naive_error(Y, valid)]
    rmse_validate = [get_naive_error(Y_validate, valid_validate)]
    for i in range(iterations):
        U, V = solve_iter(U, V, Y, valid, lambda_, rank_)
        rmse_error = get_error_rmse(Y, U, V, valid)
        rmse_validate_error = get_error_rmse(Y_validate, U, V, valid_validate)
        #print 'Iteration #{}'.format(i+1), mse_error, mse_validate_error
        rmse.append(rmse_error)
        rmse_validate.append(rmse_validate_error)
    Y_hat = np.dot(U, V.T)
    d[(lambda_, rank_)] = min(rmse_validate)
    print rmse_error, rmse_validate_error
    return U, V

def cf_test(U, V, Y_test, valid_test):
    return get_error_rmse(Y_test, U, V, valid_test)


lambdas = [0.1]
ranks = [1]
#lambdas = [0.05]
#ranks = [3]
iterations = 15

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

def run_cf_rasff():
    Y, rows, columns = np.load('{}/matrix.npy'.format(rasff_dir))
    Y = gen_neg(Y, csv_file=None, num=int(Y.sum()))
    Y_train, Y_validate, Y_test = split_matrix(Y)
    valid_train, valid_validate, valid_test = [validate(Y_train), validate(Y_validate), validate(Y_test)]
    for lambda_, rank_ in itertools.product(lambdas, ranks):
        U, V = collab_filter(Y_train, valid_train, Y_validate, valid_validate, lambda_, rank_, iterations)
    #print "Test RMSE:", cf_test(U, V, Y_test, valid_test)
    return U, V

U, V = run_cf_rasff()
Y_ = np.dot(U, V.T)
Y_predict = classify(Y_)

