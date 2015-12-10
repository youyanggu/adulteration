import time

import numpy as np
import sklearn.neighbors
from sklearn.cross_validation import train_test_split
import theano
import theano.tensor as T

from gen_embeddings import get_nearest_neighbors, print_nearest_neighbors
from nn_category import *
from scoring import calc_score
from utils import *

theano.config.floatX = 'float32'

def load_data(x, y=None):
    def shared_dataset(x, dtype, borrow=True):
        shared_x = theano.shared(np.asarray(x,
                                 dtype=dtype),
                                 borrow=borrow)
        return shared_x
    
    x_train = shared_dataset(x, str(x.dtype))
    if y is None:
        return x_train
    else:
        y_train = shared_dataset(y, str(x.dtype))
        return x_train, y_train


def calc_accuracy(pred, y_test):
    pred_cats = np.argmax(pred, axis=1)
    acc = (pred_cats == y_test).sum() * 1.0 / len(pred)
    return acc


def print_predictions(inputs, outputs, pred, counts, limit=None):
    for idx, inp in enumerate(inputs):
        print '\n============================================' 
        print 'Ingredients:', get_ingredients_from_vector(counts, inp)
        #print 'Predicted  :', 'Valid' if regr.predict(inp)[0]==1 else 'Invalid'
        print 'Predicted  :', 'Valid' if pred[idx]==1 else 'Invalid'
        print 'Actual     :', 'Valid' if outputs[idx]==1 else 'Invalid'
        if idx > limit:
            break

def track_ing_changes(predict_model, inputs, num_changes):
    probs_valid = []
    if type(num_changes) == int:
        num_changes = [num_changes]
    for n in num_changes:
        prob = np.argmax(
            predict_model(
                change_inputs(
                    inputs, n)), axis=1).sum() * 1. / len(inputs)
        print n, prob
        probs_valid.append(prob)
    return np.array(probs_valid)

def main():
    num_ingredients = 1000
    use_embeddings = False
    ings_per_prod = 5
    frac_weighted = 0.95
    invalid_multiplier = 1
    df, df_i = import_data()
    counts = df_i['ingredient'].value_counts()
    inputs_v_, outputs_v = gen_input_outputs_valid(
                        df, df_i, num_ingredients, ings_per_prod)
    inputs_i_w_, outputs_i_w = gen_input_outputs_invalid(inputs_v_,
                        invalid_multiplier*frac_weighted, 
                        num_ingredients, ings_per_prod, weighted=True)
    inputs_i_, outputs_i = gen_input_outputs_invalid(inputs_v_,
                        invalid_multiplier*(1-frac_weighted), 
                        num_ingredients, ings_per_prod, weighted=False)

    if use_embeddings:
        embeddings = np.load('embeddings/embeddings_{}.npy'.format(num_ingredients))
        #embeddings = np.load('../word2vec/word2vec_embeddings.npy')[1][:num_ingredients]
        #embeddings = 2*np.random.random((num_ingredients, 20))-1 # Try random embeddings
        embeddings = embeddings.astype('float32')
        normalize = False
        inputs_v = input_from_embeddings(inputs_v_, embeddings, 
            normalize=normalize)
        inputs_i_w = input_from_embeddings(inputs_i_w_, embeddings, 
            normalize=normalize)
        inputs_i = input_from_embeddings(inputs_i_, embeddings, 
            normalize=normalize)
    else:
        inputs_v = inputs_v_
        inputs_i_w = inputs_i_w_
        inputs_i = inputs_i_

    X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(
        inputs_v, outputs_v, test_size=1/3., random_state=42)
    X_train_i_w, X_test_i_w, y_train_i_w, y_test_i_w = train_test_split(
        inputs_i_w, outputs_i_w, test_size=1/3., random_state=42)
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
        inputs_i, outputs_i, test_size=1/3., random_state=42)

    X_train = np.vstack((X_train_v, X_train_i_w, X_train_i))
    y_train = np.hstack((y_train_v, y_train_i_w, y_train_i))
    X_test = np.vstack((X_test_v, X_test_i_w, X_test_i))
    y_test = np.hstack((y_test_v, y_test_i_w, y_test_i))

    # Scramble inputs/outputs
    np.random.seed(3)
    random_idx_tr = np.random.permutation(len(X_train))
    random_idx_te = np.random.permutation(len(X_test))
    X_train = X_train[random_idx_tr]
    y_train = y_train[random_idx_tr]
    X_test = X_test[random_idx_te]
    y_test = y_test[random_idx_te]

    print "Running models..."
    # Max entropy model
    #regr = max_entropy(X_train, y_train, X_test, y_test)
    #predict_cat(counts, regr, idx_to_cat, num_ingredients, ings)

    # Neural network model
    classifier, predict_model = run_nn(X_train, y_train, X_test, y_test, 
                                X_train.shape[1], num_outputs=2,
                                m=20, n_epochs=10, batch_size=10,
                                learning_rate=0.05, L2_reg=0.0001)

    #pred = predict_model(X_test)
    #pred_cats = np.argmax(pred, axis=1)
    #print calc_accuracy(pred, y_test)
    #print_predictions(X_test, y_test, pred_cats, counts, limit=100)

    print "Max Entropy (Valid, Invalid, Invalid weighted):"
    print calc_accuracy(regr.predict_proba(X_test_v), y_test_v)
    print calc_accuracy(regr.predict_proba(X_test_i), y_test_i)
    print calc_accuracy(regr.predict_proba(X_test_i_w), y_test_i_w)
    print "Neural Network (Valid, Invalid, Invalid weighted):"
    print calc_accuracy(predict_model(X_test_v), y_test_v)
    print calc_accuracy(predict_model(X_test_i), y_test_i)
    print calc_accuracy(predict_model(X_test_i_w), y_test_i_w)

    if not use_embeddings:
        embeddings_out = classifier.hiddenLayer.W.get_value()
        ranks, neigh = get_nearest_neighbors(embeddings_out)
        #print_nearest_neighbors(counts.index.values[:num_ingredients], ranks)
        highest_rank, score, avg_rank_of_ing_cat, random_score = calc_score(
                ranks, num_ingredients)

if __name__ == '__main__':
    main()
