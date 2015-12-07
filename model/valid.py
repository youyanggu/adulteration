import time

import numpy as np
import sklearn.neighbors
from sklearn.cross_validation import train_test_split
import theano
import theano.tensor as T

from gather_data import *
from nn_category import *
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


def calc_accuracy(pred, y_test, tensor=True):
    if tensor:
        y_test = y_test.get_value()
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
 
def main():
    num_ingredients = 1000
    use_embeddings = False
    ings_per_prod = 5
    weighted = True
    df, df_i = import_data()
    counts = df_i['ingredient'].value_counts()
    inputs_v, outputs_v = gen_input_outputs_valid(
                        df, df_i, num_ingredients, ings_per_prod)
    inputs_i, outputs_i = gen_input_outputs_invalid(
                        df_i, len(inputs_v), num_ingredients, ings_per_prod, weighted)
    inputs_ = np.vstack((inputs_v, inputs_i))
    outputs = np.hstack((outputs_v, outputs_i))

    inputs_, outputs = inputs_.astype('int32'), outputs.astype('int32')
    if use_embeddings:
        #embeddings = np.load('embeddings/embeddings_{}.npy'.format(num_ingredients))
        #embeddings = np.load('../word2vec/word2vec_embeddings.npy')[1][:num_ingredients]
        embeddings = 2*np.random.random((embeddings.shape))-1 # Try random embeddings
        inputs = input_from_embeddings(inputs_, embeddings.astype('float32'), 
            normalize=False)
    else:
        inputs = inputs_

    # Scramble inputs/outputs
    np.random.seed(3)
    random_idx = np.random.permutation(len(inputs))
    inputs = inputs[random_idx]
    outputs = outputs[random_idx]
    
    X_train, X_test, y_train, y_test = train_test_split(
        inputs, outputs, test_size=1/3., random_state=42)

    # Max entropy model
    #regr = max_entropy(X_train, y_train, X_test, y_test)
    #predict_cat(counts, regr, idx_to_cat, num_ingredients, ings)

    # Neural network model
    classifier, pred = run_nn(X_train, y_train, X_test, y_test, num_ingredients, 2,
                                m=10, n_epochs=20, batch_size=10,
                                learning_rate=0.01, L2_reg=0.0003)

    pred_cats = np.argmax(pred, axis=1)
    #print calc_accuracy(pred, y_test)
    #print_predictions(X_test, y_test, pred_cats, counts, limit=100)

if __name__ == '__main__':
    main()
