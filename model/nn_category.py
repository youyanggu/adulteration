import time

import numpy as np
import sklearn.neighbors
from sklearn.cross_validation import train_test_split
import theano
import theano.tensor as T

from gather_data import *
from gen_embeddings import get_nearest_neighbors, print_nearest_neighbors
from scoring import calc_score
from utils import *

theano.config.floatX = 'float32'

def get_batch(x_train, y_train, idx, batch_size):
    x = x_train[idx*batch_size : (idx+1)*batch_size]
    y = y_train[idx*batch_size : (idx+1)*batch_size]
    if scipy.sparse.issparse(y):
        y = y.toarray()
    return x, y

def calc_accuracy(pred, y_valid, lower_to_upper_cat=None):
    pred_cats = np.argmax(pred, axis=1)
    if lower_to_upper_cat:
        pred_cats = np.array([lower_to_upper_cat[i] for i in pred_cats])
        y_valid = np.array([lower_to_upper_cat[i] for i in y_valid])
    acc = (pred_cats == y_valid).sum() * 1.0 / len(pred)
    return acc

class OutputLayer(object):
    def __init__(self, inp, n_in, n_out):
        self.inp = inp
        self.W = theano.shared(
            value=np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='W',
            borrow=True
        )
        self.b = theano.shared(
            value=np.zeros((n_out,), dtype=theano.config.floatX),
            name='b',
            borrow=True
        )
        self.p_y_given_x = T.nnet.softmax(T.dot(inp, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

    def cost(self, y):
        t1 = T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]
        t2 = T.sum(T.log(1-self.p_y_given_x), axis=1) - \
                T.log(1-self.p_y_given_x)[T.arange(y.shape[0]), y]
        #return T.mean(-t1-t2)
        return T.sum(-t1-t2) * 1.0 / y.shape[0]

    def errors(self, y):
        return T.mean(T.neq(self.y_pred, y))

class HiddenLayer(object):
    def __init__(self, rng, inp, n_in, n_out, W=None, b=None,
                 activation=None):#T.nnet.sigmoid):
        self.inp = inp
        if W is None:
            W_values = np.asarray(
                rng.uniform(low=-1, high=1, size=(n_in, n_out)),
                dtype=theano.config.floatX
            )
            #W_values = np.zeros((n_in, n_out), dtype=theano.config.floatX)
            #print W_values[0]
            W = theano.shared(value=W_values, name='W', borrow=True)
            
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(inp, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]


class NN(object):
    def __init__(self, rng, inp, n_in, m, n_out):
        self.inp = inp
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            inp=inp,
            n_in=n_in,
            n_out=m,
            activation=T.tanh
        )
        self.outputLayer = OutputLayer(
            inp=self.hiddenLayer.output,
            n_in=m,
            n_out=n_out
        )
        self.L2 = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.hiddenLayer.b ** 2).sum()
            + (self.outputLayer.W ** 2).sum()
            + (self.outputLayer.b ** 2).sum()
        )
        
        self.cost = self.outputLayer.cost
        self.errors = self.outputLayer.errors

        self.params = self.hiddenLayer.params + self.outputLayer.params
        


def run_nn(x_train, y_train, x_valid, y_valid, num_ingredients, num_outputs, m, 
           learning_rate=0.1, L2_reg=0.0005, n_epochs=10, batch_size=10):
    print 'Building model'

    index = T.iscalar()  # index to a [mini]batch
    if x_train.dtype == 'float32':
        x = T.fmatrix('x')
    else:
        x = T.imatrix('x')
    #y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
    y = T.ivector('y')

    n_train_batches = x_train.shape[0] / batch_size

    rng = np.random.RandomState(3)

    classifier = NN(
        rng=rng,
        inp=x,
        n_in=num_ingredients,
        m=m,
        n_out=num_outputs,
    )
    cost = classifier.cost(y) + L2_reg * classifier.L2
    gparams = [T.grad(cost, param).astype('float32') for param in classifier.params]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]
        
    train_model = theano.function(
        inputs=[x, y],
        outputs=cost,
        updates=updates,
    )
    train_model_all = theano.function(
        inputs=[x],
        outputs=classifier.outputLayer.p_y_given_x,
    )
    predict_model = theano.function(
        inputs=[x],
        outputs=classifier.outputLayer.p_y_given_x,
    )
    print 'Training'

    start_time = time.time()

    epoch = 0
    while epoch < n_epochs:
        epoch = epoch + 1
        print '---------------------'
        print 'Epoch #', epoch
        costs = []
        for idx in range(n_train_batches):
            x_train_, y_train_ = get_batch(x_train, y_train, idx, batch_size)
            ret = train_model(x_train_, y_train_)
            costs.append(ret)

        print "Learning rate:", learning_rate
        learning_rate = max(0.001, learning_rate/2.)
        updates = [
                (param, param - learning_rate * gparam)
                for param, gparam in zip(classifier.params, gparams)
        ]

        print np.array(costs).mean()
        print "Train acc :", calc_accuracy(train_model_all(x_train), y_train)
        print "Valid acc  :", calc_accuracy(predict_model(x_valid), y_valid)

        #print classifier.hiddenLayer.W.get_value()[0]
        #print classifier.outputLayer.W.get_value()[0]
        #print classifier.hiddenLayer.b.get_value()
        #print classifier.outputLayer.b.get_value()

    print 'The code ran for %.2fm' % ((time.time() - start_time) / 60.)

    return classifier, predict_model

def max_entropy(X_train, y_train, X_valid, y_valid, C=1e5):
    from sklearn.linear_model import LogisticRegression
    regr = LogisticRegression(C=C)
    regr.fit(X_train, y_train)
    print regr.score(X_train, y_train)
    print regr.score(X_valid, y_valid)
    return regr

def predict_cat(counts, regr, idx_to_cat, num_ingredients, ings):
    """Given a list of ingredients (ing), predict the category."""
    if type(ings) == str:
        ings = [ings]
    ing_list = counts.index.values[:num_ingredients]
    indices = []
    for ing in ings:
        if ing in ing_list:
            #print "Found", ing
            idx = np.where(ing_list==ing)[0][0]
            indices.append(idx)
    if not indices:
        return None, None
    inp = np.zeros(num_ingredients)
    inp[np.array(indices)] = 1
    pred_class = regr.predict(inp)[0]
    # Need to get accuracy for top 3 classes
    return idx_to_cat[pred_class], np.max(regr.predict_proba(inp))

def print_predictions(inputs, outputs, pred, idx_to_cat, counts, limit=None):
    for idx, inp in enumerate(inputs):
        print '\n============================================' 
        print 'Ingredients  :', get_ingredients_from_vector(counts, inp)
        #print 'Predicted cat:', idx_to_cat[regr.predict(inp)[0]]
        print 'Predicted cat :', idx_to_cat[pred[idx]]
        print 'Actual cat    :', idx_to_cat[outputs[idx]]
        if idx > limit:
            break

def zero_embeddings(embeddings, found_ings):
    l = len(embeddings)
    found_ings = found_ings[found_ings<l]
    zero_indices = np.array([i for i in range(l) if i not in found_ings])
    new_embeddings = np.copy(embeddings)
    new_embeddings[zero_indices] = np.zeros(embeddings.shape[1])
    return new_embeddings

def prob_method(df, category, alpha=0, test_size=0.2):
    """Use unigram probabilities to predict category. Alpha is used for alpha smoothing."""
    def prob_method_helper(df_):
        ings = df_['ingredients_clean'].values
        cats = df_[category].str.lower().values
        num_categories = len(cat_to_idx)
        true_cats = np.array([cat_to_idx[c] for c in cats])
        pred_cats = []
        for i in range(len(df_)):
            cur_cat = cat_to_idx[cats[i]]
            ings_arr = []
            for ing in ings[i]:
                if ing in ing_to_cat_freq:
                    ings_arr.append(ing_to_cat_freq[ing])
                else:
                    uniform_prob = np.ones(num_categories)*1./num_categories
                    ings_arr.append(uniform_prob)
            ings_arr = np.array(ings_arr)
            if len(ings_arr)==0:
                pred_cats.append(np.log(np.zeros(num_categories)))
                continue
            pred_cat = np.sum(np.log(ings_arr), axis=0) # sum of log probs
            pred_cats.append(pred_cat)
        pred_cats = np.vstack(pred_cats)
        assert(len(pred_cats)==len(true_cats)==len(df_))

        print calc_accuracy(pred_cats, true_cats)
        if category != 'aisle':
            lower_to_upper_cat = get_upper_cat(df, category, 'aisle')
            print calc_accuracy(pred_cats, true_cats, lower_to_upper_cat)
        return pred_cats, true_cats

    cat_to_idx = {c : i for i, c in enumerate(
        np.unique(df[category].str.lower().values))}
    n = len(df)

    test_indices = np.random.choice(n, n*test_size, replace=False)
    remaining_indices = np.setdiff1d(np.arange(n), test_indices)
    train_indices = np.random.choice(remaining_indices, 
        int(len(remaining_indices)*(1-test_size)), replace=False)
    valid_indices = np.setdiff1d(remaining_indices, train_indices)
    df_train = df.ix[train_indices]
    df_valid = df.ix[valid_indices]
    df_test = df.ix[test_indices]

    ing_to_cat_freq = get_ing_cat_frequencies(df_train, category, cat_to_idx, alpha)
    print "Validation Set:"
    prob_method_helper(df_valid)
    print "Test Set:"
    pred_cats, true_cats = prob_method_helper(df_test)

    return pred_cats, true_cats

def search_smoothing(df, alphas, test_size=0.2):
    for alpha in alphas:
        print "Alpha = {}".format(alpha)
        print "Aisle"
        pred_cats, true_cats = prob_method(df, 'aisle', alpha, test_size)
        print "Shelf"
        pred_cats, true_cats = prob_method(df, 'shelf', alpha, test_size)
        print "Food Category"
        pred_cats, true_cats = prob_method(df, 'food_category', alpha, test_size)
        print "----------"

def predict_cat_from_ing(
    ing_names, num_ingredients, predict_model, idx_to_cat, top_n=3, fname=None):
    ing_names = ing_names[:num_ingredients]
    pred = predict_model(np.eye(num_ingredients, dtype='int32'))
    top_cats = np.fliplr(np.argsort(pred, axis=1))
    if fname:
        with open(fname, 'wb') as f_out:
            for i, row in enumerate(top_cats):
                cats = np.array([str(idx_to_cat[row[j]]) for j in range(top_n)])
                f_out.write('{} --> {}\n'.format(ing_names[i], 
                    np.array_str(cats, 
                        max_line_width=10000).replace('\n', '')
                    ))
    else:
        for i, row in enumerate(top_cats):
            cats = np.array([str(idx_to_cat[row[j]]) for j in range(top_n)])
            print '{} --> {}'.format(ing_names[i], 
                np.array_str(cats, 
                    max_line_width=10000).replace('\n', '')
                )

def most_popular_cat_from_ing(
    df, df_i, num_ingredients, category, freq=True, top_n=3, fname=None):
    from foodessentials import get_perc
    counts = df_i['ingredient'].value_counts()
    ing_names = counts.index.values[:num_ingredients]
    if fname:
        with open(fname, 'wb') as f_out:
            for ing in ing_names:
                x = get_perc(ing, category, df, df_i)
                if freq:
                    # Sort by freq rather than percentage.
                    x = sorted(x, key=lambda x : x[2])
                cats = np.array([str(i[1]) for i in x[-top_n:][::-1]])
                f_out.write('{} --> {}\n'.format(ing, 
                    np.array_str(cats, 
                        max_line_width=10000).replace('\n', '')
                    ))
    else:
        for ing in ing_names:
            x = get_perc(ing, category, df, df_i)
            if freq:
                # Sort by freq rather than percentage.
                x = sorted(x, key=lambda x : x[2])
            cats = np.array([str(i[1]) for i in x[-top_n:][::-1]])
            print '{} --> {}'.format(ing, 
                np.array_str(cats, 
                    max_line_width=10000).replace('\n', '')
                )
    
def main():
    num_ingredients = 1000
    ings_per_prod = None
    use_embeddings = False
    output_cat = 'shelf'
    df, df_i = import_data()
    counts = df_i['ingredient'].value_counts()
    inputs_, outputs, idx_to_cat = gen_input_outputs_cat(
                        df, counts, num_ingredients, output_cat, ings_per_prod)
    if use_embeddings:
        #embeddings = np.load('embeddings/embeddings_{}.npy'.format(num_ingredients))
        #embeddings = np.load('../word2vec/word2vec_embeddings.npy')[1][:num_ingredients]
        embeddings = 2*np.random.random((num_ingredients, 300))-1 # Try random embeddings
        embeddings = embeddings.astype('float32')
        inputs = input_from_embeddings(inputs_, embeddings, 
            normalize=False)
        #inputs = (2*np.random.random(inputs_.shape)-1).astype('float32') # Completely random inputs.
    else:
        inputs = inputs_
    num_outputs = outputs.max()+1

    print "# of data points:", len(inputs)
    # Scramble inputs/outputs
    np.random.seed(3)
    random_idx = np.random.permutation(len(inputs))
    inputs = inputs[random_idx]
    outputs = outputs[random_idx]

    test_split_idx = int(len(inputs))*0.8
    inputs, X_test = inputs[:test_split_idx], inputs[test_split_idx:]
    outputs, y_test = outputs[:test_split_idx], outputs[test_split_idx:]

    X_train, X_valid, y_train, y_valid = train_test_split(
        inputs, outputs, test_size=0.2, random_state=42)

    print "Running models..."
    # Max entropy model
    # Normalize
    #inputs_n = inputs / np.sum(inputs, axis=1)[:,None]
    #regr = max_entropy(X_train, y_train, X_valid, y_valid)
    #predict_cat(counts, regr, idx_to_cat, num_ingredients, ings)

    # Neural network model
    classifier, predict_model = run_nn(X_train, y_train, X_valid, y_valid, 
                              X_train.shape[1], num_outputs,
                              m=100, n_epochs=10, batch_size=10,
                              learning_rate=0.1, L2_reg=0.0001)

    pred_valid = predict_model(X_valid)
    pred_test = predict_model(X_test)
    if output_cat != 'aisle':
        lower_to_upper_cat = get_upper_cat(df, output_cat, 'aisle')
        print "Validation set:\n", calc_accuracy(pred_valid, y_valid)
        print calc_accuracy(pred_valid, y_valid, lower_to_upper_cat)
        print "Test set:\n", calc_accuracy(pred_test, y_test)
        print calc_accuracy(pred_test, y_test, lower_to_upper_cat)
    #print_predictions(X_valid, y_valid, 
    #    np.argmax(pred_valid, axis=1), idx_to_cat, counts, limit=100)

    if not use_embeddings:
        embeddings_out = classifier.hiddenLayer.W.get_value()
        ranks, neigh = get_nearest_neighbors(embeddings_out)
        #print_nearest_neighbors(counts.index.values[:num_ingredients], ranks)
        highest_rank, score, avg_rank_of_ing_cat, random_score = calc_score(
            ranks, num_ingredients)

if __name__ == '__main__':
    main()
