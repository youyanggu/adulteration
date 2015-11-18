import time

import numpy as np
import sklearn.neighbors
import theano
import theano.tensor as T

from gather_data import *

theano.config.floatX = 'float32'

def load_data(x, y):
    def shared_dataset(x, borrow=True):
        shared_x = theano.shared(np.asarray(x.astype('int32'),
                                 dtype='int32'),
                                 borrow=borrow)
        return shared_x
    
    x_train = shared_dataset(x)
    y_train = shared_dataset(y)
    return x_train, y_train


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
        t2 = T.sum(T.log(1-self.p_y_given_x)[T.arange(y.shape[0])]) - \
                T.log(1-self.p_y_given_x)[T.arange(y.shape[0]), y]

        return T.mean(-t1-t2)

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
            print W_values[0]
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
            activation=T.nnet.sigmoid
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
        


def run_nn(x_train, y_train, num_ingredients, m, 
           learning_rate=0.05, L2_reg=0.001, n_epochs=15, batch_size=1):
    print 'Building model'

    x_train, y_train = load_data(x_train, y_train)

    index = T.iscalar()  # index to a [mini]batch
    x = T.imatrix('x')  # the data is presented as rasterized images
    #y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
    y = T.ivector('y')

    if batch_size is None:
        batch_size = x_train.get_value(borrow=True).shape[0]
        n_train_batches = 1
    else:
        n_train_batches = x_train.get_value(borrow=True).shape[0] / batch_size

    rng = np.random.RandomState(3)

    classifier = NN(
        rng=rng,
        inp=x,
        n_in=num_ingredients,
        m=m,
        n_out=num_ingredients,
    )
    cost = classifier.cost(y) + L2_reg * classifier.L2
    #gparams = T.grad(cost, classifier.params)
    gparams = [T.grad(cost, param).astype('float32') for param in classifier.params]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]
        
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: x_train[index * batch_size: (index + 1) * batch_size],
            y: y_train[index * batch_size: (index + 1) * batch_size],
        }
    )
    predict_model = theano.function(
        inputs=[],
        outputs=classifier.outputLayer.p_y_given_x,
        givens={
            x: x_train,
        }
    )
    print 'Training'

    start_time = time.time()

    epoch = 0
    while epoch < n_epochs:
        epoch = epoch + 1
        print epoch
        costs = []
        for minibatch_index in xrange(n_train_batches):
            costs.append(train_model(minibatch_index))
        print np.array(costs).mean()
        print classifier.hiddenLayer.W.get_value()[0]
        #print classifier.outputLayer.W.get_value()[0]
        #print classifier.hiddenLayer.b.get_value()
        #print classifier.outputLayer.b.get_value()

    print >> sys.stderr, ('The code ran for %.2fm' % ((time.time() - start_time) / 60.))
    pred = predict_model()

    return classifier, pred

def max_entropy(inputs, outputs):
    from sklearn.linear_model import LogisticRegression
    from sklearn.cross_validation import train_test_split
    regr = LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(
        inputs, outputs, test_size=0.33, random_state=42)
    regr.fit(X_train, y_train)
    print regr.score(X_train, y_train)
    print regr.score(X_test, y_test)
    return regr

def predict_cat(counts, regr, idx_to_cat, num_ingredients, ings):
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
    return idx_to_cat[pred_class], np.max(regr.predict_proba(inp))

 
def main():
    num_ingredients = 1000
    output_cat = 'shelf'
    df, df_i = import_data()
    counts = df_i['ingredient'].value_counts()
    inputs, outputs, idx_to_cat = gen_input_outputs_cat(
                        df, df_i, num_ingredients, output_cat)
    
    # Max entropy model
    regr = max_entropy(inputs, outputs)
    #predict_cat(counts, regr, idx_to_cat, num_ingredients, ings)

    # Neural network model
    classifier, pred = run_nn(inputs, outputs, num_ingredients, 
                                m=500, n_epochs=20, batch_size=100)

    pred_cats = np.argmax(pred, axis=1)
    acc = (pred_cats == outputs).sum() * 1.0 / len(pred)
    print acc


if __name__ == '__main__':
    main()
