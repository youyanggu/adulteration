import time

import numpy as np
import sklearn.neighbors
import theano
import theano.tensor as T

from gather_data import *

theano.config.floatX = 'float32'

def load_data(x, y, z):
    def shared_dataset(x, borrow=True):
        shared_x = theano.shared(np.asarray(x.astype('int32'),
                                 dtype='int32'),
                                 borrow=borrow)
        return shared_x
    
    x_train = shared_dataset(x)
    y_train = shared_dataset(y)
    output_lens = shared_dataset(z)
    return x_train, y_train, output_lens


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
        self.p_y_given_x = T.dot(inp, self.W) + self.b
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

    def cost(self, y, out_len):
        t1 = T.dot(self.p_y_given_x, T.sum(y, axis=1).T)[0,0]
        t2 = T.log(T.sum(T.exp(self.p_y_given_x)))
        return T.mean(-t1+t2*out_len)

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
    def __init__(self, rng, inp, n_in, n_hidden, n_out):
        self.inp = inp
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            inp=inp,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.nnet.sigmoid
        )
        self.outputLayer = OutputLayer(
            inp=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )
        self.L2 = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.outputLayer.W ** 2).sum()
        )
        
        self.cost = self.outputLayer.cost
        self.errors = self.outputLayer.errors

        self.params = self.hiddenLayer.params + self.outputLayer.params
        


def run_nn(x_train, y_train, output_lens, num_ingredients, n_hidden, 
           learning_rate=0.0001, L2_reg=0.0001, n_epochs=5, batch_size=1):
    print 'Building model'

    x_train, y_train, output_lens = load_data(x_train, y_train, output_lens)

    index = T.iscalar()  # index to a [mini]batch
    x = T.imatrix('x')  # the data is presented as rasterized images
    #y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
    y = T.itensor3('y')
    out_len = T.ivector('out_len')

    n_train_batches = x_train.get_value(borrow=True).shape[0] / batch_size

    rng = np.random.RandomState(1234)

    classifier = NN(
        rng=rng,
        inp=x,
        n_in=num_ingredients,
        n_hidden=n_hidden,
        n_out=num_ingredients,
    )
    cost = classifier.cost(y, out_len) + L2_reg * classifier.L2
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
            out_len: output_lens[index * batch_size: (index + 1) * batch_size]
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

    print >> sys.stderr, ('The code ran for %.2fm' % ((time.time() - start_time) / 60.))
    return classifier

def main():
    num_ingredients = 100
    df, df_i = import_data()
    inputs, outputs, output_lens = gen_input_outputs(df, df_i, num_ingredients)
    assert(len(inputs)==len(outputs))
    classifier = run_nn(inputs, outputs, output_lens, num_ingredients, 100)

    embeddings = classifier.hiddenLayer.W.get_value()
    counts = df_i['ingredient'].value_counts()

    neigh = sklearn.neighbors.NearestNeighbors(2, algorithm='brute', metric='cosine')
    neigh.fit(embeddings)
    ing_names = counts.index.values
    for i in range(num_ingredients):
        print '{} --> {}'.format(ing_names[i],
            ing_names[neigh.kneighbors(embeddings[i])[1][0][1]])



if __name__ == '__main__':
    main()
