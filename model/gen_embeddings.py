import itertools
import time

import numpy as np
import sklearn.neighbors
import theano
import theano.tensor as T

from gather_data import *
from scoring import *
from utils import *

data_dir = 'data/'
embed_dir = 'embeddings/'

#theano.config.floatX = 'float32'

def subsample(x_train, y_train, output_lens, prob):
    if prob is None:
        return x_train, y_train, output_lens
    use_indices = []
    for i in range(len(x_train)):
        if np.random.random() < prob[x_train[i]]:
            use_indices.append(i)
    use_indices = np.array(use_indices)
    return x_train[use_indices], y_train[use_indices], output_lens[use_indices]


def get_batch(x_train, y_train, output_lens, idx, batch_size):
    x = x_train[idx*batch_size : (idx+1)*batch_size]
    y = y_train[idx*batch_size : (idx+1)*batch_size]
    z = output_lens[idx*batch_size : (idx+1)*batch_size]
    if scipy.sparse.issparse(y):
        y = y.toarray()
    return x, y, z

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

    def t1(self, y, out_len):
        #return T.dot(self.p_y_given_x, T.sum(y, axis=1).T).diagonal()
        return T.dot(self.p_y_given_x, y.T).diagonal()

    def t2(self, y, out_len):
        return T.log(T.sum(T.exp(self.p_y_given_x), axis=1))

    def cost(self, y, out_len):
        #t1 = T.dot(self.p_y_given_x, T.sum(y, axis=1).T).diagonal()
        t1 = T.dot(self.p_y_given_x, y.T).diagonal()
        t2 = T.log(T.sum(T.exp(self.p_y_given_x), axis=1))
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
    def __init__(self, rng, inp_idx, n_in, m, n_out, inp_all=None):
        if inp_all is None:
            inp_all_values = np.asarray(
                rng.uniform(low=-1, high=1, size=(n_out, n_in)), dtype=theano.config.floatX
                #np.zeros((n_out, n_in), dtype=theano.config.floatX)
            )
            print inp_all_values[0]
            inp_all = theano.shared(value=inp_all_values, name='inp_all', borrow=True)
        self.inp_all = inp_all

        self.inp = inp_all[inp_idx]
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            inp=self.inp,
            n_in=n_in,
            n_out=m,
            activation=T.tanh#T.nnet.sigmoid
        )
        self.outputLayer = OutputLayer(
            inp=self.hiddenLayer.output,
            n_in=m,
            n_out=n_out
        )
        self.L2 = (
            (self.inp_all ** 2).sum()
            + (self.hiddenLayer.W ** 2).sum()
            + (self.hiddenLayer.b ** 2).sum()
            + (self.outputLayer.W ** 2).sum()
            + (self.outputLayer.b ** 2).sum()
        )
        
        self.cost = self.outputLayer.cost
        self.t1 = self.outputLayer.t1
        self.t2 = self.outputLayer.t2
        self.errors = self.outputLayer.errors

        self.params = [self.inp_all] + self.hiddenLayer.params + self.outputLayer.params
        


def run_nn(x_train, y_train, output_lens, num_ingredients, m, input_size,
           learning_rate, L2_reg, n_epochs, batch_size, rng, min_count):
    print 'Building model'

    x = T.ivector('x')
    y = T.imatrix('y')
    out_len = T.ivector('out_len')

    classifier = NN(
        rng=rng,
        inp_idx=x,
        n_in=input_size,
        m=m,
        n_out=num_ingredients,
    )
    cost = classifier.cost(y, out_len) + L2_reg * classifier.L2
    t1 = classifier.t1(y, out_len)
    t2 = classifier.t2(y, out_len)
    #gparams = T.grad(cost, classifier.params)
    gparams = [T.grad(cost, param).astype('float32') for param in classifier.params]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]
        
    train_model = theano.function(
        inputs=[x, y, out_len],
        outputs=cost,
        updates=updates,
    )
    predict_model = theano.function(
        inputs=[x],
        outputs=classifier.outputLayer.p_y_given_x,
    )
    print 'Training'

    start_time = time.time()

    prob = None
    if min_count:
        prob = np.array([min(1, i) for i in (min_count*1./np.bincount(x_train))])

    epoch = 0
    while epoch < n_epochs:
        epoch = epoch + 1
        print '---------------------'
        print 'Epoch #', epoch
        x_train_sub, y_train_sub, output_lens_sub = subsample(
            x_train, y_train, output_lens, prob)
        n_train_batches = x_train_sub.shape[0] / batch_size
        costs = []
        for idx in range(n_train_batches):
            #print idx
            #ret = train_model(idx)
            x_train_, y_train_, output_lens_ = get_batch(
                    x_train_sub, y_train_sub, output_lens_sub, idx, batch_size)
            ret = train_model(x_train_, y_train_, output_lens_)
            #print ret
            costs.append(ret)
        print "Learning rate:", learning_rate
        learning_rate = max(0.005, learning_rate/2.)
        updates = [
                (param, param - learning_rate * gparam)
                for param, gparam in zip(classifier.params, gparams)
        ]
        print np.array(costs).mean()
        #print classifier.hiddenLayer.W.get_value()[0]
        #print classifier.inp_all.get_value()[0]
        #print classifier.hiddenLayer.b.get_value()
        #print classifier.outputLayer.W.get_value()[0]
        #print classifier.outputLayer.b.get_value()

        embeddings = classifier.inp_all.get_value()
        ranks_all, neigh = get_nearest_neighbors(embeddings)
        highest_rank, score, avg_rank_of_ing_cat, random_score = calc_score(
            ranks_all, num_ingredients)

    print 'The code ran for %.2fm' % ((time.time() - start_time) / 60.)

    return classifier, predict_model

def get_coocc_ranks():
    coocc = np.load(data_dir+'cooccurance.npy')
    coocc_sym = coocc*coocc.T
    ranks_all = []
    for i in range(coocc.shape[0]):
        rank = coocc.shape[1]-np.searchsorted(
            coocc_sym[i], coocc_sym[i], sorter=np.argsort(coocc_sym[i]))-1
        ranks_all.append(rank)
    return np.array(ranks_all)

def get_cooccurance_prob(ing1, ing2, df, df_i):
    prods1 = ing_utils.find_products_by_ing(ing1, df=df, df_i=df_i)
    counts1 = prods1.ingredients_clean.map(lambda x: ing2 in x).sum()
    prob1 = counts1*1.0/len(prods1)
    prods2 = ing_utils.find_products_by_ing(ing2, df=df, df_i=df_i)
    counts2 = prods2.ingredients_clean.map(lambda x: ing1 in x).sum()
    prob2 = counts2*1.0/len(prods2)
    return prob1, prob2

def get_cooccurance_matrix(ing_list):
    matrix = np.eye(len(ing_list))
    df, df_i = import_data()
    for i,j in itertools.combinations(range(len(ing_list)), 2):
        print ing_list[i], ing_list[j]
        prob1, prob2 = get_cooccurance_prob(ing_list[i], ing_list[j], df, df_i)
        print prob1, prob2
        matrix[i,j] = prob1
        matrix[j,i] = prob2
    #np.save(data_dir+'cooccurance.npy', matrix)
    return matrix

def calc_nn_coocc(pred, inputs, ranks_coocc, top_n=10, prune=10):
    def prune_ranks(ranks, prune):
        arr = np.copy(ranks)
        if prune:
            arr[arr>prune] = arr.max()
        return arr
    """Calculate correlation of cooccurance matrix with cooccurance from NN."""
    pred_coocc = {}
    for i in range(len(pred)):
        idx = np.where(inputs[i]==1)[0][0]
        if idx not in pred_coocc:
            argsort_nn = pred.shape[1]-np.searchsorted(
                pred[i], pred[i], sorter=np.argsort(pred[i]))-1
            corr = np.corrcoef(prune_ranks(argsort_nn, prune), 
                               prune_ranks(ranks_coocc[idx], prune))[0,1]
            intersects = np.intersect1d(np.argsort(pred[i])[::-1][1:top_n+1], 
                                        np.argsort(ranks_coocc[idx])[::-1][1:top_n+1])
            pred_coocc[idx] = (corr, len(intersects))
    return pred_coocc

def get_counts(inputs):
    d = {}
    s = np.sum(inputs, axis=0)
    for i, j in enumerate(counts.index.values[:100]):
        d[i] = (j, int(s[i]))
    return d

def print_nearest_neighbors(ing_names, ranks, top_n=3, fname=None):
    if fname:
        with open(fname, 'wb') as f_out:
            for i in range(ranks.shape[0]):
                nearest_neighbors = np.argsort(ranks[i])
                f_out.write('{} --> {}\n'.format(ing_names[i], 
                np.array_str(ing_names[nearest_neighbors[1:top_n+1]], 
                    max_line_width=10000).replace('\n', '')
                ))
    else:
        for i in range(ranks.shape[0]):
            nearest_neighbors = np.argsort(ranks[i])
            print '{} --> {}'.format(ing_names[i], 
                np.array_str(ing_names[nearest_neighbors[1:top_n+1]], 
                    max_line_width=10000).replace('\n', '')
                )

def compare_neighbors(neigh, embeddings, all_ings, ing, ings_to_compare):
    assert(ing in all_ings)
    for i in ings_to_compare:
        assert(i in all_ings)
    if type(ings_to_compare) == list:
        ings_to_compare = np.array(ings_to_compare)
    ing_idx = np.where(all_ings==ing)[0][0]
    nn_values, nn_idx = neigh.kneighbors(embeddings[ing_idx])
    nn_values = nn_values[0]
    nn_idx = nn_idx[0]
    ings_to_compare_idx = [np.where(all_ings==i)[0][0] for i in ings_to_compare]
    ings_to_compare_ranks = [np.where(nn_idx==i)[0][0] for i in ings_to_compare_idx]
    cosine_distances = nn_values[ings_to_compare_ranks]
    args = np.argsort(cosine_distances)
    sorted_cosine_distances = sorted(cosine_distances)
    sorted_ings_to_compare = ings_to_compare[args]
    assert(len(sorted_cosine_distances)==len(sorted_ings_to_compare))
    for i in range(len(ings_to_compare)):
        print sorted_ings_to_compare[i], sorted_cosine_distances[i]
    return sorted_cosine_distances, sorted_ings_to_compare

def get_nearest_neighbors(embeddings):
    num_ingredients = embeddings.shape[0]
    neigh = sklearn.neighbors.NearestNeighbors(
        num_ingredients, algorithm='brute', metric='cosine')
    neigh.fit(embeddings)
    ranks_all = []
    for i in range(num_ingredients):
        nearest_neighbors = neigh.kneighbors(embeddings[i:i+1])[1][0]
        ranks = np.argsort(nearest_neighbors)
        ranks_all.append(ranks)
    return np.array(ranks_all), neigh

def save_input_outputs(inputs, outputs, output_lens, suffix=''):
    if suffix:
        suffix = '_' + str(suffix)
    np.save(data_dir+'inputs{}.npy'.format(suffix), inputs)
    np.savez(data_dir+'outputs{}.npz'.format(suffix), data=outputs.data, 
            indices=outputs.indices, indptr=outputs.indptr, shape=outputs.shape)
    np.save(data_dir+'output_lens{}.npy'.format(suffix), output_lens)

def load_input_outputs(suffix=''):
    if suffix:
        suffix = '_' + str(suffix)
    inputs = np.load(data_dir+'inputs{}.npy'.format(suffix))
    loader = np.load(data_dir+'outputs{}.npz'.format(suffix))
    outputs = scipy.sparse.csr_matrix((loader['data'], 
        loader['indices'], loader['indptr']), shape=loader['shape'])
    output_lens = np.load(data_dir+'output_lens{}.npy'.format(suffix))
    return inputs, outputs, output_lens

def save_embeddings(embeddings, suffix=''):
    fname = embed_dir+'embeddings_{}.npy'.format(suffix)
    np.save(fname, embeddings)

def load_embeddings(suffix=''):
    fname = embed_dir+'embeddings_{}.npy'.format(suffix)
    return np.load(fname)

def print_embeddings(ings, embeddings, fname=None):
    if fname:
        f_out = open(fname, 'wb')
    for i in range(len(embeddings)):
        if not fname:
            print ings[i]
            print embeddings[i]
        else:
            f_out.write(ings[i]+'\n')
            f_out.write(str(embeddings[i])+'\n')
    if fname:
        f_out.close()

def print_output(ing_names, num_ingredients, predict_model, top_n=3):
    ing_names = ing_names[:num_ingredients]
    pred_outputs = predict_model(np.arange(num_ingredients).astype('int32'))
    for i, ing in enumerate(ing_names):
        out = np.argsort(pred_outputs[i])[::-1]
        print '{} --> {}'.format(ing, 
                np.array_str(ing_names[out[:top_n]], 
                    max_line_width=10000).replace('\n', '')
                )

def default_args():
    df, df_i = import_data()
    counts = df_i['ingredient'].value_counts()
    num_ingredients=120
    m=20
    input_size=10
    learning_rate=0.1
    L2_reg=0.0005
    n_epochs=10
    batch_size=100
    seed=3
    use_npy=False
    max_output_len=4
    max_rotations=5
    random_rotate=True
    rng=np.random.RandomState(seed)
    min_count=5000


def run_nn_helper(df, counts, 
         num_ingredients=120, m=20, input_size=10,
         learning_rate=0.1, L2_reg=0.0005,
         n_epochs=10, batch_size=100, seed=3, 
         use_npy=False, 
         max_output_len=4, max_rotations=5, random_rotate=True,
         min_count=5000,
         **kwargs):
    if use_npy:
        inputs, outputs, output_lens = load_input_outputs(num_ingredients)
    else:
        print "Gathering inputs/outputs..."
        inputs, outputs, output_lens = gen_input_outputs(df['ingredients_clean'].values, 
                counts, num_ingredients, max_output_len, max_rotations, random_rotate)
        save_input_outputs(inputs, outputs, output_lens, num_ingredients)

    print "# of data points:", len(inputs)
    # Scramble inputs/outputs
    np.random.seed(seed)
    random_idx = np.random.permutation(len(inputs))
    inputs = inputs[random_idx]
    outputs = outputs[random_idx]
    output_lens = output_lens[random_idx]

    classifier, predict_model = run_nn(inputs, outputs, output_lens, 
                        num_ingredients=num_ingredients, 
                        m=m, 
                        input_size=input_size,
                        learning_rate=learning_rate, 
                        L2_reg=L2_reg,
                        n_epochs=n_epochs, 
                        batch_size=batch_size,
                        rng=np.random.RandomState(seed),
                        min_count=min_count
                        )

    embeddings = classifier.inp_all.get_value()
    ranks_all, neigh = get_nearest_neighbors(embeddings)
    #print_nearest_neighbors(counts.index.values, ranks_all, 3)
    return ranks_all, classifier, predict_model

def main():
    df, df_i = import_data()
    counts = df_i['ingredient'].value_counts()
    my_product = lambda x: [dict(
        itertools.izip(x, i)) for i in itertools.product(*x.itervalues())]
    params = {}

    # one must be here
    #params['num_ingredients'] = 5000
    params['num_ingredients'] = 120

    params['use_npy'] = False
    #params['learning_rate'] = 0.1
    #params['L2_reg'] = 0.0005
    params['m'] = 10
    params['input_size'] = 10
    params['seed'] = 3
    params['n_epochs'] = 8
    params['batch_size'] = 200
    params['max_output_len'] = 5
    params['max_rotations'] = None#10
    params['random_rotate'] = True
    params['min_count'] = None

    for k,v in params.iteritems():
        if type(v) != list:
            params[k] = [v]

    param_scores = {}
    iterations = 0
    total_ranks = None
    for param in my_product(params):
        print '==========================================='
        print param
        for k in param:
            if len(params[k]) > 1:
                print '{} : {}'.format(k, param[k])
        iterations += 1
        ranks_all, classifier, predict_model = run_nn_helper(df, counts, **param)
        highest_rank, score, avg_rank_of_ing_cat, random_score = calc_score(
            ranks_all, param['num_ingredients'], print_scores=False)
        #param_scores[tuple(sorted(param.items()))] = score.mean()
        
        if total_ranks is None:
            total_ranks = ranks_all
        else:
            total_ranks += ranks_all
    if iterations > 1:
        #print param_scores
        avg_rank_raw = total_ranks / iterations
        avg_rank = np.array([
            np.searchsorted(avg_rank_raw[i], avg_rank_raw[i], 
                sorter=np.argsort(avg_rank_raw[i]))
            for i in range(avg_rank_raw.shape[0])
            ])
        print '\n==========================================================='
        print '==========================================================='
        print_nearest_neighbors(counts.index.values, avg_rank, 3)
        highest_rank, score, avg_rank_of_ing_cat, random_score = calc_score(
            avg_rank, param['num_ingredients'])

    # Only use the last param's weights for now.
    #num_ingredients = params['num_ingredients'][-1]
    #embeddings = classifier.inp_all.get_value()
    #save_embeddings(embeddings, num_ingredients)
    #print_embeddings(counts.index.values[:num_ingredients], embeddings)

if __name__ == '__main__':
    main()
