import itertools
import pickle
import time

import numpy as np
import sklearn.neighbors
import theano
import theano.tensor as T

from gather_data import *
from scoring import *
from utils import *
from nn_category import HiddenLayer, OutputLayer

data_dir = 'data/'
embed_dir = 'embeddings/'

theano.config.floatX = 'float32'

def test_model(predict_model, ings, reps, idx_to_cat, top_n=None, fname=None, target_ings=None):
    results = predict_model(reps)
    ranks = np.fliplr(np.argsort(results))
    if top_n:
        ranks = ranks[:,:top_n]
    if fname:
        f_out = open(fname, 'wb')
    for i, ing in enumerate(ings):
        include = True
        if target_ings:
            include = False
            for target_ing in target_ings:
                if target_ing in ing:
                    include = True
                    break
        if include:
            if fname:
                f_out.write('{} --> {}\n'.format(ing, [idx_to_cat[j] for j in ranks[i]]))
            else:
                print '{} --> {}'.format(ing, [idx_to_cat[j] for j in ranks[i]])
    if fname:
        f_out.close()

def subsample(x_train, y_train, prob):
    if prob is None:
        return x_train, y_train
    use_indices = []
    for i in range(len(x_train)):
        if np.random.random() < prob[x_train[i]]:
            use_indices.append(i)
    use_indices = np.array(use_indices)
    return x_train[use_indices], y_train[use_indices]


def get_batch(x_train, y_train, idx, batch_size, ing_idx_to_hier_map):
    x = x_train[idx*batch_size : (idx+1)*batch_size]
    x = np.array([ing_idx_to_hier_map[i] for i in x])
    y = y_train[idx*batch_size : (idx+1)*batch_size]
    if scipy.sparse.issparse(y):
        y = y.toarray()
    return x, y

class NN(object):
    def __init__(self, rng, inp, num_ingredients, n_in, m, n_out):
        self.inp = inp
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
            (self.hiddenLayer.W ** 2).sum()
            + (self.hiddenLayer.b ** 2).sum()
            + (self.outputLayer.W ** 2).sum()
            + (self.outputLayer.b ** 2).sum()
        )
        
        self.cost = self.outputLayer.cost
        self.errors = self.outputLayer.errors

        self.params = self.hiddenLayer.params + self.outputLayer.params
        
def run_nn(x_train, y_train, num_ingredients, num_outputs, m, input_size,
           learning_rate, L2_reg, n_epochs, batch_size, rng, min_count, ing_idx_to_hier_map):
    print 'Building model'

    x = T.fmatrix('x')
    y = T.ivector('y')

    classifier = NN(
        rng=rng,
        inp=x,
        num_ingredients=num_ingredients,
        n_in=input_size,
        m=m,
        n_out=num_outputs,
    )
    cost = classifier.cost(y) + L2_reg * classifier.L2
    #gparams = T.grad(cost, classifier.params)
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
        x_train_sub, y_train_sub = subsample(
            x_train, y_train, prob)
        n_train_batches = x_train_sub.shape[0] / batch_size
        costs = []
        for idx in range(n_train_batches):
            #print idx
            #ret = train_model(idx)
            x_train_, y_train_ = get_batch(
                    x_train_sub, y_train_sub, idx, batch_size, ing_idx_to_hier_map)
            ret = train_model(x_train_, y_train_)
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

        #embeddings = classifier.inp_all.get_value()
        #ranks_all, neigh = get_nearest_neighbors(embeddings)
        #highest_rank, score, avg_rank_of_ing_cat, random_score = calc_score(
        #    ranks_all, num_ingredients)

    print 'The code ran for %.2fm' % ((time.time() - start_time) / 60.)

    return classifier, predict_model

def save_input_outputs(inputs, outputs, suffix=''):
    if suffix:
        suffix = '_' + str(suffix)
    np.save(data_dir+'inputs_cat{}.npy'.format(suffix), inputs)
    np.save(data_dir+'outputs_cat{}.npy'.format(suffix), outputs)

def load_input_outputs(suffix=''):
    if suffix:
        suffix = '_' + str(suffix)
    inputs = np.load(data_dir+'inputs_cat{}.npy'.format(suffix))
    outputs = np.load(data_dir+'outputs_cat{}.npy'.format(suffix))
    with open('../ncim/idx_to_cat.pkl', 'rb') as f_in:
        idx_to_cat = pickle.load(f_in)
    return inputs, outputs, idx_to_cat

def gen_ing_idx_to_hier_map(ings_ordered):
    ing_to_idx_map = {ings_ordered[i] : i for i in range(len(ings_ordered))}
    ing_idx_to_hier_map = {}
    ings = np.load('../ncim/all_ings.npy')
    reps = np.load('../ncim/all_ings_reps.npy')
    assert len(ings)==len(reps)
    for i in range(len(ings)):
        ing_idx_to_hier_map[ing_to_idx_map[ings[i]]] = reps[i].astype('float32')
    return ing_idx_to_hier_map

def gen_ing_cat_pair_map(inputs, outputs):
    """Generate map that takes as input the ing/cat pair and output whether
    or not this pair has appeared."""
    ing_cat_pair_map = {}
    for inp, out in zip(inputs, outputs):
        if (inp, out) not in ing_cat_pair_map:
            ing_cat_pair_map[(inp, out)] = True
    return ing_cat_pair_map

def evaluate(valid_ing_indices, results, ing_cat_pair_map):
    """Evaluation metric via the mean average precision."""
    # Match prec: 0.448 vs 0.136 for random
    # MAP: 0.497 vs 0.166 for random
    ranks = np.fliplr(np.argsort(results))
    ranks_random = np.array([np.random.permutation(
        results.shape[1]) for i in range(results.shape[0])])
    precisions = []
    match_percs = []
    for i, rank in enumerate(ranks):
        ing_idx = valid_ing_indices[i]

        cats = set()
        for j in range(results.shape[1]):
            if (ing_idx, j) in ing_cat_pair_map:
                cats.add(j)
        num_cats = len(cats)

        c_ranks = sorted([np.where(rank==c)[0][0] for c in cats])
        mean_precision = 0
        for j, c_rank in enumerate(c_ranks):
            mean_precision += (j+1.)/(c_rank+1)
        mean_precision /= len(c_ranks)
        precisions.append(mean_precision)

        matches = 0
        for cat_rank, cat_idx in enumerate(rank[:num_cats]):
            cat_rank += 1
            if (ing_idx, cat_idx) in ing_cat_pair_map:
                matches += 1
        match_perc = matches * 1. / num_cats
        match_percs.append(match_perc)
    precisions = np.array(precisions)
    match_percs = np.array(match_percs)
    print "MAP    :", precisions.mean()
    print "Match %:", match_percs.mean()



def default_args():
    df, df_i = import_data()
    counts = df_i['ingredient'].value_counts()
    num_ingredients=5000
    m=20
    input_size=10
    learning_rate=0.1
    L2_reg=0.0005
    n_epochs=10
    batch_size=100
    seed=3
    use_npy=True
    ings_per_prod=None
    rng=np.random.RandomState(seed)
    min_count=5000

def run_nn_helper(df, counts, 
         num_ingredients=5000, m=20, input_size=10,
         learning_rate=0.1, L2_reg=0.0005,
         n_epochs=10, batch_size=100, seed=3, 
         use_npy=False, 
         ings_per_prod=5,
         min_count=None,
         **kwargs):
    ings = counts.index.values[:num_ingredients]
    if use_npy:
        inputs, outputs, idx_to_cat = load_input_outputs(num_ingredients)
        inputs, outputs = break_down_inputs(inputs, outputs)
    else:
        print "Gathering inputs/outputs..."
        output_cat = 'shelf'
        inputs_, outputs_, idx_to_cat = gen_input_outputs_cat(
                            df, counts, num_ingredients, output_cat, ings_per_prod)
        save_input_outputs(inputs_, outputs_, num_ingredients)
        inputs, outputs = break_down_inputs(inputs_, outputs_)

    num_outputs = outputs.max()+1
    print "# of data points:", len(inputs)
    # Scramble inputs/outputs
    np.random.seed(seed)
    random_idx = np.random.permutation(len(inputs))
    inputs = inputs[random_idx]
    outputs = outputs[random_idx]

    ing_idx_to_hier_map = gen_ing_idx_to_hier_map(ings)
    input_size = ing_idx_to_hier_map.values()[0].shape[0]
    inp_exist_indices = np.array(
        [i for i in range(len(inputs)) if inputs[i] in ing_idx_to_hier_map])
    inputs = inputs[inp_exist_indices]
    outputs = outputs[inp_exist_indices]

    ing_cat_pair_map = gen_ing_cat_pair_map(inputs, outputs)

    classifier, predict_model = run_nn(inputs, outputs, 
                        num_ingredients=num_ingredients, 
                        num_outputs=num_outputs,
                        m=m, 
                        input_size=input_size,
                        learning_rate=learning_rate, 
                        L2_reg=L2_reg,
                        n_epochs=n_epochs, 
                        batch_size=batch_size,
                        rng=np.random.RandomState(seed),
                        min_count=min_count,
                        ing_idx_to_hier_map=ing_idx_to_hier_map,
                        )

    valid_ing_indices = [i for i in range(num_ingredients) if i in ing_idx_to_hier_map]
    valid_ing_reps = np.array([ing_idx_to_hier_map[i] for i in valid_ing_indices])
    valid_ings = [ings[i] for i in valid_ing_indices]
    valid_ing_results = predict_model(valid_ing_reps)
    test_model(valid_ing_results, valid_ings, idx_to_cat, 3)
    evaluate(valid_ing_indices, valid_ing_results, ing_cat_pair_map)

    found_ings = np.load('../rasff/found_chems.npy')
    new_ings_reps = np.load('../rasff/found_chems_reps.npy').astype('float32')
    new_ings_results = predict_model(new_ings_reps)
    test_model(new_ings_results, found_ings, idx_to_cat, 3)
    #evaluate(valid_ing_indices, new_ings_results, ing_cat_pair_map)

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
    params['num_ingredients'] = 5000
    #params['num_ingredients'] = 120

    params['use_npy'] = True
    #params['learning_rate'] = 0.1
    #params['L2_reg'] = 0.0005
    params['m'] = 10
    params['input_size'] = 10
    params['seed'] = 3
    params['n_epochs'] = 8
    params['batch_size'] = 200
    params['ings_per_prod'] = None
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

    # Only use the last param's weights for now.
    #num_ingredients = params['num_ingredients'][-1]
    #embeddings = classifier.inp_all.get_value()
    #save_embeddings(embeddings, num_ingredients)
    #print_embeddings(counts.index.values[:num_ingredients], embeddings)

if __name__ == '__main__':
    main()

