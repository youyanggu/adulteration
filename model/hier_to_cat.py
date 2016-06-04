import itertools
import pickle
import time

import numpy as np
from sklearn.cross_validation import train_test_split
import sklearn.neighbors
import theano
import theano.tensor as T

from gather_data import *
from scoring import *
from split_data import *
from utils import *
from nn_category import HiddenLayer, OutputLayer

data_dir = 'data/'
embed_dir = 'embeddings/'

#theano.config.floatX = 'float32'

def test_model(results, ings, idx_to_cat, top_n=None, fname=None, target_ings=None, ings_wiki_links=None):
    """Prints predictions based on vector of distribution.

    target_ings : only prints ingredients that contain this word (e.g. 'oil')
    """
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
            if abs(results).sum()==0:
                preds = None
            else:
                #preds = [idx_to_cat[j] for j in ranks[i]]
                preds = ['{} ({:.3f})'.format(idx_to_cat[v], results[i][v]) for v in ranks[i]]
            if ings_wiki_links:
                s = u'{} / {} --> {}'.format(ing.decode('utf-8'), ings_wiki_links.get(ing)[0].decode('utf-8'), preds)
            else:
                s = u'{} --> {}'.format(ing.decode('utf-8'), preds)
            if fname:
                s = s+'\n'
                f_out.write(s.encode('utf8'))
            else:
                print s
    if fname:
        f_out.close()

def evaluate(valid_ing_indices, valid_ing_results, ing_cat_pair_map, select_indices=None):
    """Generate MAP and precision at N for predictions."""
    if select_indices is not None:
        valid_ing_indices_idx = [i for i,v in enumerate(valid_ing_indices) if v in select_indices]
        print '{} --> {}'.format(len(valid_ing_indices), len(valid_ing_indices_idx))
        valid_ing_indices = valid_ing_indices[valid_ing_indices_idx]
        valid_ing_results = valid_ing_results[valid_ing_indices_idx]
    avg_true_results = gen_avg_true_results(valid_ing_indices)
    print "Random:"
    evaluate_map(valid_ing_indices, valid_ing_results, ing_cat_pair_map, random=True)
    print "Avg True Results:"
    evaluate_map(valid_ing_indices, avg_true_results, ing_cat_pair_map, random=False)
    print "Model:"
    evaluate_map(valid_ing_indices, valid_ing_results, ing_cat_pair_map, random=False)

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

def gen_ing_idx_to_hier_map(ings_ordered, adulterants=False, pca_file=None):
    """Generate map of ingredient index to hierarchy representation. 

    If pca_file is specified, it will load the sklearn.decomposition.PCA model and
    transform the data.
    """
    ing_to_idx_map = {ings_ordered[i] : i for i in range(len(ings_ordered))}
    ing_idx_to_hier_map = {}
    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    
    if adulterants == True:
        ings = np.load(parent_dir+'/rasff/found_chems.npy')
        reps = np.load(parent_dir+'/rasff/found_chems_reps.npy').astype('float32')
    elif adulterants == False:
        ings = np.load(parent_dir+'/ncim/all_ings.npy')
        reps = np.load(parent_dir+'/ncim/all_ings_reps.npy').astype('float32')
    else:
        ings_adult = np.load(parent_dir+'/rasff/found_chems.npy')
        reps_adult = np.load(parent_dir+'/rasff/found_chems_reps.npy').astype('float32')
    assert len(ings)==len(reps)
    if pca_file:
        with open(pca_file, 'r') as f:
            pca = pickle.load(f)
        reps = pca.transform(reps)
        if type(adulterants) != bool:
            reps_adult = pca.transform(reps_adult)
    for i in range(len(ings)):
        if ings[i] not in ing_to_idx_map:
            continue
        ing_idx_to_hier_map[ing_to_idx_map[ings[i]]] = reps[i]
    if type(adulterants) != bool:
        for i, a in enumerate(adulterants):
            idx = np.where(ings_adult==a)[0]
            if len(idx) == 0:
                continue
            idx = idx[0]
            ing_idx_to_hier_map[len(ings_ordered)+i] = reps_adult[idx]
    return ing_idx_to_hier_map

def gen_ing_cat_pair_map(inputs, outputs):
    """Generate map that takes as input the ing/cat pair if this pair exists."""
    ing_cat_pair_map = {}
    for inp, out in zip(inputs, outputs):
        if (inp, out) not in ing_cat_pair_map:
            ing_cat_pair_map[(inp, out)] = True
    return ing_cat_pair_map

def gen_adulterant_cat_pair_map(df_=None, found_ings=None, idx_to_cat=None):
    """Generate map that contains existing adulterant/category pairs.

    df_ is generated via rasff.load_df()
    """
    if df_ is None:
        with open('../model/adulterant_cat_pair_map.pkl', 'rb') as f_in:
            adulterant_cat_pair_map = pickle.load(f_in)
        return adulterant_cat_pair_map
    if found_ings is None:
        found_ings = np.load('../rasff/all_rasff_chems.npy')
        #found_ings = np.array(df_.groupby('chemical').size().order()[::-1].index)
    if idx_to_cat is None:
        with open('../ncim/idx_to_cat.pkl', 'rb') as f_in:
            idx_to_cat = pickle.load(f_in)
    adulterant_cat_pair_map = {}
    cat_to_idx = {v: k for k, v in idx_to_cat.items()}
    chemicals = df_['chemical_'].values
    categories = df_['category_'].values
    for chem, cat in zip(chemicals, categories):
        if not cat or not chem or cat == '---':
            continue
        cat = cat.lower()
        idx = np.where(found_ings==chem)[0]
        if len(idx) == 0:
            continue
        idx = idx[0]
        cat_idx = cat_to_idx[cat]
        if (idx, cat_idx) not in adulterant_cat_pair_map:
            adulterant_cat_pair_map[(idx, cat_idx)] = 1
        else:
            adulterant_cat_pair_map[(idx, cat_idx)] += 1
    return adulterant_cat_pair_map

def default_args():
    df, df_i = import_data()
    counts = df_i['ingredient'].value_counts()
    num_ingredients=5000
    m=20 # 100 for hier
    input_size=10
    learning_rate=0.1
    L2_reg=0.0005 # 1e-7 for hier
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
    pca_file = None#'../../rcnn/code/adulteration/pca_100.pkl'
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

    adulterants = wikipedia.get_adulterants() #need to import
    ings = np.hstack([ings, adulterants])

    train_indices, dev_indices, test_indices = split_data_by_wiki(
        ings, seed)

    num_outputs = outputs.max()+1
    print "# of data points:", len(inputs)
    # Scramble inputs/outputs
    np.random.seed(seed)
    random_idx = np.random.permutation(len(inputs))
    inputs = inputs[random_idx]
    outputs = outputs[random_idx]

    ing_idx_to_hier_map = gen_ing_idx_to_hier_map(ings[:num_ingredients], adulterants, pca_file=pca_file)
    input_size = ing_idx_to_hier_map.values()[0].shape[0]
    print "Input size:", input_size
    inp_exist_indices = np.array(
        [i for i in range(len(inputs)) if inputs[i] in ing_idx_to_hier_map])
    inputs = inputs[inp_exist_indices]
    outputs = outputs[inp_exist_indices]

    #inputs_tr, outputs_tr, inputs_te, outputs_te = split_data(
    #    inputs, outputs, test_size=1/3.)
    #train_indices, test_indices = train_test_split(
    #            range(num_ingredients), test_size=1/3., random_state=42)
    #train_indices, dev_indices, test_indices = split_data_by_wiki(
    #            ings, args.seed)
    inputs_tr, outputs_tr, inputs_te, outputs_te = split_data_by_indices(
        inputs, outputs, train_indices, test_indices)
    
    ing_cat_pair_map = gen_ing_cat_pair_map(inputs, outputs)
    adulterant_cat_pair_map = gen_adulterant_cat_pair_map()

    #np.random.shuffle(inputs_tr) # benchmark/sanity check
    classifier, predict_model = run_nn(inputs_tr, outputs_tr, 
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

    """
    # Only use valid hierarchies.
    valid_ing_indices = [i for i in range(num_ingredients) if i in ing_idx_to_hier_map]
    valid_ing_reps = np.array([ing_idx_to_hier_map[i] for i in valid_ing_indices])
    valid_ings = [ings[i] for i in valid_ing_indices]
    valid_ing_results = predict_model(valid_ing_reps)
    #test_model(valid_ing_results, valid_ings, idx_to_cat, 3)
    """
    valid_ing_indices = range(len(ings))
    valid_ing_reps = np.array([ing_idx_to_hier_map.get(i, np.zeros(input_size).astype('float32')) for i in valid_ing_indices])
    valid_ing_results = predict_model(valid_ing_reps)
    
    #evaluate_map(valid_ing_indices, valid_ing_results, ing_cat_pair_map)

    found_ings = np.load('../rasff/found_chems.npy')
    new_ings_reps = np.load('../rasff/found_chems_reps.npy').astype('float32')
    if pca_file:
        with open(pca_file, 'r') as f:
            pca = pickle.load(f)
        new_ings_reps = pca.transform(new_ings_reps)
    new_ings_results = predict_model(new_ings_reps)
    #test_model(new_ings_results, found_ings, idx_to_cat, 3)
    #evaluate_map(np.arange(len(found_ings)), new_ings_results, adulterant_cat_pair_map)


    ###
    if type(valid_ing_indices) == list:
        valid_ing_indices = np.array(valid_ing_indices)
    print "======= Training evaluation ========"
    evaluate(valid_ing_indices, valid_ing_results, ing_cat_pair_map, set(train_indices))
    print "======= Validation evaluation ========"
    evaluate(valid_ing_indices, valid_ing_results, ing_cat_pair_map, set(dev_indices))
    print "======= Test evaluation ========"
    evaluate(valid_ing_indices, valid_ing_results, ing_cat_pair_map, set(test_indices))
    print "======= Adulteration evaluation ========"
    evaluate(np.arange(len(found_ings)), new_ings_results, adulterant_cat_pair_map)
    ###

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

