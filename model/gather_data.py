import itertools
import sys

import numpy as np
import pandas as pd

sys.path.append('../foodessentials')
import foodessentials


### CATEGORIES ###

def gen_input_outputs_cat(df, df_i, num_ingredients, output_cat):
    """output_cat should be: 'aisle', 'shelf' or 'food_category'."""

    counts = df_i['ingredient'].value_counts()
    counts = counts[:num_ingredients]
    onehot_ing = gen_onehot_vectors(counts.index.values, num_ingredients)

    inputs = []
    categories = []
    categories_clean = df[output_cat].str.lower().values
    cat_to_idx = {c : i for i, c in enumerate(np.unique(categories_clean))}
    ingredients_clean = df['ingredients_clean'].values
    for idx in range(len(df)):
        l = convert_ingredient_list(ingredients_clean[idx], onehot_ing)
        s = np.sum(np.array(l), axis=0)
        if type(s)==np.float64:
            assert(s==0)
            continue
        assert(len(s)==num_ingredients)
        inputs.append(s)
        categories.append(cat_to_idx[categories_clean[idx]])
    
    assert(len(inputs)==len(categories))
    outputs = categories
    idx_to_cat = {i:c for c,i in cat_to_idx.iteritems()}
    return np.array(inputs), np.array(outputs), idx_to_cat



### EMBEDDINGS ###

def get_coocc_ranks():
    coocc = np.load('cooccurance.npy')
    coocc_sym = coocc*coocc.T
    ranks_all = []
    for i in range(coocc.shape[0]):
        rank = np.searchsorted(coocc_sym[i], coocc_sym[i], sorter=np.argsort(coocc_sym[i]))
        ranks_all.append(rank)
    return np.array(ranks_all)

def get_cooccurance_prob(ing1, ing2, df, df_i):
    prods1 = foodessentials.find_products_by_ing(ing1, df=df, df_i=df_i)
    counts1 = prods1.ingredients_clean.map(lambda x: ing2 in x).sum()
    prob1 = counts1*1.0/len(prods1)
    prods2 = foodessentials.find_products_by_ing(ing2, df=df, df_i=df_i)
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
    return matrix

def gen_onehot_vectors(ings, num_ingredients):
    """Returns a dictionary mapping the name to one-hot vector representation."""
    d = {}
    for i in range(num_ingredients): # Take first num_ingredients ingredients
        vector = np.zeros(num_ingredients)
        vector[i] = 1
        d[ings[i]] = vector
    return d

def convert_ingredient_list(ing_list, d):
    return [d[i] for i in ing_list if i in d]

def gen_input_outputs(df, df_i, num_ingredients):
    counts = df_i['ingredient'].value_counts()
    counts = counts[:num_ingredients]
    d = gen_onehot_vectors(counts.index.values, num_ingredients)

    vectors = []
    ingredients_clean = df['ingredients_clean'].values
    for idx in range(len(df)):
        l = convert_ingredient_list(ingredients_clean[idx], d)
        vectors.append(l)
    print len(vectors)
    output_lens = np.array([len(v) for v in vectors])
    output_lens_new = []
    inputs = []
    outputs = []
    counter = 0
    for v in vectors:
        l = len(v)
        if l < 2:
            counter += 1
            continue
        if counter % 1000 == 1:
            print counter
        for i in range(l):
            if i >= 2:
                break # only do top x ingredients
            output_lens_new.append(l-1)
            assert(v[0].sum()==1)
            inputs.append(v[0]) # simple for now
            outputs_ = []
            for j in range(1, output_lens.max()):
                if j < l:
                    assert(v[j].sum()==1)
                    outputs_.append(v[j])
                else:
                    outputs_.append(np.zeros(num_ingredients))
            outputs.append(outputs_)
            v = np.roll(v, -1, axis=0)
        counter += 1

    assert(len(inputs)==len(outputs)==len(output_lens_new))
    return np.array(inputs), np.array(outputs), np.array(output_lens_new)

def import_data():
    df = pd.read_hdf('../foodessentials/products.h5', 'products')
    df_i = pd.read_hdf('../foodessentials/ingredients.h5', 'ingredients')
    foodessentials.add_columns(df, df_i)
    return df, df_i

    