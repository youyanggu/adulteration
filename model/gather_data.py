import itertools
import sys

import numpy as np
import pandas as pd
import scipy

sys.path.append('../foodessentials')
import ing_utils

### VALID ###

def gen_random_inp(l, num_ones):
    sample = np.random.choice(l, num_ones, replace=False)
    inp = np.zeros(l)
    inp[sample] = 1
    return inp


def get_ing_names(counts, arr):
    return counts.index.values[np.where(arr==1)[0]]

def get_combos(inputs, outputs, counts, limit=50):
    output = []
    for idx, inp in enumerate(inputs):
        output.append(regr.predict(inp), outputs[idx], get_ing_names(counts, inp))
    return output

def gen_input_outputs_invalid(length, num_ingredients, ings_per_prod):
    """Generate invalid set of ingredients from random sampling."""
    inputs = [gen_random_inp(num_ingredients, ings_per_prod) for i in range(length)]
    outputs = np.zeros(length)
    assert(len(inputs)==len(outputs))
    return np.array(inputs), outputs


def gen_input_outputs_valid(df, df_i, num_ingredients, ings_per_prod):
    """Get a set number of ingredients and set as valid combination."""

    counts = df_i['ingredient'].value_counts()
    counts = counts[:num_ingredients]
    onehot_ing = gen_onehot_vectors(counts.index.values, num_ingredients)

    inputs = []
    ingredients_clean = df['ingredients_clean'].values
    for idx in range(len(df)):
        l = convert_ingredient_list(ingredients_clean[idx], onehot_ing)
        if len(l) < ings_per_prod:
            continue
        s = np.sum(np.array(l[:ings_per_prod]), axis=0)
        assert(len(s)==num_ingredients)
        assert(s.sum()==ings_per_prod)
        inputs.append(s)
    outputs = np.ones(len(inputs))
    assert(len(inputs)==len(outputs))
    return np.array(inputs), outputs



### CATEGORIES ###

def input_from_embeddings(one_hot_inputs, embeddings):
    assert(one_hot_inputs.shape[1]==embeddings.shape[0])
    new_inputs = []
    for inp in one_hot_inputs:
        ing_indices = np.where(inp==1)[0]
        new_inp = np.sum(embeddings[ing_indices], axis=0)/len(ing_indices)
        new_inputs.append(new_inp)
    return np.array(new_inputs)

def get_ingredients_from_vector(counts, vector):
    return counts.index.values[np.where(vector==1)[0]]

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

def get_index(vec):
    return np.where(vec==1)[0][0]

def gen_onehot_vectors(ings, num_ingredients):
    """Returns a dictionary mapping the name to one-hot vector representation."""
    d = {}
    for i in range(num_ingredients): # Take first num_ingredients ingredients
        vector = np.zeros(num_ingredients)
        vector[i] = 1
        d[ings[i]] = vector
    return d

def convert_ingredient_list(ing_list, d):
    """Return one-hot representation of a list of ingredients."""
    return [d[i] for i in ing_list if i in d]

def gen_input_outputs(ingredients_clean, counts, num_ingredients, 
                      max_output_len=None, max_rotations=None, random_rotate=False):
    counts = counts[:num_ingredients]
    d = gen_onehot_vectors(counts.index.values, num_ingredients)

    vectors = []
    for idx in range(len(ingredients_clean)):
        l = convert_ingredient_list(ingredients_clean[idx], d)
        vectors.append(l)
    print len(vectors)
    output_lens = np.array([len(v) for v in vectors])
    output_lens_new = []
    inputs = []
    outputs = []
    counter = 0
    if max_output_len is None:
        max_output_len = output_lens.max()-1
    for v in vectors:
        l = len(v)
        if l < 2:
            counter += 1
            continue
        #if counter % 1000 == 0:
        #    print counter
        for i in range(l):
            if max_rotations and i >= max_rotations:
                break # only do top x ingredients
            if random_rotate:
                v = np.roll(v, np.random.randint(l), axis=0)
            elif i>0:
                v = np.roll(v, -1, axis=0)
            output_lens_new.append(min(l-1, max_output_len))
            assert(v[0].sum()==1)
            inputs.append(get_index(v[0]))
            v_sum = np.sum(v[1:max_output_len+1], axis=0)
            #if max_output_len < l-1:
            #    # Choose a subset of the ingredients.
            #    indices = np.random.choice(np.arange(1,l), max_output_len, replace=False)
            #    v_sum = np.sum(v[indices], axis=0)
            #else:
            #    v_sum = np.sum(v[1:], axis=0)
            
            #v_sum = (v_sum>0).astype(int) # if duplicates
            if num_ingredients>200:
                outputs_ = scipy.sparse.csr_matrix(v_sum)
            else:
                outputs_ = v_sum
            outputs.append(outputs_)
        counter += 1
    assert(len(inputs)==len(outputs)==len(output_lens_new))
    if scipy.sparse.issparse(outputs[0]):
        outputs = scipy.sparse.vstack(outputs)
        #outputs = scipy.sparse.csr_matrix(outputs) # need to cast for old scipy
    else:
        outputs = np.array(outputs)
    return np.array(inputs), outputs, np.array(output_lens_new)


### GENERAL ###

def import_data():
    def add_columns(df, df_i):
        df['ingredients_clean'] = ing_utils.get_ings_by_product(df, df_i)
        df['num_ingredients'] =  df['ingredients_clean'].apply(len)
        df['hier'] = df[['aisle', 'shelf', 'food_category']].values.tolist()
    df = pd.read_hdf('../foodessentials/products.h5', 'products')
    df_i = pd.read_hdf('../foodessentials/ingredients.h5', 'ingredients')
    add_columns(df, df_i)
    return df, df_i

    
