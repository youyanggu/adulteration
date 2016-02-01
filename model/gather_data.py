import itertools
import sys

import numpy as np
import pandas as pd
import scipy

sys.path.append('../foodessentials')
import ing_utils

### VALID ###

def flip_arr(arr, num_flips=1):
    assert(arr.sum() >= num_flips)
    if num_flips == 0:
        return arr
    one_indices = np.where(arr==1)[0]
    new_one_indices = []
    while len(new_one_indices)<num_flips:
        idx = np.random.choice(len(arr))
        if idx not in one_indices and idx not in new_one_indices:
            new_one_indices.append(idx)

    z = np.zeros(len(arr))
    all_one_indices = np.hstack((
            np.random.choice(one_indices, len(one_indices)-num_flips, replace=False),
            np.array(new_one_indices)))
    z[all_one_indices] = 1
    assert(arr.sum()==z.sum())
    return z

def add_arr(arr, num_adds=1):
    if num_adds == 0:
        return arr
    indices = np.where(arr==0)[0]
    new_one_indices = np.random.choice(indices, num_adds, replace=False)
    arr[new_one_indices] = 1
    return arr

def remove_arr(arr, num_removes=1):
    if num_removes == 0:
        return arr
    indices = np.where(arr!=0)[0]
    num_removes = min(num_removes, len(indices))
    new_one_indices = np.random.choice(indices, num_removes, replace=False)
    arr[new_one_indices] = 0
    return arr

def flip_inputs(inputs, num_flips=1):
    inputs = np.copy(inputs)
    return np.array([flip_arr(i, num_flips) for i in inputs]).astype('int32')

def add_inputs(inputs, num_adds=1):
    inputs = np.copy(inputs)
    return np.array([add_arr(i, num_adds) for i in inputs]).astype('int32')

def remove_inputs(inputs, num_removes=1):
    inputs = np.copy(inputs)
    return np.array([remove_arr(i, num_removes) for i in inputs]).astype('int32')

def change_inputs(inputs, num_changes):
    if num_changes>0:
        return add_inputs(inputs, num_changes)
    else:
        return remove_inputs(inputs, -num_changes)

def gen_random_inp(population, l, num_ones):
    if len(population) == l:
        sample = np.random.choice(population, num_ones, replace=False)
        inp = np.zeros(l)
        inp[sample] = 1
        return inp
    else:
        # weighted
        ones = []
        len_pop = len(population)
        while len(ones) < num_ones:
            sample = population[np.random.randint(len_pop)]
            if sample not in ones:
                ones.append(sample)
        inp = np.zeros(l)
        inp[np.array(ones)] = 1
        return inp

def get_ing_names(counts, arr):
    return counts.index.values[np.where(arr==1)[0]]

"""
def get_combos(inputs, outputs, counts, limit=50):
    output = []
    for idx, inp in enumerate(inputs):
        output.append(regr.predict(inp), outputs[idx], get_ing_names(counts, inp))
    return output
"""

def gen_input_outputs_invalid(inputs_v, frac_samples, num_ingredients, 
    ings_per_prod, weighted=True):
    """
    Generate invalid set of ingredients from random sampling.
    Generates ingredients in the same frequents as the valid inputs.
    """
    num_samples = int(frac_samples * len(inputs_v))
    if weighted:
        counts = np.sum(inputs_v, axis=0, dtype=int)
        population = []
        for i,v in enumerate(counts):
            population.extend([i]*v)
        population = np.array(population)
    else:
        population = np.arange(num_ingredients)

    inputs = [gen_random_inp(
        population, num_ingredients, ings_per_prod) for i in range(num_samples)]
    outputs = np.zeros(num_samples)
    assert(len(inputs)==len(outputs))
    return np.array(inputs).astype('int32'), outputs.astype('int32')


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
            # Ignore product if it contains too few ingredients.
            continue
        s = np.sum(np.array(l[:ings_per_prod]), axis=0)
        assert(len(s)==num_ingredients)
        #assert(s.sum()==ings_per_prod)
        inputs.append(s)
    outputs = np.ones(len(inputs))
    assert(len(inputs)==len(outputs))
    return np.array(inputs).astype('int32'), outputs.astype('int32')



### CATEGORIES ###

def break_down_inputs(inputs, outputs):
    """Give each ingredient in the input their own input. Also convert 
    input to indices only."""
    assert len(inputs)==len(outputs)
    final_inputs = []
    final_outputs = []
    l = len(inputs)
    for i in range(l):
        ones = np.where(inputs[i]==1)[0]
        for one in ones:
            final_inputs.append(one)
            final_outputs.append(outputs[i])
    return (np.array(final_inputs).astype('int32'), 
        np.array(final_outputs).astype('int32')
        )


def get_ing_cat_frequencies(df, category, cat_to_idx, alpha=0):
    ings = df['ingredients_clean'].values
    cats = df[category].str.lower().values
    num_categories = len(cat_to_idx)
    ing_to_cat_freq = {}
    cat_to_freq = [0]*num_categories
    for i in range(len(df)):
        cur_cat = cat_to_idx[cats[i]]
        cat_to_freq[cur_cat] += 1
        for ing in ings[i]:
            if ing in ing_to_cat_freq:
                ing_to_cat_freq[ing][cur_cat] += 1
            else:
                arr = [0]*num_categories
                arr[cur_cat] = 1
                ing_to_cat_freq[ing] = arr

    # Normalize + add smoothing
    cat_to_freq = np.array(cat_to_freq).astype(float)
    for k,v in ing_to_cat_freq.iteritems():
        ing_to_cat_freq[k] = np.nan_to_num(np.array(v) / cat_to_freq)

    for k,v in ing_to_cat_freq.iteritems():
        ing_to_cat_freq[k] = (v+alpha) / (np.sum(v)+alpha*num_categories)
    assert(np.isclose(np.sum(ing_to_cat_freq.values()), len(ing_to_cat_freq)))

    return ing_to_cat_freq


def get_upper_cat(df, lower_cat_name, upper_cat_name):
    lower_cat = df[lower_cat_name].str.lower().values
    upper_cat = df[upper_cat_name].str.lower().values

    lower_cat_to_idx = {c : i for i, c in enumerate(np.unique(lower_cat))}
    upper_cat_to_idx = {c : i for i, c in enumerate(np.unique(upper_cat))}

    num_categories = len(np.unique(lower_cat))
    lower_to_upper_cat = {}
    for i in range(len(df)):
        if len(lower_to_upper_cat) >= num_categories:
            break
        if lower_cat_to_idx[lower_cat[i]] in lower_to_upper_cat:
            continue
        else:
            lower_to_upper_cat[lower_cat_to_idx[lower_cat[i]]] = (
                upper_cat_to_idx[upper_cat[i]])
    return lower_to_upper_cat

def input_from_embeddings(one_hot_inputs, embeddings, normalize):
    """Map one-hot inputs to embeddings space."""
    assert(one_hot_inputs.shape[1]==embeddings.shape[0])
    new_inputs = []
    for inp in one_hot_inputs:
        ing_indices = np.where(inp==1)[0]
        new_inp = np.sum(embeddings[ing_indices], axis=0)
        if normalize:
            new_inp = new_inp / len(ing_indices)
        new_inputs.append(new_inp)
    return np.array(new_inputs)

def get_ingredients_from_vector(counts, vector):
    return counts.index.values[np.where(vector==1)[0]]

def gen_input_outputs_cat(df, counts, num_ingredients, output_cat, ings_per_prod=None):
    """output_cat should be: 'aisle', 'shelf' or 'food_category'."""

    counts = counts[:num_ingredients]
    onehot_ing = gen_onehot_vectors(counts.index.values, num_ingredients)

    inputs = []
    categories = []
    categories_clean = df[output_cat].str.lower().values
    cat_to_idx = {c : i for i, c in enumerate(np.unique(categories_clean))}
    ingredients_clean = df['ingredients_clean'].values
    for idx in range(len(df)):
        l = convert_ingredient_list(ingredients_clean[idx], onehot_ing)
        if ings_per_prod:
            s = np.sum(np.array(l[:ings_per_prod]), axis=0)
        else:
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
    return (np.array(inputs).astype('int32'), 
        np.array(outputs).astype('int32'), 
        idx_to_cat
        )


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
            if num_ingredients>0:#200:
                outputs_ = scipy.sparse.csr_matrix(v_sum)
            else:
                outputs_ = v_sum
            outputs.append(outputs_)
        counter += 1
    assert(len(inputs)==len(outputs)==len(output_lens_new))
    if scipy.sparse.issparse(outputs[0]):
        outputs = scipy.sparse.vstack(outputs)
        outputs = scipy.sparse.csr_matrix(outputs) # need to cast for old scipy
    else:
        outputs = np.array(outputs)
    return (np.array(inputs).astype('int32'), 
        outputs.astype('int32'), 
        np.array(output_lens_new).astype('int32')
        )


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

    
