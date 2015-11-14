import sys

import numpy as np
import pandas as pd

sys.path.append('../foodessentials')
import foodessentials

def gen_onehot_vectors(ings, num_ingredients):
    d = {}
    ing_names = ings.index.values
    for i in range(num_ingredients): # Take first 1500 ingredients
        vector = np.zeros(num_ingredients)
        vector[i] = 1
        d[ing_names[i]] = vector
    return d

def convert_ingredient_list(ing_list, d):
    return [d[i] for i in ing_list if i in d]

def gen_input_outputs(df, df_i, num_ingredients):
    counts = df_i['ingredient'].value_counts()
    counts = counts[:num_ingredients]
    foodessentials.add_columns(df, df_i)
    d = gen_onehot_vectors(counts, num_ingredients)

    vectors = []
    for idx, prod in df.iterrows():
        l = convert_ingredient_list(prod['ingredients_clean'], d)
        vectors.append(l)
    
    output_lens = np.array([len(v) for v in vectors])
    inputs = []
    outputs = []
    for v in vectors:
        if len(v) < 2:
            continue
        inputs.append(v[0]) # simple for now
        outputs_ = []
        for j in range(1, output_lens.max()):
            if j < len(v):
                outputs_.append(v[j])
            else:
                outputs_.append(np.zeros(num_ingredients))
        outputs.append(outputs_)

    return np.array(inputs), np.array(outputs), output_lens

def import_data():
    df = pd.read_hdf('../foodessentials/products.h5', 'products')
    df_i = pd.read_hdf('../foodessentials/ingredients.h5', 'ingredients')
    return df, df_i

    
