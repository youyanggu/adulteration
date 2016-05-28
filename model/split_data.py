import numpy as np
from sklearn.cross_validation import train_test_split
import sys

sys.path.append('../wikipedia')
from wikipedia import get_ings_wiki_links

def split_into_3(ings, seed):
    """Assumes 60/20/20 split."""
    tr, rest = train_test_split(ings, random_state=seed, test_size=0.4)
    va, te = train_test_split(rest, random_state=seed, test_size=0.5)
    return tr, va, te

def split_data_by_wiki(ings, seed=42):
    ings_wiki_links = get_ings_wiki_links()
    unique_wikis = sorted(set([i[0] for i in ings_wiki_links.values() if i[0]]))
    tr, va, te = split_into_3(unique_wikis, seed)
    tr_indices, va_indices, te_indices = [], [], []
    for i,v in enumerate(ings):
        if not v:
            continue
        wiki = ings_wiki_links[v][0]
        if wiki in tr:
            tr_indices.append(i)
        elif wiki in va:
            va_indices.append(i)
        elif wiki in te:
            te_indices.append(i)
        else:
            if wiki:
                assert False, "Ingredient not in a set: " + wiki
    return tr_indices, va_indices, te_indices

def split_data(inputs, outputs, test_size=1/3.):
    unique_inputs = np.unique(inputs)
    ing_to_cats = {}
    for inp, out in zip(inputs, outputs):
        if inp not in ing_to_cats:
            ing_to_cats[inp] = set([out])
        else:
            ing_to_cats[inp].add(out)

    ing_to_cats_tr, ing_to_cats_te = {}, {}
    for inp in unique_inputs:
        cats = sorted(ing_to_cats[inp])
        cats_tr, cats_te = train_test_split(cats, test_size=test_size, random_state=42)
        ing_to_cats_tr[inp] = set(cats_tr)
        ing_to_cats_te[inp] = set(cats_te)

    inputs_tr, outputs_tr, inputs_te, outputs_te = [],[],[],[]
    for inp, out in zip(inputs, outputs):
        if out in ing_to_cats_tr[inp]:
            inputs_tr.append(inp)
            outputs_tr.append(out)
        else:
            inputs_te.append(inp)
            outputs_te.append(out)

    return (np.array(inputs_tr), np.array(outputs_tr), 
        np.array(inputs_te), np.array(outputs_te))

def split_data_by_indices(inputs, outputs, train_indices, test_indices):
    train_idx_set = set(train_indices)
    test_idx_set = set(test_indices)

    inputs_tr = [i for i in inputs if i in train_idx_set]
    outputs_tr = [outputs[i] for i in range(len(inputs)) if inputs[i] in train_idx_set]
    inputs_te = [i for i in inputs if i in test_idx_set]
    outputs_te = [outputs[i] for i in range(len(inputs)) if inputs[i] in test_idx_set]
    return (np.array(inputs_tr), np.array(outputs_tr), 
        np.array(inputs_te), np.array(outputs_te))

    