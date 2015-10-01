import itertools
import numpy as np

def generate_random_pairs(products, chemicals, num_pairs):
    pairs = []
    n, m = len(products), len(chemicals)
    x = [i for i in itertools.product(range(n), range(m))]
    pairs_idx = np.random.choice(range(len(x)), num_pairs, replace=False)
    for idx in pairs_idx:
        p, c = products[x[idx][0]], chemicals[x[idx][1]]
        if 'and' in c:
            continue
        pairs.append((p,c))
    return pairs

