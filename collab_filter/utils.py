import csv
import itertools
import numpy as np

rasff_dir = '../../rasff_data'

def validate(Y):
    return np.isfinite(Y).astype(np.float64)

def has_and(s):
    return s.startswith('and') or ' and ' in s

def gen_random_matrix(shape, num):
    # Generate mxn matrix with 0.5*num +1's and 0.5*num -1's. Else 0.
    m,n = shape
    matrix = np.zeros((m,n))
    indices = [(x,y) for x,y in itertools.product(range(m), range(n))]
    chosen = np.random.choice(range(len(indices)), num, replace=False)
    for count, idx in enumerate(chosen):
        i,j = indices[idx]
        matrix[i,j] = 1 if count < num/2 else -1
    return matrix

def gen_random_pairs(products, chemicals, num_pairs):
    pairs = []
    n, m = len(products), len(chemicals)
    x = [i for i in itertools.product(range(n), range(m))]
    pairs_idx = np.random.choice(range(len(x)), num_pairs, replace=False)
    for idx in pairs_idx:
        p, c = products[x[idx][0]], chemicals[x[idx][1]]
        if has_and(c) or has_and(p):
            continue
        pairs.append((p,c))
    return pairs

def get_prod_chem():
    return np.load('{}/products.npy'.format(rasff_dir)), np.load('{}/chemicals.npy'.format(rasff_dir))

def gen_neg(Y, csv_file=None, num=200):
    products, chemicals = get_prod_chem()
    if csv_file is None:
        # Randomly generate some negative pairs
        all_pairs = np.array([(i,j) for i,j in zip(*np.where(Y==0))])
        indices = np.random.choice(range(len(all_pairs)), num, replace=False)
        pairs = all_pairs[indices]
    else:
        pairs = []
        f = open(csv_file, 'rU')
        reader = csv.reader(f)
        for line in reader:
            prod, chem, is_neg = line
            if is_neg != 'X':
                continue
            try:
                x_idx = np.where(products==prod)[0][0]
                y_idx = np.where(chemicals==chem)[0][0]
            except IndexError:
                print "Product or chemical not found for pair: {}, {}".format(prod, chem)
                continue
            if Y[x_idx, y_idx] != 0:
                print "Marked as False, but actually True: {}, {}".format(prod, chem)
                continue
            pairs.append((x_idx, y_idx))
        f.close()
        print "Number of negative cases:", len(pairs)
    for x,y in pairs:
        Y[x,y] = -1
    return Y


def get_pairs(Y, rows, columns, products, chemicals):
    x_, y_ = np.where(Y!=0)
    pairs = np.array([(products[rows[i]], chemicals[columns[j]]) for i,j in zip(x_, y_)])
    return pairs


def filter_p_c(df, n=1, values=True):
    #products = np.unique(df['product'].values)
    #chemicals = np.unique(df['chemical'].values)
    
    g_p = df.groupby('product').size()
    g_c = df.groupby('chemical').size()

    filtered_p = g_p[g_p>=n]
    filtered_c = g_c[g_c>=n]

    if filtered_p.index[0] == '':
        filtered_p = filtered_p[1:]
    if filtered_c.index[0] == '':
        filtered_c = filtered_c[1:]

    if values:
        return filtered_p.index.values, filtered_c.index.values
    else:
        return filtered_p, filtered_c


def filter_row(matrix, rows, columns, threshold):
    new_rows = []
    for i, row in enumerate(matrix):
        if i not in rows:
            continue
        if row[columns].sum() >= threshold:
            new_rows.append(i)
    return np.array(new_rows)


def filter_matrix(matrix, threshold):
    rows = np.arange(matrix.shape[0])
    columns = np.arange(matrix.shape[1])
    while True:
        n, m = len(rows), len(columns)
        rows = filter_row(matrix, rows, columns, threshold)
        if len(rows)==0:
            print "Error: no matrix with threshold exists."
            return None
        columns = filter_row(matrix.T, columns, rows, threshold)
        if len(rows)==n and len(columns)==m:
            break
        if len(rows)==0 or len(columns)==0:
            print "Error: no matrix with threshold exists."
            return None
    return matrix[rows][:, columns], rows, columns

def gen_matrix(df):
    """Converts df to matrix, where the rows are the unique products
    and the columns are the unique chemicals."""
    products_all = df['product'].values
    chemicals_all = df['chemical'].values
    assert(len(df)==len(products_all)==len(chemicals_all))
    products, chemicals = filter_p_c(df)
    n,m = len(products), len(chemicals)
    matrix = np.empty((n,m))
    matrix[:] = np.nan

    products_ix = {products[i]:i for i in range(len(products))}
    chemicals_ix = {chemicals[i]:i for i in range(len(chemicals))}
    for p, c in zip(products_all, chemicals_all):
        if p in products_ix and c in chemicals_ix:
            matrix[products_ix[p], chemicals_ix[c]] = 1
    Y = np.nan_to_num(matrix)
    return Y





