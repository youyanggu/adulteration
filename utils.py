import csv
import itertools
import numpy as np

def validate(Y):
    return np.isfinite(Y).astype(np.float64)

def has_and(s):
    return s.startswith('and') or ' and ' in s

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
    return np.load('../rasff_data/products.npy'), np.load('../rasff_data/chemicals.npy')

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
        reader = csv.reader(csv_file)
        for line in reader:
            prod, chem, is_neg = line
            if is_neg == '':
                continue
            x_idx, y_idx = np.where(products==prod)[0][0], np.where(chemicals==chem)[0][0]
            if Y[x_idx, y_idx] != 0:
                continue
            pairs.append((x_idx, y_idx))
    for x,y in pairs:
        Y[x,y] = -1
    return Y


def get_pairs(Y, rows, columns, products, chemicals):
    x_, y_ = np.where(Y!=0)
    pairs = np.array([(products[rows[i]], chemicals[columns[j]]) for i,j in zip(x_, y_)])
    return pairs

def clean_products(df):
    df['product'] = df['product'].str.lower()
    products = np.unique(df['product'].values)
    replace = []
    d = {i:True for i in products}
    for p in products:
        if p+'s' in d:
            replace.append(p)
    for r in replace:
        df['product'] = df['product'].str.replace(r+'s', r)
    return df

def filter_p_c(df, n=1, values=True):
    products = np.unique(df['product'].values)
    chemicals = np.unique(df['chemical'].values)
    
    g_p = df.groupby('product').size()
    g_c = df.groupby('chemical').size()

    filtered_p = g_p[g_p>=n]
    filtered_c = g_c[g_c>=n]

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





