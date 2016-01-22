import csv
import numpy as np
import pandas as pd

from parse import *
from utils import *

###################
# Load CSV file
###################

rasff_dir = '../../rasff_data'
fnames = ['co', 'fa', 'ic', 'hm', 'cc']

def load_csv(fname):
    clean_fname = '{}/rasff_{}_clean.csv'.format(rasff_dir, fname)

    f = open('{}/rasff_{}.csv'.format(rasff_dir, fname), 'rU')
    reader = csv.reader(f)

    header = next(reader)
    header = ['#', 'classification', 'date_case', 'date_last_change', 'reference', 
              'country', 'type', 'category', 'subject']

    count = 1
    first = True
    rows = []
    curRow = []
    for line in reader:
        if first:
            curRow.extend(line)
        else:
            curRow.extend(line[1:2])
        if not first:
            rows.append(curRow)
            curRow = []
        first = not first
        count += 1

    df = pd.DataFrame.from_records(np.array(rows), exclude='#', columns=header)
    food = df[df['type'].str.lower()=='food']
    food['type'] = 'food'
    food['subject'] = food['subject'].str.replace('\xe5\xb5', 'u')
    food.reset_index(drop=True, inplace=True)
    food.to_csv(clean_fname)

def run_load_csv():
    for fname in fnames:
        load_csv(fname)

##################
# Analysis
##################

def clean_df(df):
    subject = df.subject.values
    chemical, amount, product, origin = parse_subject(subject)
    df['chemical'] = chemical
    df['amount'] = amount
    df['product'] = product
    df['origin'] = origin
    df.reset_index(drop=True, inplace=True)
    df = clean_products(df)
    df = clean_chemicals(df)
    return df

def load_df():
    # combine data from different hazards
    dfs = []
    for fname in fnames:
        clean_fname = '{}/rasff_{}_clean.csv'.format(rasff_dir, fname)
        df = pd.DataFrame.from_csv(clean_fname)
        df['hazard'] = fname
        dfs.append(df)
    df = pd.concat(dfs)
    df = clean_df(df)
    return df

def main():
    df = load_df()
    #categories = np.load('{}/categories.npy'.format(rasff_dir))

    products, chemicals = filter_p_c(df)

    # create matrix
    Y = gen_matrix(df)
    Y_, rows_, columns_ = filter_matrix(Y, 2)
    #Y_, rows_, columns_ = filter_matrix(Y, 1)
    products_final = products[rows_]
    chemicals_final = chemicals[columns_]
    np.save('{}/products.npy'.format(rasff_dir), products_final)
    np.save('{}/chemicals.npy'.format(rasff_dir), chemicals_final)
    #with open('{}/products.csv'.format(rasff_dir), 'wb') as f:
    #    csv.writer(f).writerows([[i] for i in products_final])
    #with open('{}/chemicals.csv'.format(rasff_dir), 'wb') as f:
    #    csv.writer(f).writerows([[i] for i in chemicals_final])
    np.save('{}/matrix.npy'.format(rasff_dir), np.array([Y_, rows_, columns_]))

    # create actual pairs
    pairs = get_pairs(Y_, rows_, columns_, products, chemicals)
    pd.DataFrame.from_records(pairs, columns=['product', 'chemical']).to_csv('{}/pairs.csv'.format(rasff_dir), index=False)
        
    # create random pairs
    random_pairs = gen_random_pairs(products_final, chemicals_final, 600)
    pd.DataFrame.from_records(random_pairs, columns=['product', 'chemical']).to_csv('{}/pairs_random.csv'.format(rasff_dir), index=False)


    """
    # create product/chemical pairs
    d = {}
    for i in range(len(df)):
        key = (df.category.ix[i], df.chemical.ix[i])
        if key not in d:
            d[key] = 1
        else:
            d[key] += 1
    d_filtered = {k : d[k] for k in d if d[k]>1} # all category/chemical pairs that appear more than once

    pd.DataFrame.from_records(d_filtered.keys(), columns=['product', 'chemical']).to_csv('{}/pairs.csv'.format(rasff_dir), index=False)
    """
