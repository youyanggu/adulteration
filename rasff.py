import csv
import numpy as np
import pandas as pd

from parse import *
from utils import *

###################
# Load CSV file
###################

rasff_dir = '../rasff_data'
fnames = ['co', 'fa', 'ic']

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

    df = pd.DataFrame.from_records(np.array(rows), exclude='#', columns=header)
    food = df[df['type'].str.lower()=='food']
    food['type'] = 'food'
    food['subject'] = food['subject'].str.replace('\xe5\xb5', 'u')
    food.reset_index(drop=True, inplace=True)
    food.to_csv(clean_fname)

for fname in fnames:
    load_csv(fname)

##################
# Analysis
##################

def get_df(fname):
    clean_fname = '{}/rasff_{}_clean.csv'.format(rasff_dir, fname)
    df = pd.DataFrame.from_csv(clean_fname)
    subject = df.subject.values

    chemical, amount, product, origin = parse_subject(subject)
    df['chemical'] = chemical
    df['amount'] = amount
    df['product'] = product
    df['origin'] = origin
    return df

categories = np.load('{}/categories.npy'.format(rasff_dir))

# combine data from different hazards
dfs = []
for fname in fnames:
    df = get_df(fname)
    df['hazard'] = fname
    dfs.append(df)
df = pd.concat(dfs)
df.reset_index(drop=True, inplace=True)

products_ = np.unique(df.groupby('product').filter(lambda x: len(x) > 1)['product'])
chemicals_ = np.unique(df.groupby('chemical').filter(lambda x: len(x) > 1)['chemical'])

# create random pairs
pairs = generate_random_pairs(products_, chemicals_, 250)
pd.DataFrame.from_records(pairs, columns=['product', 'chemical']).to_csv('{}/pairs_random.csv'.format(rasff_dir), index=False)

# create category/chemical pairs
d = {}
for i in range(len(df)):
    key = (df.category.ix[i], df.chemical.ix[i])
    if key not in d:
        d[key] = 1
    else:
        d[key] += 1

d_filtered = {k : d[k] for k in d if d[k]>1} # all category/chemical pairs that appear more than once
pd.DataFrame.from_records(d_filtered.keys(), columns=['product', 'chemical']).to_csv('{}/pairs.csv'.format(rasff_dir), index=False)

