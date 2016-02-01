import csv
import glob

import numpy as np
import pandas as pd

REFUSAL_FOLDER =  '../fda_refusals/'

def parse_code_prefix():
    fname = REFUSAL_FOLDER + 'code_prefix.csv'
    prefix_df = pd.read_csv(fname)
    prefix_to_cat = prefix_df.set_index('code_prefix')['category'].to_dict()
    return prefix_to_cat

def parse_charges():
    fname = REFUSAL_FOLDER + 'ACT_SECTION_CHARGES.CSV'
    charges_df = pd.read_csv(fname)
    charges_df['SCTN_NAME'] = charges_df['SCTN_NAME'].str.lower()
    charges_df['is_adulteration'] = charges_df['SCTN_NAME'].str.contains("adulteration") | charges_df['SCTN_NAME'].str.contains("402")
    assert len(charges_df['ASC_ID'].unique()) == len(charges_df)
    charges_df = charges_df.sort_values('ASC_ID').set_index('ASC_ID')
    return charges_df

def is_adulteration(df, charges_df):
    is_adulteration = []
    is_adulteration_df = charges_df['is_adulteration']
    for charges in df.charges.values:
        is_adult = False
        for c in charges:
            if is_adulteration_df.ix[c]:
                is_adult = True
        is_adulteration.append(is_adult)
    return np.array(is_adulteration)

def parse_refusals():
    dfs = []
    files = glob.glob(REFUSAL_FOLDER+'REFUSAL_ENTRY*.CSV')
    for f in files:
        df_ = pd.read_csv(f)
        dfs.append(df_)
    df = pd.concat(dfs)
    return df

def run():
    charges_df = parse_charges()
    df = parse_refusals()
    df = df[~pd.isnull(df['PRODUCT_CODE'])]
    df['code_prefix'] = df['PRODUCT_CODE'].apply(lambda x: x[:2]).astype(int)
    df = df[(df['code_prefix']<=52)|(df['code_prefix']==54)]

    products = df['PRDCT_CODE_DESC_TEXT']
    charges = df['REFUSAL_CHARGES'].apply(lambda x: [int(i) for i in x.split(', ') if charges_df['is_adulteration'].ix[int(i)]])
    df['charges'] = charges
    df = df[charges.apply(len)>0] # take only cases of adulterations # df = df[is_adulteration(df, charges_df)]
    df['main_charge'] = df['charges'].apply(lambda x: x[0])
    df['charge_code'] = df['main_charge'].apply(lambda x: charges_df['CHRG_CODE'].ix[x])
    df['charge_text'] = df['main_charge'].apply(lambda x: charges_df['CHRG_STMNT_TEXT'].ix[x])
    df['adulterant'] = df['main_charge'].apply(lambda x: charges_df['adulterant'].ix[x])
    prefix_to_cat = parse_code_prefix()
    df['category'] = df['code_prefix'].apply(lambda x: prefix_to_cat[x])
    df = df.reset_index(drop=True)

    pairs = df.groupby(['adulterant', 'category']).size().sort_values()[::-1]
    print 'Number of entries  :', len(df)
    print 'Unique entries     :', len(pairs)
    print 'Unique adulterants :', len(df['adulterant'].unique())
    print 'Unique products    :', len(df['PRODUCT_CODE'].unique())
    print 'Unique categories  :', len(df['category'].unique())

    return df


