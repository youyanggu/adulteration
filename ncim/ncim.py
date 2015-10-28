import numpy as np
import pandas as pd

def select_hier(hiers):
    # TODO
    return hiers.iloc[0]

def aui_to_cui(auis, df):
    cuis = []
    for aui in auis:
        cuis.append(df[df['AUI']==aui]['CUI'].values[0])
    return np.array(cuis)

cols = 'CUI | LAT | TS | LUI | STT | SUI | ISPREF | AUI | SAUI | SCUI | SDUI | SAB | TTY | CODE | STR | SRL | SUPPRESS | 18 | 19'.split(' | ')

df = pd.read_csv('MRCONSO.RRF', sep='|', index_col=False, header=None, names=cols)
df = df[['CUI', 'AUI', 'STR', 'SAB']]
df['STR'] = df['STR'].str.lower()
df = df.dropna(subset=['STR'])
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
df.to_hdf('mrconso.h5', 'mrconso', complib='blosc', complevel=5)

df = pd.read_hdf('mrconso.h5', 'mrconso')
#str_to_cui = df.set_index('STR')['CUI'].to_dict()

# Semantic relationships
cols = ['CUI', 'TUI', 'STN', 'STY', 'ATUI', 'CVF', '6']
df_st = pd.read_csv('MRSTY.RRF', sep='|', index_col=False, header=None, names=cols)
del df['6']

# Relations
cols = ['CUI', 'AUI', 'CXN', 'PAUI', 'SAB', 'RELA', 'PTR', 'HCD', 'CVF', '10']
df_hier = pd.read_csv('MRHIER.RRF', sep='|', index_col=False, header=None)
del df['10']

# Get CUI from ingredient name
ing_to_cuis = {}
strings = df['STR'].values
df_i = pd.read_hdf('../foodessentials/ingredients.h5', 'ingredients')
counts = df_i['ingredient'].value_counts()
for ing in counts.index.values[:100]:
    matches = df[strings==ing]
    cuis = matches['CUI'].values
    ing_to_cuis[ing] = cuis
    if len(matches) == 0:
        matches = df[df['STR'].str.contains(ing)].drop_duplicates('CUI')
        if len(matches) == 0:
            print "No match for:", ing
        else:
            print "Found string match for:", ing
    else:
        print "Found direct match for:", ing


# Pick CUI
#ing_to_cui = {}
ing_to_cui = {i: ing_to_cuis[i][0] for i in ing_to_cuis if len(ing_to_cuis[i])!=0}
# TODO

# Get hierarchy for CUI
aui_to_cui = df2.set_index('AUI')['CUI'].to_dict()
ing_to_hier = {}
for ing, cui in ing_to_cui.iteritems():
    hiers = df_hier[df_hier['CUI']==cui]
    if len(hiers) == 0:
        print "No hierarchy for:", ing, cui
        continue
    row = select_hier(hiers)
    cuis = [aui_to_cui[i] for i in row['PTR'].split('.')]
    print ing, cuis
    ing_to_hier[ing] = cuis



