import numpy as np
import pandas as pd

folder = '../../Metathesaurus.RRF/META/'

def gen_mrconso():
    cols = 'CUI | LAT | TS | LUI | STT | SUI | ISPREF | AUI | SAUI | SCUI | SDUI | SAB | TTY | CODE | STR | SRL | SUPPRESS | 18 | 19'.split(' | ')
    df = pd.read_csv(folder+'MRCONSO.RRF', sep='|', index_col=False, header=None, names=cols)
    df = df[['CUI', 'AUI', 'STR', 'SAB']]
    df['STR'] = df['STR'].str.lower()
    df = df.dropna(subset=['STR'])
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_hdf('mrconso.h5', 'mrconso', complib='blosc', complevel=5)
    return df

df = pd.read_hdf('mrconso.h5', 'mrconso')

# Semantic relationships
cols = ['CUI', 'TUI', 'STN', 'STY', 'ATUI', 'CVF', '6']
df_st = pd.read_csv(folder+'MRSTY.RRF', sep='|', index_col=False, header=None, names=cols)
del df_st['6']

# Relations
cols = ['CUI', 'AUI', 'CXN', 'PAUI', 'SAB', 'RELA', 'PTR', 'HCD', 'CVF', '10']
df_hier = pd.read_csv(folder+'MRHIER.RRF', sep='|', index_col=False, header=None, names=cols)
del df_hier['10']

df_i = pd.read_hdf('../foodessentials/ingredients.h5', 'ingredients')
counts = df_i['ingredient'].value_counts()

def get_ing_to_cuis():
    # Get CUI from ingredient name
    ing_to_cuis = {}
    strings = df['STR'].values
    for i, ing in enumerate(counts.index.values[:1000]):
        if i % 100 == 0:
            print i
        matches = df[strings==ing]
        if len(matches) == 0:
            # Contains rather than strictly equal
            matches = df[df['STR'].str.contains(ing)]
            if len(matches) == 0 and ' ' in ing:
                # Remove one word at a time from the left
                split = ing.split()
                l = len(split)
                begin_idx = 1
                while len(matches) == 0 and begin_idx < l:
                    matches = df[strings==' '.join(split[begin_idx:])]
                    begin_idx += 1
                if len(matches) == 0:
                    # Remove one word at a time from the right
                    end_idx = l-1
                    while len(matches) == 0 and end_idx > 0:
                        matches = df[strings==' '.join(split[:end_idx])]
                        end_idx -= 1
                if len(matches) == 0:
                    print "{} : No match".format(ing)
                    continue
                else:
                    print "Found substring match: {} | {}".format(ing, ' '.join(split[begin_idx-1:]))
            else:
                print "Found string match   : {}".format(ing)
        else:
            print "Found direct match   : {}".format(ing)
        cuis = matches['CUI'].drop_duplicates().values
        ing_to_cuis[ing] = cuis
    return ing_to_cuis


def get_ing_to_cui():
    # Pick CUI
    flatten_cuis = np.array([i for j in ing_to_cuis.values() for i in j])
    df_hier_short = df_hier[df_hier['CUI'].isin(flatten_cuis)]
    df_st_short = df_st[df_st['CUI'].isin(flatten_cuis)]
    ing_to_cui = {}
    for ing, cuis in ing_to_cuis.iteritems():
        print ing, len(cuis)
        if len(cuis) == 0:
            # Empty: we skip this ingredient.
            continue
        if len(cuis) == 1:
            # We are good.
            ing_to_cui[ing] = cuis[0]
        else:
            semantics = df_st_short[df_st_short['CUI'].isin(cuis)]
            semantics = semantics[semantics['CUI'].isin(df_hier_short['CUI'])]
            if len(semantics) == 0:
                print "Goodbye."
                continue
            if 'Food' in semantics['STY'].values:
                all_cuis = semantics[semantics['STY'] == 'Food']
            elif 'Vitamin' in semantics['STY'].values:
                all_cuis = semantics[semantics['STY'] == 'Vitamin']
            else:
                all_cuis = semantics
            if len(all_cuis) == 1:
                ing_to_cui[ing] = all_cuis['CUI'].values[0]
            else:
                # Pick the CUI with more relationships
                num_hiers = []
                for cui_iter in np.unique(all_cuis['CUI']):
                     num_hiers.append(len(df_hier_short[df_hier_short['CUI']==cui_iter]))
                num_hiers = np.array(num_hiers)
                max_idx = np.where(num_hiers==num_hiers.max())[0][0]
                ing_to_cui[ing] = np.unique(all_cuis['CUI'])[max_idx]
                print ing, cuis
                print all_cuis
                print num_hiers
    return ing_to_cui

sources_count = {}
def get_ing_to_hiers():
    # Get hierarchy for CUI
    sources = ['NCI', 'NDFRT', 'SNOMEDCT_US', 'MSH']
    df_hier_short = df_hier[df_hier['CUI'].isin(ing_to_cui.values())]
    df_hier_short = df_hier_short[df_hier_short['SAB'].isin(sources)]
    aui_to_cui = df.set_index('AUI')['CUI'].to_dict()
    ing_to_hiers = {}
    for ing, cui in ing_to_cui.iteritems():
        hiers = df_hier_short[df_hier_short['CUI']==cui]
        if len(hiers) == 0:
            print "No hierarchy for:", ing, cui
            continue
        print len(hiers)
        
        for i in np.unique(hiers['SAB']):
            if i not in sources_count:
                sources_count[i] = 1
            else:
                sources_count[i] += 1
        #print hiers
        
        rows = hiers['PTR'].str.split('.').values
        cuis = []
        for r in rows:
            cuis.append([aui_to_cui[j] for j in r])
        #print ing, cuis
        ing_to_hiers[ing] = cuis
    return ing_to_hiers

def get_final_cuis():
    # Get all CUIs in hierarchies
    final_cuis = []
    for i in ing_to_hiers.values():
        for j in i:
            for k in j:
                final_cuis.append(k)

ing_to_cuis = get_ing_to_cuis()
ing_to_cui = get_ing_to_cui()
ing_to_hiers = get_ing_to_hiers()
final_cuis = get_final_cuis()
