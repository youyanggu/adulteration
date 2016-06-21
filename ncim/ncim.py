import numpy as np
import pandas as pd
import pickle
import sys

sys.path.append('../model')
from gen_embeddings import get_nearest_neighbors

folder = '../../Metathesaurus.RRF/META/'
prefix = '../ncim/'

def gen_raw_df():
    # CUI/AUI
    cols = 'CUI | LAT | TS | LUI | STT | SUI | ISPREF | AUI | SAUI | SCUI | SDUI | SAB | TTY | CODE | STR | SRL | SUPPRESS | 18 | 19'.split(' | ')
    df_conso = pd.read_csv(folder+'MRCONSO.RRF', sep='|', index_col=False, header=None, names=cols)
    df_conso = df_conso[['CUI', 'AUI', 'STR', 'SAB']]
    df_conso['STR'] = df_conso['STR'].str.lower()
    df_conso = df_conso.dropna(subset=['STR'])
    df_conso.drop_duplicates(inplace=True)
    df_conso.reset_index(drop=True, inplace=True)
    #df_conso.to_hdf('mrconso.h5', 'mrconso', complib='blosc', complevel=5)

    # Semantic relationships
    cols = ['CUI', 'TUI', 'STN', 'STY', 'ATUI', 'CVF', '6']
    df_st = pd.read_csv(folder+'MRSTY.RRF', sep='|', index_col=False, header=None, names=cols)
    del df_st['6']
    #df_st.to_hdf('mrsty.h5', 'mrsty', complib='blosc', complevel=5)

    # Relations
    cols = ['CUI', 'AUI', 'CXN', 'PAUI', 'SAB', 'RELA', 'PTR', 'HCD', 'CVF', '10']
    df_hier = pd.read_csv(folder+'MRHIER.RRF', sep='|', index_col=False, header=None, names=cols)
    del df_hier['10']
    #df_hier.to_hdf('mrhier.h5', 'mrhier', complib='blosc', complevel=5)

    cols = ['CUI', 'LUI', 'SUI', 'METAUI', 'STYPE', 'CODE', 'ATUI', 'SATUI',
            'ATN', 'SAB', 'ATV', 'SUPPRESS', 'CVF']
    df_sat = pd.read_csv(folder+'MRSAT.RRF', sep='|', index_col=False, header=None, names=cols)

    return df_conso, df_st, df_hier, df_sat

def get_ing_to_cuis(ings, df):
    # Get CUI from ingredient name
    direct_match = string_match = substring_match = no_match = 0
    if type(ings) == str:
        ings = [ings]
    ing_to_cuis = {}
    strings = df['STR'].values
    for i, ing in enumerate(ings):
        print i, ing
        ing = ing.strip()
        matches = df[strings==ing]
        if len(matches) == 0 and len(ing.split())==2:
            # Search for reverse of the string.
            matches = df[strings==', '.join(ing.split()[::-1])]
            if len(matches) == 0:
                matches = df[strings==' '.join(ing.split()[::-1])]
        if len(matches) == 0:
            #continue # TEMP LINE
            # Contains rather than strictly equal
            matches = df[df['STR'].str.contains(ing)]
            if len(matches) == 0:
                if ' ' in ing:
                    # Remove one word at a time from the left
                    split = ing.split()
                    l = len(split)
                    begin_idx = 1
                    cur_match = ''
                    while len(matches) == 0 and begin_idx < l:
                        cur_match = ' '.join(split[begin_idx:])
                        matches = df[strings==cur_match]
                        begin_idx += 1
                    if len(matches) == 0:
                        # Remove one word at a time from the right
                        end_idx = l-1
                        cur_match = ' '.join(split[:end_idx])
                        while len(matches) == 0 and end_idx > 0:
                            matches = df[strings==cur_match]
                            end_idx -= 1
                if len(matches) == 0 or len(cur_match)<3:
                    print "*** No match for: {}".format(ing)
                    no_match += 1
                    continue
                else:
                    print "Found substring match: {} --> {}".format(ing, cur_match)
                    substring_match += 1
            else:
                print "Found string match   : {}".format(ing) 
                string_match += 1
        else:
            print "Found direct match   : {}".format(ing)
            direct_match += 1
        cuis = matches['CUI'].drop_duplicates().values
        ing_to_cuis[ing] = cuis
    print "Direct matches   :", direct_match
    print "String matches   :", string_match
    print "Substring matches:", substring_match
    print "No matches       :", no_match
    print "{} --> {}".format(len(ings), len(ing_to_cuis))
    return ing_to_cuis


def get_ing_to_cui(ing_to_cuis, df_hier, df_st):
    # Pick CUI
    flatten_cuis = np.array([i for j in ing_to_cuis.values() for i in j])
    df_hier_short = df_hier[df_hier['CUI'].isin(flatten_cuis)]
    df_st_short = df_st[df_st['CUI'].isin(flatten_cuis)]
    ing_to_cui = {}
    for ing, cuis in ing_to_cuis.iteritems():
        #print ing, len(cuis)
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
                #print "Goodbye."
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
                #print ing, cuis
                #print all_cuis
                #print num_hiers
    print "{} --> {}".format(len(ing_to_cuis), len(ing_to_cui))
    return ing_to_cui

def get_ing_to_hiers(ing_to_cui, df_hier, sources):
    # Get hierarchy for CUI
    df_hier_short = df_hier[df_hier['CUI'].isin(ing_to_cui.values())]
    df_hier_short = df_hier_short[df_hier_short['SAB'].isin(sources)]
    
    ing_to_hiers = {}
    sources_count = {}
    for ing, cui in ing_to_cui.iteritems():
        hiers = df_hier_short[df_hier_short['CUI']==cui]
        if len(hiers) == 0:
            #print "No hierarchy for:", ing, cui
            continue
        
        for i in np.unique(hiers['SAB']):
            if i not in sources_count:
                sources_count[i] = 1
            else:
                sources_count[i] += 1
        #print hiers
        cur_auis = hiers['AUI'].values
        rows = hiers['PTR'].str.split('.').values
        auis = []
        for aui, r in zip(cur_auis, rows):
            auis.append([j for j in r] + [aui])
        #print ing, auis
        ing_to_hiers[ing] = auis
    print "{} --> {}".format(len(ing_to_cui), len(ing_to_hiers))
    return ing_to_hiers

def convert_auis_to_cuis(ing_to_hiers_aui, df):
    aui_to_cui = df.set_index('AUI')['CUI'].to_dict()
    ing_to_hiers = {}
    for k, v in ing_to_hiers_aui.iteritems():
        cuis = []
        for path in v:
            cuis.append([aui_to_cui[aui] for aui in path])
        ing_to_hiers[k] = cuis
    return ing_to_hiers


def convert_hier_to_str(ing_to_hiers_aui, df):
    aui_to_str = df.set_index('AUI')['STR'].to_dict()
    ing_to_hiers_str = {}
    for k, v in ing_to_hiers_aui.iteritems():
        strs = []
        for path in v:
            strs.append([aui_to_str[aui] for aui in path])
        ing_to_hiers_str[k] = strs
    return ing_to_hiers_str


def gen_ing_rep(ing_to_hiers, cuis_to_idx=None):
    def gen_cuis_to_idx(ing_to_hiers):
        cuis = set()
        for k,v in ing_to_hiers.iteritems():
            for path in v:
                for cui in path:
                    cuis.add(cui)
        cuis = np.array(sorted(cuis))
        cuis_to_idx = {v : i for i,v in enumerate(cuis)}
        return cuis_to_idx

    if cuis_to_idx is None:
        cuis_to_idx = gen_cuis_to_idx(ing_to_hiers)
    l = len(cuis_to_idx)
    ings, reps = [], []
    for ing, hiers in ing_to_hiers.iteritems():
        vectors = []
        for path in hiers:
            v = np.zeros(l)
            for cui in path:
                if cui in cuis_to_idx:
                    v[cuis_to_idx[cui]] = 1
            vectors.append(v)
        ings.append(ing)
        reps.append(np.mean(np.array(vectors), axis=0))
    assert(len(cuis_to_idx) == max(cuis_to_idx.values())+1)
    return np.array(ings), np.array(reps), cuis_to_idx


def calc_new_ranks(all_ings, ings, reps):
    """Calculate ranks in the original ing ordering."""
    new_reps = []
    for i in all_ings:
        idx = np.where(ings==i)[0]
        if len(idx) == 0:
            new_reps.append(np.zeros(reps.shape[1]))
        else:
            idx = idx[0]
            new_reps.append(reps[idx])
    new_ranks = get_nearest_neighbors(np.array(new_reps))
    for i,v in enumerate(new_ranks):
        v[i] = 0 # Set itself to be rank 0
    return new_ranks

def generate_nearest_neighbors(all_ings, ings, reps, print_neighbors=True, top_n=3):
    ranks, neigh = get_nearest_neighbors(reps)
    if print_neighbors:
        ing_to_nn = {}
        for i in range(ranks.shape[0]):
            nearest_neighbors = np.argsort(ranks[i])
            neighbor_names = [ing for ing in ings[nearest_neighbors[:top_n+1]] if ing != ings[i]]
            ing_to_nn[ings[i]] = neighbor_names[:top_n]
        for i in all_ings:
            if i not in ing_to_nn:
                print '{} --> N/A'.format(i)
            else:
                print '{} --> {}'.format(i, ing_to_nn[i])
    return ranks, neigh

def convert_ing_to_rep(ings, sources, cuis_to_idx):
    ing_to_cuis = get_ing_to_cuis(ings, df_conso)
    ing_to_cui = get_ing_to_cui(ing_to_cuis, df_hier, df_st)
    ing_to_hiers_aui = get_ing_to_hiers(ing_to_cui, df_hier, sources)
    ing_to_hiers = convert_auis_to_cuis(ing_to_hiers_aui, df_conso)
    if len(ing_to_hiers) == 0:
        print "Cannot find representation for ingredients."

    found_ings, reps, _ = gen_ing_rep(ing_to_hiers, cuis_to_idx)
    assert len(found_ings)==len(reps)
    return found_ings, reps

def run(fname=None, num_ingredients=5000):
    sources = ['SNOMEDCT_US', 'NCI', 'NDFRT', 'MSH']
    #sources = ['SNOMEDCT_US']

    df_conso = pd.read_hdf('{}mrconso.h5'.format(prefix), 'mrconso')
    df_st = pd.read_hdf('{}mrsty.h5'.format(prefix), 'mrsty')
    df_hier = pd.read_hdf('{}mrhier.h5'.format(prefix), 'mrhier')
    #df_sat = pd.read_hdf('{}mrsat.h5'.format(prefix), 'mrsat')
    df_i = pd.read_hdf('../foodessentials/ingredients.h5', 'ingredients')
    counts = df_i['ingredient'].value_counts()
    all_ings = counts.index.values
    
    if fname:
        with open(prefix+fname, 'rb') as f:
            ing_to_cuis = pickle.load(f)
    else:
        ing_to_cuis = get_ing_to_cuis(counts[:num_ingredients].index.values, df_conso)
    ing_to_cui = get_ing_to_cui(ing_to_cuis, df_hier, df_st)
    ing_to_hiers_aui = get_ing_to_hiers(ing_to_cui, df_hier, sources)
    ing_to_hiers = convert_auis_to_cuis(ing_to_hiers_aui, df_conso)
    ing_to_hiers_str = convert_hier_to_str(ing_to_hiers_aui, df_conso)
    ings, reps, cuis_to_idx = gen_ing_rep(ing_to_hiers)
    ranks, neigh = generate_nearest_neighbors(all_ings[:num_ingredients], ings, reps, False)
    return ings, reps, cuis_to_idx, neigh


def main():
    fname = '../ncim/ing_to_cuis_5000.pkl'
    num_ingredients = 5000
    ings, reps, cuis_to_idx, neigh = run(fname, num_ingredients)

    # Write nearest neighbors of unknown ingredients to file.
    with open('nn_{}.txt'.format(num_ingredients), 'wb') as f_out:
        new_ings = counts[num_ingredients:2*num_ingredients].index.values
        found_ings, new_ings_reps = convert_ing_to_rep(
            new_ings, sources, cuis_to_idx, neigh)
        nns = neigh.kneighbors(new_ings_reps)[1][:,:3] # top 3 neighbors
        for i in range(len(new_ings)):
            ing_idx = np.where(found_ings==new_ings[i])[0]
            if len(ing_idx)==0:
                f_out.write('{} --> N/A\n'.format(new_ings[i]))
                continue
            f_out.write('{} --> {}\n'.format(new_ings[i], ings[nns[ing_idx[0]]]))

if __name__ == '__main__':
    main()
