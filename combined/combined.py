import itertools
import sys

import numpy as np
import pandas as pd

sys.path.append('../foodessentials')
sys.path.append('../collab_filter')
sys.path.append('../model')
sys.path.append('../ncim')
import ing_utils
import rasff
import gather_data
import ncim

def count_num_formulas(counts):
    df_conso = pd.read_hdf('mrconso.h5', 'mrconso')
    df_sat = pd.read_hdf('mrsat.h5', 'mrsat')
    ing_to_cuis2 = get_ing_to_cuis(counts.index.values[:1000], df_conso)
    df_sat2 = df_sat[df_sat['ATN']=='Chemical_Formula']
    s = set(df_sat2.CUI.unique())
    n_formulas = 0
    for cuis in ing_to_cuis2.values():
        found = False
        for c in cuis:
            if c in s:
                #print df_sat2[df_sat2['CUI']==c]['ATV']
                found = True
        print found
        if found:
            n_formulas += 1

def clean_chemical(c):
    c = c.lower()
    c = c.replace('sulphite', 'sulfite')
    c = c.replace('colour', 'color')
    if ' of ' in c:
        c = c.split(' of ')[-1]
    if ' - ' in c:
        c = c.split(' - ')[-1]
    return c

def search_chemicals(counts, chemicals, print_results=False):
    """Search if chemical is found in list of ingredients."""
    found_chems = []
    d_c = {}
    ings = counts.index.values
    for c in chemicals:
        #cc = clean_chemical(c)
        if cc in ings:
            if print_results:
                print 'True  | {:<20} | {}'.format(cc, counts[cc])
            found_chems.append(c)
            d_c[c] = cc
        elif cc[:-1] in ings:
            if print_results:
                print 'True  | {:<20} | {}'.format(cc[:-1], counts[cc[:-1]])
            found_chems.append(c)
            d_c[c] = cc[:-1]
        else:
            if print_results:
                print 'False | {}'.format(cc)
    if print_results:
        print len(found_chems), len(found_chems) * 1.0 / len(chemicals)
    return found_chems, d_c

def analyze_found_chemicals(df_, found_chems):
    df_filt = df_[df_['chemical'].isin(found_chems)]
    found_prods = {}
    for idx, new_df in df_filt.groupby(['chemical_', 'category_']):#.size().order()[::-1].iteritems():
        chem, cat = idx
        if chem not in found_prods:
            print '-----'
            found_prods[chem] = np.array([i[1] for i in ing_utils.get_perc(chem, 'shelf')])
            print "Num of categories chem found in: {}, Total: {}".format(len(found_prods[chem]), counts[chem])
        print cat.lower() in found_prods[chem], idx

def gen_nearest_neighbors(chemicals, ings, df_):
    num_ingredients = 5000
    num_neighbors = 5
    ings = ings[:num_ingredients]
    cuis_to_idx, neigh = ncim.run('ing_to_cuis_5000.pkl', num_ingredients)
    sources = ['SNOMEDCT_US', 'NCI', 'NDFRT', 'MSH']
    found_ings, new_ings_reps = ncim.convert_ing_to_rep(
            chemicals, sources, cuis_to_idx)
    distances, neighbors = neigh.kneighbors(new_ings_reps)
    not_found_ings = [c for c in chemicals if c not in found_ings]

    neighbors_trunc = []
    for i in range(len(distances)):
        num_closest = (distances[i]==distances[i][0]).sum()
        if num_closest <= num_neighbors:
            neighbors_trunc.append(neighbors[i][:num_neighbors])
        else:
            # Take the n most popular ingredients if it's a tie.
            neighbors_trunc.append(sorted(neighbors[i][:num_closest])[:num_neighbors])
    neighbors_trunc = np.array(neighbors_trunc)

    # Print neighbors
    #for i in range(len(found_ings)):
    #    print '{} --> {}'.format(found_ings[i], ings[neighbors_trunc[i][:3]])

    chem_to_best_cats = {}
    cat_to_best_chems = {}
    for i, c in enumerate(found_ings):
        cat_to_perc = {}
        print '--------------'
        print '{} of {}'.format(i+1, len(found_ings))
        print '{} ({})'.format(c, c in ings)
        print '--------------'
        print 'Neighbors:'
        for nn_idx in neighbors_trunc[i]:
            nn = ings[nn_idx]
            print nn
            #cats = ing_utils.get_perc(nn, 'shelf')
            cats = get_perc(nn, 'shelf')
            if len(cats) == 0:
                continue
            for perc, cat, count in cats:
                if cat in cat_to_perc:
                    cat_to_perc[cat] = cat_to_perc[cat] + perc
                else:
                    cat_to_perc[cat] = perc
                if cat in cat_to_best_chems:
                    d = cat_to_best_chems[cat]
                else:
                    d = {}
                    cat_to_best_chems[cat] = d
                if c in d:
                    d[c] = d[c] + perc
                else:
                    d[c] = perc
        best_cats = sorted(cat_to_perc, key=cat_to_perc.get, reverse=True)
        chem_to_best_cats[c] = [(j, cat_to_perc[j]/num_neighbors) for j in best_cats]
        print "========="
        print "Predicted:"
        print chem_to_best_cats[c][:5]
        print "========="
        print "Actual:"
        print df_[df_['chemical_']==c]['category_'].value_counts()
    
    for cat in cat_to_best_chems.keys():
        d = cat_to_best_chems[cat]
        best_chems = sorted(d, key=d.get, reverse=True)
        cat_to_best_chems[cat] = [(j, d[j]/num_neighbors) for j in best_chems]
    
    return chem_to_best_cats, cat_to_best_chems

def generate_pairs(df_, categories=None, chemicals=None):
    pair_counts = df_.groupby(['category_', 'chemical_']).size().sort_values()
    chem_counts = df_.groupby('chemical_').size().sort_values()
    cat_counts = df_.groupby('category_').size().sort_values()
    if categories is None:
        categories = df_['category_'].unique()
    if chemicals is None:
        chemicals = df_['chemical_'].unique()
    pairs = []
    for cat, chem in itertools.product(categories, df_['chemical_'].unique()):
        if cat == '---' or chem == '':
            continue
        pair = (cat, chem)
        pair_count = pair_counts.get(pair, 0)
        cat_count = cat_counts.get(cat, 0)
        chem_count = chem_counts.get(chem, 0)
        pairs.append((pair[0], pair[1], pair_count, cat_count, chem_count))
    pairs_df = pd.DataFrame(pairs, 
        columns=['category', 'chemical', 'pair_count', 'cat_count', 'chem_count'])
    #pairs_df.to_csv('pairs_df.csv', index=False)
    return pairs_df
            

def main():
    mapping = pd.read_csv('mapping.csv')
    d = {a : b for a,b in zip(mapping.category.values, mapping.shelf.values)}
    df_ = rasff.load_df()
    df_['category_'] = df_['category'].replace(d)
    df_['chemical_'] = [clean_chemical(c) for c in df_['chemical'].values]
    df, df_i = gather_data.import_data()
    counts = df_i['ingredient'].value_counts()
    ings = counts.index.values
    chemicals = [i for i in df_['chemical_'].unique() if i] # remove empty string
    found_chems, d_c = search_chemicals(counts, chemicals)
    unknown_chems = [c for c in chemicals if c not in found_chems]

    chemical_counts = df_.groupby('chemical_').size().sort_values()[::-1]
    category_counts = df_.groupby('category_').size().sort_values()[::-1]
    
    #analyze_found_chemicals(df_, found_chems)
    #gen_nearest_neighbors(chemicals, ings)

    

    
