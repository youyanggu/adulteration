import numpy as np
import pandas as pd

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
    if ' of ' in c:
        c = c.split(' of ')[-1]
    if ' - ' in c:
        c = c.split(' - ')[-1]
    return c

def search_chemicals(counts, chemicals):
    found_chems = []
    d_c = {}
    ings = counts.index.values
    for c in chemicals:
        cc = clean_chemical(c)
        if cc in ings:
            print 'True  | {:<20} | {}'.format(cc, counts[cc])
            found_chems.append(c)
            d_c[c] = cc
        elif cc[:-1] in ings:
            print 'True  | {:<20} | {}'.format(cc[:-1], counts[cc[:-1]])
            found_chems.append(c)
            d_c[c] = cc[:-1]
        else:
            print 'False | {}'.format(cc)
    print len(found_chems), len(found_chems) * 1.0 / len(chemicals)
    return found_chems, d_c

def main():
    mapping = pd.read_csv('mapping.csv')
    d = {a : b for a,b in zip(mapping.category.values, mapping.shelf.values)}
    #df_ = rasff.load_df()
    df_['category_'] = df_['category'].replace(d)
    df_['chemical_'] = [clean_chemical(c) for c in df_['chemical'].values]
    #df, df_i = gather_data.import_data()
    counts = df_i['ingredient'].value_counts()
    chemicals = [i for i in df_['chemical'].unique() if i]
    found_chems, d_c = search_chemicals(counts, chemicals)
    df_filt = df_[df_['chemical'].isin(found_chems)]

    unknown_chems = [clean_chemical(c) for c in chemicals if c not in found_chems]
    df_conso = pd.read_hdf('mrconso.h5', 'mrconso')
    ing_to_cuis = get_ing_to_cuis(unknown_chems, df_conso)

    found_prods = {}
    for idx, new_df in df_filt.groupby(['chemical_', 'category_']):#.size().order()[::-1].iteritems():
        chem, cat = idx
        if chem not in found_prods:
            print '-----'
            found_prods[chem] = np.array([i[1] for i in foodessentials.get_perc(chem, 'shelf')])
            print "Num of categories chem found in: {}, Total: {}".format(len(found_prods[chem]), counts[chem])
        print cat.lower() in found_prods[chem], idx

