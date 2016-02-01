import itertools

import pandas as pd

def generate_pairs(df_, categories=None, chemicals=None):
    pair_counts = df_.groupby(['adulterant', 'food_category']).size().sort_values()
    chem_counts = df_.groupby('adulterant').size().sort_values()
    cat_counts = df_.groupby('food_category').size().sort_values()
    if categories is None:
        categories = df_['food_category'].unique()
    if chemicals is None:
        chemicals = df_['adulterant'].unique()
    pairs = []
    for chem, cat in itertools.product(df_['adulterant'].unique(), categories):
        pair = (chem, cat)
        pair_count = pair_counts.get(pair, 0)
        chem_count = chem_counts.get(chem, 0)
        cat_count = cat_counts.get(cat, 0)
        pairs.append((pair[0], pair[1], pair_count, chem_count, cat_count))
    pairs_df = pd.DataFrame(pairs, 
        columns=['adulterant', 'food_category', 'pair_count', 'chem_count', 'cat_count'])
    pairs_df.to_csv('ema_pairs_all.csv', index=False)
    return pairs_df

def split_df_list(df, target_column, separator):
    ''' df = dataframe to split,
    target_column = the column containing the values to split
    separator = the symbol used to perform the split

    returns: a dataframe with each entry for the target column separated, with each element moved into a new row. 
    The values in the other columns are duplicated across the newly divided rows.
    '''
    def split_list_to_rows(row, row_accumulator, target_column, separator):
        split_row = row[target_column].split(separator)
        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = s
            row_accumulator.append(new_row)
    new_rows = []
    df.apply(split_list_to_rows, axis=1, args=(new_rows, target_column, separator))
    new_df = pd.DataFrame(new_rows)
    return new_df

def read_ema():
    cols = ['food_category', 'year_begin', 'year_end', 'food_product',
            'adulterant', 'method', 'produced_location']

    with open('ema.html', 'rb') as f_in:
        text = f_in.read()

    df = pd.read_html(text)[0].ix[:,:7]
    df.columns = cols
    new_df = split_df_list(df, 'adulterant', '; ')
    new_df = split_df_list(new_df, 'food_product', '; ')
    new_df = new_df.sort_values('adulterant')
    new_df.reset_index(drop=True, inplace=True)
    return new_df

def read_ema_ings():
    with open('all_ingredients.html', 'rb') as f_in:
        text_ings = f_in.read()
    df_sort = pd.read_html(text_ings)[0]
    df_sort.columns = ['name', 'sort_name', 'form']
    df_sort = df_sort.sort_values('name')
    df_sort.reset_index(drop=True, inplace=True)

    cols_ings = ['ingredient', 'function', 'form', 'id_test', 'assay']
    df_type = ['high_suscept', 'low_suscept', 'high_suscept_id', 'pending']

    with open('ema_ingredients.html', 'rb') as f_in:
        text_ings = f_in.read()

    dfs = pd.read_html(text_ings)
    for i, df_ in enumerate(dfs):
        df_.columns = cols_ings
        df_['type'] = df_type[i]
    all_dfs = pd.concat(dfs)
    all_dfs = all_dfs.sort_values('ingredient')
    all_dfs = all_dfs.drop_duplicates('ingredient')
    all_dfs.reset_index(drop=True, inplace=True)
    all_dfs['sort_name'] = df_sort['sort_name']
    all_dfs = all_dfs.sort_values('sort_name')
    all_dfs.reset_index(drop=True, inplace=True)
    return all_dfs

def read_ema_data():
    return pd.read_csv('ema.csv')

def main():
    all_dfs = read_ema_ings()
    df = read_ema()
    pairs = df.groupby(['adulterant', 'food_product']).size().sort_values()[::-1]
    print 'Number of entries  :', len(df)
    print 'Unique entries     :', len(pairs)
    print 'Unique adulterants :', len(df['adulterant'].unique())
    print 'Unique products    :', len(df['food_product'].unique())
    print 'Unique categories  :', len(df['food_category'].unique())

    #all_dfs.to_csv('ema_ingredients.csv', index=False, encoding='utf-8')
    #df.to_csv('ema.csv', index=False)
    #pairs.to_csv('ema_pairs.csv')

