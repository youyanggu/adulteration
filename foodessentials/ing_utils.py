import numpy as np
import pandas as pd

def read_df(cat_min_count=10):
     df = pd.read_hdf('products.h5', 'products')
     if cat_min_count > 0:
          freq = df.groupby('food_category')['food_category'].transform('count')
          df = df[freq >= cat_min_count]
          df.reset_index(inplace=True, drop=True)
     return df

def read_df_i():
     return pd.read_hdf('ingredients.h5', 'ingredients')

def find_products_by_ing(ing, split=False, df=None, df_i=None):
     ing = ing.lower()
     if df is None:
          df = read_df()
     if df_i is None:
          df_i = read_df_i()
     if split:
          ing_split = df_i['ingredient'].apply(lambda x: x.split())
          product_ids = df_i[ing_split.apply(lambda x: ing in x)]['product_id'].values
     else:
          product_ids = df_i[df_i['ingredient'] == ing]['product_id'].values
     product_ids = np.unique(product_ids)
     products = df.ix[product_ids]
     print "Products found:", len(products)
     return products


def get_perc(ing, category='food_category', df=None, df_i=None, split=False):
     ing = ing.lower()
     if df is None:
          df = read_df()
     percs = []
     p_ing = find_products_by_ing(ing, split, df, df_i)
     cat_counts = p_ing[category].str.lower().value_counts()
     all_cat_counts = df[category].str.lower().value_counts()
     for cat in cat_counts.index.values:
          perc = cat_counts[cat]*1.0/all_cat_counts[cat]
          percs.append((perc, cat, cat_counts[cat]))
     cat_perc = sorted(percs)
     return cat_perc

def get_ings_by_product(df, df_i):
     d = df_i.groupby('product_id')['ingredient'].apply(lambda x: x.tolist()).to_dict()
     return [d.get(i, []) for i in range(len(df))]


def find_matching_ingredients():
     all_ings = {}
     for i in range(1, 10):
          ings = showingredient(i)
          if len(ings['sameasingredients']) == 0:
               continue
          arr = [(j['ingredientid'], j['name']) for j in ings['sameasingredients']]
          arr_sorted = sorted(arr, key=lambda x:x[0])
          key = arr_sorted[0][1]
          for pair in arr_sorted:
               all_ings[pair[1]] = key
     return all_ings

