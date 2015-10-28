import csv
import inflect
import numpy as np
import pandas as pd
import pickle
import re

from api import *
from parse_ingredients import *

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print x
    pd.reset_option('display.max_rows')

#took out allergens, additives, procingredients, nutrients
keys = [u'aisle', u'brand', u'food_category', u'ingredients',
        u'manufacturer', u'product_description', u'product_name', 
        u'product_size', u'serving_size', u'serving_size_uom', 
        u'servings_per_container', u'shelf', u'upc'] 

def read_df(cat_min_count=10):
     df = pd.read_hdf('products.h5', 'products')
     if cat_min_count > 0:
          freq = df.groupby('food_category')['food_category'].transform('count')
          df = df[freq >= cat_min_count]
          df.reset_index(inplace=True, drop=True)
     return df

def read_df_i():
     return pd.read_hdf('ingredients.h5', 'ingredients')

def load_file(fname):
     with open(fname, 'rb') as f_in:
          r = csv.reader(f_in)
          data = [i[0] for i in list(r)]
     return set(data)

def save_file(fname, data):
     with open(fname, 'wb') as f_out:
          wr = csv.writer(f_out)
          wr.writerows([[i] for i in sorted(data)])

def load_categories():
     return load_file('categories.csv')

def load_non_food_categories():
     return load_file('non_food_categories.csv')

def get_non_food_categories(new_cats):
     cats_to_remove = []
     old_cats = load_non_food_categories()
     non_food_cats = set([i.split('-')[0] for i in old_cats])
     for cat in new_cats:
          if cat.split('-')[0] in non_food_cats:
               if cat not in old_cats:
                    print "New non-food category:", cat
               cats_to_remove.append(cat)
     save_file('non_food_categories.csv', old_cats|set(cats_to_remove))
     return set(cats_to_remove)


def remove_categories(old_cats):
     if type(old_cats) == list:
          old_cats = set(old_cats)
     cats = load_categories()
     print "Categories: {} -> {}".format(len(cats), len(cats)-len(old_cats))
     for i in old_cats:
          cats.remove(i)
     save_file('categories.csv', cats)


def update_categories(new_cats):
     if type(new_cats) == list:
          new_cats = set(new_cats)
     cats = load_categories()
     non_food = get_non_food_categories(new_cats)
     if non_food:
          print "Not adding non-food categories:"
          for nf in sorted(non_food):
               print nf
     new_cats = new_cats - non_food
     new_cats = new_cats | cats
     print "Categories: {} -> {}".format(len(cats), len(new_cats))
     save_file('categories.csv', new_cats)


"""
Searched terms:
a-z: all
'bread, pasta, noodle, flour, dough' (100)
milk, alcohol, salt, sugar, water (200)
pear, apple, nut, water, caramel, cheese, butter, corn (200)
food, drink, meat, vegetable, egg (200)
cancer, health, vitamin, supplement, digest, pill, tablet (200)
chicken, poultry, beef, pork, duck, goat, sheep, veal (200)
ham, salami, turkey, sausage, meatloaf (200)
asian, mexican, thai, french, italian, korean, japanese, spanish (200)
coffee, tea (200)
"""
def get_categories(q, n=100, s=0, limit=100):
     def process_prods(prods, cats):
          for p in prods:
               upc = p['upc']
               if upc != '':
                    category = get_category(upc)
                    if category:
                         cats.append(category)
                         #print category
     cats = []
     try:
          data = searchprods(q, n, s)
     except ValueError:
          print "Query didn't work:", q
          return []
     numFound = data['numFound']
     if numFound == 0:
          return []
     #print "NumFound:", numFound
     limit = min(limit, numFound)
     process_prods(data['productsArray'], cats)
     s += n
     while (s < limit):
          try:
               data = searchprods(q, n, s)
          except ValueError:
               print "Query didn't work:", q
               return cats
          process_prods(data['productsArray'], cats)
          s += n
     return cats

def add_categories(search_terms, n=50, s=0, limit=50):
     set_cats = load_categories()
     l = len(set_cats)
     if type(search_terms) == str:
          search_terms = search_terms.split(', ')
     for term in search_terms:
          cats = get_categories(term, n, s, limit)
          for c in set(cats):
               if c not in set_cats:
                    print c
          set_cats.update(cats)
          print "{}: {} -> {}".format(term, l, len(set_cats))
          l = len(set_cats)
     update_categories(set_cats)

def get_category(upc, n=1):
     try:
          data = labelarray(upc, n)
     except ValueError:
          print "Invalid UPC:", upc
          return ''
     category = data['food_category']
     return category

def get_upc_from_category(category, n=10):
     s = 0
     prods = []
     found = False
     while not found:
          try:
               data = searchprods('\"{}\"'.format(category), n, s)
          except ValueError:
               print "*** Didn't work. Try again later."
               return None
          numFound = data['numFound']
          if numFound == 0:
               print "No category for search term:", category
               return None
          if s > min(20, numFound):
               break
          prods = data['productsArray']
          for p in prods:
               if p['upc'] == '':
                    continue
               if get_category(p['upc']) == category:
                    found = True
                    break
          if not found:
               print "Category not found, trying more."
          s += n
     if not found:
          print "Category {} still not found. Returning None.".format(category)
          return None
     return p['upc']

def get_cat_prods_from_upc(upc):
     try:
          data = labelarray(upc, 1)
     except ValueError:
          print "*** Didn't work. Try again later."
          return []
     numFound = data['numFound']
     category = data['food_category']
     print "{} : {} products".format(category, numFound)
     try:
          data = labelarray(upc, numFound)
     except ValueError:
          print "*** Query too large:", numFound
          return []
     prods = data['productsArray']
     return prods


def add_products(categories):
     try:
          products = []
          for c in sorted(categories):
               upc = get_upc_from_category(c)
               if not upc:
                    continue
               prods = get_cat_prods_from_upc(upc)
               if not prods:
                    print "** No products found."
                    continue
               count = 0
               for p in prods:
                    ingredients = p['ingredients']
                    if len(ingredients) <= 1:
                         continue # skip products with no ingredients
                    count += 1
                    prod = []
                    assert(c == p['food_category'])
                    for k in sorted(keys):
                         if k not in p:
                              print '**ERROR: Key not found:', k
                              prod.append(u'')
                         else:
                              prod.append(p[k])
                    #prod = create_product(p)  
                    products.append(prod)
               if count == 0:
                    print "** No products have ingredients."
          return products
     except KeyboardInterrupt as e:
          print e
          return products

def find_missing_categories():
     new_cats = load_categories()
     df = read_df(cat_min_count=0)
     old_cats = set(df['food_category'])
     missing_categories = new_cats - old_cats
     if len(missing_categories) == 0:
          print "No missing categories!"
          return None
     new_prods = add_products(missing_categories)
     new_df = pd.DataFrame.from_records(new_prods, columns=keys)
     updated_df = pd.concat([df, new_df], ignore_index=True)
     updated_df.to_hdf('products.h5', 'products', mode='w')
     return missing_categories


def convert_to_singular(df_i, regenerate=False):
     if regenerate:
          d = {}
          p = inflect.engine()
          ings = df_i['ingredient'].value_counts().index.values
          lookup = {i: True for i in ings}
          for i in ings:
               # assume it is plural, try to convert to singular
               sing = p.singular_noun(i)
               if sing is False or sing == i:
                    continue # already singular or no change
               if sing in lookup:
                    d[i] = sing
     else:
          with open('plural_to_singular.pkl', 'rb') as f_in:
               d = pickle.load(f_in)
     df_i['ingredient'] = [d[i] if i in d else i for i in df_i['ingredient']]
     return df_i


def gen_ingredients_df(df):
     prod_ingredients = df['ingredients'].values
     categories = df['food_category'].values
     ingredients = []
     for idx in range(len(df)):
          s_split, middle_ings = parse_ingredients(prod_ingredients[idx])
          for i in s_split:
               new_i = standardize_ingredient(i)
               if len(new_i) <= 1:
                    continue
               #if new_i == '2% or less of celery':
               #     print prod_ingredients[idx].lower()
               ingredients.append((new_i, idx))
     df_i = pd.DataFrame.from_records(ingredients, columns=['ingredient', 'product_id'])
     df_i = convert_to_singular(df_i)
     df_i.drop_duplicates(inplace=True)
     df_i.reset_index(inplace=True, drop=True)
     counts = df_i['ingredient'].value_counts()
     return df_i, counts


def filter_df_i(df_i, min_count=100):
     freq = df_i.groupby('ingredient')['ingredient'].transform('count')
     df_i = df_i[freq >= min_count]
     counts = df_i['ingredient'].value_counts()
     return df_i, counts


def find_products_by_ing(ing, split=False, df=None):
     ing = ing.lower()
     if df is None:
          df = read_df()
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


def get_perc(ing, df=None, split=False):
     ing = ing.lower()
     if df is None:
          df = read_df()
     percs = []
     p_ing = find_products_by_ing(ing, split, df)
     cat_counts = p_ing['food_category'].value_counts()
     all_cat_counts = df['food_category'].value_counts()
     for cat in cat_counts.index.values:
          perc = cat_counts[cat]*1.0/all_cat_counts[cat]
          if perc>0.1:
               percs.append((perc, cat, all_cat_counts[cat]))
     cat_perc = sorted(percs)
     return cat_perc

def get_ings_by_product(df, df_i):
     d = df_i.groupby('product_id')['ingredient'].apply(lambda x: x.tolist()).to_dict()
     return [d.get(i, []) for i in range(len(df))]

def add_columns(df, df_i):
     df['ingredients_clean'] = get_ings_by_product(df, df_i)
     df['num_ingredients'] =  df['ingredients_clean'].apply(len)

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

#start_session()
#categories = load_categories()
#products = add_products(categories)
#missing_categories = find_missing_categories()
#df = pd.DataFrame.from_records(products, columns=keys)
#df.to_csv('products.csv', index=False, encoding='utf-8')
#df.to_hdf('products.h5', 'products', mode='w')
#df_i.to_hdf('ingredients.h5', 'ingredients', mode='w')

df = read_df()
df_i, counts = gen_ingredients_df(df)

#df_i_filt, counts_filt = filter_df_i(df_i)

add_columns(df, df_i)





