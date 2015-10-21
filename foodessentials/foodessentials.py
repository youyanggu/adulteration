import numpy as np
import pandas as pd
import pickle
import re

from api import *
from ingredient import Ingredient
from product import Product

#took out allergens, additives, procingredients, nutrients
keys = [u'aisle', u'brand', u'food_category', u'ingredients',
        u'manufacturer', u'product_description', u'product_name', 
        u'product_size', u'serving_size', u'serving_size_uom', 
        u'servings_per_container', u'shelf', u'upc'] 

def read_df():
     return pd.read_hdf('products.h5', 'products')

def read_df_i():
     return pd.read_hdf('ingredients.h5', 'ingredients')

def load_file(fname):
     with open(fname, 'rb') as f_in:
          data = pickle.load(f_in)
     return data

def load_categories():
     return load_file('categories.pkl')

def load_cat_to_prods():
     return load_file('cat_to_prods.pkl')

def update_categories(new_cats):
     if type(new_cats) == list:
          new_cats = set(new_cats)
     cats = load_categories()
     new_cats = new_cats | cats
     print "Categories: {} -> {}".format(len(cats), len(new_cats))
     with open('categories.pkl', 'wb') as f_out:
          pickle.dump(new_cats, f_out)
     return new_cats

"""
Searched terms:
a-z: 0-100
'bread, pasta, noodle, flour, dough' (100)
milk, alcohol, salt, sugar (200)
pear, apple, nut, water, caramel, cheese, butter, corn (200)
food, drink, meat, vegetable, egg (200)
cancer, health, vitamin, supplement, digest, pill, tablet (200)
chicken, poultry, beef, pork, duck, goat, sheep, veal (200)
ham, salami, turkey, sausage, meatloaf (200)
asian, mexican, thai, french, italian, korean, japanese, spanish (200)
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

def standardize_ingredient(i):
     i = i.strip() # remove trailing spaces
     i = i.strip('.')
     i = i.strip('*')
     i = ' '.join(i.split()) # remove multiple spaces
     return i

def parse_ingredients(s):
     s = s.lower()
     ingredients_split = re.findall(r'([^,\(]*)\((.*?)\)', s)
     middle_ingredients = [i[0] for i in ingredients_split]
     subingredients = [parse_ingredients(i[1]) for i in ingredients_split]
     subingredients = list(itertools.chain.from_iterable(subingredients))
     main_ingredients = re.sub(r'\([^)]*\)', '', s)
     main_ingredients_split = main_ingredients.split(',')
     main_minus_middle_ingredients = [i for i in main_ingredients_split if i not in middle_ingredients]
     all_ingredients = main_ingredients_split + subingredients
     return all_ingredients

def gen_ingredients_df(df):
     prod_ingredients = df['ingredients'].values
     categories = df['food_category'].values
     ingredients = []
     for idx in range(len(df)):
          s = parse_ingredients(prod_ingredients[idx])
          for i in s:
               new_i = standardize_ingredient(i)
               ingredients.append((new_i, idx))
     df_i = pd.DataFrame.from_records(ingredients, columns=['ingredient', 'product_id'])
     return df_i


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
                    if ingredients == '':
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
     except Exception as e:
          print e
          return products

def find_missing_categories():
     new_cats = load_categories()
     df = pd.read_hdf('products.h5', 'products')
     old_cats = set(df['food_category'])
     missing_categories = new_cats - old_cats
     if len(missing_categories) == 0:
          print "No missing categories!"
          return None
     new_prods = add_products(missing_categories)
     new_df = pd.DataFrame.from_records(new_prods, columns=keys)
     updated_df = pd.concat([df, new_df])
     updated_df.to_hdf('products.h5', 'products', mode='w')
     return missing_categories

start_session()
categories = load_categories()
products = add_products(categories)
df = pd.DataFrame.from_records(products, columns=keys)
#df.to_csv('products.csv', index=False, encoding='utf-8')
df.to_hdf('products.h5', 'products', mode='w')

df_i = gen_ingredients_df(df)
df_i.to_hdf('ingredients.h5', 'ingredients', mode='w')




