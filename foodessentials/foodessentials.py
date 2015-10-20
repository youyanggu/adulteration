import numpy as np
import pandas as pd
import pickle
import re

from api import *
from ingredient import Ingredient
from product import Product

def load_categories():
     with open('categories.pkl', 'rb') as f_in:
          cats = pickle.load(f_in)
     return cats

def load_cat_to_prods_cats():
     with open('cat_to_prods_cats.pkl', 'rb') as f_in:
          cat_to_prods = pickle.load(f_in)
     return cat_to_prods

def load_cat_to_prods():
     with open('cat_to_prods.pkl', 'rb') as f_in:
          cat_to_prods = pickle.load(f_in)
     return cat_to_prods

def update_categories(new_cats):
     if type(new_cats) == list:
          new_cats = set(new_cats)
     cats = load_categories()
     new_cats = new_cats | cats
     print "Categories: {} -> {}".format(len(cats), len(new_cats))
     with open('categories.pkl', 'wb') as f_out:
          pickle.dump(new_cats, f_out)
     return new_cats


def save_cat_to_prods(cat_to_prods):
     cats = cat_to_prods.keys()
     with open('cat_to_prods_cats.pkl', 'wb') as f_out:
          pickle.dump(cats, f_out)
     with open('cat_to_prods.pkl', 'wb') as f_out:
          pickle.dump(cat_to_prods, f_out)

"""
Searched terms:
a: 0-350, b: 0-100, x: 0-100, z: all, 
milk: 0-100, alcohol: 0-50, salt: 0-100, sugar: 0-50

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
     data = searchprods(q, n, s)
     numFound = data['numFound']
     #print "NumFound:", numFound
     limit = min(limit, numFound)
     process_prods(data['productsArray'], cats)
     s += n
     while (s < limit):
          data = searchprods(q, n, s)
          process_prods(data['productsArray'], cats)
          s += n
     return cats

def add_categories(search_terms, n=50, s=0, limit=50):
     set_cats = load_categories()
     l = len(set_cats)
     if type(search_terms) == str:
          search_terms = [search_terms]
     for term in search_terms:
          cats = get_categories(term, n, s, limit)
          set_cats.update(cats)
          print "{}: {} -> {}".format(term, l, len(set_cats))
          l = len(set_cats)
     update_categories(set_cats)


def get_category(upc, n=1):
     try:
          data = labelarray(upc)
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

def create_product(p):
     return Product(p)

def standardize_ingredient(i):
     i = i.strip().strip('.')
     return Ingredient(i)

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

def add_products(categories):
     try:
          cat_to_prods = {}
          for c in sorted(categories):
               products = []
               upc = get_upc_from_category(c)
               if not upc:
                    continue
               prods = get_cat_prods_from_upc(upc)
               if not prods:
                    continue
               for p in prods:
                    ingredients = p['ingredients']
                    if ingredients == '':
                         continue # skip products with no ingredients
                    prod = create_product(p)  
                    products.append(prod)
               print "{} -> {} ({})".format(len(prods), len(products), len(products)*1.0/len(prods))
               cat_to_prods[c] = products 
          return cat_to_prods
     except Exception as e:
          print e
          return cat_to_prods

def find_missing_categories(cat_to_prods, categories):
     missing_categories = set(categories) - set(cat_to_prods.keys())
     new_cat_to_prods = add_products(missing_categories)
     cat_to_prods.update(new_cat_to_prods)
     return cat_to_prods

start_session()
categories = load_categories()
cat_to_prods = add_products(categories)

