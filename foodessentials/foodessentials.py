import numpy as np
import pandas as pd

from api import *
from ingredient import Ingredient
from product import Product

def get_category(upc, n=10):
     data = labelarray(upc)
     category = data['food_category']
     return category

def get_upc_from_category(category, n=10):
     s = 0
     prods = []
     
     found = False
     while not found:
          data = searchprods('\"{}\"'.format(category), n, s)
          numFound = data['numFound']
          if s > numFound:
               break
          prods = data['productsArray']
          for p in prods:
               if get_category(p['upc']) == category:
                    found = True
                    break
          s += n
     assert(found)
     return p['upc']

def get_all_prods_from_upc(upc, n=1000):
     s = 0
     prods = []
     data = labelarray(upc, n, s)
     numFound = data['numFound']
     category = data['food_category']
     prods.append(data['productsArray'])
     s += n
     while (s < numFound):
          data = labelarray(upc, n, s)
          prods.append(data['productsArray'])
          s += n
     final_prods = [i for i in itertools.chain(*prods)]
     return category, final_prods

def create_product(p):
     return Product(p)

def standardize_ingredient(i):
     return i

def parse_ingredients(s):
     return s.lower().split(', ')

createsession()
categories = ['Herbs & Spices - Nutmeg & Cinnamon']
all_prods = {}
all_ingreds = {}
entries = [] # ingredient, product, category tuplets

for c in categories:
     upc = get_upc_from_category(c)
     prods = get_all_prods_from_upc(upc)[1]

     for p in prods:
          ingredients = p['ingredients']
          if ingredients == '':
               continue # skip products with no ingredients
          upc = p['upc']
          assert(upc not in all_prods)
          all_prods[upc] = create_product(p)
          for i in parse_ingredients(ingredients):
               i_std = standardize_ingredient(i)
               if i_std in all_ingreds:
                    print "Ingredient already exists:", i_std
               else:
                    all_ingreds[i_std] = True

               entry = (i_std, p, c)
               entries.append(entry)
               print entry[0], entry[1]['product_description'], entry[2]


df = pd.DataFrame.from_records(entries)

