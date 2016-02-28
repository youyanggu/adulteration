import json
import pickle
import requests
import sys

from bs4 import BeautifulSoup

sys.path.append('../model')
import gather_data

WIKI_URL = 'https://en.wikipedia.org/w/api.php'

def get_page_html(title):
    params = {
              'action' : 'parse',
              'format' : 'json',
              'prop'   : 'text',
              'page'   : title,
             }
    resp = requests.get(WIKI_URL, params=params)
    content = resp.json()['parse']
    html = content['text']['*']
    return html

def search(term):
    params = {
              'action'   : 'query',
              'format'   : 'json',
              'list'     : 'search',
              'srsearch' : term,
             }
    resp = requests.get(WIKI_URL, params=params)
    content = resp.json()['query']['search']
    if len(content) == 0:
        return None
    hits = [i['title'] for i in content]
    return hits

def get_ing_to_title(load_saved=True):
    if load_saved:
        with open('ing_to_hits.pkl', 'rb') as f:
            ing_to_hits = pickle.load(f)
    else:
        num_ingredients = 5000
        df, df_i = gather_data.import_data()
        counts = df_i['ingredient'].value_counts()
        ings = counts.index.values[:num_ingredients]
        ing_to_hits = {}
        for i, ing in enumerate(ings):
            hits = search(ing)
            print '{} {} --> {}'.format(i, ing, hits[0])
            ing_to_hits[ing] = hits
        with open('ing_to_hits.pkl', 'wb') as f:
            pickle.dump(ing_to_hits, f)
    return ing_to_hits

def run_wiki():
    ing_to_hits = get_ing_to_title(load_saved=False)
    seen_titles = set()
    for i, ing in enumerate(ings):
        title = ing_to_hits[ing][0]
        if title in seen_titles:
            continue
        seen_titles.add(title)
        html = get_page_html(title)
        soup = BeautifulSoup(html, 'lxml')
        raw_text = soup.text
        with open('pages/{}.txt'.format(title.replace(' ', '_')), 'wb') as f:
            f.write(raw_text.encode('utf8'))

