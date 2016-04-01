import glob
import json
import nltk
import numpy as np
import os
import pandas as pd
import pickle
import requests
import string
import sys

from bs4 import BeautifulSoup
from collections import Counter
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

WIKI_URL = 'https://en.wikipedia.org/w/api.php'
DIRNAME = os.path.dirname(__file__)

def get_summary(title):
    params = {
              'action' : 'query',
              'format' : 'json',
              'prop'   : 'extracts',
              'exintro' : True,
              'explaintext' : True,
              'titles' : title,
             }
    resp = requests.get(WIKI_URL, params=params)
    content = resp.json()['query']['pages'].values()[0]['extract']
    return content

def get_category_members(category):
    params = {
              'action'  : 'query',
              'format'  : 'json',
              'list'    : 'categorymembers',
              'cmtitle' : category,
              'cmlimit' : 500,
             }
    resp = requests.get(WIKI_URL, params=params)
    content = resp.json()['query']['pages'].values()[0]['categories']
    categories = [i['title'] for i in content]
    return categories

def get_category(title):
    params = {
              'action' : 'query',
              'format' : 'json',
              'prop'   : 'categories',
              'clshow' : '!hidden',
              'titles' : title,
             }
    resp = requests.get(WIKI_URL, params=params)
    content = resp.json()['query']['pages'].values()[0]['categories']
    categories = [i['title'] for i in content]
    return categories

def get_adulterants():
    return np.load(os.path.join(DIRNAME, '../rasff/all_rasff_chems.npy'))

def get_ings(num_ingredients):
    df_i = pd.read_hdf(
        os.path.join(DIRNAME, '../foodessentials/ingredients.h5'), 'ingredients')
    counts = df_i['ingredient'].value_counts()
    if num_ingredients:
        return counts.index.values[:num_ingredients]
    else:
        return counts.index.values

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
    def preprocess(term):
        term = re.sub('.*E \d* - ', '', term)
        return term
    term = preprocess(term)
    params = {
              'action'   : 'query',
              'format'   : 'json',
              'list'     : 'search',
              'srsearch' : term,
             }
    resp = requests.get(WIKI_URL, params=params)
    content = resp.json()['query']['search']
    if len(content) == 0:
        return ['']
    hits = [i['title'].encode('utf8') for i in content]
    return hits

def get_ing_to_title(ings=None):
    """Convert ing names to their Wikipedia titles."""
    if ings is None:
        with open(os.path.join(DIRNAME, 'ing_to_hits.pkl'), 'rb') as f:
            ing_to_hits = pickle.load(f)
    else:
        ing_to_hits = {}
        for i, ing in enumerate(ings):
            if ing:
                hits = search(ing)
                print '{} {} --> {}'.format(i, ing, hits[0])
                ing_to_hits[ing] = hits
        with open(os.path.join(DIRNAME, 'ing_to_hits.pkl'), 'wb') as f:
            pickle.dump(ing_to_hits, f)
    return ing_to_hits

def clean_title(title):
        title = title.strip()
        title = title.replace('/', '-')
        title = title.replace(' ', '_')
        return title

def run_wiki(ings=None, start=0, summary=True, overwrite=False):
    """Download Wikipedia text from title."""
    ing_to_hits = get_ing_to_title(ings)
    if ings is None:
        ings = sorted(ing_to_hits.keys())
    seen_titles = set()
    for i, ing in enumerate(ings):
        if i < start:
            continue
        title = ing_to_hits[ing][0]
        print i, ing, title
        if not title or title in seen_titles:
            continue
        seen_titles.add(title)
        file_title = clean_title(title)
        if summary:
            fname = 'summary/{}.txt'.format(file_title)
        else:
            fname = 'pages/{}.txt'.format(file_title)
        if os.path.isfile(fname) and not overwrite:
            print "File already exists for: {}. overwrite is False.".format(fname)
            continue
        if summary:
            html = get_summary(title)
        else:
            html = get_page_html(title)
        soup = BeautifulSoup(html, 'lxml')
        raw_text = soup.text
        with open(fname, 'wb') as f:
            f.write(title+'\n')
            f.write(raw_text.encode('utf8'))

def clean_page(f):
    text = []
    for line in f:
        line = line.strip()
        if line in ['References[edit]', 'External Links[edit]', 'Further Reading[edit']:
            break
        if len(line) <= 1:
            continue
        text.append(line)
    return '\n'.join(text)

def clean_wiki_pages(summary=False):
    if summary:
        for fname in glob.glob('summary/*.txt'):
            with open(fname, 'r') as f_in:
                text = clean_page(f_in)
            with open('clean_summary/'+os.path.basename(fname), 'w') as f_out:
                f_out.write(text)
    else:
        for fname in glob.glob('pages/*.txt'):
            with open(fname, 'r') as f_in:
                text = clean_page(f_in)
            with open('clean/'+os.path.basename(fname), 'w') as f_out:
                f_out.write(text)

def filter_tokens(tokens):
    filtered = []
    for t in tokens:
        #if t in stopwords.words('english'):
        #    continue
        if t == '.':
            t = '</s>'
        """
        if len(t) <= 1:
            continue
        t = t.replace(u'\u2212', '-')
        try:
            float(t)
        except:
            pass
        else:
            continue
        """
        filtered.append(t)
    #print '{}-->{}'.format(len(tokens), len(filtered))
    return filtered

def tokenize(text):
    #punctuations = '"#$%&\'()*+,/:;<=>@[\\]^_`{|}~'
    punctuations = ''
    #clean_text = text.lower().translate(
    #    {i:None for i in punctuations})
    clean_text = text.lower().translate(None, punctuations)
    clean_text = clean_text.replace('?', '.').replace('!', '.')
    clean_text = clean_text.decode('utf8')
    tokens = nltk.word_tokenize(clean_text)
    tokens = filter_tokens(tokens)
    return tokens

def read_corpus():
    corpus = {}
    titles = []
    for fname in sorted(glob.glob('summary/*.txt')):
        with open(fname, 'r') as f_in:
            title = f_in.readline().replace('\n', '')
            text = ''.join(f_in.readlines())
        #title = os.path.basename(fname)[:-4]
        titles.append(title)
        #count = Counter(tokens)
        corpus[title] = text
    titles = np.array(titles)
    return corpus, titles

def get_all_tokens(corpus, titles):
    tokens = set()
    for title in titles:
        for token in tokenize(corpus[title]):
            tokens.add(token)
    return list(tokens)

def tokens_to_word2vec(tokens, model=None):
    if model is None:
        model = Word2Vec.load_word2vec_format('../word2vec/GoogleNews-vectors-negative300.bin', binary=True)
    word_to_vector = {}
    for word in tokens:
        try:
            arr = model[word]
        except KeyError:
            continue
        word_to_vector[word] = arr
    return word_to_vector

def gen_word2vec_dict(save_file=None, model=None):
    """Generates end-to-end word2vec vectors for each word in the corpus."""
    corpus, titles = read_corpus()
    tokens = get_all_tokens(corpus, titles)
    word_to_vector = tokens_to_word2vec(tokens, model)
    if save_file:
        with open(save_file, 'w') as f:
            pickle.dump(word_to_vector, f)
    return word_to_vector

def input_to_tokens(inp=None, ings=None):
    """Generates end-to-end word tokens from input index (of ingredients)."""
    if type(inp) == int:
        inp = [inp]
    all_tokens = []
    if ings is None:
        ings = get_ings(5000)
    if inp is None:
        inp = range(len(ings))
    ing_to_title = get_ing_to_title()
    for i in inp:
        if not ings[i]:
            all_tokens.append([])
            continue
        title = ing_to_title[ings[i]][0]
        if not title:
            all_tokens.append([])
        else:
            with open(os.path.join(
                DIRNAME, 'summary/{}.txt'.format(clean_title(title))), 'r') as f_in:
                f_in.readline()
                text = ''.join(f_in.readlines())
                tokens = tokenize(text)
                all_tokens.append(tokens)
    return all_tokens

def gen_inputs_to_outputs_adulterants(adulterant_cat_pair_map, save_file='input_to_outputs_adulterants.pkl'):
    input_max = max([k[0] for k in adulterant_cat_pair_map.keys()])
    output_max = 130 #max([k[1] for k in adulterant_cat_pair_map.keys()])
    input_to_outputs = {i : np.zeros(output_max+1, dtype=int) for i in range(input_max+1)}
    for inp, out in adulterant_cat_pair_map.keys():
        input_to_outputs[inp][out] += 1
    if save_file:
        with open(os.path.join(DIRNAME, save_file), 'w') as f:
            pickle.dump(input_to_outputs, f)
    return input_to_outputs

def gen_inputs_to_outputs(inputs, outputs, save_file='input_to_outputs.pkl'):
    input_to_outputs = {i : np.zeros(outputs.max()+1, dtype=int) for i in range(inputs.max()+1)}
    for inp, out in zip(inputs, outputs):
        input_to_outputs[inp][out] += 1
    if save_file:
        with open(os.path.join(DIRNAME, save_file), 'w') as f:
            pickle.dump(input_to_outputs, f)
    return input_to_outputs

def tf_idf():
    corpus, titles = read_corpus()
    num_docs = len(titles)

    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english', min_df=2)
    tfs = tfidf.fit_transform(corpus.values())
    tfidf_arr = tfs.toarray()
    feature_names = np.array(tfidf.get_feature_names())

    title_to_features = {}
    for title, vec in zip(titles, tfidf_arr):
        is_non_zero = vec>0
        feat_ranks = np.argsort(vec)[::-1]
        title_to_features[title] = feature_names[[i for i in feat_ranks if is_non_zero[i]]]
        #print title, feature_names[feat_ranks[:10]]

    k = 10
    neigh = NearestNeighbors(k, algorithm='brute', metric='cosine')
    neigh.fit(tfidf_arr)
    kneighbors_scores, kneighbors_indices = neigh.kneighbors(tfidf_arr)

    assert len(kneighbors_scores)==len(kneighbors_indices)==num_docs
    for i in range(num_docs):
        if kneighbors_indices[i,0] != i:
            print "Not the first:", titles[i], titles[kneighbors_indices[i,0]]
        print '{} --> {}'.format(titles[i], titles[kneighbors_indices[i,1:4]])
        #f.write('{} --> {}\n'.format(titles[i], titles[kneighbors_indices[i,1:4]]))



