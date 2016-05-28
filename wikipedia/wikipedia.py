# coding=utf-8
import glob
import json
import nltk
import numpy as np
import os
import pandas as pd
import pickle
import re
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
sys.path.append(os.path.join(DIRNAME, '../model'))
from gen_embeddings import get_nearest_neighbors, print_nearest_neighbors

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

def get_adulterants(get_all=False):
    """Get names of adulterants.

    If get_all is False, returns only adulterants with product categories.
    """
    adulterants = np.load(os.path.join(DIRNAME, '../rasff/all_rasff_chems.npy'))
    if get_all:
        return adulterants
    with open(os.path.join(DIRNAME, 'input_to_outputs_adulterants.pkl'), 'r') as f_in:
        input_to_outputs = pickle.load(f_in)
    assert len(adulterants) == len(input_to_outputs)
    adulterants = [v for i,v in enumerate(adulterants) if input_to_outputs[i].sum()>0]
    return adulterants

def get_ings(num_ingredients=5000):
    df_i = pd.read_hdf(
        os.path.join(DIRNAME, '../foodessentials/ingredients.h5'), 'ingredients')
    counts = df_i['ingredient'].value_counts()
    if num_ingredients:
        return counts.index.values[:num_ingredients]
    else:
        return counts.index.values

def get_products():
    with open('../ncim/idx_to_cat.pkl', 'rb') as f_in:
        idx_to_cat = pickle.load(f_in)
    return [idx_to_cat[i] for i in sorted(idx_to_cat.keys())]

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
    if ings is not None and len(ings) == 131:
        # Products
        return get_product_to_title()
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

def get_ings_wiki_links():
    ings_wiki_links = pd.read_csv(os.path.join(
        DIRNAME, 'ings_wiki_links.csv'), header=None, index_col=0).to_dict()[1]
    for k,v in ings_wiki_links.iteritems():
        if pd.isnull(v):
            clean_v = ''
        else:
            assert 'https://en.wikipedia.org/wiki/' in v
            clean_v = requests.utils.unquote(v)
            clean_v = clean_v.replace('https://en.wikipedia.org/wiki/', '')
            if '#' in clean_v:
                clean_v = clean_v[:clean_v.index('#')]
        ings_wiki_links[k] = [clean_v]
    ings_wiki_links['jalape\xc3\x91o pepper'] = ings_wiki_links['jalapeno pepper']
    return ings_wiki_links

def get_product_to_title():
    product_to_hit = pd.read_csv(os.path.join(DIRNAME, 'product_to_hit.csv'), 
        header=None, index_col=0).to_dict()[1]
    for k,v in product_to_hit.iteritems():
        product_to_hit[k] = [v.replace('https://en.wikipedia.org/wiki/', '')]
    return product_to_hit

def clean_title(title):
    title = title.strip()
    title = title.replace('/', '-')
    title = title.replace(' ', '_')
    return title

def run_wiki(ings=None, start=0, summary=True, overwrite=False):
    """Download Wikipedia text from title."""
    if ings is not None and len(ings) == 131:
        # product categories
        ing_to_hits = get_product_to_title()
    else:
        ing_to_hits = get_ings_wiki_links() #get_ing_to_title(ings)
    if ings is None:
        ings = sorted(ing_to_hits.keys())
    seen_titles = set()
    for i, ing in enumerate(ings):
        if i < start:
            continue
        title = ing_to_hits[ing][0]
        #print i, ing, title
        if not title or title in seen_titles:
            continue
        seen_titles.add(title)
        file_title = clean_title(title)
        if summary:
            fname = 'summary/{}.txt'.format(file_title)
        else:
            fname = 'pages/{}.txt'.format(file_title)
        fname = os.path.join(DIRNAME, fname)
        if os.path.isfile(fname) and not overwrite:
            #print "File already exists for: {}. overwrite is False.".format(fname)
            continue
        if summary:
            html = get_summary(title)
        else:
            html = get_page_html(title)
        soup = BeautifulSoup(html, 'lxml')
        raw_text = soup.text
        with open(fname, 'wb') as f:
            print "Writing:", title
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

def read_corpus(wanted_titles=None):
    corpus = {}
    titles = []
    assert os.path.isdir('summary')
    for fname in sorted(glob.glob(os.path.join(DIRNAME, 'summary/*.txt'))):
        with open(fname, 'r') as f_in:
            title = f_in.readline().replace('\n', '')
            text = ''.join(f_in.readlines())
        if wanted_titles:
            file_title = os.path.basename(fname)[:-4]
            file_title = file_title.replace('e\xcc\x81', '\xc3\xa9')  # Tomato_puree
            file_title = file_title.replace('n\xcc\x83', '\xc3\xb1') # Jalapeno
            file_title = file_title.replace('c\xcc\xa7ai\xcc\x81', '\xc3\xa7a\xc3\xad') # Acai_palm
            file_title = file_title.replace('a\xcc\x82', '\xc3\xa2') # Neufchâtel_cheese
            file_title = file_title.replace('e\xcc\x82', '\xc3\xaa') # Crepe
            #if file_title.endswith('pe'):
            #    print file_title, file_title in wanted_titles
            if file_title not in wanted_titles:
                continue
            titles.append(file_title)
            corpus[file_title] = text
        else:
            titles.append(title)
            corpus[title] = text
        #count = Counter(tokens)
    titles = np.array(titles)
    return corpus, titles

def get_all_tokens(corpus, titles):
    tokens = set()
    for title in titles:
        for token in tokenize(corpus[title]):
            tokens.add(token)
    return list(tokens)

def tokens_to_word2vec(tokens, model):
    if model == 'word2vec':
        model = Word2Vec.load_word2vec_format(
            os.path.join(DIRNAME, '../word2vec/GoogleNews-vectors-negative300.bin'), binary=True)
    elif model == 'glove':
        word_to_vector_glove = {}
        tokens_glove = set(tokens)
        #with open(os.path.join(DIRNAME, '../glove/glove.6B/glove.6B.300d.txt'), 'r') as f:
        with open(os.path.join(DIRNAME, '../glove/glove.42B.300d.txt'), 'r') as f:
            for line in f:
                split_index = line.index(' ')
                word = line[:split_index]
                vector = np.fromstring(line[split_index+1:], dtype=float, sep=' ')
                assert len(vector)==300
                if word == '.':
                    word = '</s>'
                if word in tokens_glove:
                    word_to_vector_glove[word] = vector
        return word_to_vector_glove
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
    if len(ings) == 131:
        ing_to_title = get_ing_to_title(ings)
    else:
        ing_to_title = get_ings_wiki_links() #get_ing_to_title()
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

def nearest_neighbors_vectors(fname, out_fname):
    """Print out the nearest neighbors of each word embedding given a dictionary 
    of word to embedding."""
    top_n=3
    with open(fname, 'r') as f:
        word_to_vector = pickle.load(f)
    ing_names = np.array(sorted(word_to_vector.keys()))
    vectors = np.array([word_to_vector[i] for i in ing_names])
    ranks, neigh = get_nearest_neighbors(vectors, k=top_n)
    print_nearest_neighbors(ing_names, ranks, top_n=3, fname=out_fname, argsort=False)


def gen_inputs_to_outputs_adulterants(adulterant_cat_pair_map, save_file='input_to_outputs_adulterants.pkl'):
    """Generates map of input index to vector of food categories with their counts.
    
    adulterat_cat_pair_map is a dict where the keys are (input index, category index) = True. It is 
    generated by hier_to_cat.gen_adulterant_cat_pair_map.

    Note that this is slightly different than in training since we assign 1 if it has occured and 0 otherwise, 
    rather than add 1 for each incidence.
    """
    input_max = max([k[0] for k in adulterant_cat_pair_map.keys()])
    output_max = 131 #max([k[1] for k in adulterant_cat_pair_map.keys()])
    input_to_outputs = {i : np.zeros(output_max, dtype=int) for i in range(input_max+1)}
    for k,v in adulterant_cat_pair_map.iteritems():
        inp, out = k
        input_to_outputs[inp][out] += v
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
    def tf_idf_tokenize(text):
        clean_text = text.replace('?', '.').replace('!', '.')
        #clean_text = clean_text.decode('utf8')
        tokens = nltk.word_tokenize(clean_text)
        tokens = filter_tokens(tokens)
        return tokens

    ings = get_ings(5000)
    ings_wiki_links = get_ings_wiki_links()
    wanted_titles = [ings_wiki_links[i][0] for i in ings]

    corpus, titles = read_corpus(wanted_titles)
    num_docs = len(titles)

    tfidf = TfidfVectorizer(tokenizer=tf_idf_tokenize, stop_words='english', min_df=2)
    tfs = tfidf.fit_transform([corpus[t] for t in wanted_titles if t])#corpus.values())
    tfidf_arr = tfs.toarray()
    feature_names = np.array(tfidf.get_feature_names())
    num_tokens = len(feature_names)

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



