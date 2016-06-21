#!/usr/bin/env python
"""
Author: Youyang Gu (yygu@mit.edu)

Web tool to display likely food category predictions given an 
ingredient/substance. Please refer to the README at: 
https://github.com/youyanggu/adulteration/web/README.md.

"""

import numpy as np
import pandas as pd
import cPickle as pickle
import sys
import theano
import theano.tensor as T

from flask import Flask
from flask import render_template
from flask import request

from nn import NN

app = Flask(__name__)

print "Loading data. Will take 1 minute."
HIER_DIMENSION = 4290 # dimension of the input vector to the neural network
sources = ['SNOMEDCT_US', 'NCI', 'NDFRT', 'MSH'] # use data from these 4 sources
df_conso = pd.read_hdf('mrconso.h5', 'mrconso')
df_hier = pd.read_hdf('mrhier.h5', 'mrhier')
with open('cuis_to_idx.pkl', 'r') as f:
    cuis_to_idx = pickle.load(f)
with open('idx_to_cat.pkl', 'rb') as f:
    idx_to_cat = pickle.load(f)
categories = [idx_to_cat[i] for i in range(len(idx_to_cat))]
aui_to_cui = df_conso.set_index('AUI')['CUI'].to_dict()
aui_to_str = df_conso.set_index('AUI')['STR'].to_dict()
print "Done loading data"

def get_cui_to_hiers(cui):
    """Look up the hierarchy for CUI in the form of AUIs."""
    df_hier_short = df_hier[df_hier['CUI']==cui]
    hiers = df_hier_short[df_hier_short['SAB'].isin(sources)]
    if len(hiers) == 0:
        return []
    cur_auis = hiers['AUI'].values
    rows = hiers['PTR'].str.split('.').values
    hiers_aui = []
    for aui, r in zip(cur_auis, rows):
        hiers_aui.append([j for j in r] + [aui])
    return hiers_aui

def convert_auis(hiers_aui):
    """Convert the hierarchy from AUIs to CUIs and strings."""
    if len(hiers_aui) == 0:
        return [], [], {}
    cuis = []
    strs = []
    cui_to_str = {}
    for path in hiers_aui:
        cuis.append([aui_to_cui[aui] for aui in path])
        strs.append([aui_to_str[aui] for aui in path])
        for aui in path:
           cui_to_str[aui_to_cui[aui]] = aui_to_str[aui]
    return cuis, strs, cui_to_str

def get_hiers_from_cui(cui):
    """Runs pipeline that converts a CUI to a hierarchy."""
    hiers_aui = get_cui_to_hiers(cui)
    return convert_auis(hiers_aui)

def convert_hiers_to_rep(hiers, cuis_to_idx, cuis_to_str):
    """Convert hierarchy into a vector representation."""
    rep_str = set()
    vectors = []
    for path in hiers:
        v = np.zeros(HIER_DIMENSION)
        for cui in path:
            if cui in cuis_to_idx:
                rep_str.add(cuis_to_str[cui])
                v[cuis_to_idx[cui]] = 1
        vectors.append(v)
    rep = np.mean(np.array(vectors), axis=0)
    return rep.astype('float32'), sorted(rep_str)

def load_model():
    """Load the pre-trained neural network model."""
    W_hid = np.load('W_hid.npy')
    b_hid = np.load('b_hid.npy')
    W_out = np.load('W_out.npy')
    b_out = np.load('b_out.npy')
    x = T.fmatrix('x')
    classifier = NN(
        inp=x,
        n_in=HIER_DIMENSION,   # length of hierarchy representation
        m=100,                 # number of hidden layers
        n_out=len(categories), # number of outputs
        W_hid=W_hid,
        b_hid=b_hid,
        W_out=W_out,
        b_out=b_out
    )
    predict_model = theano.function(
        inputs=[x],
        outputs=classifier.outputLayer.p_y_given_x,
    )
    return predict_model

def predict(predict_model, rep):
    """Generates prediction given a hierarchy representation using the 
    neural network."""
    rep = rep.reshape(1,-1)
    result = predict_model(rep)[0]
    category_to_score = [(cat.capitalize(), score) for cat, score in zip(
        categories, result)]
    category_to_score = sorted(category_to_score, key=lambda x:x[1], reverse=True)
    return category_to_score[:20]

@app.route("/")
def home():
    return render_template('main.html')

@app.route("/about", methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/show', methods=['GET'])
def show():
    cui = request.args.get('cui')
    if cui is None:
        return render_template('main.html')
    print "Request:", cui
    if len(cui) != 8 or not cui.startswith('C'):
        ret_str = 'CUI must start with C and contain exactly 8 characters/digits'
        print '{}: {}'.format(ret_str, cui)
        return render_template('main.html', result=None, ret_str=ret_str, cui=cui)
    result = df_conso[df_conso['CUI']==cui]
    if len(result) == 0:
        ret_str = 'CUI not found'
        print '{}: {}'.format(ret_str, cui)
        return render_template('main.html', result=None, ret_str=ret_str, cui=cui)
    hiers, hiers_str, cuis_to_str = get_hiers_from_cui(cui)
    if len(hiers) == 0:
        ret_str = ('A match is found, but either 1) no hierarchy exists or '
        '2) the hierarchy does not exist in our database')
        print '{}: {}'.format(ret_str, cui)
        return render_template('main.html', result=None, ret_str=ret_str, cui=cui)
    rep, rep_str = convert_hiers_to_rep(hiers, cuis_to_idx, cuis_to_str)
    num_nodes = (rep>0).sum()
    if num_nodes == 0:
        ret_str = ('The model is unable to make a prediction from this '
            'hierarchy due to lack of data:')
        print '{}: {}'.format(ret_str, cui)
        return render_template('main.html', result=None, ret_str=ret_str, cui=cui)
    predict_model = load_model()
    category_to_score = predict(predict_model, rep)
    return render_template('main.html', result=hiers_str, cui=cui, 
        category_to_score=category_to_score, num_nodes=num_nodes, rep_str=rep_str)

