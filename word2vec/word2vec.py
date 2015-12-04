import sys

from gensim.models import Word2Vec
import numpy as np

sys.path.append('../model')
from gather_data import import_data
from gen_embeddings import get_nearest_neighbors
from scoring import calc_score

#model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

def retrieve_embeddings(model, ings, limit=120):
    """Get embeddings from restricted ing list."""
    embeddings = []
    found_ings = []
    for i, ing in enumerate(ings[:limit]):
        ing_ = ing.replace(' ', '_')
        #print ing_ in model, ing
        if ing_ in model:
            found_ings.append(i)
            embeddings.append(model[ing_])
        else:
            # Assume zeros. Necessary for scoring
            embeddings.append(np.zeros(model.vector_size))
    found_ings = np.array(found_ings)
    embeddings = np.array(embeddings)
    np.save('word2vec_embeddings.npy', np.array([found_ings, embeddings]))
    return found_ings, embeddings

def get_most_similar(model, ings, limit=120, topn=3, print_nn=True):
    """Get nearest neighbors from word2vec (unrestricted)"""
    nns = []
    if type(ings)==str:
        ings = [ings]
    for ing in ings[:limit]:
        ing_ = ing.replace(' ', '_')
        if ing_ not in model:
            continue
        ret = model.most_similar(ing_, topn=topn+1)
        nn = [i[0].encode('ascii', 'ignore') for i in ret]
        nn = [i for i in nn if i.lower() != ing_][:3]
        nns.append(nn)
        if print_nn:
            print '{} --> {}'.format(ing, nn)
    if not print_nn:
        return nns

def print_nearest_neighbors(ing_names, found_ings, ranks, top_n=3):
    found_ings = set(found_ings)
    for i in range(ranks.shape[0]):
        if i not in found_ings:
            continue
        nearest_neighbors = np.argsort(ranks[i])
        print '{} --> {}'.format(ing_names[i], ing_names[nearest_neighbors[1:top_n+1]])

def get_most_similar_restricted(limit=120):
    df, df_i = import_data()
    counts = df_i['ingredient'].value_counts()
    ings = counts.index.values

    #found_ings, embeddings = retrieve_embeddings(model, ings)
    found_ings, embeddings = np.load('word2vec_embeddings.npy')
    ranks = get_nearest_neighbors(embeddings)
    print_nearest_neighbors(ings[:limit], found_ings, ranks)
    highest_ranks, avg_rankings, random_avg_rankings = calc_score(ranks, limit, 
        print_scores=False, score_path='../model/scores.csv')

    indices = found_ings[found_ings<highest_ranks.shape[0]]
    highest_ranks = highest_ranks[indices]
    avg_rankings = avg_rankings[indices]
    random_avg_rankings = random_avg_rankings[indices]
    print (highest_ranks<=3).sum(dtype=float) / np.isfinite(highest_ranks).sum()
    print highest_ranks[np.isfinite(highest_ranks)].mean()
    print avg_rankings[np.isfinite(avg_rankings)].mean()
    print random_avg_rankings[np.isfinite(random_avg_rankings)].mean()

