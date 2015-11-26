import numpy as np
import pandas as pd

def gen_random_ranks(n, total_ings):
    random_ranks = []
    range_minus_0 = np.arange(1, total_ings)
    for i in range(n):
        perm = np.random.choice(range_minus_0, n-1, replace=False)
        random_ranks.append(np.insert(perm, i, 0))
    return np.vstack(random_ranks)

def get_scores():
    df = pd.read_csv('scores.csv', header=0, index_col=0)
    df = df.ix[1:]
    df.drop('Unnamed: 1', axis=1, inplace=True)
    total = df['total']
    df.drop('total', axis=1, inplace=True)
    df = df.fillna(0)
    df = df.replace({'X' : 0}).astype(int)
    scores = df.as_matrix().T + df.as_matrix()
    return scores

def calc_score(ranks, total_ings):
    scores = get_scores()
    n = scores.shape[0]
    ranks = ranks[:n, :n]
    s = np.sum(scores, axis=1, dtype=float)
    actual_scores = np.sum(ranks*scores, axis=1)

    scores_sorted = np.fliplr(np.sort(scores, axis=1))
    perfect_rankings = np.array([np.arange(1,n+1) for i in range(n)])
    
    avg_rankings = np.sum(scores*ranks, axis=1) / s
    perfect_avg_rankings = np.sum(scores_sorted*perfect_rankings, axis=1) / s
    random_avg_rankings = np.sum(scores*gen_random_ranks(n, total_ings), axis=1) / s

    return avg_rankings, perfect_avg_rankings, random_avg_rankings