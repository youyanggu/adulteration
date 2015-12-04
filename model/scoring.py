import numpy as np
import pandas as pd

def calc_highest_ranks(ranks, scores):
    highest_ranks = []
    for i, v in enumerate(scores):
        score_indices = np.where(v>0)[0]
        if len(score_indices) > 0:
            highest_rank = ranks[i][score_indices].min()
            highest_ranks.append(highest_rank)
        else:
            highest_ranks.append(np.nan)
    highest_ranks = np.array(highest_ranks)
    return highest_ranks

def gen_random_ranks(n, total_ings):
    random_ranks = []
    range_minus_0 = np.arange(1, total_ings)
    for i in range(n):
        perm = np.random.choice(range_minus_0, n-1, replace=False)
        random_ranks.append(np.insert(perm, i, 0))
    return np.vstack(random_ranks)

def get_scores(score_path='data/scores.csv'):
    df = pd.read_csv(score_path, header=0, index_col=0)
    df = df.ix[1:]
    df.drop('Unnamed: 1', axis=1, inplace=True)
    total = df['total']
    df.drop('total', axis=1, inplace=True)
    df = df.fillna(0)
    df = df.replace({'X' : 0}).astype(int)
    scores = df.as_matrix().T + df.as_matrix()
    return scores

def calc_score(ranks, total_ings, print_scores=True, score_path='data/scores.csv'):
    scores = get_scores(score_path)
    n = scores.shape[0]
    ranks = ranks[:n, :n]
    s = np.sum(scores, axis=1, dtype=float)
    actual_scores = np.sum(ranks*scores, axis=1)

    highest_ranks = calc_highest_ranks(ranks, scores)

    scores_sorted = np.fliplr(np.sort(scores, axis=1))
    #perfect_rankings = np.array([np.arange(1,n+1) for i in range(n)])
    
    avg_rankings = np.sum(scores*ranks, axis=1) / s
    #perfect_avg_rankings = np.sum(scores_sorted*perfect_rankings, axis=1) / s
    random_avg_rankings = np.sum(scores*gen_random_ranks(n, total_ings), axis=1) / s

    if print_scores:
        print "Scores"
        print "=========="
        print (highest_ranks<=3).sum(dtype=float) / np.isfinite(highest_ranks).sum()
        print highest_ranks[np.isfinite(highest_ranks)].mean()
        print avg_rankings[np.isfinite(avg_rankings)].mean()
        #print perfect_avg_rankings[np.isfinite(perfect_avg_rankings)].mean()
        print random_avg_rankings[np.isfinite(random_avg_rankings)].mean()

    return highest_ranks, avg_rankings, random_avg_rankings


def get_ing_category(score_path='data/scores2.csv'):
    df = pd.read_csv(score_path, header=0)
    df = df.ix[:,:2]
    df = df[df['Category'].notnull()]
    df['Category'] = df['Category'].astype(int)
    df = df.replace({df['Category'].max() : sorted(df['Category'].unique())[-2]+1})
    return df
