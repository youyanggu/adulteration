import numpy as np
import pandas as pd

def calc_avg_rank_of_ing_cat(ranks, score_path='data/scores2.csv'):
    assert(ranks.shape[0]==ranks.shape[1])
    df = get_ing_category(score_path)
    n = len(df)
    ranks = ranks[:n, :n]
    categories = df['Category'].values

    mean_ranks = []
    for idx, row_ranks in enumerate(ranks):
        cur_idx = np.where(row_ranks==0)[0][0]
        cur_cat = categories[cur_idx]
        if cur_cat == categories.max():
            # Uncategorized.
            continue
        ings_same_cat = np.where(categories==cur_cat)[0]
        ings_same_cat = ings_same_cat[ings_same_cat<ranks.shape[1]]
        ings_same_cat = ings_same_cat[ings_same_cat!=idx]
        if len(ings_same_cat)==0:
            print "Warning: No other ings in category {} for ing {}.".format(cur_cat, idx)
            continue
        ranks_same_cat = row_ranks[ings_same_cat]
        mean_ranks.append(ranks_same_cat.mean())
    return np.array(mean_ranks).mean()

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

def calc_random_avg_rank(num_ings, iters=10000):
    avg_num_annotated = 7
    perc_in_top_3 = np.array([np.random.choice(np.arange(1,num_ings), 
        avg_num_annotated).min()<=3 for i in range(iters)]).mean()
    avg_best_rank = np.array([np.random.choice(np.arange(1,num_ings), 
        avg_num_annotated).min() for i in range(iters)]).mean()
    avg_rank = np.array([np.random.choice(np.arange(1,num_ings), 
        avg_num_annotated).mean() for i in range(iters)]).mean()
    return perc_in_top_3, avg_best_rank, avg_rank

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
    avg_rank_of_ing_cat = calc_avg_rank_of_ing_cat(ranks)
    #perfect_avg_rankings = np.sum(scores_sorted*perfect_rankings, axis=1) / s
    random_avg_rankings = np.sum(scores*gen_random_ranks(n, total_ings), axis=1) / s

    if print_scores:
        print "Scores"
        print "=========="
        print "% found in top 3 :", (highest_ranks<=3).sum(dtype=float) / np.isfinite(highest_ranks).sum()
        print "Avg highest rank :"highest_ranks[np.isfinite(highest_ranks)].mean()
        print "Avg rank         :", avg_rankings[np.isfinite(avg_rankings)].mean()
        #print perfect_avg_rankings[np.isfinite(perfect_avg_rankings)].mean()
        print "Avg rank of cat  :", avg_rank_of_ing_cat
        print "Random rank      :", random_avg_rankings[np.isfinite(random_avg_rankings)].mean()    

    return highest_ranks, avg_rankings, avg_rank_of_ing_cat, random_avg_rankings


def get_ing_category(score_path='data/scores2.csv'):
    df = pd.read_csv(score_path, header=0)
    df = df.ix[:,:2]
    df = df[df['Category'].notnull()]
    df['Category'] = df['Category'].astype(int)
    df = df.replace({df['Category'].max() : sorted(df['Category'].unique())[-2]+1})
    return df
