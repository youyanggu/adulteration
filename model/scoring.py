import os
import pickle

import numpy as np
import pandas as pd

DIRNAME = os.path.dirname(__file__)

def gen_avg_true_results(valid_ing_indices):
    # Another baseline is to use the average Y distribution.
    fname = os.path.join(DIRNAME, '../wikipedia/input_to_outputs.pkl')
    with open(fname, 'r') as f_in:
        input_to_outputs = pickle.load(f_in)

    avg_true_results = None
    for i in valid_ing_indices:
        if avg_true_results is None:
            avg_true_results = input_to_outputs[i] * 1. / input_to_outputs[i].sum()
        else:
            avg_true_results += input_to_outputs[i] * 1. / input_to_outputs[i].sum()
    avg_true_results /= len(valid_ing_indices)
    assert np.isclose(avg_true_results.sum(), 1, atol=1e-5)
    avg_true_results = np.array([avg_true_results for i in valid_ing_indices])
    return avg_true_results

def evaluate_map(valid_ing_indices, results, ing_cat_pair_map, random=False):
    """Evaluation metric via the mean average precision."""
    # Match prec: 0.448 vs 0.136 for random
    # MAP: 0.497 vs 0.166 for random
    if not random:
        ranks = np.fliplr(np.argsort(results))
    else:
        ranks = np.array([np.random.permutation(
            results.shape[1]) for i in range(results.shape[0])])
    precisions = []
    match_percs = []
    for i, rank in enumerate(ranks):
        ing_idx = valid_ing_indices[i]

        cats = set()
        for j in range(results.shape[1]):
            if (ing_idx, j) in ing_cat_pair_map:
                cats.add(j)
        num_cats = len(cats)

        c_ranks = sorted([np.where(rank==c)[0][0] for c in cats])
        if len(c_ranks) == 0:
            continue
        mean_precision = 0
        for j, c_rank in enumerate(c_ranks):
            mean_precision += (j+1.)/(c_rank+1)
        mean_precision /= len(c_ranks)
        precisions.append(mean_precision)

        matches = 0
        for cat_rank, cat_idx in enumerate(rank[:num_cats]):
            cat_rank += 1
            if (ing_idx, cat_idx) in ing_cat_pair_map:
                matches += 1
        match_perc = matches * 1. / num_cats
        match_percs.append(match_perc)
    precisions = np.array(precisions)
    match_percs = np.array(match_percs)
    print "MAP    :", precisions.mean()
    print "Match %:", match_percs.mean()


def calc_avg_rank_of_ing_cat(ranks, score_dir='data'):
    assert(ranks.shape[0]==ranks.shape[1])
    df = get_ing_category(score_dir)
    n = len(df)
    ranks = ranks[:n, :n]
    categories = df['category'].values

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
            #print "Warning: No other ings in category {} for ing {}.".format(cur_cat, idx)
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

def get_scores(score_dir='data'):
    df = pd.read_csv(score_dir+'/scores.csv', header=0, index_col=0)
    df = df.ix[1:]
    df.drop('Unnamed: 1', axis=1, inplace=True)
    total = df['total']
    df.drop('total', axis=1, inplace=True)
    df = df.fillna(0)
    df = df.replace({'X' : 0}).astype(int)
    scores = df.as_matrix().T + df.as_matrix()
    return scores

def calc_score(ranks, total_ings, print_scores=True, score_dir='data'):
    scores = get_scores(score_dir)
    n = scores.shape[0]
    ranks_ = ranks[:n, :n]
    s = np.sum(scores, axis=1, dtype=float)
    actual_scores = np.sum(ranks_*scores, axis=1)

    highest_ranks = calc_highest_ranks(ranks_, scores)

    scores_sorted = np.fliplr(np.sort(scores, axis=1))
    #perfect_rankings = np.array([np.arange(1,n+1) for i in range(n)])
    
    avg_rankings = np.sum(scores*ranks_, axis=1) / s
    avg_rank_of_ing_cat = calc_avg_rank_of_ing_cat(ranks, score_dir)
    #perfect_avg_rankings = np.sum(scores_sorted*perfect_rankings, axis=1) / s
    random_avg_rankings = np.sum(scores*gen_random_ranks(n, total_ings), axis=1) / s

    if print_scores:
        print "Scores"
        print "=========="
        print "% found in top 3 :", (highest_ranks<=3).sum(dtype=float) / np.isfinite(highest_ranks).sum()
        print "Avg highest rank :", highest_ranks[np.isfinite(highest_ranks)].mean()
        print "Avg rank         :", avg_rankings[np.isfinite(avg_rankings)].mean()
        #print perfect_avg_rankings[np.isfinite(perfect_avg_rankings)].mean()
        print "Avg rank of cat  :", avg_rank_of_ing_cat
        print "Random rank      :", random_avg_rankings[np.isfinite(random_avg_rankings)].mean()    

    return highest_ranks, avg_rankings, avg_rank_of_ing_cat, random_avg_rankings


def get_ing_category(score_dir='data'):
    df = pd.read_csv(score_dir+'/scores2.csv', header=0)
    df = df.ix[:,:2]
    df = df[df['category'].notnull()]
    df['category'] = df['category'].astype(int)
    df = df.replace({df['category'].max() : sorted(df['category'].unique())[-2]+1})
    return df

def predict_ing_category(
    ing_list, ings, embeddings, neigh, true_cats_train, top_n, output_str=False):
    def retrieve_cat(nn_cats):
        nn_cats = np.array([i for i in nn_cats if i not in [11]])
        nn_cats = nn_cats[:top_n]
        mode = np.argmax(np.bincount(nn_cats))
        return mode
    if type(ings) == str:
        ings = [ings]
    indices = np.array([np.where(ing_list==i)[0][0] for i in ings])
    embeddings = embeddings[indices]
    nn_values, nn_idx = neigh.kneighbors(embeddings)
    predict_cats = []
    for i, v in enumerate(indices):
        nn_cats = np.array([true_cats_train[j] for j in nn_idx[i]])
        predict_cat = retrieve_cat(nn_cats)
        predict_cats.append(predict_cat)
    predict_cats = np.array(predict_cats)
    if output_str:
        categories = np.array(['non-labeled', 'fruit/nuts', 'vegetables', 
                  'meat/fish', 'grain', 'dairy', 'vitamin', 'flavor/color', 
                  'additive', 'seasoning', 'oil', 'other'])
        return categories[predict_cats]
    return predict_cats

def run_predict_ing_category(
    ing_list, embeddings, top_n=8, train_perc=0.9, print_res=False):
    def calc_accuracy(true_cats, predict_cats):
        indices = np.array([i for i,v in enumerate(true_cats) if v != 11])
        acc = (true_cats[indices] == predict_cats[indices]).sum()*1./len(indices)
        return acc

    categories = np.array(['non-labeled', 'fruit/nuts', 'vegetables', 
                  'meat/fish', 'grain', 'dairy', 'vitamin', 'flavor/color', 
                  'additive', 'seasoning', 'oil', 'other'])
    num_ingredients = len(embeddings)
    df_score = get_ing_category()
    true_cats = df_score.category.values
    train_indices = np.random.choice(
        num_ingredients, int(train_perc*num_ingredients), replace=False)
    ing_list_train = ing_list[train_indices]
    true_cats_train = true_cats[train_indices]
    test_indices = np.setdiff1d(np.arange(num_ingredients), train_indices)
    train_embeddings = embeddings[train_indices]
    ranks, neigh = get_nearest_neighbors(train_embeddings)

    predict_cats = predict_ing_category(
        ing_list, ing_list[test_indices], embeddings, neigh, true_cats_train, top_n)
    acc = calc_accuracy(true_cats[test_indices], predict_cats)
    print acc
    if print_res:
        print '{0:<40} | {1:<12} | {2:<12}'.format('Ingredient', 'Predicted', 'Actual')
        for i,v in enumerate(test_indices):
            if true_cats[v] != 11:
                print '{0:<40} | {1:<12} | {2:<12}'.format(
                    ing_list[v], categories[predict_cats[i]], categories[true_cats[v]])
    return acc




