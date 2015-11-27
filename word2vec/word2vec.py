import sys

from gensim.models import Word2Vec

sys.path.append('../model')
from gather_data import import_data
from embeddings2 import get_nearest_neighbors, print_nearest_neighbors
from scoring import calc_score

model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

df, df_i = import_data()
counts = df_i['ingredient'].value_counts()
ings = counts.index.values
limit = 120 # number of ings to retrieve

found_ings = []
found_embeddings = []
for ing in ings[:limit]:
    ing_ = ing.replace(' ', '_')
    print ing_ in model, ing
    if ing_ in model:
        found_ings.append(ing)
        found_embeddings.append(model[ing_])
    else:
        # Assume zeros. Necessary for scoring
        found_ings.append(ing)
        found_embeddings.append(np.zeros(model.vector_size))
found_ings = np.array(found_ings)
found_embeddings = np.array(found_embeddings)
assert(len(found_ings)==len(found_embeddings))

ranks = get_nearest_neighbors(found_embeddings)
print_nearest_neighbors(found_ings, ranks)
score, perfect_score, random_score = calc_score(ranks, len(found_ings))