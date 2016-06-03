import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold

from scoring import get_ing_category

def plot_tsne(X_tsne, labels=None, only_cats=True, indices=None, limit=None):
    #colors = ['b', 'g', 'g', 'g', 'g', 'g', 'r', 'c', 'm', 'y', 'k', 'k']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'gray', 'lime', 'skyblue', 'firebrick', 'rosybrown']#, 'w']
    categories = ['fruit', 'vegetables', 'meat/fish', 'grain', 'dairy',
                  'vitamin', 'flavor', 'additive', 'seasoning',
                  'oil', 'color', 'nuts']#, 'other'] #lime and gold
    ing_mapping_fname = 'data/scores2.csv'
    """
    ing_mapping_fname = 'data/cat_mapping.csv'
    colors = ['b', 'g', 'g', 'r', 'c', 'm', 'y', 'salmon', 'gray', 'lime', 
              'r', 'skyblue', 'firebrick', 'salmon', 'm', 'rosybrown', 'k']
    categories = ['baby food', 'bakery', 'baking goods', 'bulk grocery', 
                  'cereal & breakfast foods', 'condiments, oils & salad dressings', 
                  'dairy', 'deli', 'diet & nutrition', 'drinks', 
                  'fresh fruit & vegetables', 'frozen food', 
                  'grains, pasta & side dishes', 'meat & seafood', 
                  'sauces, spices & seasonings', 'snacks, cookies & candy', 
                  'soups & canned goods']
    """
    assert len(categories) == len(colors)
    
    cats = get_ing_category(ing_mapping_fname)['category'].values-1 # 0 is non-labeled
    no_other_cat_indices = np.where(cats!=cats.max())[0]
    cats = cats[cats!=cats.max()]
    X_tsne = X_tsne[no_other_cat_indices]
    if only_cats:
        if indices is not None:
            wanted = indices[indices<len(cats)]
            X_tsne = X_tsne[wanted]
            cats = cats[wanted]
        X_tsne = X_tsne[:len(cats)]
    if cats is None:
        color_labels = 'b'
    else:
        color_labels = np.array(colors)[(cats % len(colors))]
        if len(color_labels) < len(X_tsne):
            assert(not only_cats)
            color_labels2 = np.array(['b']*(len(X_tsne)-len(color_labels)))
            color_labels = np.hstack((color_labels, color_labels2))
    
    if limit is None:
        limit = len(X_tsne)
    plt.figure(figsize=(8.5,6.5))
    plt.scatter(X_tsne[:limit,0], X_tsne[:limit,1], marker='o', color=color_labels[:limit])
    if labels is not None:
        for label, x, y in zip(labels[:limit], X_tsne[:limit,0], X_tsne[:limit,1]):
            plt.annotate(label, xy=(x,y), xytext=(0,0), textcoords='offset points', ha='right', va='bottom')
    recs = []
    for i in range(len(colors)):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=colors[i]))
    plt.legend(recs, categories, loc='upper left')
    #plt.title('t-SNE plot')
    plt.xlim([-45,25])
    plt.grid()
    plt.show(block=False)

def tsne(embeddings, plot=False):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(embeddings)
    if plot:
        plot_tsne(X_tsne)
    return X_tsne

def plot_smoothing(x, y_a, y_b, y_c):
    plt.plot(x, y_a, label='Aisle')
    plt.plot(x, y_b, label='Shelf')
    plt.plot(x, y_c, label='Food Category')
    plt.legend(loc='lower left')
    plt.xlabel('log alpha')
    plt.ylabel('accuracy')
    plt.grid()
    plt.show(block=False)


def plot_ing_changes(x, y_v, y_i, y_i_w, n=5):
    plt.plot(x, y_v, label='Valid')
    plt.plot(x, y_i, label='Invalid (random)')
    plt.plot(x, y_i_w, label='Invalid (weighted)')
    plt.axvline(n, color='k', linestyle='--')
    plt.legend(loc='upper left')
    plt.xlabel('# of ingredients')
    plt.ylabel('% predicted as valid')
    plt.grid()
    plt.show(block=False)

