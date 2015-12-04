import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold

from scoring import get_ing_category

def plot_tsne(X_tsne, labels=None, only_cats=True, indices=None, limit=None):
    #colors = ['b', 'g', 'g', 'g', 'g', 'g', 'r', 'c', 'm', 'y', 'k', 'k']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'gray', 'w', 'firebrick', 'skyblue', 'w']
    categories = ['non-labeled', 'fruit/nuts', 'vegetables', 'meat/fish', 'grain', 'dairy',
                  'vitamin', 'flavor/color', 'additive', 'seasoning',
                  'oil', 'other'] #lime
    cats = get_ing_category()['Category'].values
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
    plt.scatter(X_tsne[:limit,0], X_tsne[:limit,1], marker='o', color=color_labels[:limit])
    if labels is not None:
        for label, x, y in zip(labels[:limit], X_tsne[:limit,0], X_tsne[:limit,1]):
            plt.annotate(label, xy=(x,y), xytext=(0,0), textcoords='offset points', ha='right', va='bottom')
    recs = []
    for i in range(len(colors)):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=colors[i]))
    plt.legend(recs, categories, loc=1)
    plt.title('t-SNE plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show(block=False)

def tsne(embeddings, plot=False):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(embeddings)
    if plot:
        plot_tsne(X_tsne)
    return X_tsne