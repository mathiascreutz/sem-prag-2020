import math
import matplotlib.pyplot as plt
import numpy as np
import operator
import plot_utils
import random

from IPython.display import display, Math, Latex
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

M, mapping = plot_utils.get_embeddings()

def dot(v, w):
    # Make sure numpy arrays
    v = np.array(v)
    w = np.array(w)
    return np.sum(v*w)

def to_feature_matrix(words):
    return np.concatenate([embed(w).reshape(1, -1) for w in words])

def l2norm(v):
    # Make sure numpy array
    v = np.array(v)
    return np.sqrt(np.sum(v**2))

def cosine_similarity(v, w):
    return dot(v, w)/(l2norm(v)*l2norm(w))

def embed(w):
    if w not in mapping:
        raise ValueError("Unfortunately '" + w + "' is not in the vocabulary.")
    return M[mapping[w]]

def get_pairwise_similarities():
    n = len(mapping)
    sims = np.matmul(M, M.transpose())
    return sims

# pairwise = get_pairwise_similarities()

def get_mapping():
    return mapping

def get_words():
    return mapping.keys()

def get_top(similarities, n, mode):
    if mode == "best":
        return sorted(similarities, key=operator.itemgetter(1), reverse=True)[:n]
    elif mode == "worst":
        return sorted(similarities, key=operator.itemgetter(1))[:n]
    else:
        raise ValueError("Argument 'mode' should be 'best' or 'worst'.")
        

def plot_w2v_2d(words):
    """Plot word2vec embeddings in 2d."""
    pc_1, pc_2 = plot_utils.get_principal_comps(M, 2)
    
    words = [w for w in words if w in mapping]
    indices = [mapping[w] for w in words]
    
    xs = [pc_1[i] for i in indices]
    ys = [pc_2[i] for i in indices]
    
    plt.figure()
    plt.plot(xs, ys, marker="o", linestyle="None")
    
    [plot_utils.point_label(word, xs[i], ys[i]) for i, word in enumerate(words)]
        
    plt.show()
    
def plot_w2v_3d(words):
    pc_1, pc_2, pc_3 = plot_utils.get_principal_comps(M, 3)
    
    words = [w for w in words if w in mapping]
    indices = [mapping[w] for w in words]
    
    xs = [pc_1[i] for i in indices]
    ys = [pc_2[i] for i in indices]
    zs = [pc_3[i] for i in indices]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs, ys, zs, marker="o", linestyle="None")
    
    [plot_utils.point_label_3d(ax, word, xs[i], ys[i], zs[i]) for i, word in enumerate(words)]
    
    plt.show()
    
def plot_dendrogram(model, labels):
    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    
    labels_concat = ["/".join(tup) for tup in zip(labels, [str(l) for l in model.labels_])]
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, orientation="left", labels=labels_concat)
    
    plt.show()
    
def plot_kmeans(model, words, plot_text=True, small_points=False):
    clusters = max(model.labels_) + 1
    
    pc_1, pc_2 = plot_utils.get_principal_comps(M, 2)
    
    fig = plt.figure()
    for c in range(clusters):
        tmp = [w for w, cluster_id in zip(words, model.labels_) if cluster_id == c]
        indices = [mapping[w] for w in tmp]
        xs = [pc_1[i] for i in indices]
        ys = [pc_2[i] for i in indices]
        
        if small_points:
            plt.plot(xs, ys, marker="o", markersize=0.5, linestyle="None", label="Cluster " + str(c))
        else:
            plt.plot(xs, ys, marker="o", linestyle="None", label="Cluster " + str(c))
    
        if plot_text:
            [plot_utils.point_label(tmp[i], xs[i], ys[i]) for i, word in enumerate(tmp)]
            
        if clusters < 14:
            if small_points:
                plt.legend(markerscale=7.)
            else:
                plt.legend()
        
        
    plt.show()
    
def plot_kmeans(model, words, plot_text=True, small_points=False, mapp=None, embs=None):
    clusters = max(model.labels_) + 1
    
    if embs is not None:
        pc_1, pc_2 = plot_utils.get_principal_comps(embs, 2)
    else:
        pc_1, pc_2 = plot_utils.get_principal_comps(M, 2)
    
    fig = plt.figure()
    for c in range(clusters):
        tmp = [w for w, cluster_id in zip(words, model.labels_) if cluster_id == c]
        
        if mapp:
            indices = [mapp[w] for w in tmp]
        else:
            indices = [mapping[w] for w in tmp]
            
        xs = [pc_1[i] for i in indices]
        ys = [pc_2[i] for i in indices]
        
        if small_points:
            plt.plot(xs, ys, marker="o", markersize=0.5, linestyle="None", label="Cluster " + str(c))
        else:
            plt.plot(xs, ys, marker="o", linestyle="None", label="Cluster " + str(c))
    
        if plot_text:
            [plot_utils.point_label(tmp[i], xs[i], ys[i]) for i, word in enumerate(tmp)]
            
        if clusters < 14:
            if small_points:
                plt.legend(markerscale=7.)
            else:
                plt.legend()
        
        
    plt.show()
    
def get_by_dimension(dim, n):
    itow = {v:k for k, v in mapping.items()}
    
    best_indices = M[:,dim].argsort()[-n:][::-1]
    
    return [itow[i] for i in best_indices]

def to_dict(l):
    return {k:v for k, v in l}

def tabulate_similarities(words_and_feats):
    word_dict = to_dict(words_and_feats)
    latex = "\\textbf{Pairwise similarities} \\\\\n"
    latex += "\\begin{array}{" + "c|" + "c"*len(word_dict) + "}\n& "

    for word in word_dict.keys():
        latex += "%s & " % word
    latex = latex[:-1] + r" \\\hline" + "\n"

    for word1 in word_dict.keys():
        latex += word1
        for word2 in word_dict.keys():
            latex += r" & %.4f" % cosine_similarity(word_dict[word1], word_dict[word2])
        latex += r" \\" + "\n"

    latex += "\n\\end{array}"
    
    display(Math(latex))
    
def tabulate_angles(words_and_feats):
    word_dict = to_dict(words_and_feats)
    latex = "\\textbf{Pairwise angles} \\\\\n"
    latex += "\\begin{array}{" + "c|" + "c"*len(word_dict) + "}\n& "

    for word in word_dict.keys():
        latex += "%s & " % word
    
    latex = latex[:-1] + r" \\\hline" + "\n"

    for word1 in word_dict.keys():
        latex += word1
        for word2 in word_dict.keys():
            latex += r" & %.1f°" % math.degrees(np.arccos(round(cosine_similarity(word_dict[word1], word_dict[word2]), 10)))
        latex += r" \\" + "\n"

    latex += "\n\\end{array}"
    display(Math(latex))
    

def get_n(vec, n, mode):
    words = get_words()
    
    similarities = []
    for w in words:
        sim = cosine_similarity(embed(w), vec)
        similarities.append((w, sim))

    return get_top(similarities, n, mode)  

def sample_clusters(model, words, n, clusters):
    cluster_to_words = {c:[] for c in range(max(model.labels_) + 1)}
    [cluster_to_words[cluster_id].append(w) for w, cluster_id in zip(words, model.labels_)]
    
    for c in clusters:
        print("Cluster %d:" % c)
        print(", ".join(random.sample(cluster_to_words[c], min(n, len(cluster_to_words[c])))), end="\n\n")
        
def get_clusters_at_cutoff(model, words, cutoff):
    children = model.children_[:cutoff + 1]
    clusters = {i:[w] for i, w in enumerate(words)}
    
    tmp = len(words)
    for c1, c2 in children:
        if tmp not in clusters:
            clusters[tmp] = []
        
        for c in [c1, c2]:
            if c < len(words):
                clusters[tmp].append(words[c])
            else:
                clusters[tmp] += clusters[c]
        
        clusters[c1] = None
        clusters[c2] = None
        
        tmp +=1
        
    clusters = [w for k, w in clusters.items() if w]
    
    for i, c in enumerate(clusters):
        print("Cluster %d: %s" % (i, ", ".join(c)))