import matplotlib.pyplot as plt
import numpy as np
import pickle

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
from IPython.display import display, Math, Latex

def angle(vec1, vec2):
    """ Compute the angle in degrees between two vectors
        (The vectors are represented as lists or tuples that mark the
        end points; it is assumed that the starting point is at the
        origin)
    """
    unit1 = vec1 / np.linalg.norm(vec1)
    unit2 = vec2 / np.linalg.norm(vec2)
    
    dot_product = max(-1.0, min(1.0, np.dot(unit1, unit2)))
    
    return np.arccos(dot_product) * 180.0 / np.pi

def get_vector_color(i):
    """ An ordered range of colors used for the vectors """

    colors = [ "black", "blue", "red", "green",
               "magenta", "cyan", "brown", "yellow" ]
    
    return colors[i]

def print_angles_between_vectors(end_coords):
    """ Print all pairwise angles between a set of vectors """
    for i in range(len(end_coords)):
        c1 = get_vector_color(i).upper()
        for j in range(i + 1, len(end_coords)):
            c2 = get_vector_color(j).upper()
            print("The angle between the {:s} and {:s} vector is {:.1f} degrees."
                      .format(c1, c2, angle(end_coords[i], end_coords[j])))

def plot_vectors_2d(end_coords):
    """ Plot vectors in a 2d plane and count the angles between the vectors """

    if len(end_coords) < 1 or len(end_coords) > 8:
        print("Error: You need to plot at least one vector and at most eight vectors.")
        return
    
    fig = plt.figure()
    
    # Scale axes
    plt.axis([min(0, min(end_x for end_x, _ in end_coords))-1,
              max(0, max(end_x for end_x, _ in end_coords))+1,
              min(0, min(end_y for _, end_y in end_coords))-1,
              max(0, max(end_y for _, end_y in end_coords))+1])
    
    plt.xlabel("x")
    plt.ylabel("y")
    
    plt.grid(linestyle=":")
    
    for i, (end_x, end_y) in enumerate(end_coords):
        plt.arrow(0, 0, end_x, end_y,
                  head_width=0.2, 
                  length_includes_head=True,
                  color=get_vector_color(i))
        
    plt.show()
    
    print_angles_between_vectors(end_coords)

def plot_vectors_3d(end_coords):
    """ Plot vectors in a 3d plane and count the angles between the vectors """

    if len(end_coords) < 1 or len(end_coords) > 8:
        print("Error: You need to plot at least one vector and at most eight vectors.")
        return
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Scale axes
    ax.set_xlim3d(min(0, min(end_x for end_x, _, _ in end_coords))-1,
                  max(0, max(end_x for end_x, _, _ in end_coords))+1)
    ax.set_ylim3d(min(0, min(end_y for _, end_y, _ in end_coords))-1,
                  max(0, max(end_y for _, end_y, _ in end_coords))+1)
    ax.set_zlim3d(min(0, min(end_z for _, _, end_z in end_coords))-1,
                  max(0, max(end_z for _, _, end_z in end_coords))+1)
    
    # Label axes
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    for i, (end_x, end_y, end_z) in enumerate(end_coords):
        ax.quiver(0, 0, 0, end_x, end_y, end_z,
                  arrow_length_ratio=0.2,
                  color=get_vector_color(i))
        
    plt.show()
    
    print_angles_between_vectors(end_coords)

def plot_3d_binary(features, word_feature_tuples, from_zero=False):
    """For plotting words with three binary features."""
    x = [coord[0] for _, coord in word_feature_tuples]
    y = [coord[1] for _, coord in word_feature_tuples]
    z = [coord[2] for _, coord in word_feature_tuples]

    fig = plt.figure()
    # 3D plot
    ax = fig.add_subplot(111, projection='3d')
    
    if from_zero:
        # Set ticks
        ax.xaxis.set_ticks(list(map(lambda x: x/10.0, range(0, 15, 5))))
        ax.yaxis.set_ticks(list(map(lambda x: x/10.0, range(0, 15, 5))))
        ax.zaxis.set_ticks(list(map(lambda x: x/10.0, range(0, 15, 5))))
    else:
        ax.xaxis.set_ticks(range(-1, 2, 1))
        ax.yaxis.set_ticks(range(-1, 2, 1))
        ax.zaxis.set_ticks(range(-1, 2, 1))

    # Set label names
    ax.set_xlabel(features["x"])
    ax.set_ylabel(features["y"])
    ax.set_zlabel(features["z"])

    # Plot points
    ax.plot(x, y, z, marker="o", markersize=0, linestyle="None")
    # Plot arrows
    [ax.quiver(
        0, 0, 0, coord[0], coord[1], coord[2], length=1.0, arrow_length_ratio=0.1
     ) for _, coord in word_feature_tuples]
    
    # Plot words
    [ax.text(coord[0], coord[1], coord[2], word) for word, coord in word_feature_tuples]
    
    plt.show()
    
    
def get_principal_comps(M, n_components):
    """Get 'n' principal components of the matrix M.
    
    M is a matrix with dimensions (vocabulary, features).
    """
    if n_components not in [2, 3]:
        raise ValueError("Argument 'n_components' must be 2 or 3.")
    
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(M)

    comp_1 = pca_result[:,0]
    comp_2 = pca_result[:,1] 
    
    if n_components == 3:
        comp_3 = pca_result[:,2]
        return comp_1, comp_2, comp_3
    else:
        return comp_1, comp_2
        
        
def features_to_matrix(word_feature_tuples):
    """Convert Python list features to a numpy matrix.
    
    'word_feature_tuples' is a Python list of (word, feat_list) tuples.
    """
    feat_arrays = [np.array(feats).reshape(1, -1) for _, feats in word_feature_tuples]
    M = np.concatenate(feat_arrays)
    return M


def arrow(end_x, end_y):
    """Plot an arrow in 2d from (0, 0) to (end_x, end_y)."""
    plt.arrow(
        0, 0, 
        end_x, end_y, 
        head_width=0.1, 
        length_includes_head=True
    )
    
    
def arrow_3d(ax, end_x, end_y, end_z, color="blue", label=""):
    """Plot an arrow in 3d from (0, 0, 0) to (end_x, end_y, end_z).
    
    'ax' is an axis object from matplotlib.
    """
    ax.quiver(
        0, 0, 0,
        end_x, end_y, end_z, 
        arrow_length_ratio=0.1,
        color=color,
        label=label
    )

    
def point_label(word, x, y):
    """Plot label (i.e. word) to (x, y)."""
    plt.text(x, y, word, horizontalalignment='center', verticalalignment='bottom')
    
    
def point_label_3d(ax, word, x, y, z):
    """Plot label (i.e. word) to (x, y, z).
    
    'ax' is an axis object from matplotlib.
    """
    ax.text(x, y, z, word)
    
    
def plot_2d_binary_hd(word_feature_tuples, arrows=True):
    """Visualize high-dimensional embeddings in 2d space.
    
    'word_feature_tuples* is a Python list of (word, feat_list) tuples.
    """
    if arrows:
        msize = 0
    else:
        msize = 5
        
    mat = features_to_matrix(word_feature_tuples)
    pc_1, pc_2 = get_principal_comps(mat, 2)
    
    fig = plt.figure()
    plt.plot(pc_1, pc_2, marker="o", markersize=msize, linestyle="None")
    
    if arrows:
        [arrow(pc_1[i], pc_2[i]) for i in range(len(word_feature_tuples))]
    
    [point_label(word, pc_1[i], pc_2[i]) for i, (word, _) in enumerate(word_feature_tuples)]
    plt.show()
    
    
def plot_3d_binary_hd(word_feature_tuples, arrows=True):
    """Visualize high-dimensional embeddings in 3d space.
    
    'word_feature_tuples' is a Python list of (word, feat_list) tuples.
    """
    if arrows:
        msize = 0
    else:
        msize = 5
    mat = features_to_matrix(word_feature_tuples)
    pc_1, pc_2, pc_3 = get_principal_comps(mat, 3)
    
    fig = plt.figure()
    # 3D plot
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pc_1, pc_2, pc_3, marker="o", markersize=msize, linestyle="None")
    if arrows:
        [arrow_3d(ax, pc_1[i], pc_2[i], pc_3[i]) for i in range(len(word_feature_tuples))]
        
    [point_label_3d(ax, word, pc_1[i], pc_2[i], pc_3[i]) for i, (word, _) in enumerate(word_feature_tuples)]
    
    
def plot_w2v_2d(words, embeddings, mapping, arrows=True):
    """Plot word2vec embeddings in 2d.
    
    'embeddings' contains the word embeddings in a numpy matrix
    and 'mapping' is a {str:int} dict mapping words to row indices
    in 'embeddings'.
    """
    pc_1, pc_2 = get_principal_comps(embeddings, 2)
    
    if arrows:
        msize = 0
    else:
        msize = 5
        
    words = [w for w in words if w in mapping]
    indices = [mapping[w] for w in words]
    
    xs = [pc_1[i] for i in indices]
    ys = [pc_2[i] for i in indices]
    
    plt.figure()
    plt.plot(xs, ys, marker="o", markersize=msize, linestyle="None")
    
    if arrows:
        [plt.arrow(
            0, 0, 
            xs[i], ys[i], 
            head_width=0.01, 
            length_includes_head=True
        ) for i in range(len(xs))]
    
    [point_label(word, xs[i], ys[i]) for i, word in enumerate(words)]
        
    plt.show()
    
def plot_w2v_3d(words, embeddings, mapping, arrows=True):
    """Plot word2vec embeddings in 2d.
    
    'embeddings' contains the word embeddings in a numpy matrix
    and 'mapping' is a {str:int} dict mapping words to row indices
    in 'embeddings'.
    """
    pc_1, pc_2, pc_3 = get_principal_comps(embeddings, 3)
    
    if arrows:
        msize = 0
    else:
        msize = 5
    words = [w for w in words if w in mapping]
    indices = [mapping[w] for w in words]
    
    xs = [pc_1[i] for i in indices]
    ys = [pc_2[i] for i in indices]
    zs = [pc_3[i] for i in indices]
    
    fig = plt.figure()
        
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs, ys, zs, marker="o", markersize=msize, linestyle="None")
    
    if arrows:
        [arrow_3d(ax, xs[i], ys[i], zs[i]) for i in range(len(words))]
    
    [point_label_3d(ax, word, xs[i], ys[i], zs[i]) for i, word in enumerate(words)]
    
    plt.show()
    

def plot_sentences_3d(sentences, embeddings, mapping, embedding_fn=None):
    """Plot simple averaged sentence embeddings in 3d.
    
    'embeddings' contains the word embeddings in a numpy matrix
    and 'mapping' is a {str:int} dict mapping words to row indices
    in 'embeddings'.
    """
    if not embedding_fn:
        embedding_fn = embed_sentence
        
    embs_sent = [embedding_fn(s, embeddings, mapping) for s in sentences]
    M_sent = np.concatenate(embs_sent)

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(M_sent)

    comp_1_sent = pca_result[:,0]
    comp_2_sent = pca_result[:,1] 
    comp_3_sent = pca_result[:,2] 

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    [ax.scatter(comp_1_sent[i], comp_2_sent[i], comp_3_sent[i]) for i, _ in enumerate(comp_1_sent)]
    [ax.plot([0, comp_1_sent[i]], [0, comp_2_sent[i]], [0, comp_3_sent[i]]) for i, _ in enumerate(comp_1_sent)]

    legend_texts = [s[:20] + "..." for s in sentences]
    plt.legend(legend_texts, bbox_to_anchor=(0.6,0.6))
    plt.show()
    
    
def plot_sentences_2d(sentences, embeddings, mapping, embedding_fn=None):
    """Plot simple averaged sentence embeddings in 2d.
    
    'embeddings' contains the word embeddings in a numpy matrix
    and 'mapping' is a {str:int} dict mapping words to row indices
    in 'embeddings'.
    """
    if not embedding_fn:
        embedding_fn = embed_sentence
        
    embs_sent = [embedding_fn(s, embeddings, mapping) for s in sentences]
    M_sent = np.concatenate(embs_sent)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(M_sent)

    comp_1_sent = pca_result[:,0]
    comp_2_sent = pca_result[:,1] 

    plt.figure()
    [plt.scatter(comp_1_sent[i], comp_2_sent[i]) for i, _ in enumerate(comp_1_sent)]
    [plt.plot([0, comp_1_sent[i]], [0, comp_2_sent[i]]) for i, _ in enumerate(comp_1_sent)]
    
    legend_texts = [s[:20] + "..." for s in sentences]
    plt.legend(legend_texts, bbox_to_anchor=(0.6,0.6))
    
    plt.show()
    
    
def get_embeddings():
    """Load pretrained embeddings and word to int mappings."""
    with open("../../data/embedding-matrix-en.npy", "rb") as f:
        M = np.load(f)
    with open("../../data/wtoi-en.pickle", "rb") as f:
        wtoi = pickle.load(f)
        
    return M, wtoi

def embed_sentence(sentence, word_embeddings, mapping):
    """Embed sentence using simple word averaging method.
   
    'sentence' is an all-lowercase string with tokens separated by spaces.
   
    'word_embeddings' contains the word embeddings in a numpy matrix
    and 'mapping' is a {str:int} dict mapping words to row indices
    in 'embeddings'.
    """
    # Get the row indices of the words in the sentence
    indices = [mapping[w] for w in sentence.split() if w in mapping]
    
    # You can index a numpy array by giving it a Python list of integers.
    # word_vectors is now a (|s| x 300) array (matrix), where |s| is the
    # length of the sentence. (Actually this only holds if all the words
    # in the sentence have embeddings, so in reality it is probably not
    # |s|.)
    word_vectors = word_embeddings[indices]
    
    # Convenient averaging function, notice the correct axis (which would
    # probably be correct by default, but anyways..). "np.average" is used
    # here instead of "np.mean" because "average" could also be weighted.
    sentence_embedding = np.average(word_vectors, axis=0)
    
    # Reshape from 1d array to (1 x 300) for convenience.
    return sentence_embedding.reshape(1, -1)


def arrow_from(start_x, start_y, end_x, end_y, c="black", linestyle='solid', onlybase=True):
    """Plot an arrow in 2d from (start_x, start_y) to (end_x, end_y)."""
    if onlybase:
        a = plt.arrow(
            start_x, start_y, 
            end_x, end_y,
            width=0.00002,
            head_width=0.001, 
            length_includes_head=True,
            color=c,
            linestyle=linestyle
        )
    else:
        a = plt.arrow(
            start_x, start_y, 
            end_x, end_y,
            width=0.001,
            head_width=0.02, 
            length_includes_head=True,
            color=c,
            linestyle=linestyle
        )
    return a


def plot_w2v_algebra(embeddings, mapping, base, minus=None, plus=None, result=None):
    """Plot vector algebra of the form 'base - minus + plus = ?'"""
    words = []
    if result:
        words = [w for w in [base, minus, plus, result] if w]
    else:
        words = [w for w in [base, minus, plus, result] if w]
    
    indices = [mapping[w] for w in words]
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)

    comp_1 = pca_result[:,0][indices]
    comp_2 = pca_result[:,1][indices]
    
    wtoi = {w:i for i, w in enumerate(words)}
    itow = {i:w for i, w in enumerate(words)}
    
    base_i = wtoi[base]
    if minus:
        minus_i = wtoi[minus]
    if plus:
        plus_i = wtoi[plus]
    if result:
        equals = wtoi[result]
    
    if minus:
        base_minus_x = comp_1[base_i] - comp_1[minus_i]
        base_minus_y = comp_2[base_i] - comp_2[minus_i]

    if minus and plus:
        final_x = base_minus_x + comp_1[plus_i]
        final_y = base_minus_y + comp_2[plus_i]
    
    # closest_ind = find_true_closest(embeddings, mapping, base, minus, plus)
    # reverse_mapping = {i:w for w, i in mapping.items()}
    # print("True closest words:", [reverse_mapping[i] for i in closest_ind])

    all_x = np.concatenate([[0], comp_1])
    all_y = np.concatenate([[0], comp_2])
    
    if minus:
        all_x = np.concatenate([all_x, [base_minus_x]])
        all_y = np.concatenate([all_y, [base_minus_y]])
        
    if minus and plus:
        all_x = np.concatenate([all_x, [final_x]])
        all_y = np.concatenate([all_y, [final_y]])
    
    plt.figure()
    plt.plot(
        all_x,
        all_y, 
        marker="o", 
        markersize=2.0, 
        linestyle="None"
    )

    # origin + base
    arrow_from(0, 0, comp_1[base_i], comp_2[base_i], onlybase=(not (minus or plus)))
    if minus:
        arrow_from(0, 0, comp_1[minus_i], comp_2[minus_i], "red", onlybase=False)
    if plus:
        arrow_from(0, 0, comp_1[plus_i], comp_2[plus_i], "blue", onlybase=False)
    # base - minus
    if minus:
        arr2 = arrow_from(
            comp_1[base_i], 
            comp_2[base_i], 
            -1*comp_1[minus_i], 
            -1*comp_2[minus_i], 
            "red",
            onlybase=False
        )
    # (base - minus) + plus
    if minus and plus:
        arr3 = arrow_from(
            base_minus_x, 
            base_minus_y, 
            comp_1[plus_i], 
            comp_2[plus_i], 
            "blue",
            onlybase=False
        )
    # origin to final
    if minus and plus:
        arr4 = arrow_from(0, 0, final_x, final_y, "orange", linestyle="--", onlybase=False)

    [plt.text(comp_1[i], comp_2[i], w) for i, w in enumerate(words)]

    if minus and plus:
        plt.legend(
            [arr4],
            [base + " - " + minus + " + " + plus],
            ncol=2, fancybox=True
        )

    plt.show()

def find_true_closest(embeddings, mapping, base, minus, plus, n=5):
    true_final = embeddings[mapping[base]] - embeddings[mapping[minus]] + embeddings[mapping[plus]]
     
    distances = cosine_distances(true_final.reshape(1, -1), embeddings).reshape(-1)
    closest_indices = distances.argsort()[-n:][::-1] 
     
    return closest_indices

def to_dict(l):
    return {k:v for k, v in l}

def tabulate_angles(words_and_feats):
    word_dict = to_dict(words_and_feats)
    latex = "\\textbf{Pairwise angles} \\\\\n"
    latex += "\\begin{array}{" + "c|" + "c"*len(word_dict) + "}\n& "

    for word in word_dict.keys():
        latex += "\\textrm{%s} & " % word
    
    latex = latex[:-1] + r" \\\hline" + "\n"

    for word1 in word_dict.keys():
        latex += "\\textrm{%s}" % word1
        for word2 in word_dict.keys():
            latex += r" & %.1f°" % angle(word_dict[word1], word_dict[word2])
        latex += r" \\" + "\n"

    latex += "\n\\end{array}"
    display(Math(latex))
