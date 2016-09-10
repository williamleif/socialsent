from socialsent import util
import functools
import numpy as np
from socialsent import embedding_transformer
from scipy.sparse import csr_matrix
from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression, Ridge

from socialsent.graph_construction import similarity_matrix, transition_matrix

"""
A set of methods for inducing polarity lexicons using word embeddings and seed words.
"""

def dist(embeds, positive_seeds, negative_seeds, **kwargs):
    polarities = {}
    sim_mat = similarity_matrix(embeds, **kwargs)
    for i, w in enumerate(embeds.iw):
        if w not in positive_seeds and w not in negative_seeds:
            pol = sum(sim_mat[embeds.wi[p_seed], i] for p_seed in positive_seeds)
            pol -= sum(sim_mat[embeds.wi[n_seed], i] for n_seed in negative_seeds)
            polarities[w] = pol
    return polarities


def pmi(count_embeds, positive_seeds, negative_seeds, smooth=0.01, **kwargs):
    """
    Learns polarity scores using PMI with seed words.
    Adapted from Turney, P. and M. Littman. "Measuring Praise and Criticism: Inference of semantic orientation from assocition".
    ACM Trans. Inf. Sys., 2003. 21(4) 315-346.

    counts is explicit embedding containing raw co-occurrence counts
    """
    w_index = count_embeds.wi
    c_index = count_embeds.ci
    counts = count_embeds.m
    polarities = {}
    for w in count_embeds.iw:
        if w not in positive_seeds and w not in negative_seeds:
            pol = sum(np.log(counts[w_index[w], c_index[seed]] + smooth) 
                    - np.log(counts[w_index[seed],:].sum()) for seed in positive_seeds)
            pol -= sum(np.log(counts[w_index[w], c_index[seed]] + smooth) 
                    - np.log(counts[w_index[seed],:].sum())for seed in negative_seeds)
            polarities[w] = pol
    return polarities

def densify(embeddings, positive_seeds, negative_seeds, 
        transform_method=embedding_transformer.apply_embedding_transformation, **kwargs):
    """
    Learns polarity scores via orthogonally-regularized projection to one-dimension
    Adapted from: http://arxiv.org/pdf/1602.07572.pdf
    """
    p_seeds = {word:1.0 for word in positive_seeds}
    n_seeds = {word:1.0 for word in negative_seeds}
    new_embeddings = embeddings
    new_embeddings = embedding_transformer.apply_embedding_transformation(
            embeddings, p_seeds, n_seeds, n_dim=1,  **kwargs)
    polarities = {w:new_embeddings[w][0] for w in embeddings.iw}
    return polarities


def random_walk(embeddings, positive_seeds, negative_seeds, beta=0.9, **kwargs):
    """
    Learns polarity scores via random walks with teleporation to seed sets.
    Main method used in paper. 
    """
    def run_random_walk(M, teleport, beta, **kwargs):
        def update_seeds(r):
            r += (1 - beta) * teleport / np.sum(teleport)
        return run_iterative(M * beta, np.ones(M.shape[1]) / M.shape[1], update_seeds, **kwargs)

    if not type(positive_seeds) is dict:
        positive_seeds = {word:1.0 for word in positive_seeds}
        negative_seeds = {word:1.0 for word in negative_seeds}
    words = embeddings.iw
    M = transition_matrix(embeddings, **kwargs)
    rpos = run_random_walk(M, weighted_teleport_set(words, positive_seeds), beta, **kwargs)
    rneg = run_random_walk(M, weighted_teleport_set(words, negative_seeds), beta, **kwargs)
    return {w: rpos[i] / (rpos[i] + rneg[i]) for i, w in enumerate(words)}


def label_propagate_probabilistic(embeddings, positive_seeds, negative_seeds, **kwargs):
    """
    Learns polarity scores via standard label propagation from seed sets.
    One walk per label. Scores normalized to probabilities. 
    """
    words = embeddings.iw
    M = transition_matrix(embeddings, **kwargs)
    pos, neg = teleport_set(words, positive_seeds), teleport_set(words, negative_seeds)
    def update_seeds(r):
        r[pos] = [1, 0]
        r[neg] = [0, 1]
        r /= np.sum(r, axis=1)[:, np.newaxis]
    r = run_iterative(M, np.random.random((M.shape[0], 2)), update_seeds, **kwargs)
    return {w: r[i][0] / (r[i][0] + r[i][1]) for i, w in enumerate(words)}


def label_propagate_continuous(embeddings, positive_seeds, negative_seeds, **kwargs):
    """
    Learns polarity scores via standard label propagation from seed sets.
    One walk for both labels, continuous non-normalized scores.
    """
    words = embeddings.iw
    M = transition_matrix(embeddings, **kwargs)
    pos, neg = teleport_set(words, positive_seeds), teleport_set(words, negative_seeds)
    def update_seeds(r):
        r[pos] = 1
        r[neg] = -1
    r = run_iterative(M, np.zeros(M.shape[0]), update_seeds, **kwargs)
    return {w: r[i] for i, w in enumerate(words)}


def graph_propagate(embeddings, positive_seeds, negative_seeds, **kwargs):
    """
    Graph propagation method dapted from Velikovich, Leonid, et al. "The viability of web-derived polarity lexicons."
    http://www.aclweb.org/anthology/N10-1119
    Should be used with arccos=True
    """
    def run_graph_propagate(seeds, alpha_mat, trans_mat, T=1, **kwargs):
        def get_rel_edges(ind_set):
            rel_edges = set([])
            for node in ind_set:
                rel_edges = rel_edges.union(
                        [(node, other) for other in trans_mat[node,:].nonzero()[1]])
            return rel_edges

        for seed in seeds:
            F = set([seed])
            for t in range(T):
                for edge in get_rel_edges(F):
                    alpha_mat[seed, edge[1]] = max(
                            alpha_mat[seed, edge[1]], 
                            alpha_mat[seed, edge[0]] * trans_mat[edge[0], edge[1]])
                    F.add(edge[1])
        return alpha_mat

    M = similarity_matrix(embeddings, **kwargs)
    M = (M + M.T)/2
    print "Getting positive scores.."
    pos_alpha = M.copy()
    neg_alpha = M.copy()
    M = csr_matrix(M)
    pos_alpha = run_graph_propagate([embeddings.wi[seed] for seed in positive_seeds],
            pos_alpha, M, **kwargs)
    pos_alpha = pos_alpha + pos_alpha.T
    print "Getting negative scores.."
    neg_alpha = run_graph_propagate([embeddings.wi[seed] for seed in negative_seeds],
            neg_alpha, M, **kwargs)
    neg_alpha = neg_alpha + neg_alpha.T
    print "Computing final scores..."
    polarities = {}
    index = embeddings.wi
    pos_pols = {w:1.0 for w in positive_seeds}
    for w in negative_seeds:
        pos_pols[w] = 0.0
    neg_pols = {w:1.0 for w in negative_seeds}
    for w in positive_seeds:
        neg_pols[w] = 0.0
    for w in util.logged_loop(index):
        if w not in positive_seeds and w not in negative_seeds:
            pos_pols[w] = sum(pos_alpha[index[w], index[seed]] for seed in positive_seeds if seed in index) 
            neg_pols[w] = sum(neg_alpha[index[w], index[seed]] for seed in negative_seeds if seed in index)
    beta = np.sum(pos_pols.values()) / np.sum(neg_pols.values())
    for w in index:
        polarities[w] = pos_pols[w] - beta * neg_pols[w]
    return polarities


### HELPER METHODS #####

def teleport_set(words, seeds):
    return [i for i, w in enumerate(words) if w in seeds]

def weighted_teleport_set(words, seed_weights):
    return np.array([seed_weights[word] if word in seed_weights else 0.0 for word in words])

def run_iterative(M, r, update_seeds, max_iter=50, epsilon=1e-6, **kwargs):
    for i in range(max_iter):
        last_r = np.array(r)
        r = np.dot(M, r)
        update_seeds(r)
        if np.abs(r - last_r).sum() < epsilon:
            break
    return r

### META METHODS ###

def _bootstrap_func(embeddings, positive_seeds, negative_seeds, boot_size, score_method, seed, **kwargs):
    np.random.seed(seed)
    pos_seeds = np.random.choice(positive_seeds, boot_size)
    neg_seeds = np.random.choice(negative_seeds, boot_size)
    polarities = score_method(embeddings, pos_seeds, neg_seeds, **kwargs)
    return {word:score for word, score in polarities.iteritems() if
            not word in positive_seeds and not word in negative_seeds}

def bootstrap(embeddings, positive_seeds, negative_seeds, num_boots=10, score_method=random_walk,
        boot_size=7, return_all=False, n_procs=15, **kwargs):
    pool = Pool(n_procs)
    map_func = functools.partial(_bootstrap_func, embeddings, positive_seeds, negative_seeds,
            boot_size, score_method, **kwargs)
    polarities_list = pool.map(map_func, range(num_boots))
    if return_all:
        return polarities_list
    else:
        polarities = {}
        for word in polarities_list[0]:
            polarities[word] = np.mean([polarities_list[i][word] for i in range(num_boots)])
        return polarities
