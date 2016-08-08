import numpy as np
from scipy import sparse
from itertools import chain
from nltk.corpus import wordnet as wn


"""
Methods for constructing graphs from word embeddings.
"""

def similarity_matrix(embeddings, arccos=False, similarity_power=1, nn=25, **kwargs):
    """
    Constructs a similarity matrix from embeddings.
    nn argument controls the degree.
    """
    def make_knn(vec, nn=nn):
        vec[vec < vec[np.argsort(vec)[-nn]]] = 0
        return vec
    L = embeddings.m.dot(embeddings.m.T)
    if sparse.issparse(L):
        L = L.todense()
    if arccos:
        L = np.arccos(np.clip(-L, -1, 1))/np.pi
    else:
        L += 1
    np.fill_diagonal(L, 0)
    L = np.apply_along_axis(make_knn, 1, L)
    return L ** similarity_power

def wordnet_similarity_matrix(embeddings):
    """
    Makes a similarity matrix from WordNet.
    Embeddings argument is only used to get set of words to use.
    """
    sim_mat = np.zeros((len(embeddings.iw), len(embeddings.iw)))
    words = {word:wn.morphy(word) for word in embeddings.iw}
    lemmas = {lemma:word for word, lemma in words.iteritems()}
    for i, word in enumerate(words):
        if words[word] == None:
            continue
        synonyms = set(chain.from_iterable([o_word.lemma_names() 
            for o_word in wn.synsets(words[word])]))
        for o_word in synonyms:
            if o_word in lemmas:
                sim_mat[embeddings.wi[word], embeddings.wi[lemmas[o_word]]] = 1.
    print np.sum(sim_mat)
    np.fill_diagonal(sim_mat, 0)
    return sim_mat

def transition_matrix(embeddings, word_net=False, first_order=False, sym=False, trans=False, **kwargs):
    """
    Build a probabilistic transition matrix from word embeddings.
    """
    if word_net:
        L =  wordnet_similarity_matrix(embeddings)
    elif not first_order:
        L = similarity_matrix(embeddings, **kwargs)
    else:
        L = embeddings.m.todense().A
    if sym:
        Dinv = np.diag([1. / np.sqrt(L[i].sum()) if L[i].sum() > 0 else 0 for i in range(L.shape[0])])
        return Dinv.dot(L).dot(Dinv)
    else:
        Dinv = np.diag([1. / L[i].sum() for i in range(L.shape[0])])
        L = L.dot(Dinv)
    if trans:
        return L.T
    else:
        return L
    return L
