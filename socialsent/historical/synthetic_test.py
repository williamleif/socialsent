from socialsent import constants
from socialsent import util
import polarity_induction_methods
import time
from socialsent import seeds
from socialsent import vocab
import random
import numpy as np
from socialsent import evaluate_methods
from Queue import Empty
from multiprocessing import Process, Queue
from socialsent.representations.representation_factory import create_representation
from socialsent.representations.embedding import Embedding
from sklearn.utils.extmath import randomized_svd
#from scipy.sparse import csr_matrix, vstack
from numpy import vstack
from scipy.stats import logistic

SYNTH_FREQ = 5*10**-5.0

#NEW_POS = ["cheerful", "beautiful", "charming", "pleasant", "sweet", "favourable", "cheery"]
NEW_POS = ["cheerful", "beautiful", "charming", "merry", "pleasing"]
NEW_NEG = ["hideous", "terrible", "dreadful", "worst", "awful"]
#NEW_NEG = ["disgusting", "hideous", "terrible", "unhappy", "nasty", "repulsive", "offensive"]
OLD_POS = NEW_POS
OLD_NEG = NEW_NEG

YEARS = range(1850, 1991, 10)

"""
Runs synthetic test of amelioration and pejoration.
"""

def worker(proc_num, queue, iter):
    while True:
        time.sleep(random.random()*10)
        try:
            year = queue.get(block=False)
        except Empty:
            print proc_num, "Finished"
            return
        np.random.seed()
        positive_seeds, negative_seeds = seeds.hist_seeds()
        year = str(year)
        print proc_num, "On year", year
        words = vocab.pos_words(year, "ADJ")
        embed = create_representation("SVD", constants.COHA_EMBEDDINGS + year)
        print year, len(words)
        embed_words = set(embed.iw)
        words = words.intersection(embed_words)
        print year,  len(words)
#        counts = create_representation("Explicit", constants.COHA_COUNTS + year, normalize=False)
#        ppmi = create_representation("Explicit", constants.COHA_PPMI + year)
        weight = _make_weight(float(year))
        print year, weight
        embed = embed.get_subembed(words)
        test_embed = make_synthetic_data(embed, embed, words, weight, seed_offset=iter)
        polarities = evaluate_methods.run_method(positive_seeds, negative_seeds, 
                 test_embed,
                 method=polarity_induction_methods.random_walk, 
                 beta=0.9, nn=25,
                **evaluate_methods.DEFAULT_ARGUMENTS)
        util.write_pickle(polarities, constants.POLARITIES + year + '-synth-adj-coha-' + str(iter) + '.pkl')

def _make_weight(year):
    scaled = 2*(year-YEARS[0]) / (YEARS[-1] - YEARS[0]) - 1
    scaled *= -4
    return logistic.cdf(scaled)

def make_synthetic_data(ppmi, counts, word_subset, new_weight, num_synth=10, 
        old_pos=OLD_POS, new_pos=NEW_POS, old_neg=OLD_NEG, new_neg=NEW_NEG, dim=300, seed_offset=0):
    #print new_weight
    #ppmi = ppmi.get_subembed(word_subset, restrict_context=False)
    amel_vecs = [] 
    print "Sampling positive..."
    for i in xrange(num_synth):
        amel_vecs.append(_sample_vec2(new_pos, old_neg, counts, new_weight, seed=i+seed_offset))
    amel_mat = vstack(amel_vecs)
    pejor_vecs = []
    print "Sampling negative..."
    for i in xrange(num_synth):
        pejor_vecs.append(_sample_vec2(old_pos, new_neg, counts, 1-new_weight, seed=i+num_synth+seed_offset))
    pejor_mat = vstack(pejor_vecs)
    print "Making matrix..."
#    ppmi_mat = vstack([ppmi.m, amel_mat, pejor_mat]) 
    u = vstack([counts.m, amel_mat, pejor_mat]) 
    print "SVD on matrix..."
#    u, s, v = randomized_svd(ppmi_mat, n_components=dim, n_iter=2)
    new_vocab = ppmi.iw
    new_vocab.extend(['a-{0:d}'.format(i) for i in range(num_synth)])
    new_vocab.extend(['p-{0:d}'.format(i) for i in range(num_synth)])
    return Embedding(u, new_vocab)

def _sample_vec2(pos_words, neg_words, counts, pos_weight, seed=1):
    vec = np.zeros((counts.m.shape[1],))
    np.random.seed(seed)
    pos_weights = np.random.dirichlet(np.repeat(0.1, len(pos_words)))
    pos_weights = pos_weights / np.sum(pos_weights) 
    print pos_weights
    for i, word in enumerate(pos_words): 
        sample_vec = pos_weights[i] * pos_weight * counts.represent(word)
        vec += sample_vec
    neg_weights = np.random.dirichlet(np.repeat(0.1, len(pos_words)))
    neg_weights = neg_weights / np.sum(neg_weights) 
    for i, word in enumerate(neg_words): 
        sample_vec = neg_weights[i] * (1-pos_weight) * counts.represent(word)
        vec += sample_vec
    return vec / np.linalg.norm(vec)


def _sample_vec(pos_words, neg_words, counts, pos_weight, seed):
    sample_size = counts.m.sum() * SYNTH_FREQ / len(neg_words)
    vec = np.zeros((counts.m.shape[1],))
    np.random.seed(seed)
    pos_weights = np.random.uniform(size=len(pos_words))
    pos_weights = pos_weights / np.sum(pos_weights) 
    print pos_weights
    for i, word in enumerate(pos_words): 
        sample_vec = counts.represent(word)
        sample_vec /= float(sample_vec.sum())
        sample_vec = pos_weights[i] * pos_weight * np.random.multinomial(sample_size, sample_vec.todense().A[0])
        sample_vec = np.clip(sample_vec, 0, sample_size)
        if not np.isfinite(sample_vec.sum()):
            print "Infinite sample with", word
            continue
        vec += sample_vec
    neg_weights = np.random.uniform(size=len(neg_words))
    neg_weights = neg_weights / np.sum(neg_weights) 
    for i, word in enumerate(neg_words): 
        sample_vec = counts.represent(word)
        sample_vec /= float(sample_vec.sum())
        sample_vec = neg_weights[i] * (1-pos_weight) * np.random.multinomial(sample_size, sample_vec.todense().A[0])
        sample_vec = np.clip(sample_vec, 0, sample_size)
        if not np.isfinite(sample_vec.sum()):
            print "Infinite sample with", word
            continue
        vec += sample_vec
    vec = csr_matrix(vec)
    new_mat = vstack([counts.m, vec])
    new_mat = new_mat / new_mat.sum()
    synth_prob = new_mat[-1,:].sum()
    for neigh in vec.nonzero()[1]:
        val = max(np.log(new_mat[-1,neigh] 
                / (synth_prob * new_mat[neigh,:].sum() ** 0.75)),
                0)
        if np.isfinite(val):
            vec[0, neigh] = val
    return vec / np.sqrt((vec.multiply(vec).sum()))
         
def main(iter):
    num_procs = 20
    queue = Queue()
    for year in YEARS:
        queue.put(year)
    procs = [Process(target=worker, args=[i, queue, iter]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

if __name__ == "__main__":
    for iter in range(0,50):
        main(iter)

