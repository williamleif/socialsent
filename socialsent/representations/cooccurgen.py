from collections import Counter
import os
import numpy as np
from socialsent import util

def run(word_gen, index, window_size, out_file):
    context = []
    pair_counts = Counter()
    for word in word_gen:
        context.append(index[word])
        if len(context) > window_size * 2 + 1:
            context.pop(0)
        pair_counts = _process_context(context, pair_counts, window_size)
    import pyximport
    pyximport.install(setup_args={"include_dirs": np.get_include()})
    from representations import sparse_io
    sparse_io.export_mat_from_dict(pair_counts, out_file)

def _process_context(context, pair_counts, window_size):
    if len(context) < window_size + 1:
        return pair_counts
    target = context[window_size]
    indices = range(0, window_size)
    indices.extend(range(window_size + 1, 2 * window_size + 1))
    for i in indices:
        if i >= len(context):
            break
        pair_counts[(target, context[i])] += 1
    return pair_counts

class COHAWordGen(object):

    def __init__(self, off, index):
        self.data_dir = "/dfs/scratch0/COHA/COHA_word_lemma_pos/1990/"
        self.off = off
        self.index = index

    def __iter__(self):
        for j, fname in enumerate(os.listdir(self.data_dir)):
            if j % 2 == self.off:
                continue
            print fname
            for i, line in enumerate(open(os.path.join(self.data_dir, fname))):
                word = line.split()[1].lower()
                if word in index:
                    yield word

if __name__ == "__main__":
    index = util.load_pickle("/dfs/scratch0/COHA/cooccurs/lemma/4/index.pkl")
    word_gen = COHAWordGen(0, index)
    run(word_gen, index, 4, "/dfs/scratch0/COHA/cooccurs/lemma/testb-0-counts.bin")
#    word_gen = COHAWordGen(1, index)
#    run(word_gen, index, 4, "/dfs/scratch0/COHA/cooccurs/lemma/test-1-counts.bin")
