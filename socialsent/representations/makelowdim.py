import numpy as np

from sklearn.utils.extmath import randomized_svd
from socialsent import util
from socialsent.representations.explicit import Explicit

def run(in_file, out_path, dim=300, keep_words=None): 
        base_embed = Explicit.load(in_file, normalize=False)
        if keep_words != None:
            base_embed = base_embed.get_subembed(keep_words)
        u, s, v = randomized_svd(base_embed.m, n_components=dim, n_iter=5)
        np.save(out_path + "-u.npy", u)
        np.save(out_path + "-v.npy", v)
        np.save(out_path + "-s.npy", s)
        util.write_pickle(base_embed.iw, out_path  + "-vocab.pkl")

if __name__ == '__main__':
    print "Getting keep words..."
    counts = util.load_pickle("/dfs/scratch0/COHA/cooccurs/lemma/1990-counts.pkl") 
    keep_words = [word for word in counts if counts[word] >= 100]
    print "Running SVD.."
    run("/dfs/scratch0/COHA/cooccurs/lemma/testb-0-ppmi.bin.bin", "/dfs/scratch0/COHA/cooccurs/lemma/testb-0-svd", keep_words=keep_words)
    
