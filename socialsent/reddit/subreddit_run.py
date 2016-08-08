import sys
import os
import time
import random
import seeds
import constants
import util
from reddit import subredditgen
from multiprocessing import Queue, Process
from Queue import Empty
import polarity_induction_methods

from representations.representation_factory import create_representation


DICTS = "/dfs/scratch0/wleif/Reddit/dicts/{}-dict.pkl"
NAMES = "/dfs/scratch0/wleif/Reddit/comment_counts.txt"
POLARITIES = "/dfs/scratch0/wleif/Reddit/polarities/"

def worker(proc_num, queue):
    while True:
#        time.sleep(random.random()*10)
        try:
            name = queue.get(block=False)
        except Empty:
            print proc_num, "Finished"
            return
        if name + ".pkl" in os.listdir(POLARITIES):
            continue
        print proc_num, "Running", name
        subredditgen.main(name)
        word_dict = util.load_pickle(DICTS.format(name))
        word_dict.filter_extremes(no_above=0.1, no_below=100)
        to_keep = sorted(word_dict.dfs, key=lambda w : word_dict.dfs[w], reverse=True)[:5000]
        word_dict.filter_tokens(good_ids=to_keep)
        sub_vecs = create_representation("SVD", constants.SUBREDDIT_EMBEDDINGS.format(name))
        pos_seeds, neg_seeds = seeds.twitter_seeds()
        sub_vecs = sub_vecs.get_subembed(set(word_dict.token2id.keys()).union(pos_seeds).union(neg_seeds))
        pols = polarity_induction_methods.bootstrap(sub_vecs, pos_seeds, neg_seeds, return_all=True,
                nn=25, beta=0.9, num_boots=50, n_procs=10)
        util.write_pickle(pols, POLARITIES + name + ".pkl")

if __name__ == "__main__":
    queue = Queue()
    id = int(sys.argv[1])
    valid_ids = set(range(250, 256))
    for i, line in enumerate(util.lines(NAMES)):
        if i in valid_ids:
            name = line.split()[0]
            queue.put(name)
    print queue.qsize()
    procs = [Process(target=worker, args=[i, queue]) for i in range(1)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()


