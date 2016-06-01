import time
import random
import constants
import seeds
import util
import polarity_induction_methods

from historical import vocab
from multiprocessing import Queue, Process
from Queue import Empty
from representations.representation_factory import create_representation

"""
Makes historical sentiment lexicons for non-stop words with freq. > 10**-6.0
"""

def worker(proc_num, queue):
    while True:
        time.sleep(random.random()*10)
        try:
            year = queue.get(block=False)
        except Empty:
            print proc_num, "Finished"
            return
        positive_seeds, negative_seeds = seeds.hist_seeds()
        year = str(year)
        print proc_num, "On year", year
        words = vocab.top_words(year, 5100)
        stop_words = vocab.top_words(year, 100)
        words = words.difference(stop_words)
        embed = create_representation("SVD", constants.COHA_EMBEDDINGS + year)
        embed_words = set(embed.iw)
        words = words.intersection(embed_words)

        polarities = polarity_induction_methods.bootstrap(
                 embed.get_subembed(words.union(positive_seeds).union(negative_seeds)),
                 positive_seeds, negative_seeds,
                 score_method=polarity_induction_methods.random_walk,
                 num_boots=50, n_procs=20, return_all=True,
                 beta=0.9, nn=25)
        util.write_pickle(polarities, constants.POLARITIES + year + '-coha-freq-boot.pkl')

def main():
    num_procs = 6
    queue = Queue()
    for year in range(1850, 2010, 10):
        queue.put(year)
    procs = [Process(target=worker, args=[i, queue]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

if __name__ == "__main__":
    main()
