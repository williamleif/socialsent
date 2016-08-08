from socialsent import util

from argparse import ArgumentParser
from socialsent.representations import ppmigen, cooccurgen, makelowdim

COMMENTS = "/dfs/scratch0/wleif/Reddit/clean_comments/{}.tsv"
DICTS = "/dfs/scratch0/wleif/Reddit/dicts/{}-dict.pkl"
OUT = "/dfs/scratch0/wleif/Reddit/vecs/{}/"

def word_gen(file, gdict):
    for i, line in enumerate(open(file)):
        info = line.split("\t")
        comment = info[-1]
        for word in comment.split():
            word = word.lower()
#            if word != "<EOS>" and word in gdict.token2id:
            if word in gdict.token2id:
                yield word
        if i % 10000 == 0:
            print "Processed line", i

def main(subreddit):
    out_path = OUT.format(subreddit)
    util.mkdir(out_path)

    print "Getting and writing dictionary..."
    gdict = util.load_pickle(DICTS.format(subreddit))
    gdict.filter_extremes(no_above=0.5, no_below=100)
    gdict.compactify()
    util.write_pickle(gdict.token2id, out_path + "index.pkl")

    print "Generating word co-occurrences..."
    cooccurgen.run(word_gen(COMMENTS.format(subreddit), gdict), gdict.token2id, 4, out_path + "counts.bin")
    print "Generating PPMI vectors..."
    ppmigen.run(out_path + "counts.bin", out_path + "ppmi", cds=True)
    print "Generating SVD vectors..."
    makelowdim.run(out_path + "ppmi.bin", out_path + "vecs")

if __name__ == "__main__":
    parser = ArgumentParser("Make subreddit word vectors") 
    parser.add_argument("subreddit")
    args = parser.parse_args()
    subreddit = args.subreddit
    main(subreddit)

