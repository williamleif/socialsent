import util
from nltk.corpus import stopwords

### SYSTEM AGNOSTIC CONSTANTS 
######
DATA = '/afs/cs.stanford.edu/u/wleif/sentiment/polarity_induction/data/'
LEXICONS = DATA + 'lexicon_info/'
PROCESSED_LEXICONS = DATA + 'lexicons/'
POLARITIES = DATA + 'polarities/'
STOPWORDS = set(stopwords.words('english'))
LEXICON = 'inquirer'
YEARS = map(str, range(1850, 2000, 10))

## CONSTANTS USED FOR SPECIFIC EXPERIMENTS ###
######
## THE FOLLOWING CAN BE REPLACED BY DOWNLOADING APPROPRIATE RESOURCES AND CHANGING PATHS:

#from https://code.google.com/p/word2vec/
GOOGLE_EMBEDDINGS = '/dfs/scratch0/gigawordvecs/GoogleNews-vectors-negative300_transformed.txt'
#from http://www.cis.lmu.de/~sascha/Ultradense/
TWITTER_EMBEDDINGS = '/dfs/scratch0/gigawordvecs/twitter_lower_cw1_sg400_transformed.txt'

# The following can be constructed from the corpora cited in the paper
## UPON PUBLICATION LINKS TO ALL EMBEDDINGS WILL BE PROVIDED
STOCK_EMBEDDINGS = '/lfs/madmax9/0/stock/svd-vecs'
STOCK_COUNTS = '/lfs/madmax3/0/stock/vecs.bin'
COHA_EMBEDDINGS = '/dfs/scratch0/COHA/cooccurs/word/ppmi/lsmooth0/nFalse/neg1/cdsTrue/svd/300/50000/'
COHA_PPMI = '/dfs/scratch0/COHA/cooccurs/word/ppmi/lsmooth0/nFalse/neg1/cdsTrue/'
COHA_COUNTS = '/dfs/scratch0/COHA/cooccurs/word/4/'
COHA_SGNS_EMBEDDINGS = '/dfs/scratch0/COHA/cooccurs/word/sgns/300/'
FREQS = "/dfs/scratch0/hist_words/coha-word/freqs.pkl"
COHA_FREQS = "/dfs/scratch0/COHA/decade_freqs/{}-word.pkl"
DFS_DATA = '/dfs/scratch0/googlengrams/eng-all/decades/'
#POS = DFS_DATA + '/pos/'
POS = "/dfs/scratch0/hist_words/coha-word/pos/"
SUBREDDIT_EMBEDDINGS = '/dfs/scratch2/wleif/Reddit/vecs/{}/vecs'

def make_directories():
    util.mkdir(DATA)
    util.mkdir(PROCESSED_LEXICONS)
    util.mkdir(POLARITIES)

if __name__ == '__main__':
    make_directories()
