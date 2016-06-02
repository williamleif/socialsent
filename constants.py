import util
from nltk.corpus import stopwords

### SYSTEM AGNOSTIC CONSTANTS 
######
DATA = './data/'
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
GOOGLE_EMBEDDINGS = ""
#from http://www.cis.lmu.de/~sascha/Ultradense/
TWITTER_EMBEDDINGS = ""

# The following can be constructed from the corpora cited in the paper
## UPON PUBLICATION LINKS TO ALL DATA/EMBEDDINGS WILL BE PROVIDED
STOCK_EMBEDDINGS = ""
STOCK_COUNTS = ""
COHA_EMBEDDINGS = ""
COHA_PPMI = ""
COHA_COUNTS = ""
COHA_SGNS_EMBEDDINGS = ""
FREQS = ""
COHA_FREQS = ""
DFS_DATA = ""
POS = ""
SUBREDDIT_EMBEDDINGS = ""

def make_directories():
    util.mkdir(POLARITIES)

if __name__ == '__main__':
    make_directories()
