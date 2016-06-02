import constants
import util

def pos_tags(year):
    """
    Returns mapping from words to POS tags for year.
    AMB means tag is ambiguous.
    """
    year = str(year)
    pos_tags = util.load_pickle(constants.POS + year + "-pos.pkl")
    return pos_tags

def pos_words(year, pos):
    """
    Load all words with specified POS tag for year.
    """
    pos_tagss = pos_tags(year)
    return set([word for word in pos_tagss if pos_tagss[word] == pos])

def words_above_freq(year, freq):
    freqs = util.load_pickle(constants.COHA_FREQS.format(year)) 
    return set([word for word in freqs if freqs[word] > freq])

def top_words(year, rank):
    year = int(year)
    freqs = util.load_pickle(constants.FREQS) 
    return set(sorted(freqs, key = lambda w : freqs[w][year], reverse=True)[0:rank])


