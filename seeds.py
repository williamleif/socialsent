import random

"""
Seed words for propagating polarity scores.
"""

# From Turney and Littman (2003), probably not ideal for historical data
POSITIVE_TURNEY = ["good", "nice", "excellent", "positive", "fortunate", "correct", "superior"]
NEGATIVE_TURNEY = ["bad", "terrible", "poor", "negative", "unfortunate", "wrong", "inferior"]

POSITIVE_FINANCE = ["successful", "excellent", "profit", "beneficial", "improving", "improved", "success", "gains", "positive"]
NEGATIVE_FINANCE = ["negligent", "loss", "volatile", "wrong", "losses", "damages", "bad", "litigation", "failure", "down", "negative"]

#POSITIVE_TWEET = ["love", "awesome",  "nice", "amazing", "best", "fantastic", "correct"]
#NEGATIVE_TWEET = ["hate", "terrible",  "nasty", "awful", "worst", "horrible", "wrong"]

POSITIVE_TWEET = ["love", "loved", "loves", "awesome",  "nice", "amazing", "best", "fantastic", "correct", "happy"]
NEGATIVE_TWEET = ["hate", "hated", "hates", "terrible",  "nasty", "awful", "worst", "horrible", "wrong", "sad"]

POSITIVE_HIST = ["good", "lovely", "excellent", "fortunate", "pleasant", "delightful", "perfect", "loved", "love", "happy"] 
NEGATIVE_HIST = ["bad", "horrible", "poor",  "unfortunate", "unpleasant", "disgusting", "evil", "hated", "hate", "unhappy"]

POSITIVE_ADJ = ["good", "lovely", "excellent", "fortunate", "pleasant", "delightful", "perfect", "happy"] 
NEGATIVE_ADJ = ["bad", "horrible", "poor",  "unfortunate", "unpleasant", "disgusting", "evil", "unhappy"]


def twitter_seeds():
    return POSITIVE_TWEET, NEGATIVE_TWEET

def finance_seeds():
    return POSITIVE_FINANCE, NEGATIVE_FINANCE

def turney_seeds():
    return POSITIVE_TURNEY, NEGATIVE_TURNEY

def adj_seeds():
    return POSITIVE_ADJ, NEGATIVE_ADJ

def hist_seeds():
    return POSITIVE_HIST, NEGATIVE_HIST

def random_seeds(words, lexicon, num):
    sample_set = list(set(words).intersection(lexicon))
    seeds = random.sample(sample_set, num)
    return [s for s in seeds if lexicon[s] == 1], [s for s in seeds if lexicon[s] == -1]
