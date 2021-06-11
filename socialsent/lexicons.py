from . import constants, util
import collections
import numpy as np
import itertools


"""
Helper methods for creating and managing existing lexicons.
"""


def make_kuperman_scores_lexicon():
    polarities = {}
    for i, line in enumerate(util.lines(constants.LEXICONS + "kuperman/raw_ratings.csv")):
        if i == 0:
            continue
        info = line.split(",")
        if len(info[1].split()) == 1:
            polarities[info[1]] = float(info[2])
    util.write_json(polarities, constants.PROCESSED_LEXICONS + 'kuperman.json')

def make_twitter_scores_lexicon():
    polarities = {}
    for line in util.lines(constants.LEXICONS + "twitter/MaxDiff-Twitter-Lexicon/Maxdiff-Twitter-Lexicon_-1to1.txt"):
        info = line.split()
        if len(info[1].split()) > 1:
            continue
        polarities[info[1]] = float(info[0])
    util.write_json(polarities, constants.PROCESSED_LEXICONS + 'twitter-scores.json')

def make_140_scores_lexicon():
    polarities = {}
    for line in util.lines(constants.LEXICONS + "Sentiment140-Lexicon-v0.1/unigrams-pmilexicon.txt"):
        info = line.split()
        polarities[info[0]] = float(info[1])
    util.write_json(polarities, constants.PROCESSED_LEXICONS + '140-scores.json')

def make_qwn_scores_lexicon():
    polarities = collections.defaultdict(float)
    for line in util.lines(constants.LEXICONS + "qwn/turneyLittman_propSyn_08_mcr30-noAntGloss.dict"):
        info = line.split("\t")
        mod = float(info[3])
        for word in info[2].split(", "):
            if not "_" in word:
                polarities[word.split("#")[0]] += mod
    util.write_json(polarities, constants.PROCESSED_LEXICONS + 'qwn-scores.json')


def make_twitter_lexicon():
    polarities = {}
    for line in util.lines(constants.LEXICONS + "twitter/MaxDiff-Twitter-Lexicon/Maxdiff-Twitter-Lexicon_-1to1.txt"):
        info = line.split()
        if len(info[1].split()) > 1:
            continue
        if float(info[0]) < 0:
            polarities[info[1]] = -1
        else:
            polarities[info[1]] = 1
    util.write_json(polarities, constants.PROCESSED_LEXICONS + 'twitter.json')

def make_qwn_lexicon():
    polarities = collections.defaultdict(float)
    for line in util.lines(constants.LEXICONS + "qwn/turneyLittman_propSyn_08_mcr30-noAntGloss.dict"):
        info = line.split("\t")
        if info[1] == "neg":
            mod = -1
        else:
            mod = 1
        for word in info[2].split(", "):
            if not "_" in word:
                polarities[word.split("#")[0]] += mod
    polarities = {word:np.sign(val) for word, val in polarities.iteritems() if val != 0}
    util.write_json(polarities, constants.PROCESSED_LEXICONS + 'qwn.json')


def make_bingliu_lexicon():
    polarities = {}
    for polarity in ['positive', 'negative']:
        for line in util.lines(constants.LEXICONS + 'bl_opinion_lexicon/{:}-words.txt'
                .format(polarity)):
            try:
                line = line.strip().encode('ascii', 'ignore')
                if len(line) == 0 or line[0] == ';':
                    continue
                polarities[line] = 1 if polarity == 'positive' else -1
            except:
                print("skipping", line)
    util.write_json(polarities, constants.PROCESSED_LEXICONS + 'bingliu.json')

def make_finance_lexicon():
    fp = open(constants.LEXICONS + "finance.csv")
    fp.readline()
    polarities = {}
    for line in fp:
        info = line.split(",")
        word = info[0].lower()
        if info[7] != '0':
            polarities[word] = -1
        elif info[8] != '0':
            polarities[word] = 1
        else:
            polarities[word] = 0
    util.write_json(polarities, constants.PROCESSED_LEXICONS + "finance.json")
            

def make_concreteness_lexicon(top=75, bottom=25):
    raw_scores = {}
    fp = open(constants.LEXICONS + "concreteness/raw_ratings.csv")
    fp.readline()
    for line in fp:
        info = line.split(",")
        if len(info[0].split()) > 1:
            continue
        raw_scores[info[0]] = float(info[2])
    pos_thresh = np.percentile(raw_scores.values(), top)
    neg_thresh = np.percentile(raw_scores.values(), bottom)
    polarities = {}
    label_func = lambda s : 1 if s > pos_thresh else -1 if s < neg_thresh else 0
    for word, score in raw_scores.iteritems():
        polarities[word] = label_func(score)
    util.write_json(polarities, constants.PROCESSED_LEXICONS + "concreteness.json")
     

def make_mpqa_lexicon():
    polarities = {}
    for line in util.lines(constants.LEXICONS + 'mpqa_subjectivity.txt'):
        split = line.strip().split()
        w = split[2].split("=")[1]
        polarity = split[-1].split("=")[1]
        if polarity == 'neutral':
            polarities[w] = 0
        elif polarity == 'positive':
            polarities[w] = 1
        else:
            polarities[w] = -1
    util.write_json(polarities, constants.PROCESSED_LEXICONS + 'mpqa.json')


def make_inquirer_lexicon():
    polarities = {}
    for line in util.lines(constants.LEXICONS + 'inquirerbasic.csv'):
        for l in line.strip().split('\r'):
            split = l.split(",")
            w = split[0].lower()
            if "#" in w:
                if w.split("#")[1] != "1":
                    continue
                w = w.split("#")[0]
            polarity_neg = split[-1]
            polarity_pos = split[-2]
            if polarity_neg == 'Negativ' and polarity_pos == 'Positiv':
                continue
            elif polarity_neg == 'Negativ':
                polarities[w] = -1
            elif polarity_pos == 'Positiv':
                polarities[w] = 1
            else:
                polarities[w] = 0

    util.write_json(polarities, constants.PROCESSED_LEXICONS + 'inquirer.json')


def load_lexicon(name=constants.LEXICON, remove_neutral=True):
    lexicon = util.load_json(constants.PROCESSED_LEXICONS + name + '.json')
    return {w: p for w, p in lexicon.iteritems() if p != 0} if remove_neutral else lexicon

def compare_lexicons(print_disagreements=False):
    lexicons = {
        "inquirer": load_lexicon("inquirer", False),
        "mpqa": load_lexicon("mpqa", False),
        "bingliu": load_lexicon("bingliu", False),
    }

    for l in lexicons:
        print(l, len(lexicons[l]), len([w for w in lexicons[l] if lexicons[l][w] != 0]))

    for l1, l2 in itertools.combinations(lexicons.keys(), 2):
        ps1, ps2 = lexicons[l1], lexicons[l2]
        common_words = set(ps1.keys()) & set(ps2.keys())
        print( l1, l2, "agreement: {:.2f}".format(
                    100.0 * sum(1 if ps1[w] == ps2[w] else 0 for w in common_words) / len(common_words)))
        common_words = set([word for word in ps1.keys() if ps1[word] != 0]) & \
                       set([word for word in ps2.keys() if ps2[word] != 0])

        print(l1, l2, "agreement ignoring neutral: {:.2f}".format(
                    100.0 * sum(1 if ps1[w] * ps2[w] == 1 else 0 for w in common_words) / len(common_words)))
        
        if print_disagreements and l1 == 'opinion' and l2 == 'inquirer':
            for w in common_words:
                if lexicons[l1][w] != lexicons[l2][w]:
                    print(w, lexicons[l1][w], lexicons[l2][w])


def make_all_lexicons():
    make_bingliu_lexicon()
    make_mpqa_lexicon()
    make_inquirer_lexicon()


if __name__ == '__main__':
    make_all_lexicons()
    compare_lexicons()
