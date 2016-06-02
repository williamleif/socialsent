import constants
import util
import collections
import numpy as np
import itertools


"""
Helper methods for managing existing lexicons.
"""

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
        print l, len(lexicons[l]), len([w for w in lexicons[l] if lexicons[l][w] != 0])

    for l1, l2 in itertools.combinations(lexicons.keys(), 2):
        ps1, ps2 = lexicons[l1], lexicons[l2]
        common_words = set(ps1.keys()) & set(ps2.keys())
        print l1, l2, "agreement: {:.2f}".format(
            100.0 * sum(1 if ps1[w] == ps2[w] else 0 for w in common_words) / len(common_words))
        common_words = set([word for word in ps1.keys() if ps1[word] != 0]) & \
                       set([word for word in ps2.keys() if ps2[word] != 0])  
        print l1, l2, "agreement ignoring neutral: {:.2f}".format(
            100.0 * sum(1 if ps1[w] * ps2[w] == 1 else 0 for w in common_words) / len(common_words))
        
        if print_disagreements and l1 == 'opinion' and l2 == 'inquirer':
            for w in common_words:
                if lexicons[l1][w] != lexicons[l2][w]:
                    print w, lexicons[l1][w], lexicons[l2][w]


def make_all_lexicons():
    make_bingliu_lexicon()
    make_mpqa_lexicon()
    make_inquirer_lexicon()


if __name__ == '__main__':
    make_all_lexicons()
    compare_lexicons()
