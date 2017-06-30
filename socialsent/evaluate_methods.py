from socialsent import constants
from socialsent import util
from socialsent import polarity_induction_methods
from socialsent import seeds
from socialsent import lexicons
import sys
import random
import numpy as np
import scipy as sp
import embedding_transformer

from operator import itemgetter
from socialsent.historical import vocab
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, f1_score
from scipy.stats import kendalltau
from socialsent.representations.representation_factory import create_representation

DEFAULT_ARGUMENTS = dict(
        # for iterative graph algorithms
        similarity_power=1,
        arccos=True,
        max_iter=50,
        epsilon=1e-6,
        sym=True,

        # for learning embeddings transformation
        n_epochs=50,
        force_orthogonal=False,
        batch_size=100,
        cosine=False,

        ## bootstrap
        num_boots=1,
        n_procs=1,
)

def evaluate_methods():
    """
    Evaluates different methods on standard English.
    """
    print "Getting evalution words.."
    np.random.seed(0)
    lexicon = lexicons.load_lexicon("inquirer", remove_neutral=False)
    kuperman = lexicons.load_lexicon("kuperman", remove_neutral=False)
    eval_words = set(lexicon.keys())

    # load in WordNet lexicon and pad with zeros for missing words
    # (since these are implicitly zero for this method)
    qwn = lexicons.load_lexicon("qwn-scores")
    for word in lexicon:
        if not word in qwn:
            qwn[word] = 0

    positive_seeds, negative_seeds = seeds.hist_seeds()

    common_embed = create_representation("GIGA", constants.GOOGLE_EMBEDDINGS, 
            eval_words.union(positive_seeds).union(negative_seeds))
    embed_words = set(common_embed.iw)
    eval_words = eval_words.intersection(embed_words)

    eval_words = [word for word in eval_words 
            if not word in positive_seeds 
            and not word in negative_seeds]
    print "Evaluating with ", len(eval_words), "out of", len(lexicon)

#    print
#    print "WordNet:"
#    evaluate(qwn, lexicon, eval_words, tau_lexicon=kuperman)
#
#    print "Densifier:"
#    polarities = run_method(positive_seeds, negative_seeds, 
#            common_embed.get_subembed(set(eval_words).union(negative_seeds).union(positive_seeds)),
#            method=polarity_induction_methods.bootstrap, score_method=polarity_induction_methods.densify,
#            **DEFAULT_ARGUMENTS)
#    evaluate(polarities, lexicon, eval_words, tau_lexicon=kuperman)

    print "SentProp:"
    polarities = run_method(positive_seeds, negative_seeds, 
            common_embed.get_subembed(set(eval_words).union(negative_seeds).union(positive_seeds)),
            method=polarity_induction_methods.label_propagate_probabilistic,
            #method=polarity_induction_methods.bootstrap, 
            beta=0.99, nn=10,

            **DEFAULT_ARGUMENTS)
    evaluate(polarities, lexicon, eval_words, tau_lexicon=kuperman)
    util.write_pickle(polarities, "tmp/gi-cc-walk-pols.pkl")

def hyperparam_eval():
    print "Getting evaluation words and embeddings"
    lexicon = lexicons.load_lexicon("bingliu", remove_neutral=False)
    eval_words = set(lexicon.keys())

    positive_seeds, negative_seeds = seeds.hist_seeds()

    common_embed = create_representation("GIGA", constants.COMMON_EMBEDDINGS, 
            eval_words.union(positive_seeds).union(negative_seeds))
    common_words = set(common_embed.iw)
    eval_words = eval_words.intersection(common_words)

    hist_embed = create_representation("SVD", constants.SVD_EMBEDDINGS + "1990")
    hist_words = set(hist_embed.iw)
    eval_words = eval_words.intersection(hist_words)

    eval_words = [word for word in eval_words
            if not word in positive_seeds 
            and not word in negative_seeds] 

    print "SentProp..."
    for nn in [5, 10, 25, 50]:
        for beta in [0.8, 0.9, 0.95, 0.99]:
          print "Common"
          polarities = run_method(positive_seeds, negative_seeds, 
                    common_embed.get_subembed(set(eval_words).union(negative_seeds).union(positive_seeds)),
                    method=polarity_induction_methods.random_walk, 
                    nn=nn, beta=beta,
                    **DEFAULT_ARGUMENTS)
          evaluate(polarities, lexicon, eval_words)
          print "Hist"
          polarities = run_method(positive_seeds, negative_seeds, 
                    hist_embed.get_subembed(set(eval_words).union(negative_seeds).union(positive_seeds)),
                    method=polarity_induction_methods.random_walk, 
                    nn=nn, beta=beta,
                    **DEFAULT_ARGUMENTS)
          evaluate(polarities, lexicon, eval_words)

    print "Densify..."
    for lr in [0.001, 0.01, 0.1, 0.5]:
        for reg in [0.001, 0.01, 0.1, 0.5]:
          print "LR : ", lr, "Reg: ", reg
          print "Common"
          polarities = run_method(positive_seeds, negative_seeds, 
                    common_embed.get_subembed(set(eval_words).union(negative_seeds).union(positive_seeds)),
                    method=polarity_induction_methods.densify, 
                    lr=lr, regularization_strength=reg,
                    **DEFAULT_ARGUMENTS)
          evaluate(polarities, lexicon, eval_words, tern=False)
          print "Hist"
          polarities = run_method(positive_seeds, negative_seeds, 
                    hist_embed.get_subembed(set(eval_words).union(negative_seeds).union(positive_seeds)),
                    method=polarity_induction_methods.densify, 
                    lr=lr, regularization_strength=reg,
                    **DEFAULT_ARGUMENTS)
          evaluate(polarities, lexicon, eval_words, tern=False)


def evaluate_overlap_methods():
    """
    Evaluate different methods on standard English,
    but restrict to words that are present in the 1990s portion of historical data.
    """
    print "Getting evalution words and embeddings.."
    np.random.seed(0)
    lexicon = lexicons.load_lexicon("inquirer", remove_neutral=False)
    kuperman = lexicons.load_lexicon("kuperman", remove_neutral=False)
    eval_words = set(lexicon.keys())

    # load in WordNet lexicon and pad with zeros for missing words
    # (since these are implicitly zero for this method)
    qwn = lexicons.load_lexicon("qwn-scores")
    for word in lexicon:
        if not word in qwn:
            qwn[word] = 0

    positive_seeds, negative_seeds = seeds.hist_seeds()

#    common_embed = create_representation("GIGA", constants.COMMON_EMBEDDINGS, 
#            eval_words.union(positive_seeds).union(negative_seeds))
#    common_words = set(common_embed.iw)
#    eval_words = eval_words.intersection(common_words)

    hist_embed = create_representation("SVD", constants.COHA_EMBEDDINGS + "2000")
    hist_counts = create_representation("Explicit", constants.COHA_COUNTS + "2000", normalize=False)
    hist_words = set(hist_embed.iw)
    eval_words = eval_words.intersection(hist_words)

    eval_words = [word for word in eval_words
            if not word in positive_seeds 
            and not word in negative_seeds] 

    hist_counts = hist_counts.get_subembed(set(eval_words).union(positive_seeds).union(negative_seeds), 
            restrict_context=False)

    print "Evaluating with ", len(eval_words), "out of", len(lexicon)

    print "PMI"
    polarities = run_method(positive_seeds, negative_seeds,
            hist_counts,
            method=polarity_induction_methods.bootstrap,
            score_method=polarity_induction_methods.pmi,
            **DEFAULT_ARGUMENTS)
    evaluate(polarities, lexicon, eval_words, tau_lexicon=kuperman)

    print
    evaluate(qwn, lexicon, eval_words, tau_lexicon=kuperman)

    print "SentProp with 1990s Fic embeddings"
    polarities = run_method(positive_seeds, negative_seeds, 
                        hist_embed.get_subembed(set(eval_words).union(negative_seeds).union(positive_seeds)),
                        method=polarity_induction_methods.bootstrap,
                        score_method=polarity_induction_methods.random_walk, 
                        nn=25, beta=0.9,
                        **DEFAULT_ARGUMENTS)
    evaluate(polarities, lexicon, eval_words, tau_lexicon=kuperman)
    print
    
    print "Densifier with 1990s Fic embeddings"
    polarities = run_method(positive_seeds, negative_seeds, 
                        hist_embed.get_subembed(set(eval_words).union(negative_seeds).union(positive_seeds)),
                        method=polarity_induction_methods.bootstrap,
                        score_method=polarity_induction_methods.densify,
                        **DEFAULT_ARGUMENTS)
    evaluate(polarities, lexicon, eval_words, tau_lexicon=kuperman)
    print

    print "Velikovich with 1990s Fic embeddings"
    hist_counts.normalize()
    polarities = run_method(positive_seeds, negative_seeds, 
                        hist_counts,
                        method=polarity_induction_methods.bootstrap,
                        score_method=polarity_induction_methods.graph_propagate,
                        T=3,
                        **DEFAULT_ARGUMENTS)
    evaluate(polarities, lexicon, eval_words, tau_lexicon=kuperman)
    print

#    print "SentProp with CC"
#    polarities = run_method( positive_seeds, negative_seeds, 
#                        common_embed.get_subembed(set(eval_words).union(negative_seeds).union(positive_seeds)),
#                        method=polarity_induction_methods.bootstrap,
#                        score_method=polarity_induction_methods.random_walk,
#                        beta=0.99, nn=10,
#                        **DEFAULT_ARGUMENTS)
#    evaluate(polarities, lexicon, eval_words, tau_lexicon=kuperman)
#
#    print "Densifier with CC"
#    polarities = run_method( positive_seeds, negative_seeds, 
#                        common_embed.get_subembed(set(eval_words).union(negative_seeds).union(positive_seeds)),
#                        method=polarity_induction_methods.bootstrap,
#                        score_method=polarity_induction_methods.densify,
#                        **DEFAULT_ARGUMENTS)
#    evaluate(polarities, lexicon, eval_words, tau_lexicon=kuperman)



def evaluate_adj_methods():
    """
    Evaluate different methods on standard English,
    but restrict to words that are present in the 1990s portion of historical data.
    """
    print "Getting evalution words and embeddings.."
    np.random.seed(0)
    lexicon = lexicons.load_lexicon("inquirer", remove_neutral=False)
    kuperman = lexicons.load_lexicon("kuperman", remove_neutral=False)
    eval_words = set(lexicon.keys())
    adjs = vocab.pos_words("1990", "ADJ")

    # load in WordNet lexicon and pad with zeros for missing words
    # (since these are implicitly zero for this method)
    qwn = lexicons.load_lexicon("qwn-scores")
    for word in lexicon:
        if not word in qwn:
            qwn[word] = 0

    positive_seeds, negative_seeds = seeds.adj_seeds()

    common_embed = create_representation("GIGA", constants.COMMON_EMBEDDINGS, 
            eval_words.union(positive_seeds).union(negative_seeds))
    common_words = set(common_embed.iw)
    eval_words = eval_words.intersection(common_words)

    hist_embed = create_representation("SVD", constants.COHA_EMBEDDINGS + "2000")
    hist_counts = create_representation("Explicit", constants.COUNTS + "1990", normalize=False)
    hist_words = set(hist_embed.iw)
    eval_words = eval_words.intersection(hist_words)

    embed_words = [word for word in adjs if word in hist_words and word in common_words]
    eval_words = [word for word in eval_words if word in embed_words
            and not word in positive_seeds 
            and not word in negative_seeds] 
    
    hist_counts = hist_counts.get_subembed(set(eval_words).union(positive_seeds).union(negative_seeds), 
            restrict_context=False)

    print "Evaluating with ", len(eval_words), "out of", len(lexicon)
    print "Embeddings with ", len(embed_words)

    print "PMI"
    polarities = run_method(positive_seeds, negative_seeds,
            hist_counts,
            method=polarity_induction_methods.bootstrap,
            score_method=polarity_induction_methods.pmi,
            boot_size=6,
            **DEFAULT_ARGUMENTS)
    evaluate(polarities, lexicon, eval_words, tau_lexicon=kuperman)

    print
    evaluate(qwn, lexicon, eval_words, tau_lexicon=kuperman)

    print "Dist with 1990s Fic embeddings"
    polarities = run_method(positive_seeds, negative_seeds, 
                        hist_embed.get_subembed(set(embed_words).union(negative_seeds).union(positive_seeds)),
                        method=polarity_induction_methods.dist, 
                        **DEFAULT_ARGUMENTS)
    evaluate(polarities, lexicon, eval_words, tau_lexicon=kuperman)
    print

    print "Densifier with 1990s Fic embeddings"
    polarities = run_method(positive_seeds, negative_seeds, 
                        hist_embed.get_subembed(set(embed_words).union(negative_seeds).union(positive_seeds)),
                        method=polarity_induction_methods.bootstrap, 
                        score_method=polarity_induction_methods.densify, 
                        boot_size=6,
                        **DEFAULT_ARGUMENTS)
    evaluate(polarities, lexicon, eval_words, tau_lexicon=kuperman)
    print

    print "SentProp with 1990s Fic embeddings"
    polarities = run_method(positive_seeds, negative_seeds, 
                        hist_embed.get_subembed(set(embed_words).union(negative_seeds).union(positive_seeds)),
                        method=polarity_induction_methods.bootstrap, 
                        nn=25, beta=0.9,
                        boot_size=6,
                        **DEFAULT_ARGUMENTS)
    evaluate(polarities, lexicon, eval_words, tau_lexicon=kuperman)
    print

    print "Velikovich with 1990s Fic embeddings"
    hist_counts.normalize()
    polarities = run_method(positive_seeds, negative_seeds, 
                        hist_counts,
                        method=polarity_induction_methods.bootstrap, 
                        score_method=polarity_induction_methods.graph_propagate,
                        T=3,
                        boot_size=6,
                        **DEFAULT_ARGUMENTS)
    evaluate(polarities, lexicon, eval_words, tau_lexicon=kuperman)
    print

    print "SentProp with CC"
    polarities = run_method( positive_seeds, negative_seeds, 
                        common_embed.get_subembed(set(embed_words).union(negative_seeds).union(positive_seeds)),
                        method=polarity_induction_methods.bootstrap, 
                        score_method=polarity_induction_methods.random_walk,
                        beta=0.99, nn=10,
                        boot_size=6,
                        **DEFAULT_ARGUMENTS)
    evaluate(polarities, lexicon, eval_words, tau_lexicon=kuperman)

    print "Densifier with CC"
    polarities = run_method( positive_seeds, negative_seeds, 
                        common_embed.get_subembed(set(embed_words).union(negative_seeds).union(positive_seeds)),
                        method=polarity_induction_methods.bootstrap, 
                        score_method=polarity_induction_methods.densify,
                        boot_size=6,
                        **DEFAULT_ARGUMENTS)
    evaluate(polarities, lexicon, eval_words, tau_lexicon=kuperman)


def evaluate_finance_methods():
    np.random.seed(0)
    print "Getting evalution words and embeddings.."
    gi = lexicons.load_lexicon("inquirer", remove_neutral=False)
    lexicon = lexicons.load_lexicon("finance", remove_neutral=True)

    ### padding in neutrals from GI lexicon
    gi_neut = [word for word in gi if gi[word] == 0]
    gi_neut = np.random.choice(gi_neut, int( (float(len(gi_neut))/(len(gi)-len(gi_neut)) * len(lexicon))))
    for word in gi_neut:
        lexicon[word] = 0
    positive_seeds, negative_seeds = seeds.finance_seeds()
    stock_embed = create_representation("SVD", constants.STOCK_EMBEDDINGS)
    stock_counts = create_representation("Explicit", constants.STOCK_COUNTS)
    common_embed = create_representation("GIGA", constants.COMMON_EMBEDDINGS, set(lexicon.keys()).union(positive_seeds).union(negative_seeds))

    stock_words = set(stock_embed.iw)
    common_words = set(common_embed)
    eval_words = [word for word in lexicon if word in stock_words and
            word in common_words and
            not word in positive_seeds and  
            not word in negative_seeds]

    stock_counts = stock_counts.get_subembed(set(eval_words).union(positive_seeds).union(negative_seeds), restrict_context=False)

    print "Evaluating with ", len(eval_words), "out of", len(lexicon)

    print "Velikovich with 1990s Fic embeddings"
    stock_counts.normalize()
    polarities = run_method(positive_seeds, negative_seeds, 
                        stock_counts,
                        method=polarity_induction_methods.bootstrap, 
                        score_method=polarity_induction_methods.graph_propagate,
                        T=3,
                        boot_size=6,
                        **DEFAULT_ARGUMENTS)
    evaluate(polarities, lexicon, eval_words, tau_lexicon=None)
    print


    print "PMI"
    polarities = run_method(positive_seeds, negative_seeds,
            stock_counts,
            method=polarity_induction_methods.bootstrap, 
            score_method=polarity_induction_methods.pmi,
            **DEFAULT_ARGUMENTS)
    evaluate(polarities, lexicon, eval_words)
    print

    print "SentProp with stock embeddings"
    polarities = run_method(positive_seeds, negative_seeds, 
                        stock_embed.get_subembed(set(eval_words).union(negative_seeds).union(positive_seeds)),
                        method=polarity_induction_methods.bootstrap, 
                        beta=0.9, nn=25,
                        **DEFAULT_ARGUMENTS)
    evaluate(polarities, lexicon, eval_words)

    print "Densifier with stock embeddings"
    polarities = run_method(positive_seeds, negative_seeds, 
                        stock_embed.get_subembed(set(eval_words).union(negative_seeds).union(positive_seeds)),
                        method=polarity_induction_methods.bootstrap, 
                        score_method=polarity_induction_methods.densify, 
                        **DEFAULT_ARGUMENTS)
    evaluate(polarities, lexicon, eval_words)


def evaluate_twitter_methods():
    np.random.seed(0)

    print "Getting evalution words and embeddings.."
    gi = lexicons.load_lexicon("inquirer", remove_neutral=False)
    lexicon = lexicons.load_lexicon("twitter", remove_neutral=True)
    scores = lexicons.load_lexicon("twitter-scores", remove_neutral=True)
    sent140 = lexicons.load_lexicon("140-scores", remove_neutral=False)

    # padding lexicon with neutral from GI
    gi_neut = [word for word in gi if gi[word] == 0]
    gi_neut = np.random.choice(gi_neut, int( (float(len(gi_neut))/(len(gi)-len(gi_neut)) * len(lexicon))))
    for word in gi_neut:
        lexicon[word] = 0

    positive_seeds, negative_seeds = seeds.twitter_seeds()
    embed = create_representation("GIGA", constants.TWITTER_EMBEDDINGS, set(lexicon.keys()).union(positive_seeds).union(negative_seeds))
    print len((set(positive_seeds).union(negative_seeds)).intersection(embed.iw))
    embed_words = set(embed.iw)
    s140_words = set(sent140.keys())
    eval_words = [word for word in lexicon if word in s140_words and
            not word in positive_seeds 
            and not word in negative_seeds
            and word in embed_words] 

    print "Evaluating with ", len(eval_words), "out of", len(lexicon)

    print "Sentiment 140"
    evaluate(sent140, lexicon, eval_words, tau_lexicon=scores)
    print

    print "SentProp"
    polarities = run_method(positive_seeds, negative_seeds, 
                        embed,
                        method=polarity_induction_methods.bootstrap, 
                        score_method=polarity_induction_methods.densify,
                        lr=0.01, regularization_strength=0.5,
                        **DEFAULT_ARGUMENTS)
    util.write_pickle(polarities, "twitter-test.pkl")
    evaluate(polarities, lexicon, eval_words, tau_lexicon=scores)

    print "SentProp"
    polarities = run_method(positive_seeds, negative_seeds, 
                        embed,
                        method=polarity_induction_methods.bootstrap, 
                        score_method=polarity_induction_methods.random_walk,
                        beta=0.9, nn=25,
                        **DEFAULT_ARGUMENTS)
    evaluate(polarities, lexicon, eval_words, tau_lexicon=scores)


def run_method(positive_seeds, negative_seeds, embeddings, transform_embeddings=False, post_densify=False,
        method=polarity_induction_methods.densify, **kwargs):
    if transform_embeddings:
        print "Transforming embeddings..."
        embeddings = embedding_transformer.apply_embedding_transformation(embeddings, positive_seeds, negative_seeds, n_dim=50)
    if post_densify:
        polarities = method(embeddings, positive_seeds, negative_seeds, **kwargs)
        top_pos = [word for word in 
                sorted(polarities, key = lambda w : -polarities[w])[:150]]
        top_neg = [word for word in 
                sorted(polarities, key = lambda w : polarities[w])[:150]]
        top_pos.extend(positive_seeds)
        top_neg.extend(negative_seeds)
        return polarity_induction_methods.densify(embeddings, top_pos, top_neg)
    positive_seeds = [s for s in positive_seeds if s in embeddings]
    negative_seeds = [s for s in negative_seeds if s in embeddings]
    return method(embeddings, positive_seeds, negative_seeds, **kwargs)


def print_polarities(polarities, lexicon):
    for w, p in sorted(polarities.items(), key=itemgetter(1), reverse=True):
        print (util.GREEN if lexicon[w] == 1 else util.RED) + \
              "{:}: {:0.5f}".format(w, p) + util.ENDC

def evaluate(polarities, lexicon, eval_words, tau_lexicon=None, tern=True):
    acc, auc, avg_prec = binary_metrics(polarities, lexicon, eval_words)
    if auc < 0.5:
        polarities = {word:-1*polarities[word] for word in polarities}
        acc, auc, avg_prec = binary_metrics(polarities, lexicon, eval_words)
    print "Binary metrics:"
    print "=============="
    print "Accuracy with optimal threshold: {:.4f}".format(acc)
    print "ROC AUC Score: {:.4f}".format(auc)
    print "Average Precision Score: {:.4f}".format(avg_prec)
    
    if not tern:
        return 
    tau, cmn_f1, maj_f1, conf_mat = ternary_metrics(polarities, lexicon, eval_words, tau_lexicon=tau_lexicon)
    print "Ternary metrics:"
    print "=============="
    print "Majority macro F1 baseline {:.4f}".format(maj_f1)
    print "Macro F1 with cmn threshold: {:.4f}".format(cmn_f1)
    if tau:
        print "Kendall Tau {:.4f}".format(tau)
    print "Confusion matrix: "
    print conf_mat
    print "Neg :", float(conf_mat[0,0]) / np.sum(conf_mat[0,:])
    print "Neut :", float(conf_mat[1,1]) / np.sum(conf_mat[1,:])
    print "Pos :", float(conf_mat[2,2]) / np.sum(conf_mat[2,:])
    print
    if tau:
        print "Latex table line: {:2.1f} & {:2.1f} & {:.2f}\\\\".format(100*auc, 100*cmn_f1, tau)
    else:
        print "Latex table line: {:2.1f} & {:2.1f}\\\\".format(100*auc, 100*cmn_f1)


def binary_metrics(polarities, lexicon, eval_words, print_predictions=False, top_perc=None):
    eval_words = [word for word in eval_words if lexicon[word] != 0]
    y_prob, y_true = [], []
    if top_perc:
        polarities = {word:polarities[word] for word in 
                sorted(eval_words, key = lambda w : abs(polarities[w]-0.5), reverse=True)[:int(top_perc*len(polarities))]}
    else:
        polarities = {word:polarities[word] for word in eval_words}
    for w in polarities:
        y_prob.append(polarities[w])
        y_true.append(1 + lexicon[w] / 2)

    n = len(y_true)
    ordered_labels = [y_true[i] for i in sorted(range(n), key=lambda i: y_prob[i])]
    positive = sum(ordered_labels)
    cumsum = np.cumsum(ordered_labels)
    best_accuracy = max([(1 + i - cumsum[i] + positive - cumsum[i]) / float(n) for i in range(n)])

    return best_accuracy, roc_auc_score(y_true, y_prob), average_precision_score(y_true, y_prob)

def ternary_metrics(polarities, lexicon, eval_words, tau_lexicon=None):
    if not tau_lexicon == None:
        kendall_words = list(set(eval_words).intersection(tau_lexicon))
    y_prob, y_true = [], []
    polarities = {word:polarities[word] for word in eval_words}
    for w in polarities:
        y_prob.append(polarities[w])
        y_true.append(lexicon[w])
    y_prob = np.array(y_prob)
    y_true = np.array(y_true)
    y_prob = 2*(y_prob - np.min(y_prob)) / (np.max(y_prob) - np.min(y_prob)) - 1
    neg_prop = np.sum(np.array(lexicon.values()) == -1) / float(len(lexicon))
    pos_prop = np.sum(np.array(lexicon.values()) == 1) / float(len(lexicon))
    sorted_probs = sorted(y_prob)
    neg_thresh = sorted_probs[int(np.round(neg_prop*len(sorted_probs)))]
    pos_thresh = sorted_probs[-int(np.round(pos_prop*len(sorted_probs)))]
    cmn_labels = [1 if val >= pos_thresh else -1 if val <= neg_thresh else 0 for val in y_prob]
    if not tau_lexicon == None:
        tau = kendalltau(*zip(*[(polarities[word], tau_lexicon[word]) for word in kendall_words]))[0]
    else:
        tau = None
    maj_f1 = f1_score(y_true, np.repeat(sp.stats.mode(y_true)[0][0], len(y_true)), average="macro")
    cmn_f1 = f1_score(y_true, cmn_labels, average="macro")
    label_func = lambda entry : 1 if entry > pos_thresh else -1 if entry < neg_thresh else 0
    conf_mat = confusion_matrix(y_true, [label_func(entry) for entry in y_prob])
    return tau, cmn_f1, maj_f1, conf_mat

def optimal_tern_acc(polarities, lexicon, eval_words, threshes=np.arange(0.95, 0.0, -0.01)):
    """
    Performs grid search to determine optimal ternary accuracy.
    """
    y_prob, y_true = [], []
    polarities = {word:polarities[word] for word in eval_words}
    for w in polarities:
        y_prob.append(polarities[w])
        y_true.append(lexicon[w])
    y_prob = np.array(y_prob)
    y_true = np.array(y_true)
    y_prob = 2*(y_prob - np.min(y_prob)) / (np.max(y_prob) - np.min(y_prob)) - 1
    f1s = np.zeros((len(threshes)**2,))
    for i, pos_thresh in enumerate(threshes):
        for k, neg_thresh in enumerate(threshes):
            labels = []
            for j in range(len(y_prob)):
                if y_prob[j] > pos_thresh:
                    labels.append(1)
                elif y_prob[j] < -1*neg_thresh:
                    labels.append(-1)
                else:
                    labels.append(0)
            f1s[i*len(threshes)+k] = f1_score(y_true, labels, average="macro")
    print "(Oracle) majority baseline {:.4f}".format(
            f1_score(y_true, np.repeat(sp.stats.mode(y_true)[0][0], len(y_true)), average="macro"))
    print "Accuracy with optimal threshold: {:.4f}".format(np.max(f1s))
    best_iter = int(np.argmax(f1s))
    pos_thresh = threshes[best_iter / len(threshes)]
    neg_thresh = -1*threshes[best_iter % len(threshes)]
    print "Optimal positive threshold: {:.4f}".format(pos_thresh)
    print "Optimal negative threshold: {:.4f}".format(neg_thresh)
    print "Confusion matrix: "
    label_func = lambda entry : 1 if entry > pos_thresh else -1 if entry < neg_thresh else 0
    conf_mat = confusion_matrix(y_true, [label_func(entry) for entry in y_prob])
    print conf_mat
    print "Neg :", float(conf_mat[0,0]) / np.sum(conf_mat[0,:])
    print "Neut :", float(conf_mat[1,1]) / np.sum(conf_mat[1,:])
    print "Pos :", float(conf_mat[2,2]) / np.sum(conf_mat[2,:])


if __name__ == '__main__':
    random.seed(0)
    if sys.argv[1] == "twitter":
        evaluate_twitter_methods()
    elif sys.argv[1] == "finance":
        evaluate_finance_methods()
    elif sys.argv[1] == "overlap":
        evaluate_overlap_methods()
    elif sys.argv[1] == "adj":
        evaluate_adj_methods()
    elif sys.argv[1] == "hyper":
        hyperparam_eval()
    else:
        evaluate_methods()
