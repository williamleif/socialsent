from socialsent import seeds
from socialsent import lexicons
from socialsent.polarity_induction_methods import random_walk
from socialsent.evaluate_methods import binary_metrics
from socialsent.representations.representation_factory import create_representation

if __name__ == "__main__":
    print("Evaluting SentProp with 100 dimensional GloVe embeddings")
    print("Evaluting only binary classification performance on General Inquirer lexicon")
    lexicon = lexicons.load_lexicon("inquirer", remove_neutral=True)
    pos_seeds, neg_seeds = seeds.hist_seeds()
    embeddings = create_representation("GIGA", "data/example_embeddings/glove.6B.100d.txt",
        set(lexicon.keys()).union(pos_seeds).union(neg_seeds))
    eval_words = [word for word in embeddings.iw
            if not word in pos_seeds 
            and not word in neg_seeds]
    # Using SentProp with 10 neighbors and beta=0.99
    polarities = random_walk(embeddings, pos_seeds, neg_seeds, beta=0.99, nn=10,
            sym=True, arccos=True)

    acc, auc, avg_per  = binary_metrics(polarities, lexicon, eval_words)
    print("Accuracy with best threshold: {:0.2f}".format(acc))
    print("ROC AUC: {:0.2f}".format(auc))
    print("Average precision score: {:0.2f}".format(avg_per))

