# SocialSent: Domain-Specific Sentiment Lexicons for Computational Social Science

### Authors: William L. Hamilton and Kevin Clark
###[Project Website](http://nlp.stanford.edu/projects/socialsent)

## Overview 

SocialSent is a package for inducing and analyzing domain-specific sentiment lexicons.
A number of state-of-the-art algorithms are included, including SentProp and Densifier (http://www.cis.lmu.de/~sascha/Ultradense/).
A detailed description of the SentProp algorithm, as will as descriptions of other baselines in the SocialSent package is provided in the paper:
[Inducing Domain-Specific Sentiment Lexicons from Unlabeled Corpora](https://arxiv.org/abs/1606.02820).

The [project website](http://nlp.stanford.edu/projects/socialsent) includes pre-constructed sentiment lexicons for 150 years of historical English and 250 online communities from the social media forum Reddit.

If you make use of this work in your research please cite the following paper:

William L. Hamilton, Kevin Clark, Jure Leskovec, and Dan Jurafsky. Inducing Domain-Specific Sentiment Lexicons from Unlabeled Corpora. Proceedings of EMNLP. 2016. (to appear; arXiv:1606.02820).

## Install

`pip install socialsent` will work but has some downsides right now. In particular, if you use the pip install method, you will need to know where the installation directory is in order to modify the paths in the `constants.py` folder. You also won't have access to the examples that are discussed below if you install via pip. 

As a preferred alternative, download the source from the GitHub repo and run
`python setup.py install`

Note that you should re-run this command every time after you update the paths in `constants.py` folder. 

## Using the code

To use SentProp you will need to either build some word vector embeddings or download some that are pre-trained.
Once this is done, you would specify the path to these embeddings in `constants.py`.
The file `constants.py` also contains some links to pre-trained embeddings that were used in [the paper mentioned above.](https://arxiv.org/abs/1606.02820)
Running `example.sh` will download some pre-trained embeddings and run SentProp on them (using the code in `example.py`).
You can build embeddings yourself with the code in the `representations` directory, which is based upon code in https://github.com/williamleif/historical-embeddings
This code also illustrates how to use the SocialSent methods.

The file `polarity_induction_methods.py` contains implementations for a suite of sentiment induction algorithms, along with some comments/documentation on how to use them.
The file `evaluate_methods.py` also includes the evaluation script used in our published work, which is useful to look at to get an idea of how SentProp and the other baselines work.

NB: Right now the code uses dense numpy matrices in a (relatively) naive way and thus has memory requirements proportional to the square of the vocabulary size; with a reasonable amount of RAM, this works for vocabs of size 20000 words or less (which is reasonable for specific domain), but there are definitely optimizations that could be done, exploiting sparsity etc. I hope to get to these optimizations soon, but feel free to submit a pull request :). 

## Dependencies

**The code is not currently compatible with the newest Keras distribution (1.0); only the "denisfy"/Densifier method requires this package, however. So you can either install an older Keras (0.3) or remove all calls to the "densify" method. I aim to update this dependency in the near future**

An up-to-date Python 2.7 distribution, with the standard packages provided by the anaconda distribution is required. However, the code was only tested with some versions of these packages.

In particular, the code was tested with:
* theano (0.8.0) 
* keras (0.3.3)
* numpy (1.11.0)
* scipy (0.15.1)
* sklearn (0.18.dev0 or 0.17.1)
