# SocialSent 

### Authors: Removed for anonymity.

## Overview 

SocialSent is a package for inducing and analyzing domain-specific sentiment lexicons.
A number of state-of-the-art algorithms are included, including SentProp (URL REMOVED) and Densifier (http://www.cis.lmu.de/~sascha/Ultradense/).
A detailed description of the SentProp algorithm, as will as descriptions of other baselines in the SocialSent package is provided in (URL REMOVED).

The package also includes a set of pre-induced domain-specific lexicons for 150 years of historical English as well as for 250 online communities from reddit.

`data/lexicons` includes a number of known lexicons used for evalution etc..
`evaluate_methods.py` contains the main code used in the evaluations in the submitted EMNLP paper. This file includes hyperparameter issues etc.

## Using the code

To use SentProp you will need to either build some word vector embeddings or download some that are pre-trained.
Once this is done, you would specify the path to these embeddings in `constants.py`.
The file `constants.py` also contains some links to pre-trained embeddings that were used in (URL REMOVED).
Running `example.sh` will download some pre-trained embeddings and run SentProp on them (using the code in `example.py`).
You can build embeddings yourself with the code in the `representations` directory, which is based upon code in (URL REMOVED).

Once you have pre-trained embeddings the file `polarity_induction_methods.py` contains implementations for a suite of sentiment induction algorithms, along with some documentation on how to use them.
The file `evaluate_methods.py` also includes the evaluation script used in (URL REMOVED), which is useful to look at to get an idea of how SentProp and the other baselines work.

## Dependencies

An up-to-date Python 2.7 distribution, with the standard packages provided by the anaconda distribution is required.

In particular, the code was tested with:
* theano (0.8.0) 
* keras (0.3.3)
* numpy (1.11.0)
* scipy (0.15.1)
* sklearn (0.18.dev0 or 0.17.1)
