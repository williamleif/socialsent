#!/bin/bash

python evaluate_methods.py standard > results/standard.txt 
python evaluate_methods.py finance > results/finance.txt 
python evaluate_methods.py twitter > results/twitter.txt 
python evaluate_methods.py adj > results/adjectives.txt 
python evaluate_methods.py overlap > results/overlap.txt 
