#!/bin/bash

python3 main_mixture.py -t "none" -d "none" -g 6

python3 main_mixture.py -t "none" -d "independent" -g 6

python3 main_mixture.py -t "constant_diff_sq" -d "markov" -g 6

python3 main_mixture.py -t "scalar_diff_sq" -d "markov" -g 6

python3 main_mixture.py -t "vector_diff_sq" -d "markov" -g 6

python3 main_mixture.py -t "pos_neg_diff_sq" -d "markov" -g 6




