#!/bin/bash

python3 main_mixture.py -m "src_origin" -t "none" -d "none" -g 0

python3 main_mixture.py -m "src_origin" -t "none" -d "independent" -g 0

python3 main_mixture.py -m "src_origin" -t "constant_diff_sq" -d "markov" -g 0

python3 main_mixture.py -m "src_origin" -t "scalar_diff_sq" -d "markov" -g 0

python3 main_mixture.py -m "src_origin" -t "vector_diff_sq" -d "markov" -g 0

python3 main_mixture.py -m "src_origin" -t "pos_neg_diff_sq" -d "markov" -g 0




