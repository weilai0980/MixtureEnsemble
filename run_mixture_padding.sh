#!/bin/bash

python3 main_mixture.py -m "src_padding" -t "none" -d "none" -g $2 -p $1 -l "lk"

python3 main_mixture.py -m "src_padding" -t "none" -d "independent" -g $2 -p $1 -l "lk"

python3 main_mixture.py -m "src_padding" -t "constant_diff_sq" -d "markov" -g $2 -p $1 -l "lk"

python3 main_mixture.py -m "src_padding" -t "scalar_diff_sq" -d "markov" -g $2 -p $1 -l "lk"

python3 main_mixture.py -m "src_padding" -t "vector_diff_sq" -d "markov" -g $2 -p $1 -l "lk"

python3 main_mixture.py -m "src_padding" -t "pos_neg_diff_sq" -d "markov" -g $2 -p $1 -l "lk"


python3 main_mixture.py -m "src_padding" -t "none" -d "none" -g $2 -p $1 -l "lk_inv"

python3 main_mixture.py -m "src_padding" -t "none" -d "independent" -g $2 -p $1 -l "lk_inv"

python3 main_mixture.py -m "src_padding" -t "constant_diff_sq" -d "markov" -g $2 -p $1 -l "lk_inv"

python3 main_mixture.py -m "src_padding" -t "scalar_diff_sq" -d "markov" -g $2 -p $1 -l "lk_inv"

python3 main_mixture.py -m "src_padding" -t "vector_diff_sq" -d "markov" -g $2 -p $1 -l "lk_inv"

python3 main_mixture.py -m "src_padding" -t "pos_neg_diff_sq" -d "markov" -g $2 -p $1 -l "lk_inv"




