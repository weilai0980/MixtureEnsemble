#!/bin/bash

python3 main_mixture_speed.py -t "none" -d "none" -g 6

python3 main_mixture_speed.py -t "none" -d "independent" -g 6

python3 main_mixture_speed.py -t "constant_diff_sq" -d "markov" -g 6

python3 main_mixture_speed.py -t "scalar_diff_sq" -d "markov" -g 6

python3 main_mixture_speed.py -t "vector_diff_sq" -d "markov" -g 6


