
python3 main_mixture.py -m "src_padding" -t "none" -d "none" -g $2 -p $1 -l "heter_lk_inv" -a "../dataset/bitcoin/market1_tar5_len10/"

python3 main_mixture.py -m "src_padding" -t "vector_diff_sq" -d "markov"  -g $2 -p $1 -l "heter_lk_inv" -a "../dataset/bitcoin/market1_tar5_len10/"

python3 main_mixture.py -m "src_padding" -t "none" -d "none" -g $2 -p $1 -l "heter_lk_inv" -a "../dataset/bitcoin/market1_tar1_len10/"

python3 main_mixture.py -m "src_padding" -t "vector_diff_sq" -d "markov"  -g $2 -p $1 -l "heter_lk_inv" -a "../dataset/bitcoin/market1_tar1_len10/"



python3 main_mixture.py -m "src_padding" -t "none" -d "none" -g $2 -p $1 -l "heter_lk_inv" -a "../dataset/bitcoin/market2_tar5_len10/"

python3 main_mixture.py -m "src_padding" -t "vector_diff_sq" -d "markov"  -g $2 -p $1 -l "heter_lk_inv" -a "../dataset/bitcoin/market2_tar5_len10/"

python3 main_mixture.py -m "src_padding" -t "none" -d "none" -g $2 -p $1 -l "heter_lk_inv" -a "../dataset/bitcoin/market2_tar1_len10/"

python3 main_mixture.py -m "src_padding" -t "vector_diff_sq" -d "markov"  -g $2 -p $1 -l "heter_lk_inv" -a "../dataset/bitcoin/market2_tar1_len10/"




#python3 main_mixture.py -m "src_padding" -t "none" -d "none" -g $2 -p $1 -l "mse" -a "../dataset/bitcoin/market2_tar5_len10/"

#python3 main_mixture.py -m "src_padding" -t "none" -d "none" -g $2 -p $1 -l "mse" -a "../dataset/bitcoin/market2_tar5_len10/"

#python3 main_mixture.py -m "src_padding" -t "none" -d "none" -g $2 -p $1 -l "mse" -a "../dataset/bitcoin/market1_tar1_len10/"

#python3 main_mixture.py -m "src_padding" -t "none" -d "none" -g $2 -p $1 -l "mse" -a "../dataset/bitcoin/market2_tar1_len10/"


#python3 main_mixture.py -m "src_padding" -t "none" -d "independent" -g $2 -p $1 -l "lk_inv"

#python3 main_mixture.py -m "src_padding" -t "constant_diff_sq" -d "markov" -g $2 -p $1 -l "lk_inv"

#python3 main_mixture.py -m "src_padding" -t "scalar_diff_sq" -d "markov" -g $2 -p $1 -l "lk_inv"

#python3 main_mixture.py -m "src_padding" -t "vector_diff_sq" -d "markov" -g $2 -p $1 -l "lk_inv"

#python3 main_mixture.py -m "src_padding" -t "pos_neg_diff_sq" -d "markov" -g $2 -p $1 -l "lk_inv"




