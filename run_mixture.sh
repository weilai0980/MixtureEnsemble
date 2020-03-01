
#python3 test_main.py -g $1 -d "../dataset/bitcoin/market2_tar5_len10/"

python3 main_mixture.py -g $1 -d "../dataset/bitcoin/market2_tar5_len10/" -p "market2_tar5_w10"

python3 main_mixture.py -g $1 -d "../dataset/bitcoin/market1_tar5_len10/" -p "market1_tar5_w10"

python3 main_mixture.py -g $1 -d "../dataset/bitcoin/market2_tar1_len10/" -p "market2_tar1_w10"

python3 main_mixture.py -g $1 -d "../dataset/bitcoin/market1_tar1_len10/" -p "market1_tar1_w10"