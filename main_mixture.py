#!/usr/bin/python

import sys
import os

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.contrib import rnn

import math
import random
from random import shuffle

import json

# local packages 
from utils_libs import *
from utils_data_prep import *
from mixture import *

# ---- hyper-parameters from command line

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', help = "model", type = str, default = 'statis')
 
args = parser.parse_args()
print(args) 

# ---- hyper-parameters from config ----


import json
with open('config.json') as f:
    para_dict = json.load(f)
    print(para_dict) 

para_dim_x = []
para_steps_x = []
para_step_ahead = 0

# loss type
# optimization: em, map, bayes

# ---- hyper-parameters set-up ----

para_y_log = False
para_pred_exp = False

para_bool_bilinear = True

para_batch_size = 32
para_n_epoch = 100

para_distr_type = 'gaussian'

para_regu_positive = False
para_regu_gate =  False

#para_gate_type = 'softmax'

# epoch sample
para_val_epoch_num = int(0.05 * para_n_epoch)
para_test_epoch_num = 1


# ---- training and evalution ----
    
def train_validate(xtr, 
                   ytr, 
                   xval, 
                   yval, 
                   lr, 
                   l2, 
                   dim_x, 
                   steps_x):
    
    '''
    Args:
    
    xtr: [num_src, N, T, D]
         N: number of data samples
         T: number of steps
         D: dimension at each time step
        
    ytr: [N 1]
        
    l2: float, l2 regularization
    
    lr: float, learning rate
    
    dim_x: list of integer, corresponding to D, [num_src]
    
    steps_x: list of integer, corresponding to T, [num_src]
        
    '''
    
    # clear the graph in the current session 
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    
    # stabilize the network by fixing the random seed
    np.random.seed(1)
    tf.set_random_seed(1)
    
    with tf.Session() as sess:
        
        model = mixture_statistic(session = sess, 
                                  loss_type = para_loss_type)
        
        # initialize the network
        model.network_ini(lr, 
                          l2, 
                          dim_x_list = dim_x,
                          steps_x_list = steps_x, 
                          bool_log = para_y_log, 
                          bool_bilinear = para_bool_bilinear,
                          distr_type = para_distr_type, 
                          bool_regu_positive_mean = para_regu_positive,
                          bool_regu_gate = para_regu_gate)
        model.train_ini()
        model.inference_ini()
        
        # set up training batch parameters
        total_cnt = np.shape(xtrain)[0]
        total_batch_num = int(total_cnt/para_batch_size)
        total_idx = range(total_cnt)
        
        # log training and validation errors over epoches
        epoch_error = []
        
        st_time = time.time()
        
        #  begin training on epochs
        for epoch in range(para_epoch):
            
            # shuffle traning instances each epoch
            np.random.shuffle(total_idx)
            
            # loop over all batches
            epoch_loss = 0.0
            epoch_sq_err = 0.0
            
            for i in range(total_batch_num):
                
                # batch data
                batch_idx = total_idx[i*para_batch_size: (i+1)*para_batch_size] 
                
                batch_x = [xtr[tmp_src][batch_idx] for tmp_src in len(xtr)]
                
                # log transformation on the target
                if para_y_log == True:
                    batch_y = log(ytrain[batch_idx]+1e-5)
                    
                else:
                    batch_y = ytrain[batch_idx]
                    
                # update on the batch data
                tmp_loss, tmp_sq_err = model.train_batch(batch_x, batch_y)
                
                epoch_loss += tmp_loss
                epoch_sq_err += tmp_sq_err
                
            tmp_rmse, tmp_mae, tmp_mape, tmp_nllk, _, _ = model.inference(xval, yval, bool_indi_eval = False) 
            
            tmp_train_rmse = sqrt(1.0*tmp_sq_err/total_cnt)
            
            epoch_error.append([epoch,
                                1.0*epoch_loss/total_batch_num,
                                tmp_train_rmse, 
                                tmp_rmse, 
                                tmp_mae, 
                                tmp_mape, 
                                tmp_nllk])
            
            print("\n --- At epoch %d : \n  %s "%(epoch, str(epoch_error[-1][1:])))
            
        print("Optimization Finished!")
        
        # reset the model
        #model.model_reset()
        #clf.train_ini()
    
    ed_time = time.time()
    
    # the epoch with the lowest valdiation RMSE
    return sorted(epoch_error, key = lambda x:x[3]), 1.0*(ed_time - st_time)/para_n_epoch
 

def hyper_para_selection(hpara_log, 
                         val_epoch_num, 
                         test_epoch_num):
    
    # hpara_log - [ [hp1, hp2, ...], [[epoch, loss, train_rmse, val_rmse, val_mae, val_mape, val_nllk]] ]
    
    hp_err = []
    
    for hp_epoch_err in hpara_log:
        
        hp_err.append([hp_epoch_err[0], hp_epoch_err[1], mean([k[1][3] for k in hp_epoch_err[:val_epoch_num]])])
        
    sorted_hp = sorted(hp_err, key = lambda x:x[-1])
    
    return sorted_hp[0][0][0], sorted_hp[0][0][1], [k[0] for k in sorted_hp[0][1]][:test_epoch_num]


def log_train(path):
    
    with open(path, "a") as text_file:
        text_file.write("\n ------ \n Statistic mixture %s \n"%(train_mode))
        text_file.write("loss type: %s \n"%(para_loss_type))
        text_file.write("target distribution type: %s \n"%(para_distr_type))
        text_file.write("bi-linear: %s \n"%(para_bool_bilinear))
        text_file.write("regularization positive: %s \n"%(para_regu_positive))
        
        text_file.write("epoch num. in validation : %s \n"%(para_val_epoch_num))
        text_file.write("epoch ensemble num. in testing : %s \n"%(para_test_epoch_num))
        
        text_file.write("\n\n")
        
                                                                                     
def log_val(path, hpara_tuple, error_tuple):
    
    with open(path, "a") as text_file:
        text_file.write("\n  best hyper-parameters: %s \n"%(hpara_tuple))
        text_file.write("\n  validation performance: %s \n"%(error_tuple))
        
        
def log_test(path, error_tuple):
    
    with open(path, "a") as text_file:
        text_file.write("\n  test performance: %s \n"%(error_tuple))
    
def data_reshape(data):
    
    # data: [yi, ti, [xi_src1, xi_src2, ...]]
    src_num = len(data[0][2])

    tmpx = []
    for src_idx in range(src_num):
        tmpx.append(np.asarray([tmp[2][src_idx] for tmp in data]))
        print(np.shape(tmpx[-1]))
    
    tmpy = np.asarray([tmp[0] for tmp in data])
    
    # output shape: y [N 1], x [S N T D]
    return tmpx, tmpy

# ---- main process ----  

if __name__ == '__main__':
    
    
    # fix random seed
    np.random.seed(1)
    tf.set_random_seed(1)
    
    # ---- log
    
    path_log_error = "../bt_results/res/rolling/log_error_mix.txt"
    path_log_epoch  = "../bt_results/res/rolling/log_epoch_mix.txt"
    path_data = "../dataset/bitcoin/double_trx/"
    
    # ---- data
    
    import pickle
    tr_dta = pickle.load(open(path_data + 'train.p', "rb"), encoding='latin1')
    val_dta = pickle.load(open(path_data + 'val.p', "rb"), encoding='latin1')
    ts_dta = pickle.load(open(path_data + 'test.p', "rb"), encoding='latin1')
    
    # output from the reshape 
    # y [N 1], x [S N T D]    
    
    tr_x, tr_y = data_reshape(tr_dta)
    val_x, val_y = data_reshape(val_dta)
    ts_x, ts_y = data_reshape(ts_dta)
    
    print(len(tr_x[0]), len(tr_y))
    print(len(val_x[0]), len(val_y))
    print(len(ts_x[0]), len(ts_y))
    
    
    '''
    
    # ---- training and validation
    
    hpara_log = []
    
    # hp: hyper-parameter
    for tmp_lr in [0.001, 0.005, 0.01, 0.05]:
        for tmp_l2 in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
            
            # best validation performance
            
            # [[epoch, loss, train_rmse, val_rmse, val_mae, val_mape, val_nllk]]
            hp_epoch_error, hp_epoch_time = train_validate(xtr, 
                                                           ytr,
                                                           xval,
                                                           yval,
                                                           lr = tmp_lr,
                                                           l2 = tmp_l2,
                                                           dim_x = para_dim_x,
                                                           steps_x = para_steps_x)
                
            hpara_log.append([[tmp_lr, tmp_l2], hp_epoch_error])
            
            print 'Current parameter set-up: \n', hpara_log[-1], '\n'
            
    
    # ---- re-train
    
    best_lr, best_l2, epoch_sample = hyper_para_selection(hpara_log, 
                                                          val_epoch_num, 
                                                          test_epoch_num)
    
    
    print ' ---- Best parameters: ', best_lr, best_l2, epoch_sample, '\n'
        

    epoch_error, _ = train_validate(xtr, 
                                    ytr,
                                    xval, 
                                    yval,
                                    lr = best_lr,
                                    l2 = best_l2,
                                    dim_x = para_dim_x,
                                    steps_x = para_steps_x)    
                                    
    
    print ' ---- Re-training performance: ', epoch_error, '\n'
    
    
    
    # ---- testing
        
        
    '''        
            
