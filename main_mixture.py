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


# ---- hyper-parameters from config ----

with open('config.json') as f:
    para_dict = json.load(f)
    print(para_dict) 

para_dim_x = []
para_steps_x = []
para_step_ahead = 0


# ---- hyper-parameters set-up ----

para_y_log = False
para_pred_exp = False

para_bool_bilinear = True

para_batch_size = 32
para_epoch = 300

para_distr_type = 'gaussian'

para_pos_regu = True
para_gate_type = 'softmax'

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
        
        if method == 'linear':
            
            model = (session = sess, 
                     loss_type = para_loss_type)
            
        else:
            print "     [ERROR] Model type"
            
            
        # initialize the network
        model.network_ini(lr, 
                          l2, 
                          dim_x_list = dim_x,
                          steps_x_list = steps_x, 
                          bool_log = para_y_log, 
                          bool_bilinear = para_bool_bilinear,
                          distr_type = para_distr_type, 
                          bool_regu_positive_mean = para_pos_regu,
                          bool_regu_gate = para_gate_type )
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
                tmp_loss, tmp_sq_err = model.train_batch(batch_x, batch_y, para_keep_prob)
                
                epoch_loss += tmp_loss
                epoch_sq_err += tmp_sq_err
                
            tmp_rmse, tmp_mae, tmp_mape, tmp_nllk = model.inference(xval, yval, para_keep_prob) 
            
            
            # record for re-training the model afterwards
            
            tmp_train_rmse = sqrt(1.0*tmp_sq_err/total_cnt)
            
            epoch_error.append([epoch,
                                1.0*epoch_loss/total_batch_num
                                tmp_train_rmse, 
                                tmp_rmse, 
                                tmp_mae, 
                                tmp_mape, 
                                tmp_nllk])
            
            print("\n --- At epoch %d : \n  %s "%(epoch, str(epoch_error[-1][1:])))
            
        print "Optimization Finished!"
        
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
    
    for hp_epoch_err in error_log:
        
        hp_err.append([hp_epoch_err[0], mean([k[3] for k in hp_epoch_err[:val_epoch_num]])])
        
    sorted_hp = sorted(hp_err, key = lambda x:x[1])
    
    
    return sorted_hp[0][0][0], sorted_hp[0][0][1], [k[1][0] for k in sorted_hp][:test_epoch_num]

   
'''
def preprocess_feature_mixture(xtrain, xtest):
    
    # split training and testing data into three feature groups      
    xtr_vol =   np.asarray( [j[0] for j in xtrain] )
    xtr_feature = np.asarray( [j[1] for j in xtrain] )

    xts_vol =   np.asarray( [j[0] for j in xtest] )
    xts_feature = np.asarray( [j[1] for j in xtest] )

    # !! IMPORTANT: feature normalization

    xts = conti_normalization_test_dta(  xts_vol, xtr_vol )
    xtr = conti_normalization_train_dta( xtr_vol )

    xts_exter = conti_normalization_test_dta(  xts_feature, xtr_feature )
    xtr_exter = conti_normalization_train_dta( xtr_feature )
    
    return np.asarray(xtr), np.asarray(xtr_exter), np.asarray(xts), np.asarray(xts_exter)
'''


def log_train(path):
    
    with open(path, "a") as text_file:
        text_file.write("\n ------ \n Statistic mixture %s \n"%(train_mode))
        text_file.write("loss type: %s \n"%(para_loss_type))
        text_file.write("target distribution type: %s \n"%(para_distr_type))
        text_file.write("bi-linear: %s \n"%(para_bool_bilinear))
        text_file.write("regularization positive: %s \n"%(para_pos_regu))
        
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
    


# ---- main process ----  

if __name__ == '__main__':
    
    
    # fix random seed
    np.random.seed(1)
    tf.set_random_seed(1)
    
    # ---- log
    
    path_log_error = "../bt_results/res/rolling/log_error_mix.txt"
    path_log_epoch  = "../bt_results/res/rolling/log_epoch_mix.txt"
    
    
    # ---- data
    
    # load raw feature and target data
    features_minu = np.load("../dataset/bitcoin/training_data/feature_minu.dat" )
    rvol_hour = np.load("../dataset/bitcoin/training_data/return_vol_hour.dat" )
    all_loc_hour = np.load("../dataset/bitcoin/loc_hour.dat" )
    print '--- Start the ' + train_mode + ' training: \n', np.shape(features_minu), np.shape(rvol_hour)
    
    # prepare the set of pairs of features and targets
    x, y, var_explain = prepare_feature_target(features_minu, rvol_hour, all_loc_hour, \
                                               para_order_minu, para_order_hour, \
                                               bool_feature_selection, para_step_ahead, False)
    
    # set up the training and evaluation interval 
    interval_num = int(len(y)/interval_len)
    

    # reset the graph
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
        
    # log for predictions in each interval
    path_pred = "../bt_results/res/rolling/" + str(i-1) + "_" + str(para_step_ahead) + '_'
        
            
    tmp_x = x[ : i*interval_len]
    tmp_y = y[ : i*interval_len]
    para_train_split_ratio = 1.0*(len(tmp_x) - interval_len)/len(tmp_x)
            
    # training, validation + testing split 
    if para_bool_bilinear == True:
        
        xtrain, ytrain, xtest, ytest = training_testing_mixture_rnn(tmp_x, tmp_y, para_train_split_ratio)
    
    else:
        
        xtrain, ytrain, xtest, ytest = training_testing_mixture_mlp(tmp_x, tmp_y, para_train_split_ratio)
            
        
    # feature split, normalization READY
    xtr, xtr_exter, xtest, xtest_exter = preprocess_feature_mixture(xtrain, xtest)
        
    # build validation and testing data 
    tmp_idx = range(len(xtest))
    tmp_val_idx = []
    tmp_ts_idx = []
        
    # even sampling the validation and testing data
    for j in tmp_idx:
        if j%2 == 0:
            tmp_val_idx.append(j)
        else:
            tmp_ts_idx.append(j)
        
    xval = xtest[tmp_val_idx]
    xval_exter = xtest_exter[tmp_val_idx]
    yval = np.asarray(ytest)[tmp_val_idx]
        
    xts = xtest[tmp_ts_idx]
    xts_exter = xtest_exter[tmp_ts_idx]
    yts = np.asarray(ytest)[tmp_ts_idx]
        
    print 'shape of training, validation and testing data: \n'                            
    print np.shape(xtr), np.shape(xtr_exter), np.shape(ytrain)
    print np.shape(xval), np.shape(xval_exter), np.shape(yval)
    print np.shape(xts), np.shape(xts_exter), np.shape(yts)
        
        
    # parameter set-up
    para_order_auto = para_order_hour
        
    if para_bool_bilinear == True:
        
        para_order_x = len(xtr_exter[0][0])
        para_order_steps = len(xtr_exter[0])
        print '     !! Time-first order !! '

    elif para_bool_bilinear == False:
        
        para_order_x = len(xtr_exter[0])
        para_order_steps = 0
        print '     !! Flattened features !! '
        
    else:
        print ' [ERROR]  bi-linear '
        
    
    # ---- training and validation
    
    hpara_log = []
    
    # hp: hyper-parameter
    for para_lr in [0.001, 0.005, 0.01, 0.05]:
        for para_l2 in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
            
            # best validation performance
            
            # [epoch, loss, train_rmse, val_rmse, val_mae, val_mape, val_nllk]
            hp_epoch_error, hp_epoch_time = train_validate(xtr, 
                                                           ytr,
                                                           xval,
                                                           yval,
                                                           lr = para_lr,
                                                           l2 = para_l2,
                                                           dim_x,
                                                           steps_x)
                
            hpara_log.append([[para_lr, para_l2], hp_epoch_error])
            
            print 'Current parameter set-up: \n', hpara_log[-1], '\n'
            
    
    # ---- re-train
    
    best_lr, best_l2, epoch_sample = hyper_para_selection(hpara_log, 
                                                          val_epoch_num, 
                                                          test_epoch_num)
    
    
    print ' ---- Best parameters: ', best_lr, best_l2, epoch_sample, '\n'
        

    epoch_error, _ = train_validate(xtr, 
                                    ytr,
                                    xts, 
                                    yts,
                                    lr = best_lr,
                                    l2 = best_l2,
                                    dim_x,
                                    steps_x)    
                                    
    
        
    print ' ---- Re-training performance: ', epoch_error, '\n'
    
    
    
    # ---- testing
        
        
            
            
