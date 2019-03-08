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


# ---- training and evalution methods ----
    
def train_validate(xtr, ytr, xval, yval, lr, l2, epoch):   
    
    # stabilize the network by fixing the random seed
    np.random.seed(1)
    tf.set_random_seed(1)
    
    with tf.Session() as sess:
        
        if method == 'linear':
            
            clf = mixture_linear(sess, 
                                 lr, 
                                 l2, 
                                 para_order_x, 
                                 para_order_steps, 
                                 para_y_log, 
                                 para_bool_bilinear,
                                 para_loss_type, 
                                 para_distr_type, 
                                 para_pos_regu, 
                                 para_gate_type)
            
        else:
            print "     [ERROR] Model type"
            
        # initialize the network
        # reset the model
        clf.train_ini()
        clf.evaluate_ini()
        
        # set up training batch parameters
        total_cnt   = np.shape(xtrain)[0]
        total_batch = int(total_cnt/para_batch_size)
        total_idx   = range(total_cnt)
        
        # log training and validation errors over epoches
        tmp_err_epoch = []
        
        #  begin training epochs
        for epoch in range(para_epoch_linear):
            
            # shuffle traning instances each epoch
            np.random.shuffle(total_idx)
            
            # loop over all batches
            tmpc = 0.0
            for i in range(total_batch):
                
                batch_idx = total_idx[ i*para_batch_size: (i+1)*para_batch_size ] 
            
                batch_auto    =  xtr_auto[ batch_idx ]
                batch_x =  xtr_x[ batch_idx ]
                
                # log transformation on the target
                if para_y_log == True:
                    batch_y = log(ytrain[batch_idx]+1e-5)
                    
                else:
                    batch_y = ytrain[batch_idx]
            
                tmpc = tmpc + float(clf.train_batch( batch_auto, batch_x, batch_y, para_keep_prob ))
            
            tmp_rmse, tmp_mae, tmp_mape, tmp_nllk = clf.inference(xts_auto, xts_x, ytest, para_keep_prob) 
            
            # record for re-training the model afterwards
            epoch_error.append([epoch, tmp_train_rmse, tmp_rmse, tmp_mae, tmp_mape, tmp_nllk])
            
            # training rmse, training regularization, testing rmse, testing regularization
            print "loss on epoch ", epoch, " : ", 1.0*tmpc/total_batch, tmp_train_rmse, tmp_test_rmse
            
        print "Optimization Finished!"
        
        # reset the model
        clf.model_reset()
        #clf.train_ini()
    
    # clear the graph in the current session 
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    return min(tmp_err_epoch, key = lambda x:x[-1]), [tmp_test_rmse, tmp_test_mae, tmp_test_mape]
    

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
    
    
# ---- main process ----  

if __name__ == '__main__':
    
    
    # fix random seed
    np.random.seed(1)
    tf.set_random_seed(1)
    
    # ---- prepare the log
    
    path_log_error = "../bt_results/res/rolling/log_error_mix.txt"
    path_log_epoch  = "../bt_results/res/rolling/log_epoch_mix.txt"
    
    
    with open(log_error, "a") as text_file:
        
        text_file.write("\n %s Mixture %s  %s %s %s %s %s \n\n"%(roll_title if train_mode == 'roll' else incre_title,
                                                                 method,
                                                                     para_loss_type, 
                                                                     para_distr_type, 
                                                                     'bi-linear' if para_bool_bilinear == True else 'linear', 
                                                                     'pos_regu' if para_pos_regu == True else 'no_pos_regu'))
    
    
    # ---- prepare the data
    
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
    
    
    
    # ---- the main loop
    
    # reset the graph
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
        
    # log for predictions in each interval
    pred_file = "../bt_results/res/rolling/" + str(i-1) + "_" + str(para_step_ahead) + '_'
        
            
    tmp_x = x[ : i*interval_len]
    tmp_y = y[ : i*interval_len]
    para_train_split_ratio = 1.0*(len(tmp_x) - interval_len)/len(tmp_x)
            
        
    # training, validation+testing split 
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
        
        
    # -- training and validation phase
    
    para_log = []
        
    for para_lr in [0.001, 0.005, 0.01, 0.05]:
        for para_l2 in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
            
            # [tmp_epoch, tmp_val_rmse, tmp_val_mae, tmp_val_mape, tmp_val_nllk]]
            epoch_error_log, epoch_time = train_validate_mixture(xtr, 
                                                                 ytr,
                                                                             xval,
                                                                             yval,
                                                                             para_lr,
                                                                             para_l2,
                                                                             0)
                
            para_log.append([[para_lr, para_l2], epoch_error_log])
            
            print 'Current parameter set-up: \n', para_log[-1], '\n'
            
        
    # -- testing phase
        
    # best global hyper-parameter
    final_para = min(para_log, key = lambda x:x[1][1])
        
    best_lr = final_para[0]
    best_l2 = final_para[1]
    best_epoch = final_para[2]
        
    print ' ---- Best parameters : ', final_para, '\n'
        
    result_tuple = [final_para[3], final_para[4]]
    

    test_error = train_validate(xtr, 
                                        ytr,
                                        xts, 
                                        yts,
                                        best_lr,
                                        best_l2,
                                        best_epoch)
    result_tuple.append(test_error)
        
    print ' ---- Training, validation and testing performance: ', final_para, test_error, '\n'
        
        
    # -- log overall errors
    
    with open(log_error, "a") as text_file:
        
        text_file.write("Interval %d : %s, %s \n" %(i-1, str(final_para[:2]), str(result_tuple))).
            
            
            
