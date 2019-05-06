#!/usr/bin/python

import sys
import os

import numpy as np
import random
from random import shuffle

import time
import json

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.contrib import rnn

# local packages 
from utils_libs import *


# ----- hyper-parameters from command line

import argparse

parser = argparse.ArgumentParser()
#parser.add_argument('--model', '-m', help = "model", type = str, default = 'statistic')
parser.add_argument('--latent_prob_type', '-t', help = "latent_prob_type", type = str, default = "none")
# "none", "constant_diff_sq", "scalar_diff_sq", "vector_diff_sq"
parser.add_argument('--latent_dependence', '-d', help = "latent_dependence", type = str, default = "none")
# "none", "independent", "markov"
parser.add_argument('--data_mode', '-m', help = "source specific data or with paddning", type = str, default = "src_padding")
# "src_raw", "src_padding"

parser.add_argument('--gpu_id', '-g', help = "gpu_id", type = str, default = "0")
parser.add_argument('--target_distr', '-p', help = "target_probability_distribution", type = str, default = "gaussian")

parser.add_argument('--loss_type', '-l', help = "loss_type", type = str, default = "lk")

args = parser.parse_args()
print(args)

method_str = 'statistic'

if args.data_mode == "src_raw":
    from mixture import *

elif args.data_mode == "src_padding": 
    from mixture_padding import *
    

# ------ GPU set-up in multi-GPU environment

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


# ----- hyper-parameters from config

'''
import json
with open('config.json') as f:
    para_dict = json.load(f)
    print(para_dict) 
'''

para_step_ahead = 0


# ----- log paths

path_data = "../dataset/bitcoin/double_trx_ob_10/"
#"../dataset/bitcoin/double_trx_ob_10/"

path_log_error = "../results/mixture/log_error_mix.txt"
path_log_epoch  = "../results/mixture/log_epoch_mix.txt"
path_model = "../results/mixture/"

path_py = "../results/mixture/py_" + args.target_distr + "_" + args.loss_type + "_" + args.latent_dependence + "_" + args.latent_prob_type + ".p"

# ----- hyper-parameters set-up

para_y_log = False

para_bool_bilinear = True

para_batch_size = 64
para_n_epoch = 50

para_distr_type = args.target_distr
# gaussian, student_t
para_distr_para = [3]
# gaussian: [] 
# student_t: [nu], nu>=3

para_loss_type = args.loss_type
para_var_type = "square" # square, exp

para_regu_positive = False
para_regu_gate = False # -
para_regu_global_gate = False  
para_regu_latent_dependence = False

para_bool_bias_mean = False
para_bool_bias_var = False
para_bool_bias_gate = False

para_bool_target_seperate = False

para_latent_dependence = args.latent_dependence
para_latent_prob_type = args.latent_prob_type

para_validation_metric = 'rmse'
para_metric_map = {'rmse':3, 'mae':4, 'mape':5, 'nnllk':6}

# epoch sample
para_val_epoch_num = max(1, int(0.05*para_n_epoch))
para_test_epoch_num = 1

para_lr_range = [0.001, ]
para_l2_range = [1e-7, 0.000001, 0.00001, 0.0001, 0.001, 0.01]
#[1e-7, 0.000001, 0.00001, 0.0001, 0.001, 0.01]


# ----- training and evalution
    
def train_validate(xtr, 
                   ytr, 
                   xval, 
                   yval, 
                   lr, 
                   l2, 
                   dim_x, 
                   steps_x, 
                   retrain_epoch_set, 
                   retrain_bool,
                   retrain_best_val_err
                   ):
    
    '''
    Args:
    
    xtr: [num_src, N, T, D]
         N: number of data samples
         T: number of steps
         D: dimension at each time step
        
    ytr: [N 1]
        
    l2: float, l2 regularization
    
    lr: float, learning rate
    
    dim_x: integer, corresponding to D
    
    steps_x: integer, corresponding to T
       
    '''
    
    # clear the graph in the current session 
    tf.reset_default_graph()
    
    # stabilize the network by fixing the random seed
    np.random.seed(1)
    tf.set_random_seed(1)
    
    
    with tf.device('/device:GPU:0'):
        
        # session set-up
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        
        sess = tf.Session(config = config)
        
        # fix the random seed to stabilize the network
        np.random.seed(1)
        tf.set_random_seed(1)
        
    #with tf.Session() as sess:
        
        model = mixture_statistic(session = sess, 
                                  loss_type = para_loss_type,
                                  num_src = len(xtr) if type(xtr) == list else np.shape(xtr)[0])
        
        # -- initialize the network
        model.network_ini(lr, 
                          l2, 
                          dim_x = dim_x,
                          steps_x = steps_x, 
                          bool_log = para_y_log, 
                          bool_bilinear = para_bool_bilinear,
                          distr_type = para_distr_type, 
                          distr_para = para_distr_para,
                          bool_regu_positive_mean = para_regu_positive,
                          bool_regu_gate = para_regu_gate, 
                          bool_regu_global_gate = para_regu_global_gate, 
                          bool_regu_latent_dependence = para_regu_latent_dependence,
                          latent_dependence = para_latent_dependence,
                          latent_prob_type = para_latent_prob_type,
                          var_type = para_var_type,
                          bool_bias_mean = para_bool_bias_mean,
                          bool_bias_var = para_bool_bias_var,
                          bool_bias_gate = para_bool_bias_gate)

        model.train_ini()
        model.inference_ini()
        
        # -- set up training batch parameters
        total_cnt = len(xtr[0])
        total_batch_num = int(total_cnt/para_batch_size)
        total_idx = list(range(total_cnt))
        
        # -- begin training on epochs
        
        # log training and validation errors over epoches
        epoch_error = []
        
        saver = tf.train.Saver()
        
        st_time = time.time()
        
        for epoch in range(para_n_epoch):
            
            # shuffle traning instances each epoch
            np.random.shuffle(total_idx)
            
            # loop over all batches
            epoch_loss = 0.0
            epoch_sq_err = 0.0
            
            for i in range(total_batch_num):
                
                # batch data
                batch_idx = total_idx[i*para_batch_size: (i+1)*para_batch_size] 
                
                batch_x = [xtr[tmp_src][batch_idx] for tmp_src in range(len(xtr))]
                
                # log transformation on the target
                if para_y_log == True:
                    batch_y = log(ytr[batch_idx]+1e-5)
                    
                else:
                    batch_y = ytr[batch_idx]
                    
                # update on the batch data
                tmp_loss, tmp_sq_err = model.train_batch(batch_x, 
                                                         batch_y)
                
                epoch_loss += tmp_loss
                epoch_sq_err += tmp_sq_err
            
            
            # -- epoch-wise evaluation
            
            # nnllk: normalized negative log likelihood
            # monitor_metric = [gate, py_mean_src]
            
            val_rmse, val_mae, val_mape, val_nnllk, _, monitor_metric = model.inference(xval, 
                                                                                        yval, 
                                                                                        bool_py_eval = False)
            
            tr_rmse = sqrt(1.0*epoch_sq_err/total_cnt)
            
            epoch_error.append([epoch,
                                1.0*epoch_loss/total_batch_num,
                                tr_rmse, 
                                val_rmse, 
                                val_mae, 
                                val_mape, 
                                val_nnllk])
            
            print("\n --- At epoch %d : \n  %s "%(epoch, str(epoch_error[-1][1:])))
            print("\n gates : \n", monitor_metric[0])
            print("\n py_mean_src : \n", monitor_metric[1])
            
            # NULL value exception 
            if np.isnan(epoch_loss) == True:
                break
            
            # save the model w.r.t. the epoch in epoch_sample
            # ? val_rmse < retrain_best_val_err or
            if retrain_bool == True and (epoch in retrain_epoch_set):
                
                # path of the stored models 
                saver.save(sess, path_model + method_str + '_' + str(epoch))
                print("\n    [MODEL SAVED] \n")
            
        
        print("Optimization Finished!")
        
        ed_time = time.time()
    
    # the epoch with the lowest valdiation RMSE
    return sorted(epoch_error, key = lambda x:x[3]),\
           1.0*(ed_time - st_time)/(epoch + 1e-5)
    

def hyper_para_selection(hpara_log, 
                         val_epoch_num, 
                         test_epoch_num,
                         metric_idx):
    
    # hpara_log - [ [hp1, hp2, ...], [[epoch, loss, train_rmse, val_rmse, val_mae, val_mape, val_nnllk]] ]
    
    hp_err = []
    
    for hp_epoch_err in hpara_log:
        
        hp_err.append([hp_epoch_err[0], hp_epoch_err[1], np.mean([k[metric_idx] for k in hp_epoch_err[1][:val_epoch_num]])])
    
    sorted_hp = sorted(hp_err, key = lambda x:x[-1])
    
    # -- print out for checking
    print([(i[0], i[-1]) for i in sorted_hp])
    
    # best hp1, best hp2, epoch_sample, best validation error
    return sorted_hp[0][0][0],\
           sorted_hp[0][0][1],\
           [k[0] for k in sorted_hp[0][1]][:test_epoch_num],\
           min([tmp_epoch[3] for tmp_epoch in sorted_hp[0][1]])


def test_nn(epoch_set, 
            xts, 
            yts, 
            file_path, 
            bool_instance_eval,
            loss_type,
            num_src):
    
    # ensemble of model snapshots
    for tmp_epoch in epoch_set:
        
        # path of the stored models 
        tmp_meta = file_path + method_str + '_' + str(tmp_epoch) + '.meta'
        tmp_data = file_path + method_str + '_' + str(tmp_epoch)
        
        # clear graph
        tf.reset_default_graph()
        
        with tf.device('/device:GPU:0'):
            
            config = tf.ConfigProto()
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True
        
            sess = tf.Session(config = config)
        
            model = mixture_statistic(session = sess, 
                                      loss_type = para_loss_type,
                                      num_src = num_src)
            
            # restore the model    
            model.pre_train_restore_model(tmp_meta, tmp_data)
            
            # rmse, mae, mape, nnllk, py tuple [], monitor tuple []
            return model.inference(xts,
                                   yts, 
                                   bool_py_eval = bool_instance_eval)
            

def log_train(path):
    
    with open(path, "a") as text_file:
        
        text_file.write("\n\n ------ Statistic mixture : \n")
        
        text_file.write("data_mode : %s \n"%(args.data_mode))
        text_file.write("data source timesteps : %s \n"%(para_steps_x))
        text_file.write("data source feature dimensionality : %s \n"%(para_dim_x))
        text_file.write("data source number : %d \n"%(len(ts_x) if type(ts_x)==list else np.shape(ts_x)[0]))
        
        text_file.write("loss type : %s \n"%(para_loss_type))
        text_file.write("bi-linear : %s \n"%(para_bool_bilinear))
        
        text_file.write("target distribution type : %s \n"%(para_distr_type))
        text_file.write("target distribution para. : %s \n"%(str(para_distr_para)))
        text_file.write("target variable as a seperated data source : %s \n"%(para_bool_target_seperate))
        
        text_file.write("regularization on positive means : %s \n"%(para_regu_positive))
        text_file.write("regularization on mixture gates : %s \n"%(para_regu_gate))
        text_file.write("regularization by global gate : %s \n"%(para_regu_global_gate))
        text_file.write("regularization on latent dependence parameters : %s \n"%(para_regu_latent_dependence))
        
        text_file.write("epoch num. in validation : %s \n"%(para_val_epoch_num))
        text_file.write("epoch ensemble num. in testing : %s \n"%(para_test_epoch_num))
        
        text_file.write("batch size : %s \n"%(para_batch_size))
        text_file.write("number of epochs : %s \n"%(para_n_epoch))
        
        text_file.write("temporal dependence of latent variables : %s \n"%(para_latent_dependence))
        text_file.write("latent dependence probability type : %s \n"%(para_latent_prob_type))
        text_file.write("latent dependence as a regularization : %s \n"%(True if para_latent_prob_type != "none" else False))
        
        text_file.write("validation metric : %s \n"%(para_validation_metric))
        text_file.write("variance calculation type : %s \n"%(para_var_type))
        
        text_file.write("adding bias terms in mean: %s \n"%(para_bool_bias_mean))
        text_file.write("adding bias terms in variance: %s \n"%(para_bool_bias_var))
        text_file.write("adding bias terms in gates: %s \n"%(para_bool_bias_gate))
        
        text_file.write("\n\n")
        
def log_val_hyper_para(path, hpara, hpara_error, train_time):
    
    with open(path_log_error, "a") as text_env:
        text_env.write("%s, %s, %s\n"%(str(hpara), str(hpara_error), str(train_time)))
        
def log_val(path, hpara_tuple, error_tuple):
    
    with open(path, "a") as text_file:
        text_file.write("\n  best hyper-parameters: %s \n"%(str(hpara_tuple)))
        text_file.write("\n  validation performance: %s \n"%(str(error_tuple)))
        
def log_test(path, error_tuple):
    
    with open(path, "a") as text_file:
        text_file.write("\n  test performance: %s \n"%(str(error_tuple)))
        
def log_exception(path, message):
    
    with open(path, "a") as text_file:
        text_file.write("\n  NULL loss exception at: %s \n"%(str(message)))
        
def null_loss_exception(epoch_errors, log_path):
# epoch_errors: [[epoch, loss, train_rmse, val_rmse, val_mae, val_mape, val_nnllk]]    
    
    for i in epoch_errors:
        if np.isnan(i[1]) == True:
            log_exception(log_path, i[0])
            break
    return
    
        
def data_reshape(data, bool_target_seperate):
    
    # data: [yi, ti, [xi_src1, xi_src2, ...]]
    src_num = len(data[0][2])
    tmpx = []
    
    if bool_target_seperate == True:
        
        tmpx.append(np.asarray([tmp[2][0][:, 1:] for tmp in data]))
        print(np.shape(tmpx[-1]))
    
        tmpx.append(np.asarray([tmp[2][0][:, 0:1] for tmp in data]))
        print(np.shape(tmpx[-1]))
        
        for src_idx in range(1, src_num):
            tmpx.append(np.asarray([tmp[2][src_idx] for tmp in data]))
            print(np.shape(tmpx[-1]))
    
    else:
        
        for src_idx in range(src_num):
            tmpx.append(np.asarray([tmp[2][src_idx] for tmp in data]))
            print("src " + str(src_idx) + " : ", np.shape(tmpx[-1]))
    
    tmpy = np.asarray([tmp[0] for tmp in data])
    
    # output shape: x [S N T D],  y [N 1]
    return tmpx, np.expand_dims(tmpy, -1)

def data_padding_x(x, num_src):
    
    # shape of x: [S N T D]
    # T and D are different across sources
    
    num_samples = len(x[0])
    
    max_dim_t =  max([np.shape(x[i][0])[0] for i in range(num_src)])
    max_dim_d =  max([np.shape(x[i][0])[1] for i in range(num_src)])
    
    target_shape = [num_samples, max_dim_t, max_dim_d]
    
    target_x = []
    
    for tmp_src in range(num_src):
        
        zero_mask = np.zeros(target_shape)
        
        tmp_t = np.shape(x[tmp_src][0])[0]
        tmp_d = np.shape(x[tmp_src][0])[1]
        
        zero_mask[:, :tmp_t, :tmp_d] = x[tmp_src]
        
        target_x.append(zero_mask)
    
    # [S N T D]
    return target_x
    

# ----- main process  

if __name__ == '__main__':
    
    # fix random seed
    np.random.seed(1)
    tf.set_random_seed(1)
    
    # ----- data
    
    import pickle
    tr_dta = pickle.load(open(path_data + 'train.p', "rb"), encoding='latin1')
    val_dta = pickle.load(open(path_data + 'val.p', "rb"), encoding='latin1')
    ts_dta = pickle.load(open(path_data + 'test.p', "rb"), encoding='latin1')
    
    print(len(tr_dta), len(val_dta), len(ts_dta))
    
    # output from the reshape 
    # y [N 1], x [S N T D]    
    
    tr_x, tr_y = data_reshape(tr_dta, 
                              bool_target_seperate = para_bool_target_seperate)
    
    val_x, val_y = data_reshape(val_dta,
                                bool_target_seperate = para_bool_target_seperate)
    
    ts_x, ts_y = data_reshape(ts_dta,
                              bool_target_seperate = para_bool_target_seperate)
    
    print("training: ", len(tr_x[0]), len(tr_y))
    print("validation: ", len(val_x[0]), len(val_y))
    print("testing: ", len(ts_x[0]), len(ts_y))
    
    
    # -- steps and dimensionality of each source
    
    if args.data_mode == "src_raw":
        
        para_steps_x = []
        para_dim_x = []
        
        for tmp_src in range(len(tr_x)):
            
            tmp_shape = np.shape(tr_x[tmp_src][0])
        
            para_steps_x.append(tmp_shape[0])
            para_dim_x.append(tmp_shape[1])
            
            print("src " + str(tmp_src) + " shape: ", tmp_shape)
            
        
    elif args.data_mode == "src_padding": 
        
        tr_x = data_padding_x(tr_x, 
                              num_src = len(tr_x))
    
        val_x = data_padding_x(val_x, 
                               num_src = len(tr_x))
    
        ts_x = data_padding_x(ts_x, 
                              num_src = len(tr_x))
    
        print("Shapes after padding: ", np.shape(tr_x), np.shape(val_x), np.shape(ts_x))
        #print(tr_x[0][0], tr_x[1][0], tr_x[2][0])
        
        para_steps_x = np.shape(tr_x)[2] 
        para_dim_x = np.shape(tr_x)[3]
    
        
    # ----- training and validation
    
    log_train(path_log_error)
                           
    hpara_log = []
    
    # hp: hyper-parameter
    for tmp_lr in para_lr_range:
        for tmp_l2 in para_l2_range:
            
            # best validation performance
            
            # [[epoch, loss, train_rmse, val_rmse, val_mae, val_mape, val_nnllk]]
            hp_epoch_error, hp_epoch_time = train_validate(tr_x, 
                                                           tr_y,
                                                           val_x,
                                                           val_y,
                                                           lr = tmp_lr,
                                                           l2 = tmp_l2,
                                                           dim_x = para_dim_x,
                                                           steps_x = para_steps_x,
                                                           retrain_epoch_set = [],
                                                           retrain_bool = False, 
                                                           retrain_best_val_err = 0.0)
            
            hpara_log.append([[tmp_lr, tmp_l2], hp_epoch_error])
            
            print('\n Validation performance under the hyper-parameters: \n', hpara_log[-1][0], hpara_log[-1][1][0])
            print('\n Training time: \n', hp_epoch_time, '\n')
                           
            log_val_hyper_para(path_log_error, 
                               hpara = hpara_log[-1][0], 
                               hpara_error = hpara_log[-1][1][0],
                               train_time = hp_epoch_time)
            
            # null loss exception
            null_loss_exception(hp_epoch_error, 
                                path_log_error)
    
    
    # ----- re-train
    
    best_lr, best_l2, epoch_sample, best_val_err = hyper_para_selection(hpara_log, 
                                                                        val_epoch_num = para_val_epoch_num, 
                                                                        test_epoch_num = para_test_epoch_num,
                                                                        metric_idx = para_metric_map[para_validation_metric])
    
    print('\n----- Best hyper-parameters: ', best_lr, best_l2, epoch_sample, best_val_err, '\n')
        

    epoch_error, _ = train_validate(tr_x, 
                                    tr_y,
                                    val_x, 
                                    val_y,
                                    lr = best_lr,
                                    l2 = best_l2,
                                    dim_x = para_dim_x,
                                    steps_x = para_steps_x,
                                    retrain_epoch_set = epoch_sample, 
                                    retrain_bool = True,
                                    retrain_best_val_err = best_val_err)
    
    print('\n----- Re-training validation performance: ', epoch_error[0], '\n')
    
    
    log_val(path = path_log_error, 
            hpara_tuple = [best_lr, best_l2, epoch_sample, best_val_err], 
            error_tuple = epoch_error[0])
    
    # ----- testing
    
    print('\n----- testing ------ \n')
    
    rmse, mae, mape, nnllk, py_tuple, _ = test_nn(epoch_set = epoch_sample, 
                                                               xts = ts_x, 
                                                               yts = ts_y, 
                                                               file_path = path_model, 
                                                               bool_instance_eval = True,
                                                               loss_type = para_loss_type,
                                                               num_src = len(ts_x) if type(ts_x)==list else np.shape(ts_x)[0])
    
    
    
    print('\n testing errors: ', rmse, mae, mape, nnllk, '\n\n')  
    
    log_test(path = path_log_error, 
             error_tuple = [rmse, mae, mape, nnllk])
    
    print("--------- test ----------------- ", np.shape(py_tuple[0]), np.shape(py_tuple[1]))
    
    import pickle
    pickle.dump(py_tuple, open(path_py, "wb"))
