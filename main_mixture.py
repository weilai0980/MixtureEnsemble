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
from utils_training import *


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
    from mixture_raw import *

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
#para_step_ahead = 0


# ----- data and log paths

path_data = "../dataset/bitcoin/double_trx_ob_tar5_len10/"
#"../dataset/bitcoin/double_trx_ob_10/"

path_log_error = "../results/mixture/log_error_mix.txt"
#path_log_epoch  = "../results/mixture/log_epoch_mix.txt"

path_model = "../results/mixture/"
#path_model_posterior = "../results/mixture/model_posterior/"

path_py =    "../results/mixture/py_" + args.target_distr + "_" + args.loss_type + "_" + args.latent_dependence + "_" + args.latent_prob_type + ".p"


# ----- hyper-parameters set-up


# -- model and data

para_y_log = False
para_bool_bilinear = True

para_distr_type = args.target_distr
# gaussian, student_t
para_distr_para = [3]
# gaussian: [] 
# student_t: [nu], nu>=3

para_bool_target_seperate = False
para_var_type = "square" # square, exp


# -- optimization

para_optimization_mode = "bayesian" # map
para_burn_in_epoch = 40

para_n_epoch = 90
para_loss_type = args.loss_type

para_optimizer = "adam" # RMSprop, adam, 'sgmcmc_RMSprop'
para_optimizer_lr_decay = True
para_optimizer_lr_decay_epoch = 10


# -- training and validation

para_hpara_search = "random" # random, grid 

para_lr_range = [0.001, ]
para_batch_range = [64, 32, 16, 80]
para_l2_range = [1e-7, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]

para_hpara_range = [[0.001, 0.001], [10, 80], [1e-7, 0.01]]
para_hpara_n_trial = 15

para_validation_metric = 'rmse'
para_metric_map = {'rmse':3, 'mae':4, 'mape':5, 'nnllk':6}

# model snapshot sample: epoch-wise or step-wise
para_val_aggreg_num = max(1, int(0.1*para_n_epoch))
para_test_snapshot_num = 5

para_early_stop_bool = True
para_early_stop_window = 10


# -- regularization

para_regu_positive = False
para_regu_gate = False
para_regu_global_gate = False  
para_regu_latent_dependence = False
para_regu_weight_on_latent = True

para_bool_bias_in_mean = True
para_bool_bias_in_var = True
para_bool_bias_in_gate = True

para_latent_dependence = args.latent_dependence
para_latent_prob_type = args.latent_prob_type


def log_train(path):
    
    with open(path, "a") as text_file:
        
        text_file.write("\n\n ------ Statistic mixture : \n")
        
        text_file.write("data_mode : %s \n"%(args.data_mode))
        text_file.write("data path : %s \n"%(path_data))
        text_file.write("data source timesteps : %s \n"%(para_steps_x))
        text_file.write("data source feature dimensionality : %s \n"%(para_dim_x))
        text_file.write("data source number : %d \n"%(len(ts_x) if type(ts_x)==list else np.shape(ts_x)[0]))
        text_file.write("\n")
        
        text_file.write("bi-linear : %s \n"%(para_bool_bilinear))
        text_file.write("target distribution type : %s \n"%(para_distr_type))
        text_file.write("target distribution para. : %s \n"%(str(para_distr_para)))
        text_file.write("target variable as a seperated data source : %s \n"%(para_bool_target_seperate))
        text_file.write("variance calculation type : %s \n"%(para_var_type))
        text_file.write("\n")
        
        text_file.write("regularization on positive means : %s \n"%(para_regu_positive))
        text_file.write("regularization on mixture gates : %s \n"%(para_regu_gate))
        text_file.write("regularization by global gate : %s \n"%(para_regu_global_gate))
        text_file.write("regularization on latent dependence parameters : %s \n"%(para_regu_latent_dependence))
        text_file.write("regularization l2 on latent : %s \n"%(para_regu_weight_on_latent))
        
        text_file.write("adding bias terms in mean: %s \n"%(para_bool_bias_in_mean))
        text_file.write("adding bias terms in variance: %s \n"%(para_bool_bias_in_var))
        text_file.write("adding bias terms in gates: %s \n"%(para_bool_bias_in_gate))
        text_file.write("\n")
        
        text_file.write("temporal dependence of latent variables : %s \n"%(para_latent_dependence))
        text_file.write("latent dependence probability type : %s \n"%(para_latent_prob_type))
        text_file.write("latent dependence as a regularization : %s \n"%(True if para_latent_prob_type != "none" else False))
        text_file.write("\n")
        
        text_file.write("optimization mode : %s \n"%(para_optimization_mode))
        text_file.write("burn_in_epoch : %s \n"%(para_burn_in_epoch))
        text_file.write("loss type : %s \n"%(para_loss_type))
        text_file.write("optimizer: %s \n"%(para_optimizer))
        text_file.write("number of epochs : %s \n"%(para_n_epoch))
        text_file.write("learning rate decay: %s \n"%(str(para_optimizer_lr_decay)))
        text_file.write("learning rate decay epoch: %s \n"%(str(para_optimizer_lr_decay_epoch)))
        text_file.write("\n")
        
        text_file.write("hyper-para search: %s \n"%(para_hpara_search))
        text_file.write("hyper-para range: %s \n"%(str(para_hpara_range)))
        text_file.write("hyper-para random search trials: %s \n"%(str(para_hpara_n_trial)))
        
        text_file.write("validation metric : %s \n"%(para_validation_metric))
        text_file.write("validation aggreation num. : %s \n"%(para_val_aggreg_num))
        text_file.write("testing snapshot num. : %s \n"%(para_test_snapshot_num))
        
        text_file.write("early-stoping : %s \n"%(para_early_stop_bool))
        text_file.write("early-stoping look-back window: %s \n"%(para_early_stop_window))
        
        text_file.write("\n\n")


# ----- training and evalution
    
def training_validate(xtr, 
                   ytr, 
                   xval, 
                   yval, 
                   dim_x, 
                   steps_x, 
                   hp_lr,
                   hp_batch_size,
                   hp_l2, 
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
        
    dim_x: integer, corresponding to D
    
    steps_x: integer, corresponding to T
    
    
    - hyper-parameters (hp) :
    
       hp_l2: float, l2 regularization
    
       hp_lr: float, learning rate
    
       hp_batch_size: int
    
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
        
        
        model = mixture_statistic(session = sess, 
                                  loss_type = para_loss_type,
                                  num_src = len(xtr) if type(xtr) == list else np.shape(xtr)[0]
                                 )
        
        # -- initialize the network
        
        model.network_ini(hp_lr, 
                          hp_l2, 
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
                          bool_regu_l2_on_latent = para_regu_weight_on_latent,
                          latent_dependence = para_latent_dependence,
                          latent_prob_type = para_latent_prob_type,
                          var_type = para_var_type,
                          bool_bias_mean = para_bool_bias_in_mean,
                          bool_bias_var = para_bool_bias_in_var,
                          bool_bias_gate = para_bool_bias_in_gate,
                          optimization_method = para_optimizer,
                          optimization_lr_decay = para_optimizer_lr_decay,
                          optimization_lr_decay_steps = para_optimizer_lr_decay_epoch*int(len(xtr[0])/hp_batch_size),
                          optimization_mode = para_optimization_mode,
                          burn_in_step = para_burn_in_epoch)
                          
        model.train_ini()
        model.inference_ini()
        
        # -- set up training batch parameters
        
        total_cnt = len(xtr[0])
        total_batch_num = int(total_cnt/hp_batch_size)
        total_idx = list(range(total_cnt))
        
        
        # -- begin training on epochs
        
        # training and validation error log over epoches
        epoch_error = []
        
        
        # training time counter
        st_time = time.time()
        
        for epoch in range(para_n_epoch):
            
            # shuffle traning instances each epoch
            np.random.shuffle(total_idx)
            
            # loop over all batches
            epoch_loss = 0.0
            epoch_sq_err = 0.0
            
            for i in range(total_batch_num):
                
                # batch data
                batch_idx = total_idx[i*hp_batch_size : (i+1)*hp_batch_size] 
                
                batch_x = [xtr[tmp_src][batch_idx] for tmp_src in range(len(xtr))]
                batch_y = ytr[batch_idx]
                
                # one-step training on the batch of data
                tmp_loss, tmp_sq_err = model.train_batch(batch_x, 
                                                         batch_y,
                                                         global_step = epoch)
                epoch_loss += tmp_loss
                epoch_sq_err += tmp_sq_err
            
            
            # - epoch-wise validation
            
            # nnllk: normalized negative log likelihood
            # monitor_metric = [gate, py_mean_src]
            
            val_rmse, val_mae, val_mape, val_nnllk, monitor_metric = model.validation(xval, 
                                                                                      yval)
            
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
            
            
            # - model saver 
            
            tmp_path = path_model + method_str + '_' + str(epoch)
            
            if retrain_bool == True:

                if model.model_saver(path = tmp_path,
                                  step_id_to_store = retrain_epoch_set,
                                  early_stop_bool = para_early_stop_bool,
                                  early_stop_metric_idx = para_metric_map[para_validation_metric],
                                  early_stop_window = para_early_stop_window) == True:
                    
                    print("\n    [MODEL SAVED] \n " + tmp_path)
            
            
            # NAN value exception 
            if np.isnan(epoch_loss) == True:
                print("\n --- NAN loss !! \n" )
                break
                
                
        ed_time = time.time()
        
        
        # summary of all epochs
        bayes_error_tuple = model.validation_bayesian(xval, yval)
    
    # ? the epoch with the lowest valdiation RMSE ?
    # 
    return sorted(epoch_error, key = lambda x:x[3]),\
           1.0*(ed_time - st_time)/(epoch + 1e-5), \
           bayes_error_tuple, \
           model.model_stored_id()
           
    

def testing(model_snapshots, 
            xts, 
            yts, 
            file_path, 
            bool_instance_eval,
            loss_type,
            num_src):
    
    # ensemble of model snapshots
    infer = ensemble_inference()
    
    
    for tmp_model_id in model_snapshots:
        
        # path of the stored models 
        tmp_meta = file_path + method_str + '_' + str(tmp_model_id) + '.meta'
        tmp_data = file_path + method_str + '_' + str(tmp_model_id)
        
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
            model.model_restore(tmp_meta, tmp_data)
            
            # one-shot inference sample
            rmse, mae, mape, nnllk, py_tuple = model.inference(xts,
                                                               yts, 
                                                               bool_py_eval = bool_instance_eval)
            # store the samples
            infer.add_samples(py_mean = py_tuple[0],
                              py_var = py_tuple[1],
                              py_mean_src = py_tuple[2],
                              py_var_src = py_tuple[3],
                              py_gate_src = py_tuple[4])
    
    
    # return: [rmse, mae, mape, nnllk], [py_mean, py_var, py_mean_src, py_var_src, py_gate_src]
    if len(model_snapshots) == 0:
        
        return ["None"], ["None"]
    
    elif len(model_snapshots) == 1:
        
        # error tuple, prediction tuple 
        return [rmse, mae, mape, nnllk], py_tuple
    
    else:
        
        # ensemble inference
        return infer.bayesian_inference(yts), _
    
            
    
# ----- main process  

if __name__ == '__main__':
    
    # fix random seed
    np.random.seed(1)
    tf.set_random_seed(1)
    
    # ----- data
    
    import pickle
    tr_dta = pickle.load(open(path_data + 'train.p', "rb"), encoding = 'latin1')
    val_dta = pickle.load(open(path_data + 'val.p', "rb"), encoding = 'latin1')
    ts_dta = pickle.load(open(path_data + 'test.p', "rb"), encoding = 'latin1')
    
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
        
        para_steps_x = np.shape(tr_x)[2] 
        para_dim_x = np.shape(tr_x)[3]
    
        
    # ----- training and validation
    
    log_train(path_log_error)
    
    # -- hyper-para generator 
    
    if para_hpara_search == "random":
        
        hpara_generator = hpara_random_search(para_hpara_range, 
                                              para_hpara_n_trial)
    elif para_hpara_search == "grid":

        hpara_generator = hpara_grid_search([para_lr_range, para_batch_range, para_l2_range])
        
        
    # -- begin hyper-para search
    
    # hp_: stands for hyper-parameter
    
    hpara_log = []
    
    # sample one set-up of hyper-para
    tmp_hpara = hpara_generator.one_trial()
    
    while tmp_hpara != None:
        
        # [[epoch, loss, train_rmse, val_rmse, val_mae, val_mape, val_nnllk]]
        hp_epoch_error, hp_epoch_time, hp_bayesian_error, _ = training_validate(tr_x, 
                                                       tr_y,
                                                       val_x,
                                                       val_y,
                                                       dim_x = para_dim_x,
                                                       steps_x = para_steps_x,
                                                       hp_lr = tmp_hpara[0],
                                                       hp_batch_size = int(tmp_hpara[1]),
                                                       hp_l2 = tmp_hpara[2],
                                                       retrain_epoch_set = [],
                                                       retrain_bool = False, 
                                                       retrain_best_val_err = 0.0)
        
        # [ [tmp_lr, tmp_batch, tmp_l2], [[epoch, loss, train_rmse, val_rmse, val_mae, val_mape, val_nnllk]] ]
        hpara_log.append([tmp_hpara, hp_epoch_error])
        
        # sample the next hyper-para
        tmp_hpara = hpara_generator.one_trial()
        
        
        # -- logging
        
        log_train_val_performance(path_log_error, 
                                  hpara = hpara_log[-1][0], 
                                  hpara_error = hpara_log[-1][1][0],
                                  train_time = hp_epoch_time)
        
        
        log_train_val_bayesian_error(path_log_error, 
                                     hp_bayesian_error)
        
        
        # NAN loss exception
        log_null_loss_exception(hp_epoch_error, 
                                 path_log_error)
        
        print('\n Validation performance under the hyper-parameters: \n', hpara_log[-1][0], hpara_log[-1][1][0])
        print('\n Training time: \n', hp_epoch_time, '\n')
            
        
    # ----- re-train
    
    # best hyper-para and epoch set 
    best_hpara, epoch_sample, best_val_err = hyper_para_selection(hpara_log, 
                                                                  val_aggreg_num = para_val_aggreg_num, 
                                                                  test_snapshot_num = para_test_snapshot_num,
                                                                  metric_idx = para_metric_map[para_validation_metric])
    
    epoch_error, _, _, early_stop_id = training_validate(tr_x, 
                                    tr_y,
                                    val_x, 
                                    val_y,
                                    dim_x = para_dim_x,
                                    steps_x = para_steps_x,
                                    hp_lr = best_hpara[0],
                                    hp_batch_size = int(best_hpara[1]),
                                    hp_l2 = best_hpara[2],
                                    retrain_epoch_set = epoch_sample, 
                                    retrain_bool = True,
                                    retrain_best_val_err = best_val_err)
    
    log_val_hyper_para(path = path_log_error, 
                       hpara_tuple = [best_hpara, epoch_sample, best_val_err], 
                       error_tuple = epoch_error[0])
    
    print('\n----- Best_epoch hyper-parameters: ', best_hpara, epoch_sample, best_val_err, '\n')
    print('\n----- Re-training validation performance: ', epoch_error[0], '\n')
    
    
    # ----- testing
    
    # error tuple: rmse, mae, mape, nnllk
    
    
    # -- best one epoch 
    
    print(epoch_sample[:1])
    
    error_tuple, py_tuple = testing(model_snapshots = epoch_sample[:1], 
                                    xts = ts_x, 
                                    yts = ts_y, 
                                    file_path = path_model, 
                                    bool_instance_eval = True,
                                    loss_type = para_loss_type,
                                    num_src = len(ts_x) if type(ts_x) == list else np.shape(ts_x)[0])
    
    log_test_performance(path = path_log_error, 
                         error_tuple = error_tuple + epoch_sample[:1])
    
    
    # -- best epochs 
    
    print(epoch_sample)
    
    error_tuple, py_tuple = testing(model_snapshots = epoch_sample, 
                                    xts = ts_x, 
                                    yts = ts_y, 
                                    file_path = path_model, 
                                    bool_instance_eval = True,
                                    loss_type = para_loss_type,
                                    num_src = len(ts_x) if type(ts_x) == list else np.shape(ts_x)[0])
    
    log_test_performance(path = path_log_error, 
                         error_tuple = error_tuple + epoch_sample)
    
    
    # -- ensemble or bayesian
    
    print(list(range(para_burn_in_epoch, para_n_epoch)))
    
    error_tuple, py_tuple = testing(model_snapshots = list(range(para_burn_in_epoch, para_n_epoch)), 
                                    xts = ts_x, 
                                    yts = ts_y, 
                                    file_path = path_model, 
                                    bool_instance_eval = True,
                                    loss_type = para_loss_type,
                                    num_src = len(ts_x) if type(ts_x) == list else np.shape(ts_x)[0])
    
    log_test_performance(path = path_log_error, 
                         error_tuple = error_tuple + list(range(para_burn_in_epoch, para_n_epoch)))
    
    
    # -- early-stopping
    
    print(early_stop_id)
    
    error_tuple, py_tuple = testing(model_snapshots = early_stop_id, 
                                    xts = ts_x, 
                                    yts = ts_y, 
                                    file_path = path_model, 
                                    bool_instance_eval = True,
                                    loss_type = para_loss_type,
                                    num_src = len(ts_x) if type(ts_x) == list else np.shape(ts_x)[0])
    
    log_test_performance(path = path_log_error, 
                         error_tuple = error_tuple + early_stop_id)
    
    
    '''
    import pickle
    pickle.dump(py_tuple, open(path_py, "wb"))
    
    print('\n ----- testing performance: \n')
    print('\n testing errors: ', error_tuple, '\n\n') 
    '''
    