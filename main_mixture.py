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

# ----- arguments from command line

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--latent_prob_type', '-t', help = "latent_prob_type", type = str, default = "none")
# "none", "constant_diff_sq", "scalar_diff_sq", "vector_diff_sq"
parser.add_argument('--latent_dependence', '-d', help = "latent_dependence", type = str, default = "none")
# "none", "independent", "markov"
parser.add_argument('--data_mode', '-m', help = "source specific data or with paddning", type = str, default = "src_padding")
# "src_raw", "src_padding"
parser.add_argument('--dataset', '-a', help = "data set path", type = str, default = "../dataset/bitcoin/market2_tar5_len10/")
parser.add_argument('--gpu_id', '-g', help = "gpu_id", type = str, default = "0")
parser.add_argument('--target_distr', '-p', help = "target_probability_distribution", type = str, default = "gaussian")
parser.add_argument('--loss_type', '-l', help = "loss_type", type = str, default = "lk")

args = parser.parse_args()
print(args)

method_str = 'statistic'

from mixture_padding import *

# ------ GPU set-up in multi-GPU environment

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# ----- data and log paths

path_data = args.dataset
#"../dataset/bitcoin/market2_tar5_len10/"
path_log_error = "../results/mixture/log_error_mix.txt"
path_model = "../results/mixture/"
path_py = "../results/mixture/py_" + args.target_distr + "_" + args.loss_type + "_" + args.latent_dependence + "_" + args.latent_prob_type + ".p"

# ----- hyper-parameters set-up

# -- model

para_distr_type = args.target_distr
# gaussian, student_t
para_distr_para = []
# gaussian: [] 
# student_t: [nu], nu>=3
para_var_type = "square" # square, exp
# for one-dimensional feature, variance derivation should be re-defined? 
# always positive correlation of the feature to the variance
para_share_type_gate = "no_share"
# no_share, share, mix
para_bool_bilinear = True # for linear units

# -- data

para_y_log = False
para_bool_target_seperate = False
# if yes, the last source corresponds to the auto-regressive target variable
para_batch_augment = False
para_x_shape_acronym = ["src", "N", "T", "D"]
para_add_common_pattern = False

# -- training and validation

para_model_type = 'rnn'
para_hpara_search = "random" # random, grid 
para_hpara_n_trial = 5

para_n_epoch = 80
para_burn_in_epoch = 20

para_snapshot_type = "epoch_wise"  # batch_wise, epoch_wise
para_snapshot_Bernoulli = 0.001

para_hpara_range = {}

para_hpara_range['grid'] = {}
para_hpara_range['grid']['linear'] = {}
para_hpara_range['grid']['rnn'] = {}

para_hpara_range['grid']['linear']['lr'] = [0.001, ] 
para_hpara_range['grid']['linear']['batch_size'] = [64, 32, 16, 80, ]
para_hpara_range['grid']['linear']['l2'] = [1e-7, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, ]

para_hpara_range['grid']['rnn']['lr'] = [0.001, ]
para_hpara_range['grid']['rnn']['batch_size'] = [10, 20, 30, 40, 50, 60, 70, 80]
para_hpara_range['grid']['rnn']['l2'] = [1e-7, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, ]
para_hpara_range['grid']['rnn']['rnn_size'] =  [16, 32, 64, 128]
para_hpara_range['grid']['rnn']['dense_num'] = [1, 2]

para_hpara_range['random'] = {}
para_hpara_range['random']['linear'] = {}
para_hpara_range['random']['rnn'] = {}

para_hpara_range['random']['linear']['lr'] = [0.001, 0.001]  
para_hpara_range['random']['linear']['batch_size'] = [10, 80]
para_hpara_range['random']['linear']['l2'] = [1e-7, 0.01]

para_hpara_range['random']['rnn']['lr'] = [0.001, 0.0015]
para_hpara_range['random']['rnn']['batch_size'] = [100, 150]
para_hpara_range['random']['rnn']['l2'] = [1e-7, 0.01]
para_hpara_range['random']['rnn']['rnn_size'] =  [16, 50]
para_hpara_range['random']['rnn']['dense_num'] = [0, 0]
para_hpara_range['random']['rnn']['dropout_keep_prob'] = [1.0, 1.0]
para_hpara_range['random']['rnn']['max_norm_cons'] = [0.0, 0.0]

# model snapshot sample: epoch_wise or batch_wise
#   epoch_wise: vali. test snapshot numbers are explicited determined 
#   batch_wise: vali. test snapshot numbers are arbitary 
para_val_aggreg_num = max(1, int(0.05*para_n_epoch))
para_test_snapshot_num = para_n_epoch - para_burn_in_epoch

para_early_stop_bool = False
para_early_stop_window = 0

para_validation_metric = 'rmse' if args.loss_type == "mse" else 'nnllk'
para_metric_map = {'rmse':0, 'mae':1, 'mape':2, 'nnllk':3} 

# -- optimization

para_loss_type = args.loss_type

para_optimizer = "adam" # RMSprop, sg_mcmc_RMSprop, adam, sg_mcmc_adam, sgd, adamW 
para_optimizer_lr_decay = True
para_optimizer_lr_decay_epoch = 10 # after the warm-up
para_optimizer_lr_warmup_epoch = int(0.1*para_n_epoch)

# -- regularization

para_regu_mean = True
para_regu_var = True
para_regu_gate = False
para_regu_imbalanced_mean_var = False
para_regu_mean_positive = False
para_regu_global_gate = False  
para_regu_latent_dependence = False
para_regu_weight_on_latent = True

para_bool_bias_in_mean = True
para_bool_bias_in_var = True
para_bool_bias_in_gate = True
para_bool_global_bias = False

para_latent_dependence = args.latent_dependence
para_latent_prob_type = args.latent_prob_type

def log_train(path):
    
    with open(path, "a") as text_file:
        
        text_file.write("\n\n ------ Statistic mixture : \n")
        
        text_file.write("data_mode : %s \n"%(args.data_mode))
        text_file.write("data path : %s \n"%(path_data))
        text_file.write("data source seperated : %s \n"%(para_x_src_seperated))
        text_file.write("data source timesteps : %s \n"%(para_steps_x))
        text_file.write("data source feature dimensionality : %s \n"%(para_dim_x))
        text_file.write("data source number : %d \n"%(len(ts_x) if type(ts_x)==list else np.shape(ts_x)[0]))
        text_file.write("data added common pattern : %s \n"%(para_add_common_pattern))
        text_file.write("\n")
        
        text_file.write("model type : %s \n"%(para_model_type))
        text_file.write("bi-linear : %s \n"%(para_bool_bilinear))
        text_file.write("target distribution type : %s \n"%(para_distr_type))
        text_file.write("target distribution para. : %s \n"%(str(para_distr_para)))
        text_file.write("target variable as a seperated data source : %s \n"%(para_bool_target_seperate))
        text_file.write("variance calculation type : %s \n"%(para_var_type))
        text_file.write("para. sharing in gate logit : %s \n"%(para_share_type_gate))
        text_file.write("\n")
        
        text_file.write("regularization on mean : %s \n"%(para_regu_mean))
        text_file.write("regularization on variance : %s \n"%(para_regu_var))
        text_file.write("regularization on mixture gates : %s \n"%(para_regu_gate))
        text_file.write("regularization imbalanced l2 on mean and var. : %s \n"%(para_regu_imbalanced_mean_var))
        text_file.write("regularization on positive means : %s \n"%(para_regu_mean_positive))
        text_file.write("regularization by global gate : %s \n"%(para_regu_global_gate))
        text_file.write("regularization on latent dependence parameters : %s \n"%(para_regu_latent_dependence))
        text_file.write("regularization l2 on latent variable : %s \n"%(para_regu_weight_on_latent))
        text_file.write("\n")
        
        text_file.write("adding bias terms in mean : %s \n"%(para_bool_bias_in_mean))
        text_file.write("adding bias terms in variance : %s \n"%(para_bool_bias_in_var))
        text_file.write("adding bias terms in gates : %s \n"%(para_bool_bias_in_gate))
        text_file.write("global bias terms : %s \n"%(para_bool_global_bias))
        text_file.write("\n")
        
        text_file.write("temporal dependence of latent variables : %s \n"%(para_latent_dependence))
        text_file.write("latent dependence probability type : %s \n"%(para_latent_prob_type))
        text_file.write("latent dependence as a regularization : %s \n"%(True if para_latent_prob_type != "none" else False))
        text_file.write("\n")
        
        text_file.write("optimizer : %s \n"%(para_optimizer))
        text_file.write("loss type : %s \n"%(para_loss_type))
        text_file.write("learning rate decay : %s \n"%(str(para_optimizer_lr_decay)))
        text_file.write("learning rate decay epoch : %s \n"%(str(para_optimizer_lr_decay_epoch)))
        text_file.write("learning rate warmup epoch : %s \n"%(str(para_optimizer_lr_warmup_epoch)))
        text_file.write("\n")
        
        text_file.write("hyper-para search : %s \n"%(para_hpara_search))
        text_file.write("hyper-para range : %s \n"%(str(para_hpara_range[para_hpara_search][para_model_type])))
        text_file.write("hyper-para random search trials : %s \n"%(str(para_hpara_n_trial)))
        
        text_file.write("\n")
        text_file.write("epochs in total : %s \n"%(para_n_epoch))
        text_file.write("burn_in_epoch : %s \n"%(para_burn_in_epoch))
        
        text_file.write("snapshot type : %s \n"%(para_snapshot_type))
        text_file.write("snapshot_Bernoulli : %s \n"%(para_snapshot_Bernoulli))
        text_file.write("num. snapshots in validation : %s \n"%(para_val_aggreg_num))
        text_file.write("num. snapshots in testing : %s \n"%(para_test_snapshot_num))
        text_file.write("validation metric : %s \n"%(para_validation_metric))
        
        text_file.write("early-stoping : %s \n"%(para_early_stop_bool))
        text_file.write("early-stoping look-back window : %s \n"%(para_early_stop_window))
        
        text_file.write("\n\n")

# ----- training and evalution
    
def training_validating(xtr,
                        ytr,
                        xval,
                        yval,
                        dim_x,
                        steps_x,
                        x_src_seperated,
                        hyper_para_dict,
                        training_dict,
                        retrain_snapshot_steps, 
                        retrain_bayes_steps,
                        retrain_bool):
    '''
    Args:
    
    xtr: [num_src, N, T, D]
         S: num_src
         N: number of data samples
         T: number of steps
         D: dimension at each time step
        
    ytr: [N 1]
        
    dim_x: integer, corresponding to D
    
    steps_x: integer, corresponding to T
    
    hyper_para_dict: 
       "lr": float,
       "batch_size": int
       "l2": float,
                           
       "lstm_size": int,
       "dense_num": int,
       "use_hidden_before_dense": bool
       
    training_dict:
       "batch_per_epoch": int
       "tr_idx": list of integer
       
    '''
    
    # clear the graph in the current session 
    tf.reset_default_graph()
    
    # stabilize the network by fixing random seeds
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
                                  num_src = len(xtr) if type(xtr) == list else np.shape(xtr)[0])
        
        # -- initialize the network
        
        model.network_ini(hyper_para_dict,
                          x_dim = dim_x,
                          x_steps = steps_x, 
                          x_src_seperated = x_src_seperated,
                          model_type = para_model_type, 
                          model_distr_type = para_distr_type,
                          model_distr_para = para_distr_para,
                          model_var_type = para_var_type,
                          model_bool_bilinear = para_bool_bilinear,
                          model_para_share_type = para_share_type_gate,
                          bool_regu_mean = para_regu_mean,
                          bool_regu_var = para_regu_var,
                          bool_regu_gate = para_regu_gate,
                          bool_regu_positive_mean = para_regu_mean_positive,
                          bool_regu_global_gate = para_regu_global_gate, 
                          bool_regu_latent_dependence = para_regu_latent_dependence,
                          bool_regu_l2_on_latent = para_regu_weight_on_latent,
                          bool_regu_imbalance = para_regu_imbalanced_mean_var,
                          latent_dependence = para_latent_dependence,
                          latent_prob_type = para_latent_prob_type,
                          bool_bias_mean = para_bool_bias_in_mean,
                          bool_bias_var = para_bool_bias_in_var,
                          bool_bias_gate = para_bool_bias_in_gate,
                          bool_bias_global_src = para_bool_global_bias,
                          optimization_method = para_optimizer,
                          optimization_lr_decay = para_optimizer_lr_decay,
                          optimization_lr_decay_steps = para_optimizer_lr_decay_epoch*int(len(xtr[0])/hyper_para_dict["batch_size"]),
                          optimization_burn_in_step = para_burn_in_epoch,
                          optimization_warmup_step = para_optimizer_lr_warmup_epoch*training_dict["batch_per_epoch"] - 1
                         )
        
        model.train_ini()
        model.inference_ini()
        
        # -- set up training batch parameters
        
        tr_batch_num = training_dict["batch_per_epoch"] 
        tr_idx = training_dict["tr_idx"]
        
        # -- begin training
        
        # training and validation error log
        step_error = []
        global_step = 0
        
        # training time counter
        st_time = time.time()
        
        for epoch in range(para_n_epoch):
            
            # shuffle traning instances each epoch
            np.random.shuffle(tr_idx)
            
            # loop over all batches            
            for i in range(tr_batch_num):
                
                # batch data
                batch_idx = tr_idx[i*int(hyper_para_dict["batch_size"]) : (i+1)*int(hyper_para_dict["batch_size"])] 
                
                # shape: [S B T D]
                # B: number of data instances in one batch
                batch_x = [xtr[tmp_src][batch_idx] for tmp_src in range(len(xtr))]
                # [B 1]
                batch_y = ytr[batch_idx]
                
                '''
                # bath-wise data augmentation: for inbalanced data
                if para_batch_augment == True:
                    
                    batch_x, batch_y = batch_augment(batch_x, 
                                                     batch_y, 
                                                     num_src = len(xtr) if type(xtr) == list else np.shape(xtr)[0])
                '''
                
                # one-step training on a batch of training data
                model.train_batch(batch_x, 
                                  batch_y,
                                  x_src_seperated = x_src_seperated,
                                  global_step = epoch)
                
                # - batch-wise validation
            
                # val_metric: [val_rmse, val_mae, val_mape, val_nnllk]
                # nnllk: normalized negative log likelihood
                
                val_metric, monitor_metric = model.validation(xval,
                                                              yval,
                                                              x_src_seperated = x_src_seperated,
                                                              snapshot_type = para_snapshot_type,
                                                              snapshot_Bernoulli = para_snapshot_Bernoulli,
                                                              step = global_step,
                                                              bool_end_of_epoch = (True if i == tr_batch_num -1 else False))
                if val_metric:
                    
                    # tr_metric [tr_rmse, tr_mae, tr_mape, tr_nnllk]
                    tr_metric, _ = model.inference(xtr,
                                                   ytr, 
                                                   x_src_seperated = x_src_seperated,
                                                   bool_py_eval = False)
                    
                    # para_metric_map[] defined on
                    step_error.append([global_step,
                                       tr_metric, 
                                       val_metric, 
                                       epoch])
                
                # - model saver 
                
                if retrain_bool == True and model.model_saver(path = path_model + method_str + '_' + str(global_step),
                                                              epoch = epoch,
                                                              step = global_step,
                                                              snapshot_steps = retrain_snapshot_steps,
                                                              bayes_steps = retrain_bayes_steps,
                                                              early_stop_bool = para_early_stop_bool,
                                                              early_stop_window = para_early_stop_window) == True:
                    
                    print("\n    [MODEL SAVED] \n " + path_model + method_str + '_' + str(global_step))
                        
                global_step += 1
            
            # -- epoch-wise
            
            print("\n --- At epoch %d : \n  %s "%(epoch, str(step_error[-1])))
            print("\n gates : \n", monitor_metric)
            print("\n py_mean_src : \n", monitor_metric)
            
            # NAN value exception 
            if np.isnan(monitor_metric[-1]) == True:
                print("\n --- NAN loss !! \n" )
                break
                
        ed_time = time.time()
        
    # ? the epoch with the lowest para_validation_metric ?
    # [global_step, tr_metric, val_metric, epoch]
    return sorted(step_error, key = lambda x:x[2][para_metric_map[para_validation_metric]]),\
           1.0*(ed_time - st_time)/(epoch + 1e-5), \

def testing(model_snapshots,
            xts,
            yts,
            x_src_seperated,
            file_path,
            bool_instance_eval,
            loss_type,
            num_src):
    
    # ensemble of model snapshots
    infer = ensemble_inference()
    
    with tf.device('/device:GPU:0'):
        
        for tmp_model_id in model_snapshots:
            
            # path of the stored models 
            tmp_meta = file_path + method_str + '_' + str(tmp_model_id) + '.meta'
            tmp_data = file_path + method_str + '_' + str(tmp_model_id)
        
            # clear graph
            tf.reset_default_graph()
            
            config = tf.ConfigProto()
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True
        
            sess = tf.Session(config = config)
        
            model = mixture_statistic(session = sess, 
                                      loss_type = para_loss_type,
                                      num_src = num_src)
            # restore the model
            model.model_restore(tmp_meta, 
                                tmp_data)
            
            # one-shot inference sample
            # [rmse, mae, mape, nnllk],  [py_mean, py_var, py_mean_src, py_var_src, py_gate_src]
            error_tuple, py_tuple = model.inference(xts,
                                                    yts, 
                                                    x_src_seperated = x_src_seperated,
                                                    bool_py_eval = bool_instance_eval)
            if bool_instance_eval == True:
                
                # store the samples
                infer.add_samples(py_mean = py_tuple[0],
                                  py_var = py_tuple[1],
                                  py_mean_src = py_tuple[2],
                                  py_var_src = py_tuple[3],
                                  py_gate_src = py_tuple[4])
            
    # return: error tuple [rmse, mae, mape, nnllk], 
    #         prediction tuple []
    
    if len(model_snapshots) == 0:
        
        return ["None"], ["None"]
    
    elif len(model_snapshots) == 1:
        
        return error_tuple, py_tuple
    
    else:
        # ensemble inference
        return infer.bayesian_inference(yts)
    
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
    # y [N 1], x [S [N T D]]    
    
    # if para_bool_target_seperate = yes, the last source corresponds to the auto-regressive target variable
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
        # y [N 1], x [S [N T D]]
        
        if para_add_common_pattern == True:
            # x [S [N T D]]
            # T is the same across sources
            # [N T D*S]
            common_x = np.concatenate(tr_x, -1)
            # [S+1, ]
            tr_x.append(common_x)
        
            common_x = np.concatenate(val_x, -1)
            val_x.append(common_x)
            
            common_x = np.concatenate(ts_x, -1)
            ts_x.append(common_x)
        
            print("\n ----- test ----- \n")
            for tmp in tr_x:
                print(np.shape(tmp))
            
            for tmp in val_x:
                print(np.shape(tmp))
            
            for tmp in ts_x:
                print(np.shape(tmp))
        
        para_steps_x = []
        para_dim_x = []
        para_x_src_seperated = True
        
        for tmp_src in range(len(tr_x)):
            
            tmp_shape = np.shape(tr_x[tmp_src][0])
        
            para_steps_x.append(tmp_shape[0])
            para_dim_x.append(tmp_shape[1])
            
            print("src " + str(tmp_src) + " shape: ", tmp_shape)
        
        shape_tr_x_dict = dict({"N": len(tr_x[tmp_src])})
            
    elif args.data_mode == "src_padding":
        # y [N 1], x [S N T D]
        
        # padding to normalized feature data
        tr_x = data_padding_x(tr_x, 
                              num_src = len(tr_x))
        
        val_x = data_padding_x(val_x, 
                               num_src = len(tr_x))
        
        ts_x = data_padding_x(ts_x, 
                              num_src = len(tr_x))
        
        print("Shapes after padding: ", np.shape(tr_x), np.shape(val_x), np.shape(ts_x))
        
        para_steps_x = np.shape(tr_x)[2] 
        para_dim_x = np.shape(tr_x)[3]
        para_x_src_seperated = False
    
        # y [N 1], x [S N T D]  
        shape_tr_x_dict =  dict(zip(para_x_shape_acronym, np.shape(tr_x)))
    
    # ----- training and validation
    
    log_train(path_log_error)
    
    # -- hyper-para generator 
    
    if para_hpara_search == "random":
        
        hpara_generator = hyper_para_random_search(para_hpara_range[para_hpara_search][para_model_type], 
                                                   para_hpara_n_trial)
    elif para_hpara_search == "grid":

        hpara_generator = hyper_para_grid_search(para_hpara_range[para_hpara_search][para_model_type])
            
    # -- begin hyper-para search
    
    hpara_log = []
    
    # sample one set-up of hyper-para
    hpara_dict = hpara_generator.one_trial()
    tr_dict = {} # training para dictionary 
                                                 
    while hpara_dict != None:
        
        tr_dict = parameter_manager(shape_x_dict = shape_tr_x_dict, 
                                    hpara_dict = hpara_dict)
        # hp_: hyper-parameter
        # hp_step_error: [ [step, train_metric, val_metric, epoch] ]
        
        hp_step_error, hp_epoch_time = training_validating(tr_x,
                                                           tr_y,
                                                           val_x,
                                                           val_y,
                                                           dim_x = para_dim_x,
                                                           steps_x = para_steps_x,
                                                           x_src_seperated = para_x_src_seperated,
                                                           hyper_para_dict = hpara_dict,
                                                           training_dict = tr_dict,
                                                           retrain_bool = False,
                                                           retrain_snapshot_steps = [],
                                                           retrain_bayes_steps = [])
        
        #[ dict{lr, batch, l2, ..., burn_in_steps}, [[step, tr_metric, val_metric, epoch]] ]
        ''' ? '''
        hpara_dict["burn_in_steps"] = para_burn_in_epoch*tr_dict["batch_per_epoch"] - 1
        hpara_log.append([hpara_dict, hp_step_error])
        
        # -- prepare for the next trial
        
        # stabilize the network by fixing random seeds
        #np.random.seed(1)
        #tf.set_random_seed(1)
        
        # sample the next hyper-para
        hpara_dict = hpara_generator.one_trial()
        
        # -- logging
        
        log_train_val_performance(path_log_error, 
                                  hpara = hpara_log[-1][0], 
                                  hpara_error = hpara_log[-1][1][0],
                                  train_time = hp_epoch_time)
        
        # NAN loss exception
        log_null_loss_exception(hp_step_error, 
                                path_log_error)
        
        print('\n Validation performance under the hyper-parameters: \n', hpara_log[-1][0], hpara_log[-1][1][0])
        print('\n Training time: \n', hp_epoch_time, '\n')
        
    # ----- re-train
    
    # best hyper-para and snapshot set 
    best_hpara, snapshot_steps, bayes_steps, snapshot_steps_features, bayes_steps_features = hyper_para_selection(hpara_log, 
                                                                   val_aggreg_num = para_val_aggreg_num, 
                                                                   test_snapshot_num = para_test_snapshot_num,
                                                                   metric_idx = para_metric_map[para_validation_metric])
    tr_dict = parameter_manager(shape_x_dict = shape_tr_x_dict, 
                                hpara_dict = best_hpara)
    
    step_error, _ = training_validating(tr_x, 
                                        tr_y,
                                        val_x, 
                                        val_y,
                                        dim_x = para_dim_x,
                                        steps_x = para_steps_x,
                                        x_src_seperated = para_x_src_seperated,
                                        hyper_para_dict = best_hpara,
                                        training_dict = tr_dict,
                                        retrain_bool = True,
                                        retrain_snapshot_steps = snapshot_steps, 
                                        retrain_bayes_steps = bayes_steps)
    
    log_val_hyper_para(path = path_log_error, 
                       hpara_tuple = [best_hpara, snapshot_steps], 
                       error_tuple = step_error[0])
    
    print('\n----- Best epoch hyper-parameters: ', best_hpara, snapshot_steps, '\n')
    print('\n----- Re-training validation performance: ', step_error[0], '\n')
    
    # ----- testing
    # error tuple: rmse, mae, mape, nnllk
    
    # -- best one snapshot 
    
    error_tuple, _ = testing(model_snapshots = snapshot_steps[:1],
                             xts = ts_x,
                             yts = ts_y,
                             x_src_seperated = para_x_src_seperated,
                             file_path = path_model,
                             bool_instance_eval = True,
                             loss_type = para_loss_type,
                             num_src = len(ts_x) if type(ts_x) == list else np.shape(ts_x)[0])
    
    log_test_performance(path = path_log_error,
                         error_tuple = [error_tuple],
                         ensemble_str = "One-shot")
    
    # -- best snapshot steps 
    
    error_tuple, _ = testing(model_snapshots = snapshot_steps,
                             xts = ts_x,
                             yts = ts_y,
                             x_src_seperated = para_x_src_seperated,
                             file_path = path_model,
                             bool_instance_eval = True,
                             loss_type = para_loss_type,
                             num_src = len(ts_x) if type(ts_x) == list else np.shape(ts_x)[0])
    
    log_test_performance(path = path_log_error,
                         error_tuple = [error_tuple],
                         ensemble_str = "Top-rank")
    
    # -- bayesian steps
    
    error_tuple, py_tuple = testing(model_snapshots = bayes_steps,
                                    xts = ts_x,
                                    yts = ts_y,
                                    x_src_seperated = para_x_src_seperated,
                                    file_path = path_model,
                                    bool_instance_eval = True,
                                    loss_type = para_loss_type,
                                    num_src = len(ts_x) if type(ts_x) == list else np.shape(ts_x)[0])
    
    log_test_performance(path = path_log_error,
                         error_tuple = [error_tuple],
                         ensemble_str = "Bayesian")
    
    # -- dump predictions on testing data
    
    import pickle
    pickle.dump(py_tuple, open(path_py, "wb"))
    
    # -- for test
    
    print("--- !!! --- \n", snapshot_steps_features, bayes_steps_features)