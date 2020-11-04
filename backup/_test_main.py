#!/usr/bin/python

import sys
import os
import numpy as np
import random
from random import randint
from random import shuffle
import time
import json
import pickle

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.contrib import rnn

# local packages 
from utils_libs import *
from utils_training import *
from utils_inference import *

# ----- arguments from command line

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', help = "data set path", type = str, default = "../dataset/bitcoin/market2_tar5_len10/")
parser.add_argument('--gpu_id', '-g', help = "gpu_id", type = str, default = "0")

args = parser.parse_args()
print(args)

from mixture_models import *

# ------ GPU set-up in multi-GPU environment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# ----- data and log paths
path_data = args.dataset
path_log_error = "../results/mixture/log_error_mix_" + str(args.gpu_id) + ".txt"
path_model = "../results/mixture/"
path_py = "../results/mixture/py_" + str(args.dataset)[-10:-1] + "_" + ".p"

# ----- set-up

# -- model
para_distr_type = "gaussian"
# gaussian, student_t
para_distr_para = []
# gaussian: [] 
# student_t: [nu], nu >= 3
para_var_type = "square" # square, exp
# [Note] for one-dimensional feature, variance derivation should be re-defined? 
# always positive correlation of the feature to the variance
para_share_type_gate = "no_share"
# no_share, share, mix

para_model_type = 'linear'

# -- data

if para_model_type == 'rnn':
    para_x_src_padding = False
    para_add_common_factor = False
    para_common_factor_type = "pool" if para_add_common_factor == True else ""
    
elif para_model_type == 'linear':
    para_x_src_padding = True
    para_add_common_factor = True
    para_common_factor_type = "factor" if para_add_common_factor == True else ""

para_bool_target_seperate = False # [Note] if yes, the last source corresponds to the auto-regressive target variable
para_x_shape_acronym = ["src", "N", "T", "D"]

# -- training

# [Note] if best epoch is close to "para_n_epoch", possible to increase "para_n_epoch".
# [Note] if best epoch is around the middle place of the training trajectory, ensemble expects to take effect. 
para_n_epoch = 80
para_burn_in_epoch = 60
para_vali_snapshot_num = max(1, int(0.05*para_n_epoch))
para_test_snapshot_num = para_n_epoch - para_burn_in_epoch

para_hpara_search = "random" # random, grid 
para_hpara_train_trial_num = 10
para_hpara_retrain_num = 10
para_hpara_ensemble_num = 5

# optimization
para_loss_type = "heter_lk_inv"
para_optimizer = "adam"
# RMSprop, adam, sgd, adamW 
# sg_mcmc_RMSprop, sg_mcmc_adam
# [Note] for sg_mcmc family, "para_n_epoch" could be set to higher values

# [Note] training heuristic: re-set the following for training on new data
# [Note] if lr_decay is on, "lr" and "para_n_epoch" can be set to higher values
para_optimizer_lr_decay = True 
para_optimizer_lr_decay_epoch = 10 # after the warm-up
# [Note] when sg_mcmc is on, turn off the learning rate warm-up
para_optimizer_lr_warmup_epoch = max(1, int(0.1*para_n_epoch))

para_early_stop_bool = False
para_early_stop_window = 0

para_validation_metric = 'rmse'
para_metric_map = {'rmse':0, 'mae':1, 'mape':2, 'nnllk':3} 

# regularization
para_regu_mean = True
para_regu_var = True
para_regu_gate = False
para_regu_mean_positive = False

para_bool_bias_in_mean = True
para_bool_bias_in_var = True
para_bool_bias_in_gate = True

# -- hpara: hyper parameter

para_hpara_range = {}
para_hpara_range['random'] = {}
para_hpara_range['random']['linear'] = {}
para_hpara_range['random']['rnn'] = {}

# - linear
if para_add_common_factor == True:
    para_hpara_range['random']['linear']['factor_size'] = [10, 10]
para_hpara_range['random']['linear']['lr'] = [0.001, 0.001]  
para_hpara_range['random']['linear']['batch_size'] = [10, 80]
# source-wise
para_hpara_range['random']['linear']['l2_mean'] = [1e-7, 1e-3]
para_hpara_range['random']['linear']['l2_var'] = [1e-7, 1e-3]
if para_regu_gate == True:
    para_hpara_range['random']['linear']['l2_gate'] = [1e-7, 1e-3]

# - rnn
# source-wise
para_hpara_range['random']['rnn']['rnn_size'] =  [16, 16]
para_hpara_range['random']['rnn']['dense_num'] = [0, 3] # inproper value leads to non-convergence in training

para_hpara_range['random']['rnn']['lr'] = [0.001, 0.001]
para_hpara_range['random']['rnn']['batch_size'] = [100, 140]

# source-wise
para_hpara_range['random']['rnn']['l2_mean'] = [1e-7, 1e-3]
para_hpara_range['random']['rnn']['l2_var'] = [1e-7, 1e-3]
if para_regu_gate == True:
    para_hpara_range['random']['linear']['l2_gate'] = [1e-7, 1e-3]
para_hpara_range['random']['rnn']['dropout_keep_prob'] = [0.7, 1.0]
para_hpara_range['random']['rnn']['max_norm_cons'] = [0.0, 0.0]

# -- log
def log_train(path):
    with open(path, "a") as text_file:
        text_file.write("\n\n ------ Bayesian mixture : \n")
        
        text_file.write("data source padding : %s \n"%(para_x_src_padding))
        text_file.write("data path : %s \n"%(path_data))
        text_file.write("data source timesteps : %s \n"%(para_steps_x))
        text_file.write("data source feature dimensionality : %s \n"%(para_dim_x))
        text_file.write("data source number : %d \n"%( len(src_ts_x) ))
        text_file.write("data common factor : %s \n"%(para_add_common_factor))
        text_file.write("data common factor type : %s \n"%(para_common_factor_type))
        text_file.write("\n")
        
        text_file.write("model type : %s \n"%(para_model_type))
        text_file.write("target distribution type : %s \n"%(para_distr_type))
        text_file.write("target distribution para. : %s \n"%(str(para_distr_para)))
        text_file.write("target variable as a seperated data source : %s \n"%(para_bool_target_seperate))
        text_file.write("variance calculation type : %s \n"%(para_var_type))
        text_file.write("para. sharing in gate logit : %s \n"%(para_share_type_gate))
        text_file.write("\n")
        
        text_file.write("regularization on mean : %s \n"%(para_regu_mean))
        text_file.write("regularization on variance : %s \n"%(para_regu_var))
        text_file.write("regularization on mixture gates : %s \n"%(para_regu_gate))
        text_file.write("regularization on positive means : %s \n"%(para_regu_mean_positive))
        text_file.write("\n")
        
        text_file.write("adding bias terms in mean : %s \n"%(para_bool_bias_in_mean))
        text_file.write("adding bias terms in variance : %s \n"%(para_bool_bias_in_var))
        text_file.write("adding bias terms in gates : %s \n"%(para_bool_bias_in_gate))
        text_file.write("\n")
        
        text_file.write("optimizer : %s \n"%(para_optimizer))
        text_file.write("loss type : %s \n"%(para_loss_type))
        text_file.write("learning rate decay : %s \n"%(str(para_optimizer_lr_decay)))
        text_file.write("learning rate decay epoch : %s \n"%(str(para_optimizer_lr_decay_epoch)))
        text_file.write("learning rate warm-up epoch : %s \n"%(str(para_optimizer_lr_warmup_epoch)))
        text_file.write("\n")
        
        text_file.write("hyper-para search : %s \n"%(para_hpara_search))
        text_file.write("hyper-para range : %s \n"%(str(para_hpara_range[para_hpara_search][para_model_type])))
        text_file.write("hyper-para training trial num : %s \n"%(str(para_hpara_train_trial_num)))
        text_file.write("hyper-para retraining num.: %s \n"%(str(para_hpara_retrain_num)))
        text_file.write("random seed ensemble num.: %s \n"%(str(para_hpara_ensemble_num)))
        text_file.write("\n")
        
        text_file.write("epochs in total : %s \n"%(para_n_epoch))
        text_file.write("burn_in_epoch : %s \n"%(para_burn_in_epoch))
        text_file.write("num. snapshots in validating : %s \n"%(para_vali_snapshot_num))
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
                        hyper_para_dict,
                        training_dict,
                        retrain_top_steps, 
                        retrain_bayes_steps,
                        retrain_bool,
                        retrain_idx,
                        random_seed):
    '''
    Argu.:
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
    
    with tf.device('/device:GPU:0'):
        
        # clear the graph in the current session 
        tf.reset_default_graph()
        
        # fix the random seed to stabilize the network
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        random.seed(random_seed)  # `python` built-in pseudo-random generator
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
        
        # session set-up
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        sess = tf.Session(config = config)
        
        model = mixture_statistic(session = sess, 
                                  loss_type = para_loss_type,
                                  num_src = len(xtr),
                                  hyper_para_dict = hyper_para_dict, 
                                  model_type = para_model_type)
        
        # -- initialize the network
        model.network_ini(hyper_para_dict,
                          x_dim = dim_x,
                          x_steps = steps_x, 
                          x_bool_common_factor = para_add_common_factor,
                          model_type = para_model_type, 
                          model_distr_type = para_distr_type,
                          model_distr_para = para_distr_para,
                          model_var_type = para_var_type,
                          model_para_share_type = para_share_type_gate,
                          bool_regu_mean = para_regu_mean,
                          bool_regu_var = para_regu_var,
                          bool_regu_gate = para_regu_gate,
                          bool_regu_positive_mean = para_regu_mean_positive,
                          bool_bias_mean = para_bool_bias_in_mean,
                          bool_bias_var = para_bool_bias_in_var,
                          bool_bias_gate = para_bool_bias_in_gate,
                          optimization_method = para_optimizer,
                          optimization_lr_decay = para_optimizer_lr_decay,
                          optimization_lr_decay_steps = para_optimizer_lr_decay_epoch*int(len(xtr[0])/int(hyper_para_dict["batch_size"])),
                          optimization_burn_in_step = para_burn_in_epoch,
                          optimization_warmup_step = para_optimizer_lr_warmup_epoch*training_dict["batch_per_epoch"] - 1)
        
        # !! the order of Saver
        saver = tf.train.Saver(max_to_keep = None)
        
        model.train_ini()
        model.inference_ini()
        #tf.get_default_graph().finalize()
        
        # -- set up training batch parameters
        batch_gen = data_loader(x = xtr,
                                y = ytr,
                                batch_size = int(hyper_para_dict["batch_size"]), 
                                num_ins = training_dict["tr_num_ins"],  
                                num_src = len(xtr))
        # -- begin training
        
        # training and validation error log
        step_error = []
        global_step = 0
        
        # training time counter
        st_time = time.time()
        
        for epoch in range(para_n_epoch):
            # shuffle traning instances each epoch
            batch_gen.re_shuffle()
            batch_x, batch_y, bool_last = batch_gen.one_batch()
            
            # loop over all batches
            while batch_x != None:
                    
                # one-step training on a batch of training data
                model.train_batch(batch_x, 
                                  batch_y,
                                  global_step = epoch)
                
                # - batch-wise validation
                # val_metric: [val_rmse, val_mae, val_mape, val_nnllk]
                # nnllk: normalized negative log likelihood
                val_metric, monitor_metric = model.validation(xval,
                                                              yval,
                                                              step = global_step,
                                                              bool_end_of_epoch = bool_last)
                if val_metric:
                    # tr_metric [tr_rmse, tr_mae, tr_mape, tr_nnllk]
                    tr_metric, _ = model.inference(xtr,
                                                   ytr, 
                                                   bool_py_eval = False)
                    #step_error.append([global_step, tr_metric, val_metric, epoch])
                    step_error.append([epoch, tr_metric, val_metric, epoch])
                    
                # - next batch
                batch_x, batch_y, bool_last = batch_gen.one_batch()
                global_step += 1
                    
            # -- model saver 
            model_saver_flag = model.model_saver(path = path_model + para_model_type + '_' + str(retrain_idx) + '_' + str(epoch),
                                                 epoch = epoch,
                                                 step = global_step,
                                                 top_snapshots = retrain_top_steps,
                                                 bayes_snapshots = retrain_bayes_steps,
                                                 early_stop_bool = para_early_stop_bool,
                                                 early_stop_window = para_early_stop_window, 
                                                 tf_saver = saver)
            # epoch-wise
            print("\n --- At epoch %d : \n  %s "%(epoch, str(step_error[-1])))
            print("\n   loss and regualization : \n", monitor_metric)
            
            # NAN value exception 
            if np.isnan(monitor_metric[0]) == True:
                print("\n --- NAN loss !! \n" )
                break
                
            if retrain_bool == True and model_saver_flag != None:
                print("\n    [MODEL SAVED] " + model_saver_flag + " \n " + path_model + para_model_type + '_' + str(retrain_idx) + '_' + str(epoch))
                
        ed_time = time.time()
        
    # ? sorted training log ?
    # step_error: [global_step, tr_metric, val_metric, epoch]
    # sort step_error based on para_validation_metric
    sort_step_error = sorted(step_error, key = lambda x:x[2][para_metric_map[para_validation_metric]])
    
    return sort_step_error,\
           1.0*(ed_time - st_time)/(epoch + 1e-5), \

def testing(retrain_snapshots,
            retrain_ids,
            xts,
            yts,
            file_path,
            bool_instance_eval,
            loss_type,
            num_src,
            snapshot_features, 
            hpara_dict):
    
    # ensemble of model snapshots
    infer = ensemble_inference()
    
    with tf.device('/device:GPU:0'):
        
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        
        for tmp_idx, tmp_retrain_id in enumerate(retrain_ids):
            
            for tmp_model_id in retrain_snapshots[tmp_idx]:
                
                # path of the stored models 
                tmp_meta = file_path + para_model_type + '_' + str(tmp_retrain_id) + '_' + str(tmp_model_id) + '.meta'
                tmp_data = file_path + para_model_type + '_' + str(tmp_retrain_id) + '_' + str(tmp_model_id)
        
                # clear graph
                tf.reset_default_graph()
                saver = tf.train.import_meta_graph(tmp_meta, 
                                                   clear_devices = True)
                sess = tf.Session(config = config)
                
                model = mixture_statistic(session = sess, 
                                          loss_type = para_loss_type,
                                          num_src = num_src,
                                          hyper_para_dict = hpara_dict, 
                                          model_type = para_model_type)
                # restore the model
                model.model_restore(tmp_meta, 
                                    tmp_data, 
                                    saver)
                
                # one-shot inference sample
                # error_tuple: [rmse, mae, mape, nnllk],  
                # py_tuple: [py_mean, py_var, py_mean_src, py_var_src, py_gate_src]
                error_tuple, py_tuple = model.inference(xts,
                                                        yts, 
                                                        bool_py_eval = bool_instance_eval)
                if bool_instance_eval == True:
                    # store the samples
                    infer.add_samples(py_mean = py_tuple[0],
                                      py_var = py_tuple[1],
                                      py_mean_src = py_tuple[2],
                                      py_var_src = py_tuple[3],
                                      py_gate_src = py_tuple[4])
    
    num_snapshots = sum([len(i) for i in retrain_snapshots])
    
    # return: error tuple, prediction tuple
    if num_snapshots == 0:
        return ["None"], ["None"]  
    else:
        # ensemble inference
        if len(snapshot_features) == 0 or num_snapshots == 1:
            return infer.bayesian_inference(yts)
        else:
            return infer.importance_inference(snapshot_features = snapshot_features, 
                                              y = yts)
# ----- main process  

if __name__ == '__main__':
    
    # ----- data
    
    import pickle
    tr_dta = pickle.load(open(path_data + 'train.p', "rb"), encoding = 'latin1')
    val_dta = pickle.load(open(path_data + 'val.p', "rb"), encoding = 'latin1')
    ts_dta = pickle.load(open(path_data + 'test.p', "rb"), encoding = 'latin1')
    print(len(tr_dta), len(val_dta), len(ts_dta))
    
    # if para_bool_target_seperate = yes, the last source corresponds to the auto-regressive target variable
    tr_x, tr_y = data_reshape(tr_dta, 
                              bool_target_seperate = para_bool_target_seperate)
    val_x, val_y = data_reshape(val_dta,
                                bool_target_seperate = para_bool_target_seperate)
    ts_x, ts_y = data_reshape(ts_dta,
                              bool_target_seperate = para_bool_target_seperate)
    # output from the reshape
    # y [N 1], x [S [N T D]]
    print("training: ", len(tr_x[0]), len(tr_y))
    print("validation: ", len(val_x[0]), len(val_y))
    print("testing: ", len(ts_x[0]), len(ts_y))
    
    # for test
    tr_x = [i[:2] for i in tr_x]
    tr_y = tr_y[:2]
    val_x = [i[:2] for i in val_x]
    val_y = val_y[:2]
    ts_x = [i[:2] for i in ts_x]
    ts_y = ts_y[:2]
    
    
    # -- source-wise data preparation 

    if para_x_src_padding == True:
        # T and D different across data sources
        # padding to same T and D
        # y: [N 1], x: [S [N T D]]
        src_tr_x = data_padding_x(tr_x,
                                  num_src = len(tr_x))
        src_val_x = data_padding_x(val_x,
                                   num_src = len(tr_x))
        src_ts_x = data_padding_x(ts_x,
                                  num_src = len(tr_x))
        print("Shapes after padding: ", np.shape(src_tr_x), np.shape(src_val_x), np.shape(src_ts_x))
        
    else:
        src_tr_x = tr_x
        src_val_x = val_x
        src_ts_x = ts_x
        
    if para_add_common_factor == True:
        # x: [S [N T D]]
        # assume T is same across data sources
        
        # [N T sum(D)]
        tr_x_concat = np.concatenate(tr_x, -1)
        val_x_concat = np.concatenate(val_x, -1)
        ts_x_concat = np.concatenate(ts_x, -1)
        
        if para_common_factor_type == "pool":
            tr_x_factor = tr_x_concat
            val_x_factor = val_x_concat
            ts_x_factor = ts_x_concat
            
        elif para_common_factor_type == "factor":
            tmp_dim = np.shape(tr_x_concat)[-1]
            tmp_step = np.shape(tr_x_concat)[1]
            
            from sklearn.decomposition import FactorAnalysis
            transformer = FactorAnalysis(n_components = 10, 
                                         random_state = 0)
            # [N T d]
            tr_x_factor = []
            for tmp_x in tr_x_concat:
                # tmp_x: [T sum(D)] -> [T d]
                tr_x_factor.append(transformer.fit_transform(tmp_x))
                
            val_x_factor = []
            for tmp_x in val_x_concat:
                # tmp_x: [T sum(D)] -> [T d]
                val_x_factor.append(transformer.fit_transform(tmp_x))
            
            ts_x_factor = []
            for tmp_x in ts_x_concat:
                # tmp_x: [T sum(D)] -> [T d]
                ts_x_factor.append(transformer.fit_transform(tmp_x))
        
        # [S+1 [N T d]]
        src_tr_x.append(np.asarray(tr_x_factor))
        src_val_x.append(np.asarray(val_x_factor))
        src_ts_x.append(np.asarray(ts_x_factor))
    
    # steps and dimensionality of each source
    para_steps_x = []
    para_dim_x = []
    for tmp_src in range(len(src_tr_x)):
        tmp_shape = np.shape(src_tr_x[tmp_src][0])
        para_steps_x.append(tmp_shape[0])
        para_dim_x.append(tmp_shape[1])
        print("src " + str(tmp_src) + " shape: ", tmp_shape)
    
    shape_tr_x_dict = dict({"N": len(tr_x[0])})
    
    print(para_steps_x, para_dim_x)