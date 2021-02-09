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
from tensorflow.contrib import rnn

# local packages 
from utils_training import *
from utils_inference import *
from mixture_models import *

def prepare_data(para_train):
    
    # ------ data
    tr_dta =  pickle.load(open(para_train['path_data'] + 'train_dese.p', "rb"), encoding = 'latin1')
    val_dta = pickle.load(open(para_train['path_data'] + 'val_dese.p', "rb"), encoding = 'latin1')
    ts_dta =  pickle.load(open(para_train['path_data'] + 'test_dese.p', "rb"), encoding = 'latin1')
    print(len(tr_dta), len(val_dta), len(ts_dta))
    
    # if para_bool_target_seperate = yes, the last source corresponds to the auto-regressive target variable
    tr_x, tr_y = data_reshape(tr_dta, 
                              bool_target_seperate = para_train['para_bool_target_seperate'])
    val_x, val_y = data_reshape(val_dta,
                                bool_target_seperate = para_train['para_bool_target_seperate'])
    ts_x, ts_y = data_reshape(ts_dta,
                              bool_target_seperate = para_train['para_bool_target_seperate'])
    
    # --- log transformation of y
        
    # output from the reshape
    # y [N 1], x [S [N T D]]
    print("training: ", len(tr_x[0]), len(tr_y))
    print("validation: ", len(val_x[0]), len(val_y))
    print("testing: ", len(ts_x[0]), len(ts_y))
    
    # --- source-wise data preparation 

    if para_train['para_x_src_padding'] == True:
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
        
    if para_train['para_add_common_factor'] == True:
        # x: [S [N T D]]
        # assume T is same across data sources
        
        # [N T sum(D)]
        tr_x_concat = np.concatenate(tr_x, -1)
        val_x_concat = np.concatenate(val_x, -1)
        ts_x_concat = np.concatenate(ts_x, -1)
        
        if para_train['para_common_factor_type'] == "pool":
            tr_x_factor = tr_x_concat
            val_x_factor = val_x_concat
            ts_x_factor = ts_x_concat
            
        elif para_train['para_common_factor_type'] == "factor":
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
    
    para_train['x_steps'] = para_steps_x
    para_train['x_dims'] = para_dim_x
    para_train['y_dim'] = len(tr_y[0])
    para_train['tr_num_ins'] = len(tr_x[0])    
    
    return src_tr_x, tr_y, src_val_x, val_y, src_ts_x, ts_y,

# ------ 

def train_validate_process(xtr,
                        ytr,
                        xval,
                        yval,
                        hyper_para,
                        para_train,
                        retrain_top_steps, 
                        retrain_bayes_steps,
                        retrain_bool,
                        retrain_iter_idx,
                        random_seed):
    '''
    Argu.:
      xtr: [num_src, N, T, D]
         S: num_src
         N: number of data samples
         T: number of steps
         D: dimension at each time step
      ytr: [N 1]
      
      hyper_para_dict: 
       "lr": float,
       "batch_size": int
       "l2": float,
                
       "lstm_size": int,
       "dense_num": int,
       "use_hidden_before_dense": bool
    '''
    # clear the graph in the current session 
    tf.reset_default_graph()
    
    with tf.device('/device:GPU:0'):
        
        # -- initialize the network
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
                                  para_train = para_train)        
        model.network_ini(hyper_para = hyper_para)
        
        # !! the order of Saver
        saver = tf.train.Saver(max_to_keep = None)
        
        model.train_ini()
        model.inference_ini()
        #tf.get_default_graph().finalize()
        
        # -- set up training batch parameters
        batch_gen = data_loader(x = xtr,
                                y = ytr,
                                batch_size = int(hyper_para["batch_size"]), 
                                num_src = int(para_train['para_num_source']))
        # -- begin training
        
        # training and validation error log
        step_error = []
        
        # training time counter
        st_time = time.time()
        
        for epoch in range(para_train['para_n_epoch']):
            
            # shuffle traning instances each epoch
            batch_gen.re_shuffle()
            batch_x, batch_y, bool_last = batch_gen.one_batch()
            
            # - loop over all batches
            while batch_x != None:
                # one-step training on a batch of training data
                model.train_batch(batch_x, 
                                  batch_y,)                
                # next batch
                batch_x, batch_y, bool_last = batch_gen.one_batch()
                
            # - epoch-wise validating
            val_metric, _, monitor_metric = model.inference(xval,
                                                            yval,
                                                            bool_instance_eval = False)
            tr_metric, _, _ = model.inference(xtr,
                                              ytr,
                                              bool_instance_eval = False)
            step_error.append([epoch, tr_metric, val_metric])
                    
            # - model saver 
            model_saver_flag = model.model_saver(path = para_train['path_model'] + para_train['para_model_type'] + '_' + str(retrain_iter_idx) + '_' + str(epoch),
                                                 epoch = epoch,
                                                 top_snapshots = retrain_top_steps,
                                                 bayes_snapshots = retrain_bayes_steps,
                                                 early_stop_bool = para_train['para_early_stop_bool'],
                                                 early_stop_window = para_train['para_early_stop_window'], 
                                                 tf_saver = saver)
            # epoch-wise
            print("\n --- At epoch %d : \n  %s "%(epoch, str(step_error[-1])))
            print("\n   loss and regualization : \n", monitor_metric)
            
            # NAN value exception 
            if np.isnan(monitor_metric[0]) == True:
                print("\n --- NAN loss !! \n" )
                break
            # model save message    
            if retrain_bool == True and model_saver_flag != None:
                print("\n    [MODEL SAVED] " + model_saver_flag + " \n " + para_train['path_model'] + para_train['para_model_type'] + '_' + str(retrain_iter_idx) + '_' + str(epoch))
                
        ed_time = time.time()
        
    # sort step_error based on para_validation_metric
    sort_step_error = sorted(step_error, key = lambda x:x[2][para_train['para_metric_map'][para_train['para_validation_metric']]])
    
    return sort_step_error,\
           1.0*(ed_time - st_time)/(epoch + 1e-5),\

# ------
    
def test_process(retrain_snapshots,
                 retrain_ids,
                 xts,
                 yts,
                 snapshot_features, 
                 para_train):
    
    # ensemble of model snapshots
    infer = ensemble_inference()
    
    with tf.device('/device:GPU:0'):
        
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        
        for tmp_idx, tmp_retrain_id in enumerate(retrain_ids):
            
            for tmp_model_id in retrain_snapshots[tmp_idx]:
                
                # path of the stored models 
                tmp_meta = para_train['path_model'] + para_train['para_model_type'] + '_' + str(tmp_retrain_id) + '_' + str(tmp_model_id) + '.meta'
                tmp_data = para_train['path_model'] + para_train['para_model_type'] + '_' + str(tmp_retrain_id) + '_' + str(tmp_model_id)
        
                # clear graph
                tf.reset_default_graph()
                
                # restore the model
                saver = tf.train.import_meta_graph(tmp_meta, 
                                                   clear_devices = True)
                sess = tf.Session(config = config)
                
                model = mixture_statistic(session = sess,
                                          para_train = para_train)
                model.model_restore(tmp_data, 
                                    saver)
                # one-shot inference
                error_tuple, py_tuple, _ = model.inference(xts,
                                                        yts, 
                                                        bool_instance_eval = True)
                infer.add_samples(py_mean = py_tuple[0],
                                  py_var = py_tuple[1],
                                  py_mean_src = py_tuple[2],
                                  py_var_src = py_tuple[3],
                                  py_gate_src = py_tuple[4],
                                  py_lk = py_tuple[5])
    
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
# ------ 

def train_validate_test(src_tr_x,
                        tr_y,
                        src_val_x,
                        val_y,
                        src_ts_x,
                        ts_y,
                        hyper_para_range, 
                        para_train):
    
    # -- hyper-para generator 
    if para_train['para_hpara_search'] == "random":        
        hpara_generator = hyper_para_random_search(hyper_para_range[para_train['para_hpara_search']][para_train['para_model_type']], 
                                                   para_train['para_hpara_train_trial_num'])
    elif para_train['para_hpara_search'] == "grid":
        hpara_generator = hyper_para_grid_search(hyper_para_range[para_train['para_hpara_search']][para_train['para_model_type']])
            
    # -- begin hyper-para search
    hpara_log = []
    
    # sample one hyper-para instance
    hpara_instance = hpara_generator.one_trial()
     
    # ------ train and validate for each hyper-para instance
    while hpara_instance != None:
        
        # hp_step_error: [[step, train_metric, val_metric, epoch]]
        hp_step_error, hp_epoch_time = train_validate_process(src_tr_x,
                                                           tr_y,
                                                           src_val_x,
                                                           val_y,
                                                           hyper_para = hpara_instance,
                                                           para_train = para_train,
                                                           retrain_bool = False,
                                                           retrain_top_steps = [],
                                                           retrain_bayes_steps = [],
                                                           retrain_iter_idx = 0,
                                                           random_seed = 1)
        hpara_log.append([hpara_instance, hp_step_error])
        
        # sample the next hyper-para
        hpara_instance = hpara_generator.one_trial()
        
        # log
        log_train_val_performance(para_train['path_log_error'],
                                  hpara = hpara_log[-1][0],
                                  hpara_error = hpara_log[-1][1][0],
                                  train_time = hp_epoch_time)
        # NAN loss exception
        log_null_loss_exception(hp_step_error, 
                                para_train['path_log_error'])
        
        print('\n Validation performance under the hyper-parameters: \n', hpara_log[-1][0], hpara_log[-1][1][0])
        print('\n Training time: \n', hp_epoch_time, '\n')
        
    # ------ re-train
    # save all epoches in re-training, then select snapshots
    
    # best hyper-para
    best_hpara = hyper_para_selection(hpara_log,
                                      val_snapshot_num = para_train['para_vali_snapshot_num'], 
                                      metric_idx = para_train['para_metric_map'][para_train['para_validation_metric']])                                                                
    retrain_hpara_steps = []
    retrain_hpara_step_error = []
    retrain_random_seeds = [1] + [randint(0, 1000) for _ in range(para_train['para_hpara_retrain_num']-1)]
    
    for tmp_retrain_id in range(para_train['para_hpara_retrain_num']):
                                                                                                   
        step_error, _ = train_validate_process(src_tr_x,
                                            tr_y,
                                            src_val_x,
                                            val_y,
                                            hyper_para = best_hpara,
                                            para_train = para_train,
                                            retrain_bool = True,
                                            retrain_top_steps = list(range(para_train['para_n_epoch'])), # top_steps,
                                            retrain_bayes_steps = list(range(para_train['para_n_epoch'])), # bayes_steps,
                                            retrain_iter_idx = tmp_retrain_id,
                                            random_seed = retrain_random_seeds[tmp_retrain_id])
        
        top_steps, bayes_steps, top_steps_features, bayes_steps_features, val_error, step_error_pairs = snapshot_selection(train_log = step_error,
                                                                                                                           snapshot_num = para_train['para_test_snapshot_num'],
                                                                                                                           total_step_num = para_train['para_n_epoch'],
                                                                                                                           metric_idx = para_train['para_metric_map'][para_train['para_validation_metric']],
                                                                                                                           val_snapshot_num = para_train['para_vali_snapshot_num'])
        if len(top_steps) != 0:
            retrain_hpara_steps.append([top_steps, bayes_steps, top_steps_features, bayes_steps_features, tmp_retrain_id, val_error])
            retrain_hpara_step_error.append([step_error_pairs, tmp_retrain_id])
        
        log_val_hyper_para(path = para_train['path_log_error'],
                           hpara_tuple = [best_hpara, top_steps],
                           error_tuple = step_error[0], 
                           log_string = "-- " + str(tmp_retrain_id))
    
        print('\n----- Retrain hyper-parameters: ', best_hpara, top_steps, '\n')
        print('\n----- Retrain validation performance: ', step_error[0], '\n')
    
    # sort each re-train trial by the validation error
    sort_retrain_hpara_steps = sorted(retrain_hpara_steps, 
                                      key = lambda x:x[-1])
    
    log_test_performance(path = para_train['path_log_error'], 
                         error_tuple = [i[-2:] for i in sort_retrain_hpara_steps], 
                         ensemble_str = "Retrain Ids and Vali. Errors: ")
    
    log_test_performance(path = para_train['path_log_error'], 
                         error_tuple = [i[-2:] for i in sort_retrain_hpara_steps[:para_train['para_hpara_ensemble_trial_num']]], 
                         ensemble_str = "Retrain Ids for ensemble: ")
    
    # ------ test
    
    # -- one snapshot from one retrain
    error_tuple, py_tuple = test_process(retrain_snapshots = [sort_retrain_hpara_steps[0][0][:1]],
                                         retrain_ids = [ sort_retrain_hpara_steps[0][-2] ],
                                         xts = src_ts_x, 
                                         yts = ts_y, 
                                         snapshot_features = [],
                                         para_train = para_train)
    log_test_performance(path = para_train['path_log_error'], 
                         error_tuple = [error_tuple], 
                         ensemble_str = "One-shot-one-retrain")
    pickle.dump(py_tuple, 
                open(para_train['path_py'] + "_one_one" + ".p", "wb"))
    
    # -- one snapshot from multi retrain
    error_tuple, py_tuple = test_process(retrain_snapshots = [tmp_steps[0][:1] for tmp_steps in sort_retrain_hpara_steps], 
                                         retrain_ids = [i[-2] for i in sort_retrain_hpara_steps[:para_train['para_hpara_ensemble_trial_num']]],
                                         xts = src_ts_x,
                                         yts = ts_y, 
                                         snapshot_features = [],
                                         para_train = para_train)
    log_test_performance(path = para_train['path_log_error'], 
                         error_tuple = [error_tuple], 
                         ensemble_str = "One-shot-multi-retrain")
    pickle.dump(py_tuple, 
                open(para_train['path_py'] + "_one_multi" + ".p", "wb"))
    
    # -- top snapshots from one retrain
    error_tuple, py_tuple = test_process(retrain_snapshots = [sort_retrain_hpara_steps[0][0]], 
                                         retrain_ids = [ sort_retrain_hpara_steps[0][-2] ], 
                                         xts = src_ts_x, 
                                         yts = ts_y, 
                                         snapshot_features = [], 
                                         para_train = para_train)
    log_test_performance(path = para_train['path_log_error'],
                         error_tuple = [error_tuple],
                         ensemble_str = "Top-shots-one-retrain")
    pickle.dump(py_tuple, 
                open(para_train['path_py'] + "_top_one" + ".p", "wb"))
    
    # -- top snapshots multi retrain
    error_tuple, py_tuple = test_process(retrain_snapshots = [tmp_steps[0] for tmp_steps in sort_retrain_hpara_steps], 
                                         retrain_ids = [i[-2] for i in sort_retrain_hpara_steps[:para_train['para_hpara_ensemble_trial_num']]], 
                                         xts = src_ts_x,
                                         yts = ts_y,
                                         snapshot_features = [], 
                                         para_train = para_train)
    log_test_performance(path = para_train['path_log_error'], 
                         error_tuple = [error_tuple], 
                         ensemble_str = "Top-shots-multi-retrain")
    pickle.dump(py_tuple, 
                open(para_train['path_py'] + "_top_multi" + ".p", "wb"))
    
    # -- bayesian snapshots one retrain
    error_tuple, py_tuple = test_process(retrain_snapshots = [sort_retrain_hpara_steps[0][1]], 
                                         retrain_ids = [ sort_retrain_hpara_steps[0][-2] ], 
                                         xts = src_ts_x, 
                                         yts = ts_y,
                                         snapshot_features = [],
                                         para_train = para_train)
    log_test_performance(path = para_train['path_log_error'], 
                         error_tuple = [error_tuple], 
                         ensemble_str = "Bayesian-one-retrain")
    pickle.dump(py_tuple, 
                open(para_train['path_py'] + "_bayes_one" + ".p", "wb"))
    
    # -- bayesian snapshots multi retrain
    error_tuple, py_tuple = test_process(retrain_snapshots = [tmp_steps[1] for tmp_steps in sort_retrain_hpara_steps],
                                         retrain_ids = [i[-2] for i in sort_retrain_hpara_steps[:para_train['para_hpara_ensemble_trial_num']]],
                                         xts = src_ts_x,
                                         yts = ts_y,
                                         snapshot_features = [], 
                                         para_train = para_train)
    log_test_performance(path = para_train['path_log_error'],
                         error_tuple = [error_tuple],
                         ensemble_str = "Bayesian-multi-retrain")
    pickle.dump(py_tuple, 
                open(para_train['path_py'] + "_bayes_multi" + ".p", "wb"))
    
    # -- global top1 and topK steps
    retrain_ids, retrain_id_steps = global_top_steps_multi_retrain(retrain_step_error = retrain_hpara_step_error, 
                                                                   num_step = int(para_train['para_test_snapshot_num']*para_train['para_hpara_ensemble_trial_num']))    
    log_test_performance(path = para_train['path_log_error'], 
                         error_tuple = [retrain_ids, retrain_id_steps], 
                         ensemble_str = "Global-top-steps: ")
    
    error_tuple, py_tuple = test_process(retrain_snapshots = retrain_id_steps, 
                                         retrain_ids = retrain_ids,
                                         xts = src_ts_x,
                                         yts = ts_y, 
                                         snapshot_features = [], 
                                         para_train = para_train)
    log_test_performance(path = para_train['path_log_error'], 
                         error_tuple = [error_tuple], 
                         ensemble_str = "Global-top-steps-multi-retrain ")
    pickle.dump(py_tuple, 
                open(para_train['path_py'] + "_global" + ".p", "wb"))
    