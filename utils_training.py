#!/usr/bin/python

import numpy as np
import tensorflow as tf

# fix random seed
np.random.seed(1)
tf.set_random_seed(1)

# local 
from utils_libs import *

# ----- data preparation 

def data_reshape(data, 
                 bool_target_seperate):
    
    # S: source
    # N: instance
    # T: time steps
    # D: dimensionality at each step
    
    # data: [yi, ti, [xi_src1, xi_src2, ...]]
    # by default, the first element in the xi_src1 is the auto-regressive target
    
    src_num = len(data[0][2])
    tmpx = []
    
    if bool_target_seperate == True:
        
        tmpx.append(np.asarray([tmp[2][0][:, 1:] for tmp in data]))
        print(np.shape(tmpx[-1]))
    
        for src_idx in range(1, src_num):
            tmpx.append(np.asarray([tmp[2][src_idx] for tmp in data]))
            print(np.shape(tmpx[-1]))
            
        tmpx.append(np.asarray([tmp[2][0][:, 0:1] for tmp in data]))
        print(np.shape(tmpx[-1]))
    
    else:
        
        for src_idx in range(src_num):
            tmpx.append(np.asarray([tmp[2][src_idx] for tmp in data]))
            print("src " + str(src_idx) + " : ", np.shape(tmpx[-1]))
    
    tmpy = np.asarray([tmp[0] for tmp in data])
    
    # output shape: x [S N T D],  y [N 1]
    return tmpx, np.expand_dims(tmpy, -1)


def data_padding_x(x, 
                   num_src):
    
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


def func_mape(y, 
              yhat):
    
    tmp_list = []
    
    for idx, val in enumerate(y):
        
        if abs(val) > 1e-5:
            tmp_list.append(abs(1.0*(yhat[idx]-val)/val))
    
    return np.mean(tmp_list)

def func_mae(y, 
             yhat):
    
    return np.mean(np.abs(np.asarray(y) - np.asarray(yhat)))
    
def func_rmse(y, 
              yhat):
    
    return np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat))**2))


def batch_augment(x, 
                  y, 
                  num_src):
    
    # x: [S B T D], 
    # y: [B 1]
    
    # [B [1]]
    idx_y = [[idx, i] for idx, i in enumerate(y)]
    
    sort_idx_y = sorted(idx_y, key = lambda x: x[1][0], reverse = True)
    
    aug_num = int(0.2*len(y))
    
    for i in range(aug_num):
        
        tmp_idx = sort_idx_y[i][0]
        
        for j in range(num_src):
            x[j] = np.append(x[j], x[j][tmp_idx:tmp_idx+1], axis = 0)
            x[j] = np.append(x[j], x[j][tmp_idx:tmp_idx+1], axis = 0)
        
        y = np.append(y, y[tmp_idx : tmp_idx+1], axis = 0)
        y = np.append(y, y[tmp_idx : tmp_idx+1], axis = 0)
    
    return x, y



# ----- logging


def log_train_val_performance(path, 
                              hpara, 
                              hpara_error, 
                              train_time):
    
    with open(path, "a") as text_env:
        text_env.write("%s, %s, %s\n"%(str(hpara), str(hpara_error), str(train_time)))
        
        
def log_train_val_bayesian_error(path, 
                                 error):
    
    with open(path, "a") as text_env:
        text_env.write("          %s\n\n"%(str(error)))
        
        
def log_val_hyper_para(path, 
                       hpara_tuple, 
                       error_tuple):
    
    with open(path, "a") as text_file:
        text_file.write("\n  best hyper-parameters: %s \n"%(str(hpara_tuple)))
        text_file.write("\n  validation performance: %s \n"%(str(error_tuple)))
     
    
def log_test_performance(path, 
                         error_tuple,
                         ensemble_str):
    
    with open(path, "a") as text_file:
        text_file.write("\n  test performance %s : %s \n"%(ensemble_str, str(error_tuple)))
        
        
def log_null_loss_exception(epoch_errors, 
                            log_path):
    
    # epoch_errors: [ [step, tr_metric, val_metric, epoch] ]    
    for i in epoch_errors:
        
        if np.isnan(i[1][0]) == True:
            
            with open(log_path, "a") as text_file:
                text_file.write("\n  NULL loss exception at: %s \n"%(str(i[0])))
            
            break
    return


# ----- hyper-parameter searching 


def parameter_manager(shape_x_dict, 
                      hpara_dict):
    
    #hpara_dict = dict(zip(hyper_para_names, hyper_para_sample))
    #hpara_dict["batch_size"] = int(hpara_dict["batch_size"])
    
    tr_dict = {}
    
    ''' ? np.ceil ? '''
    tr_dict["batch_per_epoch"] = int(np.ceil(1.0*shape_x_dict["N"]/int(hpara_dict["batch_size"])))
    tr_dict["tr_idx"] = list(range(shape_x_dict["N"]))
        
    return tr_dict


# hpara: hyper-parameter    
class hyper_para_grid_search(object):
    
    def __init__(self, 
                 hpara_range):
        
        # lr_range, batch_size_range, l2_range
        self.n_hpara = len(hpara_range)
        self.hpara_range = hpara_range
        
        self.ini_flag = True
        
        self.idx = [0 for _ in range(self.n_hpara)]
        
    def one_trial(self):
        
        if self.ini_flag == True or self.trial_search(self.idx, 0, False) == True:
            
            self.ini_flag = False
            
            return [self.hpara_range[i][self.idx[i]] for i in range(self.n_hpara)]
        
        else:
            return None
        
    def trial_search(self, 
                     idx, 
                     cur_n, 
                     bool_restart):
        
        if cur_n >= self.n_hpara:
            return False
        
        if bool_restart == True:
            
            self.idx[cur_n] = 0
            self.trial_search(idx, cur_n + 1, True)
            
            return True
        
        else:
            
            if self.trial_search(self.idx, cur_n + 1, False) == False:
                
                if self.idx[cur_n] + 1 < len(self.hpara_range[cur_n]):
                    
                    self.idx[cur_n] += 1
                    self.trial_search(self.idx, cur_n + 1, True)
                    
                    return True
                
                else:
                    return False
            else:
                return True
            
            
class hyper_para_random_search(object):
    
    def __init__(self, 
                 hpara_range_dict, 
                 n_trial):
        # ?
        # np.random.seed(1)
        
        self.n_trial = n_trial
        self.cur_trial = 0
        
        # lr_range, batch_size_range, l2_range        
        # [[lower_boud, up_bound]]
        # self.hpara_range = hpara_range
        
        self.hpara_names = []
        self.hpara_range = []
        for tmp_name in hpara_range_dict:
            self.hpara_range.append(hpara_range_dict[tmp_name])
            self.hpara_names.append(tmp_name)
            
        self.n_hpara = len(self.hpara_range)
        
        # no duplication
        self.hpara_set = set()
        
        self.ini_flag = True
        
    def one_trial(self):
        
        if self.cur_trial < self.n_trial:
            
            self.cur_trial += 1
            return self.trial_search()
        
        else:
            return None
        
    def trial_search(self):
        
        # Return:
        # a name-value hyper-para dictionary
        
        bool_duplicate = True
        
        while bool_duplicate == True:
            
            tmp_hpara = ()
            for i in self.hpara_range:
                
                tmp_hpara = tmp_hpara + (i[0] + (i[1] - i[0])*np.random.random(), )
            
            # -- reconstruct the name-value hyper-para dictionary
            if tmp_hpara not in self.hpara_set:
                
                bool_duplicate = False
                self.hpara_set.add(tmp_hpara)
                
                hpara_instance = {} # {hpara names: values}
                for idx, tmp_hpara_val in enumerate(list(tmp_hpara)):
                    hpara_instance[self.hpara_names[idx]] = tmp_hpara_val
                
                return hpara_instance
                
        return
        
    
def hyper_para_selection(hpara_log, 
                         val_aggreg_num, 
                         test_snapshot_num,
                         metric_idx):
    
    # hpara_log - [ dict{lr, batch, l2, ..., burn_in_steps}, [[step, tr_metric, val_metric, epoch]] ]
    
    hp_err = []
    
    for hp_epoch_err in hpara_log:
        hp_err.append([hp_epoch_err[0], hp_epoch_err[1], np.mean([k[2][metric_idx] for k in hp_epoch_err[1][:val_aggreg_num]])])
    
    
    # sorted_hp[0]: hyper-para with the best validation performance
    sorted_hp = sorted(hp_err, key = lambda x:x[-1])
    
    
    # -- bayes steps
    full_steps = [k[0] for k in sorted_hp[0][1]]
    
    tmp_burn_in_step = sorted_hp[0][0]["burn_in_steps"]
    bayes_steps = [i for i in full_steps if i >= tmp_burn_in_step]
    bayes_steps_features = [ [k[1], k[2]] for k in sorted_hp[0][1] if k[0] >= tmp_burn_in_step ]
    
    
    # -- snapshot steps
    snapshot_steps = full_steps[:len(bayes_steps)]
    snapshot_steps_features = [ [k[1], k[2]] for k in sorted_hp[0][1][:len(bayes_steps)] ]
    
    best_hyper_para_dict = sorted_hp[0][0]
    # best hp, snapshot_steps, bayes_steps
    
    return best_hyper_para_dict,\
           snapshot_steps,\
           bayes_steps,\
           snapshot_steps_features,\
            bayes_steps_features
            
            