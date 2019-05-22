#!/usr/bin/python

import numpy as np

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


def mape(y, 
         yhat):
    
    tmp_list = []
    
    for idx, val in enumerate(y):
        
        if abs(val) > 1e-5:
            tmp_list.append(abs(1.0*(yhat[idx]-val)/val))
    
    return np.mean(tmp_list)

def mae(y, 
        yhat):
    
    return np.mean(np.abs(np.asarray(y) - np.asarray(yhat)))
    
def rmse(y, 
         yhat):
    
    return np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat))**2))


# ----- logging


def log_train_val_performance(path, 
                              hpara, 
                              hpara_error, 
                              train_time):
    
    with open(path, "a") as text_env:
        text_env.write("%s, %s, %s\n"%(str(hpara), str(hpara_error), str(train_time)))
        
        
def log_train_val_bayesian_error(path, 
                                 error, 
                                 ):
    
    with open(path, "a") as text_env:
        text_env.write("          %s\n\n"%(str(error)))
        
        
def log_val_hyper_para(path, 
                       hpara_tuple, 
                       error_tuple):
    
    with open(path, "a") as text_file:
        text_file.write("\n  best hyper-parameters: %s \n"%(str(hpara_tuple)))
        text_file.write("\n  validation performance: %s \n"%(str(error_tuple)))
     
    
def log_test_performance(path, 
                         error_tuple):
    
    with open(path, "a") as text_file:
        text_file.write("\n  test performance: %s \n"%(str(error_tuple)))
        
        
def log_null_loss_exception(epoch_errors, 
                            log_path):
    
    # epoch_errors: [[epoch, loss, train_rmse, val_rmse, val_mae, val_mape, val_nnllk]]    
    for i in epoch_errors:
        
        if np.isnan(i[1]) == True:
            
            with open(log_path, "a") as text_file:
                text_file.write("\n  NULL loss exception at: %s \n"%(str(i[0])))
            
            break
    return


# ----- hyper-parameter searching 

# hpara: hyper-parameter
    
class hpara_grid_search(object):
    

    def __init__(self, 
                 hpara_range):
        
        # lr_range, batch_size_range, l2_range
        self.n_hpara = len(hpara_range)
        self.hpara_range = hpara_range
        
        self.ini_flag = True
        
        self.idx = [0 for _ in range(self.n_hpara)]
        
    def hpara_trial(self):
        
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
            
            
class hpara_random_search(object):
    
    def __init__(self, 
                 hpara_range, 
                 n_trial):
        # ?
        # np.random.seed(1)
        
        self.n_hpara = len(hpara_range)
        # lr_range, batch_size_range, l2_range
        
        self.n_trial = n_trial
        self.cur_trial = 0
        
        # [[lower_boud, up_bound]]
        self.hpara_range = hpara_range
        
        # no duplication
        self.hpara_set = set()
        
        self.ini_flag = True
        
    def hpara_trial(self):
        
        if self.cur_trial < self.n_trial:
            
            self.cur_trial += 1
            return self.trial_search()
        
        else:
            return None
        
        
    def trial_search(self):
        
        bool_duplicate = True
        
        while bool_duplicate == True:
            
            tmp_hpara = ()
            for i in self.hpara_range:
                
                tmp_hpara = tmp_hpara + (i[0] + (i[1] - i[0])*np.random.random(), )
            
            
            # 
            if tmp_hpara not in self.hpara_set:
                
                bool_duplicate = False
                self.hpara_set.add(tmp_hpara)
                
                return tmp_hpara
                
        return
        
        
    
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
    
    # best hp, epoch_sample, best validation error
    return sorted_hp[0][0],\
           [k[0] for k in sorted_hp[0][1]][:test_epoch_num],\
           min([tmp_epoch[3] for tmp_epoch in sorted_hp[0][1]])