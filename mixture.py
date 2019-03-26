#!/usr/bin/python

import gzip
import os
import tempfile

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.contrib import rnn

import math
import random

import collections
import hashlib
import numbers

# library for bayesian inference
#import edward as ed
#from edward.models import Categorical, Dirichlet, Empirical, InverseGamma, MultivariateNormalDiag, Normal, ParamMixture, Beta, Bernoulli, Mixture

# local packages
from utils_libs import *

# reproducibility by fixing the random seed
# np.random.seed(1)
# tf.set_random_seed(1)

# ---- utilities functions ----

def linear(x, 
           dim_x, 
           scope, 
           bool_bias):
    
    with tf.variable_scope(scope):
        
        w = tf.get_variable('w', 
                        tf.random_normal([dim_x, 1], stddev = math.sqrt(1.0/float(dim_x))))
        
        b = tf.get_variable('b',
                        tf.zeros([1,]))
        
        if bool_bias == True:
            h = tf.matmul(x, w) + b
        else:
            h = tf.matmul(x, w)
           
        #l2
        regularizer = tf.reduce_sum(tf.square(w_v))
           
           # [B]
    return tf.squeeze(h), regularizer

def bilinear(x, 
             shape_x, 
             scope,
             bool_bias):
    
    # shape of x: [b, l, r]
    # shape_x: [l, r]
    with tf.variable_scope(scope):
        
        w_l = tf.Variable(name = 'w_left', 
                          initial_value = tf.random_normal([shape_x[0], 1], stddev = math.sqrt(1.0/float(shape_x[0]))))
        
        w_r = tf.Variable(name = 'w_right', 
                          initial_value = tf.random_normal([shape_x[1], 1], stddev = math.sqrt(1.0/float(shape_x[1]))))
        
        b = tf.Variable(tf.zeros([1,]))
        
        tmph = tf.tensordot(x, w_r, 1)
        tmph = tf.squeeze(tmph, [2])
        
        if bool_bias == True:
            h = tf.matmul(tmph, w_l) + b
        else:
            h = tf.matmul(tmph, w_l)
            
           # [B] 
    return tf.squeeze(h), tf.reduce_sum(tf.square(w_t)) + tf.reduce_sum(tf.square(w_v)) 

# ---- Mixture linear ----

class mixture_statistic():
    
    def __init__(self, 
                 session, 
                 loss_type):
        
        '''
        Args:
        
        session: tensorflow session
        
        loss_type: string, type of loss functions, {mse, lk, lk_inv}
        
        '''
        
        # build the network graph 
        self.lr = 0.0
        self.l2 = 0.0
        
        self.epsilon = 1e-3
        
        self.sess = session
        
        self.bool_log = ''
        self.loss_type = loss_type
        self.distr_type = ''
        

    def network_ini(self, 
                    lr, 
                    l2, 
                    dim_x_list,
                    steps_x_list, 
                    bool_log, 
                    bool_bilinear,
                    distr_type, 
                    bool_regu_positive_mean,
                    bool_regu_gate):
        
        '''
        Args:
        
        lr: float, learning rate
        
        l2: float, l2 regularization
        
        dim_x_list: list of dimension values for each component in X
        
        steps_x_list: list of sequence length values for each component in X
        
        bool_log: if log operation is on the targe variable Y
        
        bool_bilinear: if bilinear function is used on the components in X
        
        
        
        distr_type: string, type of the distribution of the target variable
        
        bool_regu_positive_mean: if regularization of positive mean 
        
        bool_regu_gate: if regularization the gate functions
        
        '''
        
        # bool_hidden_depen
        
        # ---- fix the random seed to reproduce the results
        np.random.seed(1)
        tf.set_random_seed(1)
        
        # ---- ini
        
        # build the network graph 
        self.lr = lr
        self.l2 = l2
        
        self.epsilon = 1e-3
        
        #self.sess = session
        
        self.bool_log = bool_log
        self.loss_type = loss_type
        self.distr_type = distr_type
        
        # initialize placeholders
        self.y = tf.placeholder(tf.float32, [None, 1], name = 'y')
        
        # ph: placeholder 
        x_ph_list = []
        num_compt = len(dim_x_list)
        
        for i in range(num_compt):
            
            if bool_bilinear == True:
                x_ph_list.append(tf.placeholder(tf.float32, [None, steps_x_list[i], dim_x_list[i]], name = 'x' + str(i)))
            else:
                x_ph_list.append(tf.placeholder(tf.float32, [None, steps_x_list[i]*dim_x_list[i]], name = 'x' + str(i)))
                
                
        # ---- prediction of individual models
        
        mean_list = []
        var_list = []
        logit_list = []
        
        regu_mean = 0.0
        regu_var = 0.0
        regu_gate = 0.0
        
        for i in range(num_compt):
            
            if bool_bilinear == True:
                
                #[B]
                tmp_mean, tmp_regu_mean = bilinear(x_ph_list[i], 
                                                   [steps_x_list[i], dim_x_list[i]], 
                                                   'mean' + str(i), 
                                                   True)
                
                tmp_var, tmp_regu_var = bilinear(x_ph_list[i], 
                                                 [steps_x_list[i], dim_x_list[i]], 
                                                 'var' + str(i), 
                                                 True)
                
                tmp_logit, tmp_regu_gate = bilinear(x_ph_list[i],
                                                    [steps_x_list[i], dim_x_list[i]], 
                                                    'gate' + str(i),
                                                    True)
                
            else:
                
                #[B]
                tmp_mean, tmp_regu_mean = linear(x_ph_list[i], 
                                                 dim_x_list[i], 
                                                 'mean' + str(i),
                                                 True)
                
                tmp_var, tmp_regu_var = linear(x_ph_list[i], 
                                               dim_x_list[i], 
                                               'var' + str(i), 
                                               True)
                
                tmp_logit, tmp_regu_gate = linear(x_ph_list[i], 
                                                  dim_x_list[i], 
                                                  'gate' + str(i), 
                                                  True)
                
            # [C B]
            mean_list.append(tmp_mean)
            var_list.append(tf.square(tmp_var))
            logit_list.append(tmp_logit)
            
            regu_mean += tmp_regu_mean
            regu_var += tmp_regu_var
            regu_gate += tmp_regu_gate
            
        # concatenate individual means and variance
        # [B C]
        mean_stack = tf.stack(mean_list, 1)
        var_stack = tf.stack(var_list, 1)
        inv_var_stack = tf.stack(var_list, 1)
            
        # [B C]
        self.logit = tf.stack(logit_list, 1)
        self.gates = tf.nn.softmax(self.logit, axis = -1)
        
        # ---- regularization
        
        # temporal coherence, diversity 
        # gate smoothness 
        # gate diversity
        
        # mean diversity
        
        # mean non-negative  
        # regu_mean_pos = tf.reduce_sum( tf.maximum(0.0, -1.0*mean_v) + tf.maximum(0.0, -1.0*mean_x) )
        
        self.regularization = l2*regu_mean
        
        '''
        if loss_type == 'sq' and distr_type == 'gaussian':
            self.regularization = l2*(regu_mean)
                
        elif loss_type == 'lk' and distr_type == 'gaussian':
            self.regularization = l2*(regu_mean + regu_var)
                
        else:
            print '[ERROR] loss type'
        '''
        
        # activation and hinge regularization 
        if bool_regu_positive_mean == True:
            self.regularization += 0.1*l2*regu_mean_pos
            
        if bool_regu_gate == True:
            self.regularization += 0.1*l2*regu_gate
            
            
        # ---- negative log likelihood 
        
        # Dictionary
        #   nllk: negative log likelihood
        #   hetero: heteroskedasticity
        #   inv: inversed
        #   const: constant
        #   indi: individual
        #   py: predicted y

        
        # lk    
        # [B C]
        tmpllk_indi_hetero = tf.exp(-0.5*tf.square(self.y - mean_stack)/(var_stack + 1e-5))/(2.0*np.pi*(var_stack + 1e-5))**0.5
            
        llk_hetero = tf.multiply(tmpllk_indi_hetero, self.gates) 
        self.nllk_hetero = tf.reduce_sum(-1.0*tf.log(tf.reduce_sum(llk_hetero, axis = -1) + 1e-5))
        
        # lk_inv
        # [B C]
        tmpllk_indi_hetero_inv = tf.exp(-0.5*tf.square(self.y - mean_stack)*inv_var_stack)*(0.5/np.pi*inv_var_stack)**0.5
            
        llk_hetero_inv = tf.multiply(tmpllk_indi_hetero_inv, self.gates) 
        self.nllk_hetero_inv = tf.reduce_sum(-1.0*tf.log(tf.reduce_sum(llk_hetero_inv, axis = -1) + 1e-5))
        
        # mse
        # [B C]
        tmpllk_indi_const = tf.exp(-0.5*tf.square(self.y - mean_stack))/(2.0*np.pi)**0.5
            
        llk_const = tf.multiply(tmpllk_indi_const, self.gates) 
        self.nllk_const = tf.reduce_sum(-1.0*tf.log(tf.reduce_sum(llk_const, axis = -1) + 1e-5))
        
        # ---- prediction 
        
        if distr_type == 'gaussian':
            
            # component mean
            self.py_mean_indi = mean_stack
            
            # mixture mean
            # [B]
            self.py = tf.reduce_sum(tf.multiply(mean_stack, self.gates), 1)
            
            # mixture variance
            if self.loss_type == 'lk':
                
                # component variance
                self.py_var_indi = var_stack
                
                sq_mean_stack = var_stack + tf.square(mean_stack)
                mix_sq_mean = tf.reduce_sum(tf.multiply(sq_mean_stack, self.gates), 1)
            
            elif self.loss_type == 'lk_inv':
                
                # component variance
                self.py_var_indi = inv_var_stack
                
                sq_mean_stack = 1.0/(inv_var_stack + 1e-5) + tf.square(mean_stack)
                mix_sq_mean = tf.reduce_sum(tf.multiply(sq_mean_stack, self.gates), 1)
                
            elif self.loss_type == 'mse':
                
                # component variance
                self.py_var_indi = 1.0
                
                sq_mean_stack = 1.0 + tf.square(mean_stack)
                mix_sq_mean = tf.reduce_sum(tf.multiply(sq_mean_stack, self.gates), 1)
                
            # [B]
            self.py_var = mix_sq_mean - tf.square(self.py)
        
        else:
            print('[ERROR] distribution type')
        

    #   initialize loss and optimization operations for training
    def train_ini(self):
        
        self.sq_error = tf.reduce_sum(tf.square(self.y - self.py))
        
        # loss, nllk, py_mean, py_std
        
        # loss
        if self.loss_type == 'mse':
            
            self.loss = tf.reduce_mean(tf.square(self.y - self.py)) + self.regularization
            self.nllk = self.nllk_const
            
        elif self.loss_type == 'lk_inv':
            
            self.loss = self.nllk_hetero_inv + self.regularization + 0.1*self.l2*regu_var
            self.nllk = self.nllk_hetero_inv
            
        elif self.loss_type == 'lk':
            
            self.loss = self.nllk_hetero + self.regularization + 0.1*self.l2*regu_var
            self.nllk = self.nllk_hetero
            
        else:
            print('[ERROR] loss type')
        
        # [B]
        self.py_mean = self.py
        self.py_std = tf.sqrt(self.py_var)
        
        
        self.train = tf.train.AdamOptimizer(learning_rate = self.lr)
        self.optimizer =  self.train.minimize(self.loss)
        
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        
        
    #   training on batch of data
    def train_batch(self, x_list, y):
        
        data_dict = {}
        for idx, i in enumerate(self.x_ph_list):
            data_dict[i] = x_list[idx]
        
        data_dict[self.y] = y
        
        _, tmp_loss, tmp_sq_err = self.sess.run([self.optimizer, self.loss, self.sq_error],
                                                feed_dict = data_dict)
                             
        return tmp_loss, tmp_sq_err
    
    
    #   error metric
    def inference_ini(self):
        
        # RMSE
        self.rmse = tf.sqrt(tf.losses.mean_squared_error(self.y, self.py))
        
        # MAE
        self.mae  = tf.reduce_mean(tf.abs(self.y - self.py))
        
        # MAPE
        # based on ground-truth y
        mask = tf.greater(tf.abs(self.y), 1e-5)
        
        y_mask = tf.boolean_mask(self.y, mask)
        y_hat_mask = tf.boolean_mask(self.py, mask)
        
        self.mape = tf.reduce_mean(tf.abs((y_mask - y_hat_mask)/(y_mask + 1e-10)))
        
    
    #   infer givn testing data
    def inference(self, x_list, y, bool_indi_eval):
        
        # rmse, mae, mape, nllk, py_mean, py_std
        
        data_dict = {}
        for idx, i in enumerate(self.x_ph_list):
            data_dict[i] = x_list[idx]
        
        data_dict[self.y] = y
        
        rmse, mae, mape, nllk = self.sess.run([self.rmse, self.mae, self.mape, self.nllk], feed_dict = data_dict)
        
        if bool_indi_eval == True:
            
            py_mean, py_std = self.sess.run([self.py_mean, self.py_std], feed_dict = data_dict)
            
        else:
            py_mean = None
            py_std = None
            
        return rmse, mae, mape, nllk, py_mean, py_std
    
    
    # collect the optimized variable values
    def collect_coeff_values(self, vari_keyword):
        return [tf_var.name for tf_var in tf.trainable_variables() if (vari_keyword in tf_var.name)],
    [tf_var.eval() for tf_var in tf.trainable_variables() if (vari_keyword in tf_var.name)]
    
    
    # restore the model from the files
    def pre_train_restore_model(self, 
                                path_meta, 
                                path_data):
        
        saver = tf.train.import_meta_graph(path_meta, clear_devices=True)
        saver.restore(self.sess, path_data)
        
        return
    
