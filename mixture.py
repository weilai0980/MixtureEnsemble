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

from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops.rnn_cell_impl import * 

# library for bayesian inference
import edward as ed
from edward.models import Categorical, Dirichlet, Empirical, InverseGamma, MultivariateNormalDiag, Normal, ParamMixture, Beta, Bernoulli, Mixture

# local packages
from utils_libs import *
from ts_mv_rnn_basics import *

# reproducibility by fixing the random seed
np.random.seed(1)
tf.set_random_seed(1)

# ---- utilities functions ----

def linear(x, 
           dim_x, 
           scope, 
           bool_bias):
    
    with tf.variable_scope(scope):
        
        w = tf.Variable(name = 'w', 
                        tf.random_normal([dim_x, 1], stddev = math.sqrt(1.0/float(dim_x))))
        
        b = tf.Variable(name = 'b', 
                        tf.zeros([1,]))
        
        if bool_bias == True:
            h = tf.matmul(x, w) + b
        else:
            h = tf.matmul(x, w)
           
        #l2
        regularizer = tf.nn.l2_loss(w)
        
    return tf.squeeze(h), regularizer

def bilinear(x, 
             shape_x, 
             scope,
             bool_bias):
    
    # shape of x: [b, t, v]

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
            
    # l2, l1 regularization
    #tf.nn.l2_loss(w_t) tf.reduce_sum(tf.abs(w_v)
    return tf.squeeze(h), tf.reduce_sum(tf.square(w_t)) + tf.reduce_sum(tf.abs(w_v)) 

# ---- Mixture linear ----

class mixture_linear():

    def __init__(self, 
                 session, 
                 lr, 
                 l2, 
                 batch_size, 
                 dim_x_list,
                 steps_x_list, 
                 bool_log, 
                 bool_bilinear,
                 loss_type, 
                 distr_type, 
                 activation_type, 
                 pos_regu, 
                 gate_type):
        
        # bool_hidden_depen
        
        # build the network graph 
        self.LEARNING_RATE = lr
                
        self.N_BATCH = batch_size
        self.L2 = l2
        
        self.MAX_NORM = 0.0
        self.epsilon = 1e-3
        
        self.sess = session
        
        self.bool_log = bool_log
        self.loss_type = loss_type
        self.distr_type = distr_type
        
        # initialize placeholders
        self.y = tf.placeholder(tf.float32, [None, 1], name = 'y')
        self.keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
        
        # ph: placeholder 
        x_ph_list = []
        num_compt = len(dim_x_list)
        
        for i in range(num_compt):
            
            if bool_bilinear == True:
                x_ph_list.append(tf.placeholder(tf.float32, [None, steps_x_list[i], dim_x_list[i]], name = 'x' + str(i)))
            else:
                x_ph_list.append(tf.placeholder(tf.float32, [None, steps_x_list[i]*dim_x_list[i]], name = 'x' + str(i)))
                
                
        # --- prediction of individual models
        
        mean_list = []
        var_list = []
        logit_list = []
        
        regu_mean = 0.0
        regu_var = 0.0
        regu_gate = 0.0
        
        for i in range(num_compt):
            
            if bool_bilinear == True:
                
                tmp_mean, tmp_regu_mean = bilinear(x_ph_list[i], 
                                                   [steps_x_list[i], dim_x_list[i]], 
                                                   'mean' + str(i) , 
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
                                                  'var' + str(i), 
                                                  True)
                
            # [C B 1]
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
            
        # [B C]
        self.logit = tf.stack(logit_list, 1)
        self.gates = tf.nn.softmax(self.logit, axis = -1)
        
        
        # --- negative log likelihood 
        
        # nllk: negative log likelihood
            
        #[B C]
        tmpllk_indi_hetero = tf.exp(-0.5*tf.square(self.y - mean_stack)/(var_stack + 1e-5))/(2.0*np.pi*(var_stack + 1e-5))**0.5
            
        llk_hetero = tf.multiply(tmpllk_indi, self.gates) 
        self.nllk_hetero = tf.reduce_sum(-1.0*tf.log(tf.reduce_sum(llk, axis = -1) + 1e-5))
        
        #[B C]
        tmpllk_indi_const = tf.exp(-0.5*tf.square(self.y - mean_stack))/(2.0*np.pi)**0.5
            
        llk_const = tf.multiply(tmpllk_indi, self.gates) 
        self.nllk_const = tf.reduce_sum(-1.0*tf.log(tf.reduce_sum(llk, axis = -1) + 1e-5))
        
        
        # --- regularization
        
        # temporal coherence, diversity 
        # gate smoothness 
        # gate diversity
        
        # mean diversity
        
        # mean non-negative  
        regu_mean_pos = tf.reduce_sum( tf.maximum(0.0, -1.0*mean_v) + tf.maximum(0.0, -1.0*mean_x) )
        
        self.regularization = 0.0
        
        if loss_type == 'sq' and distr_type == 'gaussian':
            self.regularization = l2 * (regu_mean)
                
        elif loss_type == 'lk' and distr_type == 'gaussian':
            self.regularization = l2 * (regu_mean + regu_var)
                
        else:
            print '[ERROR] loss type'
        
        # activation and hinge regularization 
        if para_regu_positive_mean == True:
            self.regularization += l2*regu_mean_pos
            
        if para_regu_gate == True:
            self.regularization += 0.1*l2*regu_gate
        
        # --- prediction 
        
        if distr_type == 'gaussian':
            
            # component mean
            self.py_mean_indi = mean_stack
            
            # component variance
            self.py_var_indi = mean_var
            
            # mixture mean
            self.py = tf.reduce_sum(tf.multiply(mean_stack, self.gates), 1)
            
            # mixture variance
            sq_mean_stack =  var_stack + tf.square(mean_stack)
            mix_sq_mean = tf.reduce_sum(tf.multiply(sq_mean_stack, self.gates), 1)
            
            self.py_var = mix_sq_mean - tf.square(self.py)
        
        else:
            print '[ERROR] distribution type'
        

#   initialize loss and optimization operations for training
    def train_ini(self):
        
        # loss
        if self.loss_type == 'mse':
            
            self.mse = tf.reduce_mean(tf.square(self.y - self.py))
            
            self.loss = self.mse + self.regularization
            
            self.nllk = 
            
            self.py = 
            self.py_std = 
            
        elif self.loss_type == 'lk':
            
            self.loss = self.nllk + self.regularization
            
            self.nllk = 
            
            self.py = 
            self.py_std = 
        
        else:
            print '[ERROR] loss type'
        
        self.train = tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE)
        self.optimizer =  self.train.minimize(self.loss)
        
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        
    #   training on batch of data
    def train_batch(self, auto_train, x_train, y_train, keep_prob ):
        
        # !
        _, c = self.sess.run([self.optimizer, self.loss],\
                             feed_dict={self.auto: auto_train, \
                                        self.x:x_train, self.y:y_train, self.keep_prob:keep_prob })
        return c
    
    
    def inference_ini(self):
        
        # --- error metric
        
        # RMSE
        self.rmse = tf.sqrt(tf.losses.mean_squared_error(self.y, self.y_hat))
        # MAE
        self.mae  = tf.reduce_mean(tf.abs(self.y - self.y_hat))
        
        # MAPE
        mask = tf.greater(tf.abs(self.y), 0.00001)
        
        y_mask = tf.boolean_mask(self.y, mask)
        y_hat_mask = tf.boolean_mask(self.y_hat, mask)
        
        self.mape = tf.reduce_mean(tf.abs((y_mask - y_hat_mask)/(y_mask+1e-10)))
    
    
    #   infer givn testing data
    def inference(self, auto_test, x_test, y_test, keep_prob):
        
        return self.sess.run([self.rmse, self.mae, self.mape, self.nllk], 
                             feed_dict = {self.auto:auto_test, \
                                          self.x:x_test,  self.y:y_test, self.keep_prob:keep_prob })
    
    # collect the optimized variable values
    def collect_coeff_values(self, vari_keyword):
        return [ tf_var.name for tf_var in tf.trainable_variables() if (vari_keyword in tf_var.name) ],\
    [ tf_var.eval() for tf_var in tf.trainable_variables() if (vari_keyword in tf_var.name) ]
    
    
    # restore the model from the files
    def pre_train_restore_model(self, 
                                path_meta, 
                                path_data):
        
        saver = tf.train.import_meta_graph(path_meta, clear_devices=True)
        saver.restore(self.sess, path_data)
        
        return
    
    
        
