#!/usr/bin/python

import tensorflow as tf

# local 
from utils_libs import *

# ----- utilities functions -----

def multi_src_predictor_linear(x, 
                               n_src, 
                               steps, 
                               dim, 
                               bool_bias, 
                               bool_scope_reuse, 
                               str_scope,
                               para_share_logit):
    '''
    Argu.:
      x: [S B T D]
      bool_bias: [bool_bias_mean, bool_bias_var, bool_bias_gate]
      bool_scope_reuse: [mean, var, gate]
    
    '''
    #[S B]
    tmp_mean, regu_mean = multi_src_bilinear(x,
                                             [steps, dim],
                                             str_scope + "mean",
                                             bool_bias = bool_bias[0],
                                             bool_scope_reuse = bool_scope_reuse[0], 
                                             num_src = n_src)
        
    tmp_var, regu_var = multi_src_bilinear(x,
                                           [steps, dim],
                                           str_scope + "var",
                                           bool_bias = bool_bias[1],
                                           bool_scope_reuse = bool_scope_reuse[1],
                                           num_src = n_src)
    
    tmp_logit, regu_gate = multi_src_logit_bilinear(x,
                                                    [steps, dim],
                                                    str_scope + 'gate',
                                                    bool_bias = bool_bias[2],
                                                    bool_scope_reuse = bool_scope_reuse[2],
                                                    num_src = n_src,
                                                    para_share_type = para_share_logit)
    
    return tmp_mean, regu_mean, tmp_var, regu_var, tmp_logit, regu_gate
    
    
def multi_src_logit_bilinear(x, 
                            shape_x, 
                            scope, 
                            bool_bias,
                            bool_scope_reuse, 
                            num_src,
                            para_share_type):
    
    # x: [S, B, T, D]
    # shape_x: [T, D]
    with tf.variable_scope(scope, 
                           reuse = bool_scope_reuse):
        
        if para_share_type == "no_share":
            
            # [S  1  T  1]
            w_l = tf.get_variable('w_left', 
                                  [num_src, 1, shape_x[0], 1],
                                  initializer = tf.contrib.layers.xavier_initializer())
            # [S 1 D]
            w_r = tf.get_variable('w_right', 
                                  [num_src, 1, shape_x[1]],
                                  initializer = tf.contrib.layers.xavier_initializer())
            # [S 1]
            b = tf.get_variable("b", 
                                shape = [num_src, 1], 
                                initializer = tf.zeros_initializer())
            
            # [S B T D] * [S 1 T 1] -> [S B D]
            tmp_h = tf.reduce_sum(x * w_l, 2)
            
            # [S B D]*[S 1 D] - > [S B]
            h = tf.reduce_sum(tmp_h * w_r, 2)
        
        elif para_share_type == "share":
            
            # [1  1  T  1]
            w_l = tf.get_variable('w_left', 
                                  [1, 1, shape_x[0], 1],
                                  initializer = tf.contrib.layers.xavier_initializer())
            # [1 1 D]
            w_r = tf.get_variable('w_right', 
                                  [1, 1, shape_x[1]],
                                  initializer = tf.contrib.layers.xavier_initializer())
            # [1 1]
            b = tf.get_variable("b", 
                                shape = [1, 1], 
                                initializer = tf.zeros_initializer())
            
            # [S B T D] * [1 1 T 1] -> [S B D]
            tmp_h = tf.reduce_sum(x * w_l, 2)
            
            # [S B D]*[1 1 D] - > [S B]
            h = tf.reduce_sum(tmp_h * w_r, 2)
            
        elif para_share_type == "mix":
            
            # x: [S, B, T, D] -> [B, T, D, S]
            tmpx = tf.transpose(x, [1, 2, 3, 0])
            # [B T D*S]
            tmpx_mix = tf.reshape(x, [-1, shape_x[0], shape_x[1]*num_src])
            
            
            # [1 T 1]
            w_l = tf.get_variable('w_left', 
                                  [1, shape_x[0], 1],
                                  initializer = tf.contrib.layers.xavier_initializer())
            # [S*D S]
            w_r = tf.get_variable('w_right', 
                                  [shape_x[1]*num_src, num_src],
                                  initializer = tf.contrib.layers.xavier_initializer())
            # [S 1]
            b = tf.get_variable("b", 
                                shape = [num_src, 1], 
                                initializer = tf.zeros_initializer())
            
            # [B T D*S] * [1 T 1] -> [B S*D]
            tmp_h = tf.reduce_sum(tmpx_mix * w_l, 1)
            
            # [B S*D]*[S*D S] - > [S B]
            h = tf.transpose(tf.matmul(tmp_h, w_r), [1,0])
            
        if bool_bias == True:
            
            # [S B]
            h = h + b
            
           # [S B] 
    return h, tf.reduce_sum(tf.square(w_l)) + tf.reduce_sum(tf.square(w_r))

def multi_src_linear(x, 
                     dim_x, 
                     scope, 
                     bool_bias,
                     bool_scope_reuse, 
                     num_src):
    # x: [S, B, T*D]
    # dim_x: T*D
    with tf.variable_scope(scope, 
                           reuse = bool_scope_reuse):
        
        # [S 1 T*D]
        w = tf.get_variable('w', 
                            shape = [num_src, 1, dim_x],
                            initializer = tf.contrib.layers.xavier_initializer())
        
        b = tf.get_variable("b", 
                            shape = [num_src, 1], 
                            initializer = tf.zeros_initializer())
        
        if bool_bias == True:
            
            # [S, B, T*D] * [S 1 T*D] -> [S B] + [S 1]
            h = tf.reduce_sum(x * w, -1) + b
            
        else:
            h = tf.reduce_sum(x * w, -1)
        
           # [S B]          l2: regularization
    return h, tf.reduce_sum(tf.square(w))

def multi_src_bilinear(x, 
                       shape_x, 
                       scope,
                       bool_bias,
                       bool_scope_reuse,
                       num_src):
    # x: [S, B, T, D]
    # shape_x: [T, D]
    with tf.variable_scope(scope, 
                           reuse = bool_scope_reuse):
        # [S  1  T  1]
        w_l = tf.get_variable('w_left', 
                              [num_src, 1, shape_x[0], 1],
                              initializer = tf.contrib.layers.xavier_initializer())
        # [S 1 D]
        w_r = tf.get_variable('w_right', 
                              [num_src, 1, shape_x[1]],
                              initializer = tf.contrib.layers.xavier_initializer())
        # [S 1]
        b = tf.get_variable("b", 
                            shape = [num_src, 1], 
                            initializer = tf.zeros_initializer())
        
        # [S B T D] * [S 1 T 1] -> [S B D]
        tmp_h = tf.reduce_sum(x * w_l, 2)
        
        if bool_bias == True:
            
            # [S B D]*[S 1 D] - > [S B]
            h = tf.reduce_sum(tmp_h * w_r, 2) + b
            
        else:
            h = tf.reduce_sum(tmp_h * w_r, 2)
            
         # [S B] 
    return h, tf.reduce_sum(tf.square(w_l)) + tf.reduce_sum(tf.square(w_r))

def linear(x, 
           dim_x, 
           scope, 
           bool_bias,
           bool_scope_reuse):
    
    # x: [B D]
    # dim_x: D
    
    with tf.variable_scope(scope, 
                           reuse = bool_scope_reuse):
        
        w = tf.get_variable('w', 
                            [dim_x, 1],
                            initializer = tf.contrib.layers.xavier_initializer())
        
        b = tf.get_variable("b", 
                            shape = [1,], 
                            initializer = tf.zeros_initializer())
        
        if bool_bias == True:
            h = tf.matmul(x, w) + b
        else:
            h = tf.matmul(x, w)
           
           # [B]          l2: regularization
    return tf.squeeze(h), tf.reduce_sum(tf.square(w))

def bilinear(x, 
             shape_x, 
             scope,
             bool_bias,
             bool_scope_reuse):
    
    # shape of x: [b, l, r]
    # shape_x: [l, r]
    with tf.variable_scope(scope, 
                           reuse = bool_scope_reuse):
        
        w_l = tf.get_variable('w_left', 
                              [shape_x[0], 1],
                              initializer = tf.contrib.layers.xavier_initializer())
        
        w_r = tf.get_variable('w_right', 
                              [shape_x[1], 1],
                              initializer = tf.contrib.layers.xavier_initializer())
        
        b = tf.get_variable("b", 
                            shape = [1,], 
                            initializer = tf.zeros_initializer())
        
        tmph = tf.tensordot(x, w_r, 1)
        tmph = tf.squeeze(tmph, [2])
        
        if bool_bias == True:
            h = tf.matmul(tmph, w_l) + b
        else:
            h = tf.matmul(tmph, w_l)
            
           # [B] 
    return tf.squeeze(h), tf.reduce_sum(tf.square(w_l)) + tf.reduce_sum(tf.square(w_r)) 
