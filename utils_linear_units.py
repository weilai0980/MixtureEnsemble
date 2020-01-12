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
                               para_share_logit,
                               bool_common_factor,
                               common_factor_dim):
    '''
    Argu.:
      x: [S [B T D]] when bool_common_factor = False, or
         [S+1 [B T D]] when bool_common_factor = True
      bool_bias: [bool_bias_mean, bool_bias_var, bool_bias_gate]
      bool_scope_reuse: [mean, var, gate]
    '''
    
    if bool_common_factor == True:
        # [B T sum(D)]
        x_common = x[-1]
        # [S B T D]
        x_src = tf.stack(x[:-1], 0)
    else:
        # [S [B T D]] -> [S B T D]
        x_src = tf.stack(x, 0)
    
    #[S B]
    tmp_mean, regu_mean = multi_src_bilinear(x_src,
                                             [steps, dim],
                                             str_scope + "mean",
                                             bool_bias = bool_bias[0],
                                             bool_scope_reuse = bool_scope_reuse[0], 
                                             num_src = n_src)
    tmp_var, regu_var = multi_src_bilinear(x_src,
                                           [steps, dim],
                                           str_scope + "var",
                                           bool_bias = bool_bias[1],
                                           bool_scope_reuse = bool_scope_reuse[1],
                                           num_src = n_src)
    tmp_logit, regu_gate = multi_src_logit_bilinear(x_src,
                                                    [steps, dim],
                                                    str_scope + 'gate',
                                                    bool_bias = bool_bias[2],
                                                    bool_scope_reuse = bool_scope_reuse[2],
                                                    num_src = n_src,
                                                    para_share_type = para_share_logit)
    if bool_common_factor == True:
        
        factorCell = tempFactorCell(num_units = common_factor_dim, 
                                    initializer = tf.contrib.layers.xavier_initializer())
        # [B F] F:factor dimension
        factor, state = tf.nn.dynamic_rnn(cell = factorCell, 
                                          inputs = x_common, 
                                          dtype = tf.float32)
        # [B 1]
        facor_mean, regu_factor_mean = linear(x = factor, 
                                              dim_x = common_factor_dim, 
                                              scope = "factor_mean", 
                                              bool_bias = True,
                                              bool_scope_reuse = False)
        facor_var, regu_factor_var = linear(x = factor, 
                                            dim_x = common_factor_dim, 
                                            scope = "factor_var", 
                                            bool_bias = True,
                                            bool_scope_reuse = False)
        facor_logit, regu_factor_logit = linear(x = factor, 
                                                dim_x = common_factor_dim, 
                                                scope = "factor_logit", 
                                                bool_bias = True,
                                                bool_scope_reuse = False)
        # [S+1 B]
        tmp_mean = tf.concat([tmp_mean, tf.transpose(facor_mean, [1, 0])], 0)
        tmp_var = tf.concat([tmp_var, tf.transpose(facor_var, [1, 0])], 0)
        tmp_logit = tf.concat([tmp_logit, tf.transpose(facor_logit, [1, 0])], 0)
      
        regu_mean += facor_mean
        regu_var += facor_var
        regu_logit += facor_logit
      
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
    '''
     x: [B D]
     dim_x: D
    '''
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
    '''
     shape of x: [b, l, r]
     shape_x: [l, r]
    '''
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
    
# ------ linear factor process

import sys
import collections
import hashlib
import numbers

from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops.rnn_cell_impl import * 
from tensorflow.python.ops import nn_ops

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"
    
def _linear_transition(args,
                       output_size,
                       bias,
                       kernel_initializer, 
                       bias_initializer = None):
    """
    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_initializer: starting value to initialize the bias (default is all zeros).
      kernel_initializer: starting value to initialize the weight.
    Returns:
      A 2D Tensor with shape [batch output_size]
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("args must be specified")
        
    if not nest.is_sequence(args):
        args = [args]

    # calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)
        if shape[1].value is None:
            raise ValueError("linear expects shape[1] to be provided for shape %s, ""but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[1].value
    
    dtype = [a.dtype for a in args][0]
    
    # --- begin linear update ---  
    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
        # define relevant dimensions
        input_dim  = args[0].get_shape()[1].value
        #hidden_dim = args[1].get_shape()[1].value
        hidden_dim = output_size
        
        # --- hidden update ---
        #[D H]
        weights_IH = vs.get_variable('input_hidden', 
                                     [input_dim, hidden_dim],
                                     dtype = dtype,
                                     initializer = kernel_initializer)
        #[H H]
        weights_HH = vs.get_variable('hidden_hidden', 
                                     [hidden_dim, hidden_dim],
                                     dtype = dtype,
                                     initializer = kernel_initializer)
        # [B D]
        tmp_input = args[0] 
        # [B H]
        tmp_h = args[1]
        # [B H]
        new_h = math_ops.matmul(tmp_input, weights_IH) + math_ops.matmul(tmp_h, weights_HH)
      
        # --- bias ---
        if not bias:
            return new_h
          
        with vs.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            if bias_initializer is None:
                bias_initializer = init_ops.constant_initializer(0.0,
                                                                 dtype = dtype)
            biases = vs.get_variable(_BIAS_VARIABLE_NAME, 
                                     [hidden_dim],
                                     dtype = dtype,
                                     initializer = bias_initializer)
            
        return nn_ops.bias_add(new_h, biases)

class tempFactorCell(RNNCell):
    
    def __init__(self, 
                 num_units, 
                 initializer, 
                 reuse = None):
        """
        Args:
          num_units: int, The number of units in the tempFactor cell.
        """
        super(tempFactorCell, self).__init__(_reuse = reuse)
        self._num_units = num_units
        self._linear = None
        self._kernel_ini = initializer
        
    @property
    def state_size(self):
        return self._num_units
    @property
    def output_size(self):
        return self._num_units
    
    def call(self, 
             inputs, 
             state):
        """
        Args:
          inputs: `2-D` tensor with shape `[batch_size x input_size]`.
        """
        h = state
        
        if self._linear is None:
            self._new_h = _linear_transition([inputs, h], 
                                             self._num_units, 
                                             True, 
                                             kernel_initializer = self._kernel_ini)
        else:
            print('[ERROR]  factor cell type')
            
        _new_state = _new_h
        return _new_h, _new_state