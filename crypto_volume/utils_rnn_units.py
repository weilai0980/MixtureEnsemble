import sys
import collections
import hashlib
import numbers

from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops.rnn_cell_impl import *
import tensorflow as tf

# local 
from utils_libs import *

# stabilize the network by fixing random seeds
np.random.seed(1)
tf.set_random_seed(1)

def multi_src_predictor_rnn(x, 
                            n_src, 
                            bool_bias, 
                            bool_scope_reuse, 
                            str_scope,
                            rnn_size_layers,
                            rnn_cell_type,
                            dropout_keep,
                            dense_num,
                            max_norm_cons):
    
    '''
    Argu.:
      x: [S [B T D]] when bool_common_factor = False, or
         [S+1 [B T D]] when bool_common_factor = True
      bool_bias: [bool_bias_mean, bool_bias_var, bool_bias_gate]
      bool_scope_reuse: [mean, var, gate]
    '''
    np.random.seed(1)
    tf.set_random_seed(1)
    
    x_list = x
    
    h_list = []
    for i in range(n_src):
        h, _  = plain_rnn(x = x_list[i],
                          dim_layers = rnn_size_layers,
                          scope = str_scope + "_rnn_" + str(i),
                          dropout_keep_prob = dropout_keep,
                          cell_type = rnn_cell_type)
        # obtain the last hidden state
        # [B T d] -> [T B d]
        tmp_h = tf.transpose(h, [1,0,2])
        # [S, B d]
        h_list.append(tmp_h[-1])
    
    # [S B d]
    h_src = tf.stack(h_list, 0)
    
    # --- mean
    # [S B d]
    h_mean, reg_mean_h, dim_src_mean = multi_mv_dense(num_layers = dense_num,
                                                      keep_prob = dropout_keep,
                                                      h_vari = h_src,
                                                      dim_vari = rnn_size_layers[-1],
                                                      scope = str_scope + "_mean_h",
                                                      num_vari = n_src,
                                                      activation_type = "relu",
                                                      max_norm_regul = max_norm_cons,
                                                      regul_type = "l2")
    # output layer
    # [S B 1] 
    # no dropout on output layer
    tmp_mean, regu_mean_pred = mv_dense(h_vari = h_mean, 
                                        dim_vari = dim_src_mean, 
                                        scope = str_scope + "_mean_pred", 
                                        num_vari = n_src, 
                                        dim_to = 1, 
                                        activation_type = "", 
                                        max_norm_regul = max_norm_cons, 
                                        regul_type = "l2")
    
    regu_mean = reg_mean_h + regu_mean_pred
    
    # --- variance
    # !! watch out for the explosion of variance due to square aferwards !!
    # activation_type = "tanh"
    # [S B d]
    h_var, reg_var_h, dim_src_var = multi_mv_dense(num_layers = dense_num,
                                                   keep_prob = dropout_keep,
                                                   h_vari = h_src,
                                                   dim_vari = rnn_size_layers[-1],
                                                   scope = str_scope + "_var_h",
                                                   num_vari = n_src,
                                                   activation_type = "tanh",
                                                   max_norm_regul = max_norm_cons,
                                                   regul_type = "l2")
    # output layer
    # [S B 1]
    # no dropout on output layer
    tmp_var, regu_var_pred = mv_dense(h_vari = h_var, 
                                      dim_vari = dim_src_var, 
                                      scope = str_scope + "_var_pred", 
                                      num_vari = n_src, 
                                      dim_to = 1, 
                                      activation_type = "", 
                                      max_norm_regul = max_norm_cons, 
                                      regul_type = "l2")
    
    regu_var = reg_var_h + regu_var_pred
    
    # --- gate
    '''
    # [S B d]
    h_logit, reg_logit_h, dim_src_logit = multi_mv_dense(num_layers = dense_num,
                                                         keep_prob = dropout_keep,
                                                         h_vari = h_src,
                                                         dim_vari = rnn_size_layers[-1],
                                                         scope = str_scope + "_logit_h",
                                                         num_vari = n_src,
                                                         bool_activation = True,
                                                         max_norm_regul = max_norm_cons,
                                                         regul_type = "l2")
    # [S B d] -> [S B 1]
    # ? tanh activation
    tmp_logit, regu_logit_pred = mv_dense(h_vari = h_logit, 
                                          dim_vari = dim_src_logit, 
                                          scope = str_scope + "_logit", 
                                          num_vari = n_src, 
                                          dim_to = 1, 
                                          bool_activation = False, 
                                          max_norm_regul = max_norm_cons, 
                                          regul_type = "l2")
    
    regu_logit = reg_logit_h + regu_logit_pred
    '''
    # output layer
    # [S B d] -> [S B 1]
    # ? tanh activation
    tmp_logit, regu_logit = mv_dense(h_vari = h_src, 
                                     dim_vari = rnn_size_layers[-1], 
                                     scope = str_scope + "_logit", 
                                     num_vari = n_src, 
                                     dim_to = 1, 
                                     activation_type = "", 
                                     max_norm_regul = max_norm_cons, 
                                     regul_type = "l2")
    
    return tf.squeeze(tmp_mean), regu_mean, tf.squeeze(tmp_var), regu_var, tf.squeeze(tmp_logit), regu_logit

# ---- Multi variable dense layers ---- 

def multi_mv_dense(num_layers,
                   keep_prob,
                   h_vari,
                   dim_vari,
                   scope,
                   num_vari,
                   activation_type,
                   max_norm_regul,
                   regul_type):
    '''
    Argu.:
      h_vari: [V B D] -> [V B d]
      dim_vari: int
      num_vari: int
      dim_to: int
    '''
    in_dim_vari = dim_vari
    out_dim_vari = int(dim_vari/2)
    h_mv_input = h_vari
    
    reg_mv_dense = 0.0
    
    for i in range(num_layers):
        
        if out_dim_vari <= 1:
            break
        
        with tf.variable_scope(scope + str(i)):
            
            # no dropout on the input
            if i != 0:
                # h_mv [V B d]
                h_mv_input = tf.nn.dropout(h_mv_input, 
                                           keep_prob, 
                                           seed = 1)
            # ? max norm constrains
            h_mv_input, tmp_regu_dense = mv_dense(h_vari = h_mv_input, 
                                                  dim_vari = in_dim_vari,
                                                  scope = scope + str(i),
                                                  num_vari = num_vari,
                                                  dim_to = out_dim_vari,
                                                  activation_type = activation_type, 
                                                  max_norm_regul = max_norm_regul, 
                                                  regul_type = regul_type)
            reg_mv_dense += tmp_regu_dense
            
            in_dim_vari = out_dim_vari
            out_dim_vari = int(out_dim_vari/2)
            
    return h_mv_input, reg_mv_dense, in_dim_vari
            
# with max-norm regularization 
def mv_dense(h_vari, 
             dim_vari, 
             scope, 
             num_vari, 
             dim_to, 
             activation_type, 
             max_norm_regul, 
             regul_type):
    '''
    Argu.:
      h_vari: [V B D] -> [V B d]
      dim_vari: int
      num_vari: int
      dim_to: int
    '''
    with tf.variable_scope(scope):
        
        # [V 1 D d]
        w = tf.get_variable('w',  
                            [num_vari, 1, dim_vari, dim_to], 
                            initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        # [V 1 1 d]
        b = tf.get_variable("b", 
                            shape = [num_vari, 1, 1, dim_to], 
                            initializer = tf.zeros_initializer())
        # [V B D 1]
        h_expand = tf.expand_dims(h_vari, -1)
        
        if max_norm_regul > 0:
            clipped = tf.clip_by_norm(w, 
                                      clip_norm = max_norm_regul, 
                                      axes = 2)
            clip_w = tf.assign(w, clipped)
            
            tmp_h =  tf.reduce_sum(h_expand * clip_w + b, 2)
            
        else:
            tmp_h =  tf.reduce_sum(h_expand * w + b, 2)
            
        # [V B D 1] * [V 1 D d] -> [V B d]
        if activation_type == "relu":
            h = tf.nn.relu(tmp_h)
        elif activation_type == "tanh":
            h = tf.nn.tanh(tmp_h)
        elif activation_type == "sigmoid":
            h = tf.nn.sigmoid(tmp_h)
        elif activation_type == "leaky_relu":
            h = tf.nn.leaky_relu(tmp_h, 
                                 alpha = 0.2)
        else:
            h = tmp_h
        
        # weight regularization    
        if regul_type == 'l2':
            return h, tf.nn.l2_loss(w) 
        elif regul_type == 'l1':
            return h, tf.reduce_sum(tf.abs(w)) 
        else:
            return '[ERROR] regularization type'

# ----- RNN layers -----  
    
def res_lstm(x, 
             hidden_dim, 
             n_layers, 
             scope, 
             dropout_keep_prob):
    '''
    Argu.:
      x: [B T D] 
      hidden_dim: int
      n_layers: int
      dropout_keep_prob: float 
    '''
    with tf.variable_scope(scope + str(0)):
            #Deep lstm: residual or highway connections 
            lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_dim, \
                                                initializer = tf.contrib.keras.initializers.glorot_normal(seed = 1))
            hiddens, state = tf.nn.dynamic_rnn(cell = lstm_cell, 
                                               inputs = x, 
                                               dtype = tf.float32)
    for i in range(1, n_layers):
        
        with tf.variable_scope(scope + str(i)):
            
            tmp_h = hiddens
            
            lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_dim, \
                                                initializer = tf.contrib.keras.initializers.glorot_normal(seed = 1))
            hiddens, state = tf.nn.dynamic_rnn(cell = lstm_cell, 
                                               inputs = hiddens, 
                                               dtype = tf.float32)
            hiddens = hiddens + tmp_h 
             
    return hiddens, state

def cudnn_rnn(x, 
              dim_layers, 
              scope, 
              dropout_keep_prob, 
              cell_type):
    
    tmp_cell = tf.contrib.cudnn_rnn.CudnnGRU(num_layers = len(dim_layers),
                                             num_units = dim_layers[0],
                                             dropout = 1.0 - dropout_keep_prob,
                                             kernel_initializer = tf.contrib.keras.initializers.glorot_normal(seed = 1),
                                             name = scope)
    
    hiddens, state = tf.nn.dynamic_rnn(cell = tmp_cell, 
                                       inputs = x, 
                                       dtype = tf.float32)
    return hiddens, state
    
def plain_rnn(x, 
              dim_layers, 
              scope, 
              dropout_keep_prob, 
              cell_type):
    '''
    Argu.:
      x: [B T D] 
      dim_layers: [int]
      dropout_keep_prob: float 
      cell_type: lstm, gru
    '''
    # stabilize the network by fixing random seeds
    np.random.seed(1)
    tf.set_random_seed(1)
    
    with tf.variable_scope(scope):
        
        if cell_type == 'lstm':
            tmp_cell = tf.nn.rnn_cell.LSTMCell(dim_layers[0], 
                                               initializer = tf.contrib.keras.initializers.glorot_normal(seed = 1))
        elif cell_type == 'gru':
            tmp_cell = tf.nn.rnn_cell.GRUCell(dim_layers[0],
                                              kernel_initializer = tf.contrib.keras.initializers.glorot_normal(seed = 1))
        # !! only dropout on hidden states !!
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(tmp_cell,
                                                 state_keep_prob = dropout_keep_prob, 
                                                 seed = 1)
            
        hiddens, state = tf.nn.dynamic_rnn(cell = rnn_cell, 
                                           inputs = x, 
                                           dtype = tf.float32)
    for i in range(1, len(dim_layers)):
        
        with tf.variable_scope(scope + str(i)):
            
            if cell_type == 'lstm':
                tmp_cell = tf.nn.rnn_cell.LSTMCell(dim_layers[i], 
                                                   initializer= tf.contrib.keras.initializers.glorot_normal(seed = 1))
            elif cell_type == 'gru':
                tmp_cell = tf.nn.rnn_cell.GRUCell(dim_layers[i],
                                                  kernel_initializer= tf.contrib.keras.initializers.glorot_normal(seed = 1))
            
            # dropout on both input and hidden states
            rnn_cell = tf.nn.rnn_cell.DropoutWrapper(tmp_cell,
                                                     input_keep_prob = dropout_keep_prob,
                                                     state_keep_prob = dropout_keep_prob, 
                                                     seed = 1)
            
            hiddens, state = tf.nn.dynamic_rnn(cell = rnn_cell, 
                                               inputs = hiddens, 
                                               dtype = tf.float32)
    return hiddens, state
