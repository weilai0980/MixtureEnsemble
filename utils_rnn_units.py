import sys

import collections
import hashlib
import numbers

from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops.rnn_cell_impl import *
import tensorflow as tf

# local 
from utils_libs import *


def multi_src_predictor_rnn(x, 
                            x_src_seperated,
                            n_src, 
                            bool_bias, 
                            bool_scope_reuse, 
                            str_scope,
                            rnn_size_layers,
                            rnn_cell_type,
                            dropout_keep,
                            dense_num,
                            max_norm_cons):
    
    
    if x_src_seperated == True:
        x_list = x
        
    else:
        # shape: [S, B T D]
        tmp_x_list = tf.split(x, 
                              num_or_size_splits = n_src, 
                              axis = 0)
        x_list = [tf.squeeze(tmp_x, 0) for tmp_x in tmp_x_list]
        
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
                                                      bool_activation = True,
                                                      max_norm_regul = max_norm_cons,
                                                      regul_type = "l2")
    # [S B 1] 
    # no dropout on output layer
    tmp_mean, regu_mean_pred = mv_dense(h_vari = h_mean, 
                                        dim_vari = dim_src_mean, 
                                        scope = str_scope + "_mean_pred", 
                                        num_vari = n_src, 
                                        dim_to = 1, 
                                        bool_activation = False, 
                                        max_norm_regul = max_norm_cons, 
                                        regul_type = "l2")
    
    regu_mean = reg_mean_h + regu_mean_pred
    
    # --- variance
    
    # [S B d]
    h_var, reg_var_h, dim_src_var = multi_mv_dense(num_layers = dense_num,
                                                   keep_prob = dropout_keep,
                                                   h_vari = h_src,
                                                   dim_vari = rnn_size_layers[-1],
                                                   scope = str_scope + "_var_h",
                                                   num_vari = n_src,
                                                   bool_activation = True,
                                                   max_norm_regul = max_norm_cons,
                                                   regul_type = "l2")
    # [S B 1]
    # no dropout on output layer
    tmp_var, regu_var_pred = mv_dense(h_vari = h_var, 
                                      dim_vari = dim_src_var, 
                                      scope = str_scope + "_var_pred", 
                                      num_vari = n_src, 
                                      dim_to = 1, 
                                      bool_activation = False, 
                                      max_norm_regul = max_norm_cons, 
                                      regul_type = "l2")
    
    regu_var = reg_var_h + regu_var_pred
    
    # --- gate
    
    # [S B d] -> [S B 1]         
    tmp_logit, regu_logit = mv_dense(h_vari = h_src, 
                                     dim_vari = rnn_size_layers[-1], 
                                     scope = str_scope + "_logit", 
                                     num_vari = n_src, 
                                     dim_to = 1, 
                                     bool_activation = True, 
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
                   bool_activation,
                   max_norm_regul,
                   regul_type):
    
    '''
    h_vari: [V B D] -> [V B d] 

    Argu.:
      h_vari: [V B D]
      dim_vari: int
      num_vari: int
      dim_to: int
    '''
    
    in_dim_vari = dim_vari
    out_dim_vari = int(dim_vari/2)
    h_mv_input = h_vari
    
    reg_mv_dense = 0.0
    
    for i in range(num_layers):
        
        with tf.variable_scope(scope + str(i)):
            
            # ? dropout
            h_mv_input = tf.nn.dropout(h_mv_input, 
                                       keep_prob)
            # h_mv [V B d]
            # ? max norm constrains
            h_mv_input, tmp_regu_dense = mv_dense(h_vari = h_mv_input, 
                                                  dim_vari = in_dim_vari,
                                                  scope = scope + str(i),
                                                  num_vari = num_vari,
                                                  dim_to = out_dim_vari,
                                                  bool_activation = False, 
                                                  max_norm_regul = max_norm_regul, 
                                                  regul_type = regul_type)
            
            reg_mv_dense += tmp_regu_dense
            
            in_dim_vari  = out_dim_vari
            out_dim_vari = int(out_dim_vari/2)
            
    return h_mv_input, reg_mv_dense, in_dim_vari
            
# with max-norm regularization 
def mv_dense(h_vari, 
             dim_vari, 
             scope, 
             num_vari, 
             dim_to, 
             bool_activation, 
             max_norm_regul, 
             regul_type):
    
    '''
    h_vari: [V B D] -> [V B d] 

    Argu.:
      h_vari: [V B D]
      dim_vari: int
      num_vari: int
      dim_to: int
    '''
    
    with tf.variable_scope(scope):
        
        # [V 1 D d]
        w = tf.get_variable('w', 
                            [num_vari, 1, dim_vari, dim_to], 
                            initializer=tf.contrib.layers.xavier_initializer())
        # [V 1 1 d]
        b = tf.Variable(tf.random_normal([num_vari, 1, 1, dim_to]))
        
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
        # ?
        if bool_activation == True:
            h = tf.nn.relu(tmp_h)
        else:
            h = tmp_h
        
        # weight regularization    
        if regul_type == 'l2':
            return h, tf.nn.l2_loss(w) 
        
        elif regul_type == 'l1':
            return h, tf.reduce_sum(tf.abs(w)) 
        
        else:
            return '[ERROR] regularization type'

# with max-norm regularization 
def mv_dense_share(h_vari, 
                   dim_vari, 
                   scope,
                   num_vari, 
                   dim_to, 
                   bool_no_activation, 
                   max_norm_regul, 
                   regul_type):
    
    # argu [V B D]
    
    with tf.variable_scope(scope):
        
        # [D d]
        w = tf.get_variable('w', 
                            [dim_vari, dim_to], 
                            initializer=tf.contrib.layers.xavier_initializer())
        # [ d]
        b = tf.Variable(tf.random_normal([dim_to]))
        
        
        if max_norm_regul > 0:
            
            clipped = tf.clip_by_norm(w, clip_norm = max_norm_regul, axes = 0)
            clip_w = tf.assign(w, clipped)
            
            # [V B d]
            tmp_h = tf.tensordot(h_vari, w, 1) + b
            #tmp_h = tf.reduce_sum(h_expand * clip_w + b, 2)
            
        else:
            tmp_h = tf.tensordot(h_vari, w, 1) + b
            #tmp_h =  tf.reduce_sum(h_expand * w + b, 2)
            
        # [V B D 1] * [V 1 D d] -> [V B d]
        # ?
        if bool_no_activation == True:
            h = tmp_h
        else:
            h = tf.nn.relu(tmp_h) 
            
        if regul_type == 'l2':
            return h, tf.nn.l2_loss(w) 
        
        elif regul_type == 'l1':
            return h, tf.reduce_sum( tf.abs(w) ) 
        
        else:
            return '[ERROR] regularization type'

# ----- RNN layers -----  
    
def res_lstm(x, 
             hidden_dim, 
             n_layers, 
             scope, 
             dropout_keep_prob):
    
    #dropout
    #x = tf.nn.dropout(x, dropout_keep_prob)
    
    with tf.variable_scope(scope):
            #Deep lstm: residual or highway connections 
            lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_dim, \
                                                initializer= tf.contrib.keras.initializers.glorot_normal())
            hiddens, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = x, dtype = tf.float32)
            
    for i in range(1, n_layers):
        
        with tf.variable_scope(scope+str(i)):
            
            tmp_h = hiddens
            
            lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_dim, \
                                                    initializer= tf.contrib.keras.initializers.glorot_normal())
            hiddens, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = hiddens, dtype = tf.float32)
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
                                             kernel_initializer = tf.contrib.keras.initializers.glorot_normal(),
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
    
    with tf.variable_scope(scope):
        
        if cell_type == 'lstm':
            
            tmp_cell = tf.nn.rnn_cell.LSTMCell(dim_layers[0], 
                                               initializer = tf.contrib.keras.initializers.glorot_normal())
        elif cell_type == 'gru':
            
            # tf.nn.rnn_cell.GRUCell, tf.contrib.cudnn_rnn.CudnnGRU
            tmp_cell = tf.nn.rnn_cell.GRUCell(dim_layers[0],
                                              kernel_initializer = tf.contrib.keras.initializers.glorot_normal())
        # dropout on hidden states
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(tmp_cell,
                                                 state_keep_prob = dropout_keep_prob)
            
        hiddens, state = tf.nn.dynamic_rnn(cell = rnn_cell, 
                                           inputs = x, 
                                           dtype = tf.float32)
        
    for i in range(1, len(dim_layers)):
        
        with tf.variable_scope(scope + str(i)):
            
            if cell_type == 'lstm':
                
                tmp_cell = tf.nn.rnn_cell.LSTMCell(dim_layers[i], 
                                                   initializer= tf.contrib.keras.initializers.glorot_normal())
            elif cell_type == 'gru':
                
                tmp_cell = tf.nn.rnn_cell.GRUCell(dim_layers[i],
                                                  kernel_initializer= tf.contrib.keras.initializers.glorot_normal())
            
            # dropout on both input and hidden states
            rnn_cell = tf.nn.rnn_cell.DropoutWrapper(tmp_cell,
                                                     input_keep_prob = dropout_keep_prob,
                                                     state_keep_prob = dropout_keep_prob)
            
            hiddens, state = tf.nn.dynamic_rnn(cell = rnn_cell, 
                                               inputs = hiddens, 
                                               dtype = tf.float32)
    return hiddens, state 

# ----- DENSE layers -----  

def res_dense(x, 
              x_dim, 
              hidden_dim, 
              n_layers, 
              scope, 
              dropout_keep_prob):
    
        #dropout
        x = tf.nn.dropout(x, dropout_keep_prob)
        
        with tf.variable_scope(scope):
            
            # initilization
            w = tf.get_variable('w', 
                                [x_dim, hidden_dim], 
                                dtype = tf.float32,
                                initializer = tf.contrib.layers.variance_scaling_initializer())
                                #initializer = tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.zeros([hidden_dim]))
            h = tf.nn.relu(tf.matmul(x, w) + b)
            
            regularization = tf.nn.l2_loss(w)
        #dropout
        #h = tf.nn.dropout(h, dropout_keep_prob)
        
        for i in range(1, n_layers):
            
            with tf.variable_scope(scope+str(i)):
                
                w = tf.get_variable('w', [hidden_dim, hidden_dim], 
                                    initializer = tf.contrib.layers.variance_scaling_initializer())
                                    #initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.zeros(hidden_dim))
                
                # residual connection
                tmp_h = h
                h = tf.nn.relu(tf.matmul(h, w) + b)
                h = tmp_h + h
                
                regularization += tf.nn.l2_loss(w)
        
        return h, regularization
    
def plain_dense(x, 
                x_dim, 
                dim_layers, 
                scope, 
                dropout_keep_prob, 
                max_norm_regul):
    
        #dropout
        x = tf.nn.dropout(x, dropout_keep_prob)
        
        with tf.variable_scope(scope):
            # initilization
            w = tf.get_variable('w', [x_dim, dim_layers[0]], dtype=tf.float32,\
                                    initializer = tf.contrib.layers.variance_scaling_initializer())
                                    #initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.zeros([dim_layers[0]]))
            
            # max norm constraints
            if max_norm_regul > 0:
                
                clipped = tf.clip_by_norm(w, 
                                          clip_norm = max_norm_regul, 
                                          axes = 1)
                clip_w = tf.assign(w, clipped)
                    
                h = tf.nn.relu( tf.matmul(x, clip_w) + b )
            else:
                h = tf.nn.relu( tf.matmul(x, w) + b )
                    
            #?
            regularization = tf.nn.l2_loss(w)
            #regularization = tf.reduce_sum(tf.abs(w))
                
        #dropout
        h = tf.nn.dropout(h, dropout_keep_prob)
        
        for i in range(1, len(dim_layers)):
            
            with tf.variable_scope(scope + str(i)):
                
                w = tf.get_variable('w', 
                                    [dim_layers[i-1], dim_layers[i]], 
                                    dtype = tf.float32,
                                    initializer = tf.contrib.layers.variance_scaling_initializer())
                                    #initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.zeros(dim_layers[i]))
                
                # max norm constraints 
                if max_norm_regul > 0:
                    
                    clipped = tf.clip_by_norm(w, 
                                              clip_norm = max_norm_regul, 
                                              axes = 1)
                    clip_w = tf.assign(w, clipped)
                    
                    h = tf.nn.relu(tf.matmul(h, clip_w) + b)
                    
                else:
                    h = tf.nn.relu(tf.matmul(h, w) + b)
                
                #?
                regularization += tf.nn.l2_loss(w)
                #regularization += tf.reduce_sum(tf.abs(w))
                
        return h, regularization

def plain_dense_leaky(x, 
                      x_dim, 
                      dim_layers, 
                      scope, 
                      dropout_keep_prob, 
                      alpha):
    
        #dropout
        x = tf.nn.dropout(x, dropout_keep_prob)
        
        with tf.variable_scope(scope):
                # initilization
                w = tf.get_variable('w', [x_dim, dim_layers[0]], dtype=tf.float32,\
                                    initializer = tf.contrib.layers.variance_scaling_initializer())
                                    #initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.zeros([dim_layers[0]]))
                
                # ?
                tmp_h = tf.matmul(x, w) + b 
                h = tf.maximum( alpha*tmp_h, tmp_h )

                #?
                regularization = tf.nn.l2_loss(w)
                #regularization = tf.reduce_sum(tf.abs(w))
                
        #dropout
        #h = tf.nn.dropout(h, dropout_keep_prob)
        
        for i in range(1, len(dim_layers)):
            
            with tf.variable_scope(scope+str(i)):
                w = tf.get_variable('w', [dim_layers[i-1], dim_layers[i]], dtype=tf.float32,\
                                    initializer = tf.contrib.layers.variance_scaling_initializer())
                                    #initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.zeros(dim_layers[i]))
                
                # ?
                tmp_h = tf.matmul(h, w) + b 
                h = tf.maximum( alpha*tmp_h, tmp_h )
                
                #?
                regularization += tf.nn.l2_loss(w)
                #regularization += tf.reduce_sum(tf.abs(w))
                
        return h, regularization
    
    
def multi_dense(x, 
                x_dim, 
                num_layers, 
                scope, 
                dropout_keep_prob, 
                max_norm_regul):
    
        in_dim = x_dim
        out_dim = int(in_dim/2)
        
        h = x
        regularization = 0.0
        
        for i in range(num_layers):
            
            with tf.variable_scope(scope+str(i)):
                
                #dropout
                h = tf.nn.dropout(h, dropout_keep_prob)
                
                w = tf.get_variable('w', [ in_dim, out_dim ], dtype=tf.float32,\
                                    initializer = tf.contrib.layers.variance_scaling_initializer())
                                    #initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.zeros( out_dim ))
                
                # max norm constraints 
                if max_norm_regul > 0:
                    
                    clipped = tf.clip_by_norm(w, clip_norm = max_norm_regul, axes = 1)
                    clip_w = tf.assign(w, clipped)
                    
                    h = tf.nn.relu( tf.matmul(h, clip_w) + b )
                    
                else:
                    h = tf.nn.relu( tf.matmul(h, w) + b )
                
                #?
                regularization += tf.nn.l2_loss(w)
                #regularization += tf.reduce_sum(tf.abs(w))
                
                in_dim = out_dim
                out_dim = int(out_dim/2)
                
        return h, regularization, in_dim    