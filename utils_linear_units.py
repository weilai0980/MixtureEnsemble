#!/usr/bin/python

# ----- utilities functions -----

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
        
        w = tf.get_variable('w', 
                            shape = [num_src, 1, dim_x],
                            initializer = tf.contrib.layers.xavier_initializer())
        
        b = tf.get_variable("b", 
                            shape = [num_src, 1], 
                            initializer = tf.zeros_initializer())
        
        if bool_bias == True:
            
            # [S, B, T*D] * [S 1 T*D] -> [S B]
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

