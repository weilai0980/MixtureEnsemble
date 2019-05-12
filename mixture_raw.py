#!/usr/bin/python

import numpy as np
import tensorflow as tf

# local packages
from utils_libs import *

# reproducibility by fixing the random seed
# np.random.seed(1)
# tf.set_random_seed(1)

# ----- utilities functions -----

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


# ----- Mixture statistic -----

class mixture_statistic():
    
    def __init__(self, 
                 session, 
                 loss_type,
                 num_src):
        
        '''
        Args:
        
        session: tensorflow session
        
        loss_type: string, type of loss functions, {mse, lk, lk_inv}
        
        '''
        
        # build the network graph 
        self.lr = 0.0
        self.l2 = 0.0
        
        self.sess = session
        
        self.bool_log = ''
        self.loss_type = loss_type
        self.distr_type = ''
        
        # number of components or sources in X
        self.num_src = num_src
        

    def network_ini(self, 
                    lr, 
                    l2, 
                    dim_x,
                    steps_x, 
                    bool_log, 
                    bool_bilinear,
                    distr_type,
                    distr_para, 
                    bool_regu_positive_mean,
                    bool_regu_gate,
                    bool_regu_global_gate,
                    bool_regu_latent_dependence,
                    latent_dependence,
                    latent_prob_type,
                    var_type,
                    bool_bias_pred):
        
        '''
        Arguments:
        
        lr: float, learning rate
        
        l2: float, l2 regularization
        
        dim_x_list: list of dimension values for each component in X
        
        steps_x_list: list of sequence length values for each component in X
        
        bool_log: if log operation is on the targe variable Y
        
        bool_bilinear: if bilinear function is used on the components in X
        
        distr_type: string, type of the distribution of the target variable
        
        distr_para: set of parameters of the associated distribution
        
                    "gaussian": []
                    "t-distr": [nu]
        
        bool_regu_positive_mean: if regularization of positive mean 
        
        bool_regu_gate: if regularization the gate functions
        
        bool_regu_global_gate: if global gate regularization is applied
        
        latent_dependence: string, dependence of latent logits, "none", "independent", "markov"
        
        latent_prob_type: string, probability calculation of latent logits, "none", "scalar", "vector", "matrix"
                    
        var_type: square or exponential   
        
        bool_bias_pred: only on mean and variance
        
        '''

        # Dictionary of abbreviation
        #   nllk: negative log likelihood
        #   hetero: heteroskedasticity
        #   inv: inversed
        #   const: constant
        #   indi: individual
        #   py: predicted y
        #   src: source
        
        
        # ----- fix the random seed to reproduce the results
        
        np.random.seed(1)
        tf.set_random_seed(1)
        
        # ----- ini
        
        # build the network graph 
        self.lr = lr
        self.l2 = l2
        
        self.bool_log = bool_log
        self.distr_type = distr_type
        
        # initialize placeholders
        
        self.y = tf.placeholder(tf.float32, [None, 1], name = 'y')
        
        # ph: placeholder
        # shape: [C B T D]
        self.x_ph_list = []
        
        for i in range(self.num_src):
            self.x_ph_list.append(tf.placeholder(tf.float32, [None, steps_x[i], dim_x[i]], name = 'x' + str(i)))
        
        
        # for regularization
        # [1 S]
        self.global_logits = tf.get_variable('global_logits', 
                                             [1, self.num_src],
                                             initializer = tf.contrib.layers.xavier_initializer())
        
        
        # [S B T-1 D]
        pre_x = [tf.slice(self.x_ph_list[i], [0, 0, 0], [-1, steps_x[i]-1, -1]) for i in range(self.num_src)]
        # [S B T-1 D]
        curr_x = [tf.slice(self.x_ph_list[i], [0, 1, 0], [-1, steps_x[i]-1, -1]) for i in range(self.num_src)]
        
        
        # ----- individual models
        
        mean_list = []
        var_list = []
        
        logit_list = []
        pre_logit_list = []
        curr_logit_list = []
        
        regu_mean = 0.0
        regu_var = 0.0
        regu_gate = 0.0
        
        for i in range(self.num_src):
            
            if bool_bilinear == True:
                
                if latent_dependence != "none" :
                    
                    #[B]
                    tmp_mean, tmp_regu_mean = bilinear(curr_x[i],
                                                       [steps_x[i] - 1, dim_x[i]],
                                                       'mean' + str(i),
                                                       bool_bias = bool_bias_pred,
                                                       bool_scope_reuse = False)
                    
                    
                    tmp_var, tmp_regu_var = bilinear(curr_x[i],
                                                     [steps_x[i] - 1, dim_x[i]],
                                                     'var' + str(i),
                                                     bool_bias = bool_bias_pred,
                                                     bool_scope_reuse = False)
                    
                    #[B]
                    tmp_curr_logit, tmp_regu_gate = bilinear(curr_x[i],
                                                             [steps_x[i] - 1, dim_x[i]],
                                                             'gate' + str(i),
                                                             bool_bias = True,
                                                             bool_scope_reuse = False)
                    
                    #[B]
                    tmp_pre_logit, _ = bilinear(pre_x[i],
                                                [steps_x[i] - 1, dim_x[i]],
                                                'gate' + str(i),
                                                bool_bias = True,
                                                bool_scope_reuse = True)
                    
                    
                else:
                    
                    #[B]
                    tmp_mean, tmp_regu_mean = bilinear(self.x_ph_list[i],
                                                       [steps_x[i], dim_x[i]],
                                                       'mean' + str(i),
                                                       bool_bias = bool_bias_pred,
                                                       bool_scope_reuse = False)
                
                    tmp_var, tmp_regu_var = bilinear(self.x_ph_list[i],
                                                     [steps_x[i], dim_x[i]],
                                                     'var' + str(i),
                                                     bool_bias = bool_bias_pred,
                                                     bool_scope_reuse = False)
                    
                    tmp_logit, tmp_regu_gate = bilinear(self.x_ph_list[i],
                                                        [steps_x[i], dim_x[i]],
                                                        'gate' + str(i),
                                                        bool_bias = True,
                                                        bool_scope_reuse = False)
                
            '''
            else:
            
                if latent_dependence != "none":
                
                    tmp_pre_x = tf.reshape(pre_x[i], [-1, (steps_x_list[i]-1) * dim_x_list[i]])
                    tmp_curr_x = tf.reshape(curr_x[i], [-1, (steps_x_list[i]-1) * dim_x_list[i]])
                    
                    #[B]
                    tmp_mean, tmp_regu_mean = linear(tmp_curr_x, 
                                                     steps_x_list[i]*dim_x_list[i], 
                                                     'mean' + str(i), 
                                                     bool_bias = True,
                                                     bool_scope_reuse = False)
                
                    tmp_var, tmp_regu_var = linear(tmp_curr_x, 
                                                   steps_x_list[i]*dim_x_list[i], 
                                                   'var' + str(i), 
                                                   bool_bias = True,
                                                   bool_scope_reuse = False)
                    
                    
                    #[B]
                    tmp_curr_logit, tmp_regu_gate = linear(tmp_curr_x,
                                                           steps_x_list[i]*dim_x_list[i], 
                                                           'gate' + str(i),
                                                           bool_bias = True, 
                                                           bool_scope_reuse = False) 
                    
                    #[B]
                    tmp_pre_logit, _ = linear(tmp_pre_x,
                                              steps_x_list[i]*dim_x_list[i], 
                                              'gate' + str(i),
                                              bool_bias = True,
                                              bool_scope_reuse = True)
                    
                    
                else:
                    
                    tmp_x = tf.reshape(self.x_ph_list[i], [-1, steps_x_list[i] * dim_x_list[i]])
                    
                    #[B]
                    tmp_mean, tmp_regu_mean = linear(tmp_x, 
                                                     steps_x_list[i]*dim_x_list[i], 
                                                     'mean' + str(i), 
                                                     bool_bias = True,
                                                     bool_scope_reuse = False)
                
                    tmp_var, tmp_regu_var = linear(tmp_x, 
                                                   steps_x_list[i]*dim_x_list[i], 
                                                   'var' + str(i), 
                                                   bool_bias = True,
                                                   bool_scope_reuse = False)
                    
                    tmp_logit, tmp_regu_gate = linear(tmp_x,
                                                      steps_x_list[i]*dim_x_list[i], 
                                                      'gate' + str(i),
                                                      bool_bias = True,
                                                      bool_scope_reuse = False)
            '''
            
            # [S B]
            mean_list.append(tmp_mean)
            
            if var_type == "square":
                # square
                var_list.append(tf.square(tmp_var))
            
            elif var_type == "exp":
                var_list.append(tf.exp(tmp_var))
                
                
            if latent_dependence != "none" :
                
                # [S B]
                pre_logit_list.append(tmp_pre_logit)
                curr_logit_list.append(tmp_curr_logit)
                
            else:    
                logit_list.append(tmp_logit)
                
            regu_mean += tmp_regu_mean
            regu_var += tmp_regu_var
            regu_gate += tmp_regu_gate
        
        
        # -- individual means and variance
        
        # [B S]
        mean_stack = tf.stack(mean_list, axis = 1)
        var_stack = tf.stack(var_list, 1)
        inv_var_stack = tf.stack(var_list, 1)
        
        # ----- gates
        
        regu_latent_dependence = 0.0
        
        # -- latent logits
        
        if latent_dependence == "markov":
            
            # [S B] -> [B S]
            pre_logit = tf.stack(pre_logit_list, [1, 0])
            curr_logit = tf.stack(curr_logit_list, [1, 0])
        
            if latent_prob_type == "constant_diff_sq":
                
                # [B 1]
                latent_prob_logits = tf.reduce_sum(tf.square(curr_logit - pre_logit), 1, keep_dims = True)
                
                # regularization
                regu_latent_dependence = 0.0
        
            elif latent_prob_type == "scalar_diff_sq":
                
                # [1]
                w_logit = tf.get_variable('w_logit',
                                          [],
                                          initializer = tf.contrib.layers.xavier_initializer())
            
                latent_prob_logits = w_logit*tf.reduce_sum(tf.square(curr_logit - pre_logit), 1, keep_dims = True)
                
                # regularization
                regu_latent_dependence = tf.square(w_logit) 
            
            
            elif latent_prob_type == "vector_diff_sq":
                
                # [1]
                w_logit = tf.get_variable('w_logit',
                                          [self.num_src, 1],
                                          initializer = tf.contrib.layers.xavier_initializer())
            
                latent_prob_logits = tf.matmul(tf.square(curr_logit - pre_logit), w_logit)
            
                # regu
                regu_latent_dependence = tf.square(w_logit) 
            
            
            elif latent_prob_type == "pos_neg_diff_sq":
                
                # [1]
                w_pos = tf.get_variable('w_pos',
                                        [],
                                        initializer = tf.contrib.layers.xavier_initializer())
                
                w_neg = tf.get_variable('w_neg',
                                        [],
                                        initializer = tf.contrib.layers.xavier_initializer())
                # [B 1]
                pos_logits = 1.0*tf.square(w_pos)*tf.reduce_sum(tf.square(curr_logit - pre_logit), 1, keep_dims = True)
                # [B 1]
                neg_logits = -1.0*tf.square(w_neg)*tf.reduce_sum(tf.square(curr_logit - pre_logit), 1, keep_dims = True)
                
                # [B 1]
                latent_prob = 0.5*tf.sigmoid(pos_logits) + 0.5*tf.sigmoid(neg_logits)
                
                # regularization
                regu_latent_dependence = tf.square(w_pos) + tf.square(w_neg)
            
            
            '''
            
            elif latent_prob_type == "pos_scalar_diff_sq":
                
                # [1]
                w_logit = tf.get_variable('w_logit',
                                          [],
                                          initializer = tf.contrib.layers.xavier_initializer())
            
                latent_prob_logits = 1.0*tf.square(w_logit)*tf.reduce_sum(tf.square(curr_logit-pre_logit), 1, keep_dims = True)
            
                # [B 1]
                latent_prob = tf.sigmoid(latent_prob_logits)
            
            
            elif latent_prob_type == "neg_scalar_diff_sq":
                
                # [1]
                w_logit = tf.get_variable('w_logit',
                                          [],
                                          initializer = tf.contrib.layers.xavier_initializer())
            
                latent_prob_logits = -1.0*tf.square(w_logit)*tf.reduce_sum(tf.square(curr_logit-pre_logit),1,keep_dims = True)
            
                # [B 1]
                latent_prob = tf.sigmoid(latent_prob_logits) 
            
            
            elif latent_prob_type == "vector_diff":
            
                # [C 1]
                w_logit = tf.get_variable('w_logit',
                                          [self.num_src, 1], 
                                          initializer = tf.contrib.layers.xavier_initializer())
            
                # regu? 
            
                # [B 1]                        [B C]                    [C 1]
                latent_prob_logits = tf.matmul(curr_logit - pre_logit, w_logit)
            
                # [B 1]                                                            
                latent_prob = tf.sigmoid(latent_prob_logits)
            
            elif latent_prob_type == "vector_dense":
            
                # [C 1]
                w_logit = tf.get_variable('w_logit',
                                          [self.num_src*2, 1], 
                                          initializer = tf.contrib.layers.xavier_initializer())
            
                # regu? 
            
                # [B 1]                            [B 2C]                    [2C 1]
                latent_prob_logits = tf.matmul(tf.concat([curr_logit, pre_logit], 1), w_logit)
            
                # [B 1]                                                            
                latent_prob = tf.sigmoid(latent_prob_logits)
            
            
            elif latent_prob_type == "matrix":
            
                # [C C]
                w_logit = tf.get_variable('w_logit',
                                          [self.num_src, self.num_src],
                                          initializer = tf.contrib.layers.xavier_initializer())
            
                # [B C]                             [B C]                  [C C]                                      
                latent_prob_logits = tf.matmul(curr_logit - pre_logit, w_logit)
            
                # [B 1]                                                            
                latent_prob = tf.sigmoid(latent_prob_logits)
        
        
            elif latent_prob_type == "none":
            
                latent_prob = 0.0
        '''
        
        
        # -- latent_dependence
        
        if latent_dependence == "independent" or latent_dependence == "markov":
            
            # [B C]
            pre_logit = tf.stack(pre_logit_list, 1)
            curr_logit = tf.stack(curr_logit_list, 1)
            
            # ? 
            # [B C]
            gate_logits = curr_logit
        
        elif latent_dependence == "none":
            
            # [B C]
            gate_logits = tf.stack(logit_list, 1)
        
        '''
        elif latent_dependence == "markov":
            
            # ??
            gate_logits = (1.0 - latent_prob)*curr_logit + latent_prob*pre_logit    
            
        '''
        
        self.gates = tf.nn.softmax(gate_logits, axis = -1)
        
        
        # ----- prediction 
        
        if distr_type == 'gaussian':
            
            # -- mean
            
            # component mean
            self.py_mean_src = mean_stack
            
            # mixed mean
            # [B 1]                      [B S]      [B S]
            self.py_mean = tf.reduce_sum(mean_stack * self.gates, 1, keepdims = True)
            
            # -- variance
            
            # variance of mixture
            if self.loss_type == 'lk':
                
                # component variance
                self.py_var_src = var_stack
                
                sq_mean_stack = var_stack + tf.square(mean_stack)
                # [B]                                   [B S]          [B S]
                mix_sq_mean = tf.reduce_sum(tf.multiply(sq_mean_stack, self.gates), 1)
                
                # [B]
                self.py_var = mix_sq_mean - tf.square(self.py_mean)
            
            elif self.loss_type == 'lk_inv' or self.loss_type == 'elbo':
                
                # component variance
                self.py_var_src = 1.0/(inv_var_stack + 1e-5)
                
                sq_mean_stack = 1.0/(inv_var_stack + 1e-5) + tf.square(mean_stack)
                mix_sq_mean = tf.reduce_sum(tf.multiply(sq_mean_stack, self.gates), 1)
                
                # [B]
                self.py_var = mix_sq_mean - tf.square(self.py_mean)
                
            elif self.loss_type == 'mse':
                
                # component variance
                self.py_var_src = 1.0
                
                sq_mean_stack = 1.0 + tf.square(mean_stack)
                mix_sq_mean = tf.reduce_sum(tf.multiply(sq_mean_stack, self.gates), 1)
                
                # [B]
                self.py_var = mix_sq_mean - tf.square(self.py_mean)
                
            elif self.loss_type == 'simple_mix':
                
                # mixed variance
                # [B 1]                     [B S]     [B S]
                self.py_var = tf.reduce_sum(var_stack * self.gates, 1, keepdims = True)
                
            elif self.loss_type == 'simple_mix_inv':
                
                # mixed variance
                # [B 1]                     [B S]         [B S]
                self.py_var = tf.reduce_sum(inv_var_stack * self.gates, 1, keepdims = True)\
                
                
            # -- standard deviation
            # [B]
            self.py_std = tf.sqrt(self.py_var)
            
        
        
        elif distr_type == 't-distr':
            
            # -- mean
            
            # component mean
            self.py_mean_src = mean_stack
            
            # mixed mean
            # [B 1]                      [B S]      [B S]
            self.py_mean = tf.reduce_sum(mean_stack * self.gates, 1, keepdims = True)
            
            # -- variance

                
                # component variance
                self.py_var_src = distr_para[0] # "nu"
                
                sq_mean_stack = 1.0 + tf.square(mean_stack)
                mix_sq_mean = tf.reduce_sum(tf.multiply(sq_mean_stack, self.gates), 1)
                
                # [B]
                self.py_var = mix_sq_mean - tf.square(self.py_mean)
                
                
                
            # -- standard deviation
            # [B]
            self.py_std = tf.sqrt(self.py_var)
        
        
        
        #else:
        #    print('[ERROR] distribution type')
            
        
        # ----- regularization
        
        self.regularization = 0
        self.regu_var = regu_var 
        self.regu_mean = regu_mean 
        
        
        # -- non-negative hinge regularization 
        
        if bool_regu_positive_mean == True:
            
            # regu_mean_pos = tf.reduce_sum(tf.maximum(0.0, -1.0*mean_v) + tf.maximum(0.0, -1.0*mean_x))
            self.regularization += regu_mean_pos
        
        
        # -- gate smoothing
        
        if latent_prob_type != "none":
            
            if latent_prob_type == "pos_neg_diff_sq":
                
                # exact llk
                #self.latent_depend_regu = -1.0*tf.reduce_sum(tf.log(latent_prob))
                
                # lower bound
                self.latent_depend_regu = 0.5*(tf.reduce_sum(tf.log(1.0 + tf.exp(-1.0*tf.abs(pos_logits)))\
                                          + tf.maximum(0.0, -1.0*pos_logits)))\
                                                                + \
                                          0.5*(tf.reduce_sum(tf.log(1.0 + tf.exp(-1.0*tf.abs(neg_logits)))\
                                          + tf.maximum(0.0, -1.0*neg_logits)))                 
            else:
                
                # [B 1]
                # ! numertical stable version of log(sigmoid())
                self.latent_depend_regu = (tf.reduce_sum(tf.log(1.0 + tf.exp(-1.0*tf.abs(latent_prob_logits))) \
                                           + tf.maximum(0.0, -1.0*latent_prob_logits))) 
        else:
            self.latent_depend_regu = 0.0
            
            
        # -- latent dependence
        
        if bool_regu_latent_dependence == True:
            self.regularization += regu_latent_dependence
            
        # -- weights in gates
        
        if bool_regu_gate == True:
            self.regularization += regu_gate
        
        
        # -- global logits
        # implicitly regularization on weights of gate functions
        
        if bool_regu_global_gate == True:
            #                                        [B C]        [1 C]
            logits_diff_sq = tf.reduce_sum(tf.square(gate_logits - self.global_logits), 1)
            regu_global_logits = tf.reduce_sum(logits_diff_sq) + tf.nn.l2_loss(self.global_logits)
            
            self.regularization += regu_global_logits
        
        
        # ----- negative log likelihood 
                
        # lk: likelihood
        # llk: log likelihood
        # nllk: negative log likelihood
        
        # -- lk
        # [B C]
        tmp_lk_indi_hetero = tf.exp(-0.5*tf.square(self.y - mean_stack)/(var_stack + 1e-5))/(2.0*np.pi*(var_stack + 1e-5))**0.5
            
        lk_hetero = tf.multiply(tmp_lk_indi_hetero, self.gates) 
        self.nllk_hetero = tf.reduce_sum(-1.0*tf.log(tf.reduce_sum(lk_hetero, axis = -1) + 1e-5))
        
        # -- lk_inv
        # [B C]
        tmp_lk_indi_hetero_inv = tf.exp(-0.5*tf.square(self.y - mean_stack)*inv_var_stack)*(0.5/np.pi*inv_var_stack)**0.5
            
        lk_hetero_inv = tf.multiply(tmp_lk_indi_hetero_inv, self.gates) 
        self.nllk_hetero_inv = tf.reduce_sum(-1.0*tf.log(tf.reduce_sum(lk_hetero_inv, axis = -1) + 1e-5))
        
        
        # -- ELBO
        # evidence lower bound optimization
        # based on lk_inv
        
        # [B 1] - [B C]
        tmp_nllk_inv = 0.5*tf.square(self.y - mean_stack)*inv_var_stack - 0.5*tf.log(inv_var_stack+1e-5) + 0.5*tf.log(2*np.pi)
        
        self.nllk_elbo = tf.reduce_sum(tf.reduce_sum(self.gates * tmp_nllk_inv, -1)) 
        
        # -- mse
        # [B C]
        tmp_lk_indi_const = tf.exp(-0.5*tf.square(self.y - mean_stack))/(2.0*np.pi)**0.5
            
        lk_const = tf.multiply(tmp_lk_indi_const, self.gates) 
        self.nllk_const = tf.reduce_sum(-1.0*tf.log(tf.reduce_sum(lk_const, axis = -1) + 1e-5))
        
        # -- simple_mix
        
        #self.nllk_gate = -1.0*tf.reduce_sum(tf.log(latent_prob))
        
        # variance
        # [B]
        tmp_nllk_mix = 0.5*tf.square(self.y - self.py_mean)/(self.py_var + 1e-5) + 0.5*tf.log(self.py_var + 1e-5)\
                       + 0.5*tf.log(2*np.pi)
            
        self.nllk_mix = tf.reduce_sum(tmp_nllk_mix) 
        
        # variance inverse
        # [B]
        # self.py_var is inversed
        tmp_nllk_mix_inv = 0.5*tf.square(self.y - self.py_mean)*self.py_var - 0.5*tf.log(self.py_var + 1e-5)\
                           + 0.5*tf.log(2*np.pi)
            
        self.nllk_mix_inv = tf.reduce_sum(tmp_nllk_mix_inv) 
       

    #   initialize loss and optimization operations for training
    def train_ini(self):
        
        self.sq_error = tf.reduce_sum(tf.square(self.y - self.py_mean))
        
        # loss, nllk
        if self.loss_type == 'mse':
            
            self.loss = tf.reduce_mean(tf.square(self.y - self.py_mean)) + \
                        0.1*self.l2*self.regularization + self.l2*self.regu_mean
                
            self.nllk = self.nllk_const
        
        elif self.loss_type == 'lk_inv':
            
            self.loss = self.nllk_hetero_inv + 0.1*self.l2*(self.regularization + self.regu_var) + self.l2*self.regu_mean
            self.nllk = self.nllk_hetero_inv
            
        elif self.loss_type == 'lk':
            
            # ?
            self.loss = self.nllk_hetero + 0.1*self.l2*self.regularization + self.l2*(self.regu_mean + self.regu_var)\
                        + self.latent_depend_regu
                
            self.nllk = self.nllk_hetero
        
        
        elif self.loss_type == 'elbo':
            
            self.loss = self.nllk_elbo + 0.1*self.l2*(self.regularization + self.regu_var) + self.l2*self.regu_mean
            
            # negative log likelihood calculated through nllk_hetero_inv
            self.nllk = self.nllk_hetero_inv
            
        elif self.loss_type == 'simple_mix':
            
            # ?
            self.loss = self.nllk_mix + 0.1*self.l2*(self.regularization + self.regu_var) + self.l2*self.regu_mean
                        #self.nllk_gate + \
                        
            self.nllk = self.nllk_mix
         
        elif self.loss_type == 'simple_mix_inv':
            
            # ?
            self.loss = self.nllk_mix_inv + 0.1*self.l2*(self.regularization + self.regu_var) + self.l2*self.regu_mean
                        #self.nllk_gate + \
            
            self.nllk = self.nllk_mix_inv
            
        else:
            print('[ERROR] loss type')
        
        
        self.train = tf.train.AdamOptimizer(learning_rate = self.lr)
        self.optimizer = self.train.minimize(self.loss)
        
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        
        
    #   training on batch of data
    def train_batch(self,
                    x,
                    y):
        
        # x: [S, [B, T D]]
        # y: [B, 1]
        
        data_dict = {}
        for idx in range(self.num_src):
            data_dict["x" + str(idx) + ":0"] = x[idx]
        
        data_dict['y:0'] = y
        
        _, tmp_loss, tmp_sq_err = self.sess.run([self.optimizer, self.loss, self.sq_error],
                                                feed_dict = data_dict)
        return tmp_loss, tmp_sq_err
    
    
    #   evaluation metric
    def inference_ini(self):
        
        # RMSE
        self.rmse = tf.sqrt(tf.losses.mean_squared_error(self.y, self.py_mean))
        
        # MAE
        self.mae = tf.reduce_mean(tf.abs(self.y - self.py_mean))
        
        # MAPE
        # based on ground-truth y
        mask = tf.greater(tf.abs(self.y), 1e-5)
        
        y_mask = tf.boolean_mask(self.y, mask)
        y_hat_mask = tf.boolean_mask(self.py_mean, mask)
        
        self.mape = tf.reduce_mean(tf.abs((y_mask - y_hat_mask)/(y_mask + 1e-10)))
        
        # NNLLK 
        # nnllk - normalized negative log likelihood by the number of data samples
        self.nnllk = self.nllk / tf.to_float(tf.shape(self.x_ph_list[0])[0])
        
        # for restored models
        tf.add_to_collection("rmse", self.rmse)
        tf.add_to_collection("mae", self.mae)
        tf.add_to_collection("mape", self.mape)
        tf.add_to_collection("nnllk", self.nnllk)
        
        tf.add_to_collection("py_mean", self.py_mean)
        tf.add_to_collection("py_std", self.py_std)
        tf.add_to_collection("py_gate", self.gates)
        
    
    #   infer givn testing data
    def inference(self,
                  x,
                  y,
                  bool_indi_eval):
        
        # x: shape [S N T D] 
        # each element corresponds to one data soucre and is of shape [N T D]
        
        # rmse, mae, mape, nnllk, py_mean, py_std
        
        data_dict = {}
        for idx in range(self.num_src):
            data_dict["x" + str(idx) + ":0"] = x[idx]
        
        data_dict['y:0'] = y
        
        rmse, mae, mape, nnllk = self.sess.run([tf.get_collection('rmse')[0],
                                                tf.get_collection('mae')[0],
                                                tf.get_collection('mape')[0],
                                                tf.get_collection('nnllk')[0]], 
                                                feed_dict = data_dict)
        
        # test: for the observing during the training
        py_gate = self.sess.run([tf.get_collection('py_gate')[0]], 
                             feed_dict = data_dict)
        
        if bool_indi_eval == True:
            
            py_mean, py_std = self.sess.run([tf.get_collection('py_mean')[0], 
                                             tf.get_collection('py_std')[0]], 
                                             feed_dict = data_dict)
        else:
            py_mean = None
            py_std = None
            
        return rmse, mae, mape, nnllk, py_mean, py_std, py_gate
    
    
    # collect the optimized variable values
    def collect_coeff_values(self,
                             vari_keyword):
        
        return [tf_var.name for tf_var in tf.trainable_variables() if (vari_keyword in tf_var.name)],\
               [tf_var.eval() for tf_var in tf.trainable_variables() if (vari_keyword in tf_var.name)]
    
    
    # restore the model from the files
    def pre_train_restore_model(self,
                                path_meta,
                                path_data):
        
        saver = tf.train.import_meta_graph(path_meta, 
                                           clear_devices = True)
        saver.restore(self.sess, 
                      path_data)
        
        return
    
