#!/usr/bin/python

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from scipy.stats import truncnorm
from scipy.optimize import fmin_slsqp

# local packages
from utils_libs import *
from utils_linear_units import *
from utils_training import *

from utils_optimization import *

# reproducibility by fixing the random seed
# np.random.seed(1)
# tf.set_random_seed(1)


# ----- Mixture statistic -----


class mixture_statistic():
    
    def __init__(self, 
                 session, 
                 loss_type,
                 num_src,
                 bool_jump_boosting):
        
        '''
        Args:
        
        session: tensorflow session
        
        loss_type: string, type of loss functions, {mse, lk, lk_inv, elbo, ...}
        
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
        
        # for SG-MCMC during both training and testing phases
        # [A B S]
        # A: number of samples
        self.py_mean_src_samples = []
        self.py_var_src_samples = []
        self.py_gate_src_samples = []
        
        self.py_mean_samples = []
        
        
        self.log_step_error = []
        self.log_error_up_flag = False
        
        self.stored_step_id = []
        
        self.bool_jump_boosting = bool_jump_boosting
        

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
                    bool_regu_l2_on_latent,
                    latent_dependence,
                    latent_prob_type,
                    var_type,
                    bool_bias_mean,
                    bool_bias_var,
                    bool_bias_gate,
                    optimization_method,
                    optimization_lr_decay,
                    optimization_lr_decay_steps,
                    optimization_mode,
                    burn_in_step,
                    bool_global_bias_src,
                    bool_imbalance_l2,
                    bool_regu_mean_var,
                    bool_jump_boosting
                    ):
        

        
        '''
        Arguments:
        
        lr: float, learning rate
        
        l2: float, l2 regularization
        
        dim_x:  dimension values for each component in X
        
        steps_x: sequence length values for each component in X
        
        bool_log: if log operation is on the targe variable Y
        
        bool_bilinear: if bilinear function is used on the components in X
        
        distr_type: string, type of the distribution of the target variable
        
        distr_para: set of parameters of the associated distribution
        
                    "gaussian": []
                    "student_t": [nu] >= 3
        
        bool_regu_positive_mean: if regularization of positive mean 
        
        bool_regu_gate: if regularization the gate functions
        
        bool_regu_global_gate: if global gate regularization is applied
        
        latent_dependence: string, dependence of latent logits, "none", "independent", "markov"
        
        latent_prob_type: string, probability calculation of latent logits, "none", "scalar", "vector", "matrix"
        
        var_type: square or exponential   
        
        bool_bias_mean: have bias in mean prediction
        
        bool_bias_var: have bias in variance prediction
        
        bool_bias_gate: have bias in gate prediction
        
        optimization_mode: bayesian, map
        
        '''
        
        # Dictionary of abbreviation
        #   nllk: negative log likelihood
        #   hetero: heteroskedasticity
        #   inv: inversed
        #   const: constant
        #   indi: individual
        #   py: predicted y
        #   src: source
        #   var: variance
        
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

        # shape: [S B T D]
        self.x = tf.placeholder(tf.float32, [self.num_src, None, steps_x, dim_x], name = 'x')
        
        self.bool_regu_l2_on_latent = bool_regu_l2_on_latent
        self.bool_imbalance_l2 = bool_imbalance_l2
        
        
        self.optimization_method =   optimization_method
        self.optimization_lr_decay = optimization_lr_decay
        self.optimization_lr_decay_steps = optimization_lr_decay_steps 
        
        self.bool_regu_mean_var = bool_regu_mean_var
        
        self.optimization_mode = optimization_mode
        self.training_step = 0
        self.burn_in_step = burn_in_step
        
        self.bool_jump_boosting = bool_jump_boosting
        
        # ----- individual models
        
        if latent_dependence != "none" :
            
            # [S B T-1 D]
            pre_x = tf.slice(self.x, [0, 0, 0, 0], [-1, -1, steps_x - 1, -1])
            # [S B T-1 D]
            curr_x = tf.slice(self.x, [0, 0, 1, 0], [-1, -1, steps_x - 1, -1])
            
            if bool_bilinear == True:
                
                #[S B]
                tmp_mean, regu_mean, tmp_var, regu_var, tmp_curr_logit, regu_gate = \
                multi_src_predictor_linear(x = curr_x, 
                                           n_src = self.num_src, 
                                           steps = steps_x - 1, 
                                           dim = dim_x, 
                                           bool_bias = [bool_bias_mean, bool_bias_var, bool_bias_gate], 
                                           bool_scope_reuse= [False, False, False], 
                                           str_scope = "")
                
                
                #[S B]
                _, _, _, _, tmp_pre_logit, _ = \
                multi_src_predictor_linear(x = pre_x, 
                                           n_src = self.num_src, 
                                           steps = steps_x - 1, 
                                           dim = dim_x, 
                                           bool_bias = [bool_bias_mean, bool_bias_var, bool_bias_gate], 
                                           bool_scope_reuse= [True, True, True], 
                                           str_scope = "")
                
        else:
            
            if bool_bilinear == True:
                
                #[S B]
                tmp_mean, regu_mean, tmp_var, regu_var, tmp_logit, regu_gate = \
                multi_src_predictor_linear(x = self.x, 
                                           n_src = self.num_src, 
                                           steps = steps_x, 
                                           dim = dim_x, 
                                           bool_bias = [bool_bias_mean, bool_bias_var, bool_bias_gate], 
                                           bool_scope_reuse= [False, False, False], 
                                           str_scope = "")
                
                
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
        
        
        # ----- individual means and variance
        
        # -- mean
        if bool_global_bias_src == True:
            
            # global bias term
        
            global_b = tf.get_variable('global_b', 
                                   shape = [1, ],
                                   initializer = tf.zeros_initializer())
            
            #[1 B]                    [S B]
            tmp_target_src = tf.slice(tmp_mean, [0, 0], [1, -1]) 
            #[S-1 B]
            tmp_rest_src = tf.slice(tmp_mean, [1, 0], [-1, -1])
        
            tmp_target_src = tmp_target_src + global_b
            # [B S]            [S B]
            mean_stack = tf.transpose(tf.concat([tmp_target_src, tmp_rest_src], axis = 0), [1, 0])
        
        else:
            
            # [B S]
            mean_stack = tf.transpose(tmp_mean, [1, 0])
        
        
        # -- variance
        if var_type == "square":
            
            # square
            var_stack = tf.transpose(tf.square(tmp_var), [1, 0])
            inv_var_stack = tf.transpose(tf.square(tmp_var), [1, 0])
        
        elif var_type == "exp":
            
            # exp
            var_stack = tf.transpose(tf.exp(tmp_var), [1, 0])
            inv_var_stack = tf.transpose(tf.exp(tmp_var), [1, 0])
            
        
        # ----- gates
        
        regu_latent_dependence = 0.0
        
        # -- latent logits dependence
        
        if latent_dependence == "markov":
            
            # [B S]
            pre_logit = tf.transpose(tmp_pre_logit, [1, 0])
            curr_logit = tf.transpose(tmp_curr_logit, [1, 0])
            
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
            
                # [B 1]
                latent_prob_logits = w_logit*tf.reduce_sum(tf.square(curr_logit - pre_logit), 1, keep_dims = True)
            
                # regularization
                regu_latent_dependence = tf.square(w_logit)
                
                
            elif latent_prob_type == "vector_diff_sq":
                
                # [1]
                w_logit = tf.get_variable('w_logit',
                                          [self.num_src, 1],
                                          initializer = tf.contrib.layers.xavier_initializer())
            
                latent_prob_logits = tf.matmul(tf.square(curr_logit - pre_logit), w_logit)
                
                # regularization
                regu_latent_dependence = tf.reduce_sum(tf.square(w_logit))
                
            
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
                
                
        # -- latent logits 
        
        if latent_dependence == "independent" or latent_dependence == "markov":
            
            # [B S]
            pre_logit = tf.transpose(tmp_pre_logit, [1, 0])
            curr_logit = tf.transpose(tmp_curr_logit, [1, 0])
            
            # [B S]
            gate_logits = curr_logit
        
        elif latent_dependence == "none":
            
            # [B S]
            gate_logits = tf.transpose(tmp_logit, [1, 0])
        
        self.gates = tf.nn.softmax(gate_logits, axis = -1)
        
        
        # ----- mixture mean, variance and nllk  
        
        if distr_type == 'gaussian':
            
            # -- mean
            
            # component mean
            # [B S] 
            self.py_mean_src = mean_stack
            
            # mixture mean
            # [B 1]                      [B S]        [B S]
            self.py_mean = tf.reduce_sum(mean_stack * self.gates, 1, keepdims = True)
            
            # -- variance
            
            if self.loss_type == 'lk':
                
                # component variance
                self.py_var_src = var_stack
                
                # variance
                sq_mean_stack = var_stack + tf.square(mean_stack)
                # [B 1]                                   [B S]          [B S]
                mix_sq_mean = tf.reduce_sum(tf.multiply(sq_mean_stack, self.gates), 1, keepdims = True)
                
                # [B 1]
                self.py_var = mix_sq_mean - tf.square(self.py_mean)
                
                # negative log likelihood
                # [B S]
                lk_hetero_src = tf.exp(-0.5*tf.square(self.y-mean_stack)/(var_stack+1e-5))/(tf.sqrt(2.0*np.pi*var_stack)+1e-5)
                                                                                                 
                lk_hetero = tf.multiply(lk_hetero_src, self.gates) 
                self.nllk_hetero = tf.reduce_sum(-1.0*tf.log(tf.reduce_sum(lk_hetero, axis = -1) + 1e-5))
                
                #self.nllk_loss
                #self.nllk
            
            elif self.loss_type == 'lk_inv':
                
                # component variance
                self.py_var_src = 1.0/(inv_var_stack + 1e-5)
                
                # variance
                sq_mean_stack = 1.0/(inv_var_stack + 1e-5) + tf.square(mean_stack)
                mix_sq_mean = tf.reduce_sum(tf.multiply(sq_mean_stack, self.gates), 1, keepdims = True)
                
                # [B 1]
                self.py_var = mix_sq_mean - tf.square(self.py_mean)
                
                # negative log likelihood
                # [B S]
                lk_hetero_src_inv = tf.exp(-0.5*tf.square(self.y-mean_stack)*inv_var_stack)*tf.sqrt(0.5/np.pi*inv_var_stack)
            
                lk_hetero_inv = tf.multiply(lk_hetero_src_inv, self.gates) 
                self.nllk_hetero_inv = tf.reduce_sum(-1.0*tf.log(tf.reduce_sum(lk_hetero_inv, axis = -1) + 1e-5))
            
            
            # elbo: evidence lower bound optimization    
            elif self.loss_type == 'elbo':
                
                # component variance
                self.py_var_src = 1.0/(inv_var_stack + 1e-5)
                
                # variance
                sq_mean_stack = 1.0/(inv_var_stack + 1e-5) + tf.square(mean_stack)
                mix_sq_mean = tf.reduce_sum(tf.multiply(sq_mean_stack, self.gates), 1, keepdims = True)
                
                # [B 1]
                self.py_var = mix_sq_mean - tf.square(self.py_mean)
                
                # negative log likelihood
                # based on lk_inv
        
                # [B 1] - [B S]
                tmp_nllk_inv = 0.5*tf.square(self.y - mean_stack)*inv_var_stack - 0.5*tf.log(inv_var_stack + 1e-5) + 0.5*tf.log(2*np.pi)
        
                self.nllk_elbo = tf.reduce_sum(tf.reduce_sum(self.gates*tmp_nllk_inv, -1)) 
                
                
                
            elif self.loss_type == 'mse':
                
                # component variance
                self.py_var_src = 1.0
                
                # variance
                sq_mean_stack = 1.0 + tf.square(mean_stack)
                # [B 1]
                mix_sq_mean = tf.reduce_sum(tf.multiply(sq_mean_stack, self.gates), 1, keepdims = True)
                
                # [B 1]
                self.py_var = mix_sq_mean - tf.square(self.py_mean)
                
                # negative log likelihood
                # [B S]
                tmp_lk_const_src = tf.exp(-0.5*tf.square(self.y - mean_stack))/(2.0*np.pi)**0.5
            
                lk_const = tf.multiply(tmp_lk_const_src, self.gates) 
                self.nllk_const = tf.reduce_sum(-1.0*tf.log(tf.reduce_sum(lk_const, axis = -1) + 1e-5))
            
                
            elif self.loss_type == 'lk_var_mix':
                
                # mixed variance
                # [B 1]                     [B S]     [B S]
                self.py_var = tf.reduce_sum(var_stack * self.gates, 1, keepdims = True)
                
                # negative log likelihood
                # [B]
                tmp_nllk_var_mix = 0.5*tf.square(self.y - self.py_mean)/(self.py_var + 1e-5) + 0.5*tf.log(self.py_var + 1e-5)\
                           + 0.5*tf.log(2*np.pi)
            
                self.nllk_var_mix = tf.reduce_sum(tmp_nllk_mix) 
            
            
            elif self.loss_type == 'lk_var_mix_inv':
                
                # mixed variance
                # [B 1]                     [B S]         [B S]
                py_var_inv = tf.reduce_sum(inv_var_stack * self.gates, 1, keepdims = True)
                # [B 1]
                self.py_var = tf.reduce_sum(1.0/(inv_var_stack + 1.0) * self.gates, 1, keepdims = True)
                
                # negative log likelihood
                # [B]
                # self.py_var is inversed
                tmp_nllk_var_mix_inv = 0.5*tf.square(self.y - self.py_mean)*py_var_inv - 0.5*tf.log(py_var_inv + 1e-5)\
                               + 0.5*tf.log(2*np.pi)
            
                self.nllk_var_mix_inv = tf.reduce_sum(tmp_nllk_mix_inv)
                
                
            # -- standard deviation
            # [B 1]
            self.py_std = tf.sqrt(self.py_var)
            
            
        elif distr_type == 'student_t':
            
            # for negative log likelihood
            t_distr_constant = 1.0/(np.sqrt(distr_para[0])*sp.special.beta(0.5,distr_para[0]/2.0)+ 1e-5)
            
            # -- mean
            
            # component mean
            self.py_mean_src = mean_stack
            
            # mixture mean
            # [B 1]                      [B S]       [B S]
            self.py_mean = tf.reduce_sum(mean_stack * self.gates, 1, keepdims = True)
            
            
            # -- variance
            
            # variance of mixture
            if self.loss_type == 'lk':
                
                # component variance
                # [B S]    
                self.py_var_src = var_stack*1.0*distr_para[0]/(distr_para[0]-2.0) # "nu"
                
                # variance
                sq_mean_stack = var_stack*1.0*distr_para[0]/(distr_para[0]-2.0) + tf.square(mean_stack)
                
                # [B 1]                                   [B S]          [B S]
                mix_sq_mean = tf.reduce_sum(tf.multiply(sq_mean_stack, self.gates), 1,  keepdims = True)
                
                # [B 1]
                self.py_var = mix_sq_mean - tf.square(self.py_mean)
                
            
                # [B S]
                # self.x: [S B T D]
            
                # [B S]
                normalizer_hetero_src = t_distr_constant/(tf.sqrt(var_stack) + 1e-5)
                #1.0/(tf.sqrt(distr_para[0]*var_stack)*sp.special.beta(0.5, distr_para[0]/2.0) + 1e-5)
            
                base_hetero_src = 1.0 + 1.0*tf.square(self.y - mean_stack)/distr_para[0]/(var_stack + 1e-5)
            
                lk_hetero_src = normalizer_hetero_src*tf.keras.backend.pow(base_hetero_src, -(distr_para[0] + 1)/2)
        
                lk_hetero = tf.multiply(lk_hetero_src, self.gates) 
                self.nllk_hetero = tf.reduce_sum(-1.0*tf.log(tf.reduce_sum(lk_hetero, axis = -1) + 1e-5))
                
            
            elif self.loss_type == 'lk_inv' :
                
                # component variance
                # [B S]    
                self.py_var_src = 1.0*distr_para[0]/(distr_para[0]-2.0)/(inv_var_stack + 1e-5) # "nu"
                
                # variance
                sq_mean_stack = self.py_var_src + tf.square(mean_stack)
                mix_sq_mean = tf.reduce_sum(tf.multiply(sq_mean_stack, self.gates), keepdims = True)
                
                # [B 1]
                self.py_var = mix_sq_mean - tf.square(self.py_mean)
                
                
                # negative log likelihood
                # [B S]
                normalizer_hetero_src_inv = t_distr_constant*tf.sqrt(inv_var_stack)
            
                base_hetero_src_inv = 1.0 + 1.0*tf.square(self.y - mean_stack)/distr_para[0]*inv_var_stack
            
                lk_hetero_src_inv = normalizer_hetero_src_inv*tf.keras.backend.pow(base_hetero_src_inv, -(distr_para[0] + 1)/2)
            
            
                lk_hetero_inv = tf.multiply(lk_hetero_src_inv, self.gates) 
                self.nllk_hetero_inv = tf.reduce_sum(-1.0*tf.log(tf.reduce_sum(lk_hetero_inv, axis = -1) + 1e-5))
                
                
            '''     
            elif self.loss_type == 'elbo':
                
                # component variance
                # [B S]    
                self.py_var_src = 1.0*distr_para[0]/(distr_para[0]-2.0)/(inv_var_stack + 1e-5) # "nu"
                
                sq_mean_stack = self.py_var_src + tf.square(mean_stack)
                mix_sq_mean = tf.reduce_sum(tf.multiply(sq_mean_stack, self.gates), keepdims = True)
                
                # [B 1]
                self.py_var = mix_sq_mean - tf.square(self.py_mean)
                
               
            elif self.loss_type == 'mse':
                
                # component variance
                self.py_var_src = 1.0
                
                sq_mean_stack = 1.0 + tf.square(mean_stack)
                # [B 1]
                mix_sq_mean = tf.reduce_sum(tf.multiply(sq_mean_stack, self.gates), 1, keepdims = True)
                
                # [B 1]
                self.py_var = mix_sq_mean - tf.square(self.py_mean)
                
            elif self.loss_type == 'var_mix':
                
                # mixed variance
                # [B 1]                     [B S]     [B S]
                self.py_var = tf.reduce_sum(var_stack * self.gates, 1, keepdims = True)
                
            elif self.loss_type == 'var_mix_inv':
                
                # mixed variance
                # [B 1]                     [B S]         [B S]
                self.py_var = tf.reduce_sum(inv_var_stack * self.gates, 1, keepdims = True)
            '''
            
            
            # -- standard deviation
            # [B 1]
            self.py_std = tf.sqrt(self.py_var)
            

        # ----- regularization
        
        self.regu_var = regu_var 
        self.regu_mean = regu_mean         
        
        self.regularization = 0
        
        # -- non-negative hinge regularization 
        
        if bool_regu_positive_mean == True:
            # regu_mean_pos = tf.reduce_sum(tf.maximum(0.0, -1.0*mean_v) + tf.maximum(0.0, -1.0*mean_x))
            self.regularization += regu_mean_pos
        
        # -- latent dependence parameter
        
        if bool_regu_latent_dependence == True:
            self.regularization += regu_latent_dependence
        
        # -- weights in gates  
        
        if bool_regu_gate == True:
            self.regularization += regu_gate
        
        # -- global logits
        
        # implicitly regularization on weights of gate functions
        if bool_regu_global_gate == True:
            
            # for regularization
            # [1 S]
            self.global_logits = tf.get_variable('global_logits', 
                                                 [1, self.num_src],
                                                 initializer = tf.contrib.layers.xavier_initializer())
            
            #                                        [B S]        [1 S]
            logits_diff_sq = tf.reduce_sum(tf.square(gate_logits - self.global_logits), 1)
            regu_global_logits = tf.reduce_sum(logits_diff_sq) + tf.nn.l2_loss(self.global_logits)
            
            self.regularization += regu_global_logits
        
        
        # -- gate smoothing
        
        if latent_prob_type != "none":
            
            if latent_prob_type == "pos_neg_diff_sq":
                
                # exact llk
                # self.latent_depend = -1.0*tf.reduce_sum(tf.log(latent_prob))
                
                # lower bound, comparable perforamcne to exact llk 
                self.latent_depend = 0.5*(tf.reduce_sum(tf.log(1.0 + tf.exp(-1.0*pos_logits))))\
                                                                + \
                                     0.5*(tf.reduce_sum(tf.log(1.0 + tf.exp(neg_logits)) - 1.0*neg_logits))
            
            else:
                
                # ! numertical stable version of log(sigmoid())
                # avoid the overflow of exp(-x) in sigmoid, when -x is positive large 
                # [B 1]
                self.latent_depend = (tf.reduce_sum(tf.log(1.0 + tf.exp(-1.0*tf.abs(latent_prob_logits))) \
                                           + tf.maximum(0.0, -1.0*latent_prob_logits))) 
        else:
            self.latent_depend = 0.0
            
            
        # ----- jump boosting 
        # firtst source, first element is the target 
        
        
        if bool_jump_boosting == True:
            
            # shape: [B 1]
            self.y_jump = tf.placeholder(tf.float32, [None, 1], name = 'y_jump')
            
            # self.x [S B T D]
            # [1 B T D]
            jump_src_x = tf.slice(self.x, [0, 0, 0, 0], [1, -1, -1, -1])
                    
            #[1 B]
            jump_magni, jump_regu_magni = multi_src_bilinear(jump_src_x,
                                                             [steps_x, dim_x],
                                                             'jump_mean',
                                                             bool_bias = True,
                                                             bool_scope_reuse = False,
                                                             num_src = 1)
            #[1 B]
            jump_logit, jump_regu_logit = multi_src_bilinear(jump_src_x,
                                                            [steps_x, dim_x],
                                                            'jump_gate',
                                                            bool_bias = True,
                                                            bool_scope_reuse = False,
                                                            num_src = 1)
            # [1 B] - > [B 1]
            self.py_jump = tf.transpose(tf.sigmoid(jump_logit)*jump_magni, [1, 0])
            
            self.loss_jump_boosting = tf.reduce_sum(tf.square(self.py_jump - self.y_jump)) + \
                                      self.l2*(jump_regu_magni + jump_regu_logit)
            
          
        
        
    #   initialize loss and optimization operations for training
    def train_ini(self):
        
        
        # ----- jump boosting
        
        if self.bool_jump_boosting == True:
            
            
            if self.optimization_lr_decay == True:
                
                global_step_boosting = tf.Variable(0, 
                                      trainable = False)
            
                learning_rate_boosting = tf.train.exponential_decay(self.lr, 
                                                           global_step_boosting,
                                                           decay_steps = self.optimization_lr_decay_steps, 
                                                           decay_rate = 0.96, 
                                                           staircase = True)
            else:
                
                learning_rate_boosting = self.lr
            
            
            optimizer_jump_boosting = myAdamOptimizer(learning_rate = learning_rate_boosting)
            
            
            if self.optimization_lr_decay == True:
                
                self.minimize_ops_boosting = optimizer_jump_boosting.minimize(self.loss_jump_boosting, 
                                                                              global_step = global_step_boosting)
            else:
                self.minimize_ops_boosting = optimizer_jump_boosting.minimize(self.loss_jump_boosting)
            
            
        
        
        
        
        
        # ----- loss 
        
        # loss, nllk
        if self.loss_type == 'mse':
            
            self.loss = tf.reduce_mean(tf.square(self.y - self.py_mean)) + \
                        0.1*self.l2*self.regularization + self.l2*self.regu_mean
                
            self.nllk = self.nllk_const
            
        elif self.loss_type == 'lk':
            
            self.loss = self.nllk_hetero + 0.1*self.l2*self.regularization + self.l2*(self.regu_mean + self.regu_var)
                        
            if self.bool_regu_l2_on_latent == True:
                self.loss += self.l2*self.latent_depend
            else:
                self.loss += self.latent_depend
            
            self.nllk = self.nllk_hetero
            
        # ?
        elif self.loss_type == 'lk_inv':
            
            self.loss = self.nllk_hetero_inv + 0.1*self.l2*self.regularization 
            
            
            
            if self.bool_regu_mean_var == True:
                
                if self.bool_imbalance_l2 == True:
                    self.loss += (self.l2*self.regu_mean + 100*self.l2*self.regu_var)
                else:
                    self.loss += self.l2*(self.regu_mean + self.regu_var)

                    
            
            if self.bool_regu_l2_on_latent == True:
                self.loss += self.l2*self.latent_depend
                
            else:
                self.loss += self.latent_depend
            
            
            self.nllk = self.nllk_hetero_inv
            
        '''
        elif self.loss_type == 'elbo':
            
            self.loss = self.nllk_elbo + 0.1*self.l2*self.regularization + self.l2*(self.regu_mean + self.regu_var)
            
            # negative log likelihood calculated through nllk_hetero_inv
            self.nllk = self.nllk_hetero_inv
            
        elif self.loss_type == 'simple_mix':
            
            # ?
            self.loss = self.nllk_mix + 0.1*self.l2*self.regularization + self.l2*(self.regu_mean + self.regu_var)
                        #self.nllk_gate + \
                        
            self.nllk = self.nllk_mix
         
        elif self.loss_type == 'simple_mix_inv':
            
            # ?
            self.loss = self.nllk_mix_inv + 0.1*self.l2*self.regularization + self.l2*(self.regu_mean + self.regu_var)
                        #self.nllk_gate + \
            
            self.nllk = self.nllk_mix_inv
            
        else:
            print('[ERROR] loss type')
        '''
        
        
        
        # ----- learning decay
        
        if self.optimization_lr_decay == True:
            
            global_step = tf.Variable(0, 
                                      trainable = False)
            
            tmp_learning_rate = tf.train.exponential_decay(self.lr, 
                                                           global_step,
                                                           decay_steps = self.optimization_lr_decay_steps, 
                                                           decay_rate = 0.96, 
                                                           staircase = True)
        else:
            tmp_learning_rate = self.lr
            
        
        
        # ----- optimizer
        
        
        if self.optimization_method == 'adam':
            
            tmp_train = myAdamOptimizer(learning_rate = tmp_learning_rate)
            
            #tmp_train = tf.train.AdamOptimizer(learning_rate = tmp_learning_rate)
        
        elif self.optimization_method == 'RMSprop':
            tmp_train = tf.train.RMSPropOptimizer(learning_rate = tmp_learning_rate)
        
        elif self.optimization_method == 'sgmcmc_RMSprop':
            
            tmp_train = tfp.optimizer.StochasticGradientLangevinDynamics(learning_rate = tmp_learning_rate,
                                                                          preconditioner_decay_rate = 0.99)
        
        if self.optimization_lr_decay == True:
            
            self.optimizer = tmp_train.minimize(self.loss, 
                                                global_step = global_step)
        else:
            self.optimizer = tmp_train.minimize(self.loss)
            
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        
        
    #   training on batch of data
    def train_batch(self, 
                    x, 
                    y,
                    global_step):
        
        data_dict = {}
        data_dict["x:0"] = x
        data_dict['y:0'] = y
        
        # record the global training step 
        self.training_step = global_step
        
        _, tmp_loss = self.sess.run([self.optimizer, self.loss],
                                    feed_dict = data_dict)
        
        # ----- alternating optimization for the boosting
        
        if self.bool_jump_boosting == True:
            
            tmp_py = self.sess.run(tf.get_collection('py_mean')[0],
                                        feed_dict = data_dict)
            
            #print("--------- test", np.shape(y), np.shape(tmp_py))
            
            y_jump = y - tmp_py
            data_dict["y_jump:0"] = y_jump
            
            _ = self.sess.run([self.minimize_ops_boosting],
                               feed_dict = data_dict)
        
        
        
        
        return tmp_loss
    
    
    def inference_ini(self):
        
        # evaluation metric
        
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
        
        # shape of x: [S, B, T, D]
        self.nnllk = self.nllk / tf.to_float(tf.shape(self.x)[1])
        
        # for model restore and inference
        tf.add_to_collection("rmse", self.rmse)
        tf.add_to_collection("mae", self.mae)
        tf.add_to_collection("mape", self.mape)
        tf.add_to_collection("nnllk", self.nnllk)
        
        tf.add_to_collection("py_mean", self.py_mean)
        tf.add_to_collection("py_var", self.py_var)
        tf.add_to_collection("py_gate", self.gates)
        
        tf.add_to_collection("py_mean_src", self.py_mean_src)
        tf.add_to_collection("py_var_src", self.py_var_src)
        
        if self.bool_jump_boosting == True:
            tf.add_to_collection("py_jump", self.py_jump)
        
    # epoch-wise
    def validation(self,
                   x,
                   y):
    
        # x: shape [S B T D]
        # y: [B 1]
        
        # -- validation inference
        
        data_dict = {}
        data_dict["x:0"] = x
        data_dict['y:0'] = y
        
        rmse, mae, mape, nnllk = self.sess.run([tf.get_collection('rmse')[0],
                                                tf.get_collection('mae')[0],
                                                tf.get_collection('mape')[0],
                                                tf.get_collection('nnllk')[0]],
                                                feed_dict = data_dict)
        
        # monitoring tuple during the training
        # [B S]      [B S]
        py_gate_src, py_mean_src, py_mean = self.sess.run([tf.get_collection('py_gate')[0], tf.get_collection('py_mean_src')[0],
                                                  tf.get_collection('py_mean')[0]],
                                                 feed_dict = data_dict)
        
        
        tmp_rmse, tmp_mae = 0.0, 0.0
        
        if self.bool_jump_boosting == True:
            
            py_jump = self.sess.run(tf.get_collection('py_jump')[0],
                                    feed_dict = data_dict)
            
            
            #print("--------- test", np.shape(y), np.shape(py_jump), np.shape(py_mean))
            
            tmp_rmse = func_rmse(np.squeeze(y), np.squeeze(py_mean + py_jump))
            tmp_mae = func_mae(np.squeeze(y), np.squeeze(py_mean + py_jump))
            
            # py_gate_src, py_mean_src = tmp_rmse, tmp_mae
            
        
        
        # -- validation monitoring
        
        # validation error log for early stopping
        # epoch-wise 
        self.log_step_error.append([self.training_step, [rmse, mae, mape, nnllk]])
        
        
        
        # -- validation for SG-MCMC
        
        if self.optimization_mode == "bayesian" and self.training_step >= self.burn_in_step:
            
            # [B 1]          [B S]
            py_mean, py_var, py_var_src = self.sess.run([tf.get_collection('py_mean')[0],
                                                         tf.get_collection('py_var')[0],
                                                         tf.get_collection('py_var_src')[0],
                                                         ],
                                                         feed_dict = data_dict)
            
            
            
            # for SG-MCMC
            self.py_mean_src_samples.append(py_mean_src)
            self.py_var_src_samples.append(py_var_src)
            self.py_gate_src_samples.append(py_gate_src)
            
            self.py_mean_samples.append(py_mean)
        
        # error metric tuple [rmse, mae, mape, nnllk], monitoring tuple []
        return rmse, mae, mape, nnllk, [tmp_rmse, tmp_mae]
    
    # hyper-para wise
    def validation_bayesian(self,
                            x, 
                            y):
        
        if len(self.py_mean_src_samples) == 0:
            return None
        
        # [A B S]
        # A: number of samples
        m_src_sample = np.asarray(self.py_mean_src_samples)
        v_src_sample = np.asarray(self.py_var_src_samples)
        g_src_sample = np.asarray(self.py_gate_src_samples)
        
        # [A B 1]
        m_sample = np.asarray(self.py_mean_samples)
        
        # [B]
        bayes_mean_src = np.mean(np.sum(m_src_sample*g_src_sample, axis = 2), axis = 0)
        
        bayes_mean = np.squeeze(np.mean(m_sample, axis = 0))
        
        #tmp_m_src_sample = m_src_sample**2
        #py_var = 
        
        tmpy = np.squeeze(y)
        
        return [func_rmse(tmpy, bayes_mean_src), func_mae(tmpy, bayes_mean_src), func_mape(tmpy, bayes_mean_src), \
                func_rmse(tmpy, bayes_mean), func_mae(tmpy, bayes_mean), func_mape(tmpy, bayes_mean) ]
    
    
    #   infer givn testing data
    def inference(self, 
                  x, 
                  y, 
                  bool_py_eval, 
                  ):
        
        # x: shape [S B T D]
        # y: [B 1]
        
        data_dict = {}
        data_dict["x:0"] = x
        data_dict['y:0'] = y
        
        rmse, mae, mape, nnllk = self.sess.run([tf.get_collection('rmse')[0],
                                                tf.get_collection('mae')[0],
                                                tf.get_collection('mape')[0],
                                                tf.get_collection('nnllk')[0]],
                                                feed_dict = data_dict)
        
        
        if bool_py_eval == True:
            
            # [B 1]  [B 1]   [B S]
            py_mean, py_var, py_gate_src, py_mean_src, py_var_src = self.sess.run([tf.get_collection('py_mean')[0],
                                                                                   tf.get_collection('py_var')[0],
                                                                                   tf.get_collection('py_gate')[0],
                                                                                   tf.get_collection('py_mean_src')[0],
                                                                                   tf.get_collection('py_var_src')[0]
                                                                                   ],
                                                                                   feed_dict = data_dict)
            
            
            if self.bool_jump_boosting == True:
                # [B 1]
                py_jump = self.sess.run(tf.get_collection('py_jump')[0],
                                        feed_dict = data_dict)
                
                py_mean += py_jump
                
                
        else:
            py_mean = None
            py_var = None
            py_gate_src = None
            py_mean_src = None
            py_var_src = None
            
        
        # error metric tuple [rmse, mae, mape, nnllk], py tuple []
        return [rmse, mae, mape, nnllk], [py_mean, py_var, py_mean_src, py_var_src, py_gate_src]
    
    
    def model_stored_id(self):
        
        return self.stored_step_id
    
    
    def model_saver(self, 
                    path,
                    step_id_to_store,
                    early_stop_bool,
                    early_stop_window,
                    ):
        
        
        # -- early stopping
        
        # self.log_step_error: [self.training_step, [rmse, mae, mape, nnllk]]
        
        if early_stop_bool == True:
            
            if len(self.stored_step_id) < 5 and self.training_step >= early_stop_window:
                
                tmp_last_error = self.log_step_error[-1][1][0]
                tmp_window_error = np.mean([i[1][0] for i in self.log_step_error[-1*(early_stop_window + 1):-1]])
                
                if tmp_window_error < tmp_last_error:
                    
                    if self.log_error_up_flag == False:
                        
                        self.stored_step_id.append(self.training_step - 1)
                    
                        saver = tf.train.Saver()
                        saver.save(self.sess, path)
                        
                        #  avoid consecutive upward 
                        self.log_error_up_flag = True
                
                        return True
                else:
                    
                    self.log_error_up_flag = False
        
        
        # -- best snapshots
        
        if len(step_id_to_store) != 0 and self.training_step in step_id_to_store:
            
            saver = tf.train.Saver()
            saver.save(self.sess, path)
            
            return True
        
        
        # -- bayesian ensembles
        
        elif self.optimization_mode == "bayesian" and self.training_step >= self.burn_in_step:
            
            saver = tf.train.Saver()
            saver.save(self.sess, path)
            
            return True
        
        return False
        
    
    #   restore the model from the files
    def model_restore(self,
                      path_meta, 
                      path_data):
        
        saver = tf.train.import_meta_graph(path_meta, 
                                           clear_devices = True)
        saver.restore(self.sess, 
                      path_data)
        return
    
    #   collect the optimized variable values
    def collect_coeff_values(self, 
                             vari_keyword):
        
        return [tf_var.name for tf_var in tf.trainable_variables() if (vari_keyword in tf_var.name)],\
               [tf_var.eval() for tf_var in tf.trainable_variables() if (vari_keyword in tf_var.name)]
    
    
    
class ensemble_inference(object):

    def __init__(self):
        
        # for SG-MCMC
        # [A B S]
        # A: number of samples
        self.py_mean_src_samples = []
        self.py_var_src_samples = []
        self.py_gate_src_samples = []
        
        self.py_mean_samples = []
        self.py_var_samples = []
        
        
    def add_samples(self, 
                    py_mean, 
                    py_var,
                    py_mean_src,
                    py_var_src, 
                    py_gate_src):
        
        # [A B S]         
        self.py_mean_src_samples.append(py_mean_src)
        self.py_var_src_samples.append(py_var_src)
        self.py_gate_src_samples.append(py_gate_src)
        
        # [A B 1]
        self.py_mean_samples.append(py_mean)
        self.py_var_samples.append(py_var)
        
        return
    
            
    def bayesian_inference(self, 
                           y):
        # y: [B 1]
        
        # [A B S]
        # A: number of samples
        
        m_src_sample = np.asarray(self.py_mean_src_samples)
        v_src_sample = np.asarray(self.py_var_src_samples)
        g_src_sample = np.asarray(self.py_gate_src_samples)
        
        # [A B 1]
        m_sample = np.asarray(self.py_mean_samples)
        
        
        # -- mean
        # [B]
        bayes_mean = np.mean(np.sum(m_src_sample*g_src_sample, axis = 2), axis = 0)
        #bayes_mean = np.squeeze(np.mean(m_sample, axis = 0))
        
        
        # -- total variance
        # [B]
        sq_mean = bayes_mean**2
        # [A B S]
        var_plus_sq_mean_src = v_src_sample + m_src_sample**2
        
        # [B]
        bayes_total_var = np.mean(np.sum(g_src_sample*var_plus_sq_mean_src, -1), 0) - sq_mean
        
        
        # -- volatility
        # heteroskedasticity
        # [B]                       [A B S]
        bayes_vola = np.mean(np.sum(g_src_sample*v_src_sample, -1), 0)
        
        
        # -- uncertainty 
        # without heteroskedasticity
        # [B]                       [A B S]
        bayes_unc = np.mean(np.sum(g_src_sample*(m_src_sample**2), -1), 0) - sq_mean
        
        
        # -- nnllk
        # normalized negative log-likelihood
        
        # [1 B 1]
        aug_y = np.expand_dims(y, axis=0)
        
        # [A B S]
        tmp_lk_src = np.exp(-0.5*(aug_y - m_src_sample)**2/(v_src_sample + 1e-5))/(np.sqrt(2.0*np.pi*v_src_sample) + 1e-5)
        # [B]                   [A B S]
        tmp_lk = np.mean(np.sum(g_src_sample*tmp_lk_src, -1), 0)
                                                                                                 
        nnllk = np.mean(-1.0*np.log(tmp_lk + 1e-5))
        
        
        # -- gate
        # [B S]                 [A B S]
        bayes_gate_src = np.mean(g_src_sample, axis = 0)
        
        bayes_gate_src_var = np.var(g_src_sample, axis = 0)
        
        
        '''
        # -- gate uncertainty 
        # infer by truncated Gaussian [0.0, 1.0]
        
        print("-----------------  begin to infer gate uncertainty...\n")
        
        
        # [B S]
        loc_ini = np.mean(g_src_sample, axis = 0) 
        scale_ini = np.std(g_src_sample, axis = 0)
        
        real_left = 0.0
        real_right = 1.0
        
        # [B S]
        left_ini = 1.0*(real_left - loc_ini)/(scale_ini + 1e-5)
        right_ini = 1.0*(real_right - loc_ini)/(scale_ini + 1e-5)
        
        
        # [A B S]
        norm_g_src_sample = 1.0*(g_src_sample - loc_ini)/(scale_ini + 1e-5)
        
        
        def func(p, r, xa, xb):
            return truncnorm.nnlf(p, r)
        
        def constraint(p, r, xa, xb):
            a, b, loc, scale = p
            return np.array([a*scale + loc - xa, b*scale + loc - xb])
        

        # [B S A]       
        batch_src_sample = np.transpose(norm_g_src_sample, [1, 2, 0])
        
        num_batch = np.shape(batch_src_sample)[0]
        num_src = np.shape(batch_src_sample)[1]
        
        
        # [B S 2]: [B S 0] mean, [B S 1] var
        tr_gau_batch_src = []
        
        for tmp_ins in range(num_batch):
            
            tr_gau_batch_src.append([])
            
            for tmp_src in range(num_src):
                
                tmp_para = fmin_slsqp(func, 
                                      [left_ini[tmp_ins][tmp_src], right_ini[tmp_ins][tmp_src], loc_ini[tmp_ins][tmp_src], scale_ini[tmp_ins][tmp_src]], 
                                      f_eqcons = constraint, 
                                      args = (batch_src_sample[tmp_ins][tmp_src], real_left, real_right),
                                      iprint = False, 
                                      iter = 1000)
                
                tr_gau_batch_src[-1].append([tmp_para[2], tmp_para[3]])
                
        '''
        
        # -- output
        tmpy = np.squeeze(y)
        
        # error tuple [], prediction tuple []
        return [func_rmse(tmpy, bayes_mean), func_mae(tmpy, bayes_mean), func_mape(tmpy, bayes_mean), nnllk],\
               [bayes_mean, bayes_total_var, bayes_vola, bayes_unc, bayes_gate_src, bayes_gate_src_var, g_src_sample]
        
    
    
    
    
    
    