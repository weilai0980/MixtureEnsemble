#!/usr/bin/python

import numpy as np
from scipy.stats import truncnorm
from scipy.optimize import fmin_slsqp
from scipy.stats import norm

import tensorflow as tf
# import tensorflow_probability as tfp

from sklearn.neighbors.kde import KernelDensity

# local packages
from utils_libs import *
from utils_linear_units import *
from utils_rnn_units import *
from utils_training import *
from utils_optimization import *

# reproducibility by fixing the random seed
#np.random.seed(1)
#tf.set_random_seed(1)

# ----- Mixture statistic -----

class mixture_statistic():
    
    def __init__(self, 
                 session, 
                 loss_type,
                 num_src, 
                 hyper_para_dict,
                 model_type):
        '''
        Argu.:
          session: tensorflow session
          loss_type: string, type of loss functions, {mse, lk, lk_inv, elbo, ...}
          num_src: number of sources in X
          hyper_para_dict:
          model_type:
        '''
        self.sess = session
        self.loss_type = loss_type
        self.num_src = num_src 
        self.model_type = model_type
        
        self.hyper_para_dict = hyper_para_dict
        
        self.stored_step_id = []
        
    def network_ini(self,
                    hyper_para_dict,
                    x_dim,
                    x_steps,
                    x_bool_common_factor,
                    model_type,
                    model_distr_type,
                    model_distr_para,
                    model_var_type,
                    model_para_share_type,
                    bool_regu_mean,
                    bool_regu_var,
                    bool_regu_gate,
                    bool_regu_positive_mean,
                    bool_bias_mean,
                    bool_bias_var,
                    bool_bias_gate,
                    optimization_method,
                    optimization_lr_decay,
                    optimization_lr_decay_steps,
                    optimization_burn_in_step,
                    optimization_warmup_step):
        '''
        Argu.:
        
        hyper_para_dict:
        
           lr: float, learning rate
           l2: float, l2 regularization
           batch_size: int
           
           lstm_size: int
           dense_num: int
           use_hidden_before_dense: bool
        
        x_dim:  dimension values for each component in X
        x_steps: sequence length values for each component in X
        model_distr_type: string, type of the distribution of the target variable
        model_distr_para: set of parameters of the associated distribution
                          "gaussian": []
                          "student_t": [nu] >= 3
                          
        model_var_type: square or exponential
        bool_regu_positive_mean: if regularization of positive mean 
        bool_regu_gate: if regularization the gate functions
        
        latent_dependence: string, dependence of latent logits, "none", "independent", "markov"
        latent_prob_type: string, probability calculation of latent logits, "none", "scalar", "vector", "matrix"
        bool_bias_mean: have bias in mean prediction
        bool_bias_var: have bias in variance prediction
        bool_bias_gate: have bias in gate prediction
        
        Dictionary of abbreviation:
           nllk: negative log likelihood
           hetero: heteroskedasticity
           inv: inversed
           const: constant
           indi: individual
           py: predicted y
           src: source
           var: variance
         
           A: number of samples
           S: source 
           B: batch size
           T: time steps
           D: data dimensionality at each time step
        '''
        # ----- fix the random seed to reproduce the results
        #np.random.seed(1)
        #tf.set_random_seed(1)
        
        # ----- ini 
        self.hyper_para_dict = hyper_para_dict
        
        # build the network graph 
        self.lr = self.hyper_para_dict["lr"]
        self.distr_type = model_distr_type
        
        # initialize placeholders
        # y: [B 1]
        self.y = tf.placeholder(tf.float32, 
                                [None, 1], 
                                name = 'y')
        # x: [S, [B T D]]
        self.x = []
        for i in range(self.num_src):
            self.x.append(tf.placeholder(tf.float32, 
                                         [None, x_steps[i], x_dim[i]], 
                                         name = 'x' + str(i)))
        if model_type == "rnn":
            self.keep_prob = tf.placeholder(tf.float32, 
                                            shape = (), 
                                            name = 'keep_prob')
        # -- hyper-parameters
        self.optimization_method = optimization_method
        self.optimization_lr_decay = optimization_lr_decay
        self.optimization_lr_decay_steps = optimization_lr_decay_steps 
        self.optimization_warmup_step = optimization_warmup_step
        
        self.bool_regu_mean = bool_regu_mean
        self.bool_regu_var = bool_regu_var
        self.bool_regu_gate = bool_regu_gate
        
        self.training_step = 0
        
        # ----- individual models
        '''
            # x: [S, [B T D]]
            self.pre_x = []
            self.cur_x = []
            for i in range(self.num_src):
                self.pre_x.append(tf.slice(self.x[i], [0, 0, 0], [-1, x_steps[i]-1, -1]))
                self.cur_x.append(tf.slice(self.x[i], [0, 1, 0], [-1, x_steps[i]-1, -1]))
        '''
        if model_type == "linear":
            #[S B]
            tmp_mean, regu_mean, tmp_var, regu_var, tmp_logit, regu_gate = multi_src_predictor_linear(x = self.x, 
                                                                                                      n_src = self.num_src, 
                                                                                                      steps = x_steps, 
                                                                                                      dim = x_dim, 
                                                                                                      bool_bias = [bool_bias_mean, bool_bias_var, bool_bias_gate], 
                                                                                                      bool_scope_reuse= [False, False, False], 
                                                                                                      str_scope = "linear", 
                                                                                                      para_share_logit = model_para_share_type, 
                                                                                                      bool_common_factor = x_bool_common_factor,
                                                                                                      common_factor_dim = 0)
            #int(self.hyper_para_dict['factor_size'])
        elif model_type == "rnn":
            #[S B]
            tmp_mean, regu_mean, tmp_var, regu_var, tmp_logit, regu_gate = multi_src_predictor_rnn(x = self.x,
                                                                                                   n_src = self.num_src,
                                                                                                   bool_bias = [bool_bias_mean, bool_bias_var, bool_bias_gate],
                                                                                                   bool_scope_reuse = [False, False, False],
                                                                                                   str_scope = "rnn",
                                                                                                   rnn_size_layers = [int(self.hyper_para_dict['rnn_size'])],
                                                                                                   rnn_cell_type = "lstm",
                                                                                                   dropout_keep = self.keep_prob,
                                                                                                   dense_num = int(self.hyper_para_dict['dense_num']),
                                                                                                   max_norm_cons = self.hyper_para_dict['max_norm_cons'])
        # ----- individual means and variance
        
        # -- mean
        mean_stack = tf.transpose(tmp_mean, [1, 0])
        
        # -- variance
        if model_var_type == "square":
            # square
            var_stack = tf.transpose(tf.square(tmp_var), [1, 0])
            inv_var_stack = tf.transpose(tf.square(tmp_var), [1, 0])
        elif model_var_type == "exp":
            # exp
            var_stack = tf.transpose(tf.exp(tmp_var), [1, 0])
            inv_var_stack = tf.transpose(tf.exp(tmp_var), [1, 0])
            
        # ----- gates
        
        # [B S]
        gate_logits = tf.transpose(tmp_logit, [1, 0])
        # obtain the gate values
        self.gates = tf.nn.softmax(gate_logits, axis = -1)
        
        # ----- mixture mean, variance and nllk
        
        if model_distr_type == 'normal':
            
            # -- mean
            # component mean
            # [B S] 
            self.py_mean_src = mean_stack
            # mixture mean
            # [B 1]                      [B S]        [B S]
            self.py_mean = tf.reduce_sum(mean_stack * self.gates, 1, keepdims = True)
            
            # --
            if self.loss_type == 'heter_lk':
                
                # --- loss
                # negative log likelihood
                # [B S]
                lk_src = tf.exp(-0.5*tf.square(self.y-mean_stack)/(var_stack+1e-5))/(tf.sqrt(2.0*np.pi*var_stack)+1e-5)
                lk = tf.reduce_sum(tf.multiply(lk_src, self.gates), axis = -1) 
                self.nnllk_loss = tf.reduce_mean(-1.0*tf.log(lk + 1e-5))
                
                # --- evaluation
                # nnllk
                self.nnllk = self.nnllk_loss
                
                # component variance
                self.py_var_src = var_stack
                # variance
                var_plus_sq_mean_src = self.py_var_src + tf.square(self.py_mean_src)
                # [B 1]                                 [B S]          [B S]
                self.py_var = tf.reduce_sum(tf.multiply(var_plus_sq_mean_src, self.gates), 1, keepdims = True) - tf.square(self.py_mean)
                
            elif self.loss_type == 'heter_lk_inv':
                
                # --- loss
                # negative log likelihood
                # [B S]
                lk_src = tf.exp(-0.5*tf.square(self.y - mean_stack)*inv_var_stack)*tf.sqrt(0.5/np.pi*inv_var_stack)
                lk = tf.reduce_sum(tf.multiply(lk_src, self.gates) , axis = -1) 
                self.nnllk_loss = tf.reduce_mean(-1.0*tf.log(lk+ 1e-5))
                
                # --- evaluation
                # nnllk
                self.nnllk = self.nnllk_loss
                
                # component variance
                self.py_var_src = 1.0/(inv_var_stack + 1e-5)
                # variance
                var_plus_sq_mean_src = self.py_var_src + tf.square(self.py_mean_src)
                # [B 1]
                self.py_var = tf.reduce_sum(tf.multiply(var_plus_sq_mean_src, self.gates), 1, keepdims = True) - tf.square(self.py_mean)
                
            # elbo: evidence lower bound optimization    
            elif self.loss_type == 'heter_elbo':
                
                # component variance
                self.py_var_src = 1.0/(inv_var_stack + 1e-5)
                
                # variance
                sq_mean_stack = 1.0/(inv_var_stack + 1e-5) + tf.square(mean_stack)
                mix_sq_mean = tf.reduce_sum(tf.multiply(sq_mean_stack, self.gates), 1, keepdims = True)
                # [B 1]
                self.py_var = mix_sq_mean - tf.square(self.py_mean)
                
                # negative log likelihood
                # based on lk_inv
                # [B S]
                lk_src = tf.exp(-0.5*tf.square(self.y-mean_stack)*inv_var_stack)*tf.sqrt(0.5/np.pi*inv_var_stack)
            
                lk = tf.multiply(lk_src, self.gates) 
                self.nllk = tf.reduce_mean(-1.0*tf.log(tf.reduce_sum(lk, axis = -1) + 1e-5))
                
                # [B 1] - [B S]
                tmp_nllk_bound = .5*tf.square(self.y - mean_stack)*inv_var_stack - 0.5*tf.log(inv_var_stack + 1e-5) + 0.5*tf.log(2*np.pi)
        
                self.nllk_bound = tf.reduce_sum(tf.reduce_sum(self.gates*tmp_nllk_bound, -1)) 
            
            elif self.loss_type == 'mse':
                
                # component variance
                self.py_var_src = tf.constant(1.0, shape = [1, self.num_src])
                
                # variance
                sq_mean_stack = 1.0 + tf.square(mean_stack)
                # [B 1]
                mix_sq_mean = tf.reduce_sum(tf.multiply(sq_mean_stack, self.gates), 1, keepdims = True)
                # [B 1]
                self.py_var = mix_sq_mean - tf.square(self.py_mean)
                
                # negative log likelihood
                # [B S]
                # ? variance of constant 1 
                lk_src = tf.exp(-0.5*tf.square(self.y - mean_stack))/(2.0*np.pi)**0.5
            
                lk = tf.multiply(lk_src, self.gates) 
                self.nllk = tf.reduce_sum(-1.0*tf.log(tf.reduce_sum(lk, axis = -1) + 1e-5))
                
        elif model_distr_type == 'log_normal_logOpt_linearComb':
            
            self.log_y = tf.math.log(self.y + 1e-5)
            
            if self.loss_type == 'heter_lk_inv':
                
                # --- loss
                # nnllk: normalized negative log likelihood
                # [B S]
                # in the log scale
                lk_src = tf.exp(-0.5*tf.square(self.log_y - mean_stack)*inv_var_stack)*tf.sqrt(0.5/np.pi*inv_var_stack)
                lk = tf.multiply(lk_src, self.gates) 
                self.nnllk_loss = tf.reduce_mean(-1.0*tf.log(tf.reduce_sum(lk, axis = -1) + 1e-5))
                
                # --- evaluation
                log_py_var_src = 1.0/(inv_var_stack + 1e-5)
                log_py_var_src_inv = inv_var_stack
                log_py_mean_src = mean_stack
                
                # component mean
                # [B S]
                self.py_mean_src = tf.exp(log_py_mean_src + log_py_var_src/2.0)
                # mixture mean
                # [B 1]                      [B S]        [B S]
                self.py_mean = tf.reduce_sum(self.py_mean_src*self.gates, 1, keepdims = True)
                
                # component variance
                # [B S]
                self.py_var_src = (tf.exp(log_py_var_src) - 1.0)*tf.exp(2.0*log_py_mean_src + log_py_var_src)
                # mixture variance
                # [B 1]
                var_plus_sq_mean_src = self.py_var_src + tf.square(self.py_mean_src)
                self.py_var = tf.reduce_sum(tf.multiply(var_plus_sq_mean_src, self.gates), 1, keepdims = True) - tf.square(self.py_mean)
                
                # lk: likelihood
                # in the linear scale
                tmp_lk_src = tf.exp(-0.5*tf.square(self.log_y - log_py_mean_src)*log_py_var_src_inv)*tf.sqrt(0.5/np.pi*log_py_var_src_inv)/(1.0*self.y+1e-5)
                # [B]
                self.lk = tf.reduce_sum(tf.multiply(tmp_lk_src, self.gates), axis = -1)
                self.nnllk = tf.reduce_mean(-1.0*tf.log(self.lk + 1e-5))
                
                # --- temporary
                
                # mixture mean of log_y
                # [B 1]                      [B S]        [B S]
                self.log_py_mean = tf.reduce_sum(log_py_mean_src*self.gates, 1, keepdims = True)
                
                # mixture variance of log_y
                # [B 1]
                var_plus_sq_mean_src = log_py_var_src + tf.square(log_py_mean_src)
                self.log_py_var = tf.reduce_sum(tf.multiply(var_plus_sq_mean_src, self.gates), 1, keepdims = True) - tf.square(self.log_py_mean)
                
                self.log_py_mean_src = log_py_mean_src
                self.log_py_var_src  = log_py_var_src
                
        elif model_distr_type == 'log_normal_linearOpt_linearComb':
            
            self.log_y = tf.math.log(self.y + 1e-5)
            
            if self.loss_type == 'heter_lk_inv':
                
                # --- loss
                # nnllk: normalized negative log likelihood
                # [B S]
                # in the log scale 
                lk_src = tf.exp(-0.5*tf.square(self.log_y - mean_stack)*inv_var_stack)*tf.sqrt(0.5/np.pi*inv_var_stack)/(1.0*self.y+1e-5)
                lk = tf.multiply(lk_src, self.gates) 
                self.nnllk_loss = tf.reduce_mean(-1.0*tf.log(tf.reduce_sum(lk, axis = -1) + 1e-5))
                
                # --- evaluation
                log_py_var_src = 1.0/(inv_var_stack + 1e-5)
                log_py_var_src_inv = inv_var_stack
                log_py_mean_src = mean_stack
                
                # component mean
                # [B S]
                self.py_mean_src = tf.exp(log_py_mean_src + log_py_var_src/2.0)
                # mixture mean
                # [B 1]                      [B S]        [B S]
                self.py_mean = tf.reduce_sum(self.py_mean_src*self.gates, 1, keepdims = True)
                
                # component variance
                # [B S]
                self.py_var_src = (tf.exp(log_py_var_src) - 1.0)*tf.exp(2.0*log_py_mean_src + log_py_var_src)
                # mixture variance
                # [B 1]
                var_plus_sq_mean_src = self.py_var_src + tf.square(self.py_mean_src)
                self.py_var = tf.reduce_sum(tf.multiply(var_plus_sq_mean_src, self.gates), 1, keepdims = True) - tf.square(self.py_mean)
                
                # lk: likelihood
                # in the linear scale
                tmp_lk_src = tf.exp(-0.5*tf.square(self.log_y - log_py_mean_src)*log_py_var_src_inv)*tf.sqrt(0.5/np.pi*log_py_var_src_inv)/(1.0*self.y+1e-5)
                # [B]
                self.lk = tf.reduce_sum(tf.multiply(tmp_lk_src, self.gates), axis = -1)
                self.nnllk = tf.reduce_mean(-1.0*tf.log(self.lk + 1e-5))
                
        elif model_distr_type == 'log_normal_logOpt_logComb':
            
            self.log_y = tf.math.log(self.y + 1e-5)
            
            if self.loss_type == 'heter_lk_inv':
                
                # --- loss
                # nnllk: normalized negative log likelihood
                # [B S]
                lk_src = tf.exp(-0.5*tf.square(self.log_y - mean_stack)*inv_var_stack)*tf.sqrt(0.5/np.pi*inv_var_stack)
                lk = tf.multiply(lk_src, self.gates) 
                self.nnllk_loss = tf.reduce_mean(-1.0*tf.log(tf.reduce_sum(lk, axis = -1) + 1e-5))
                
                # --- evaluation
                log_py_var_src = 1.0/(inv_var_stack + 1e-5)
                log_py_var_src_inv = inv_var_stack
                log_py_mean_src = mean_stack
                
                # mixture mean of log_y
                # [B 1]                      [B S]        [B S]
                log_py_mean = tf.reduce_sum(log_py_mean_src*self.gates, 1, keepdims = True)
                
                # mixture variance of log_y
                # [B 1]
                var_plus_sq_mean_src = log_py_var_src + tf.square(log_py_mean_src)
                log_py_var = tf.reduce_sum(tf.multiply(var_plus_sq_mean_src, self.gates), 1, keepdims = True) - tf.square(log_py_mean)
                
                self.py_mean_src = tf.exp(log_py_mean_src + log_py_var_src/2.0)
                self.py_mean = tf.exp(log_py_mean + log_py_var/2.0)
                
                self.py_var_src = (tf.exp(log_py_var_src) - 1.0)*tf.exp(2.0*log_py_mean_src + log_py_var_src)
                self.py_var = (tf.exp(log_py_var) - 1.0)*tf.exp(2.0*log_py_mean + log_py_var)
                
                # lk: likelihood
                # [B]
                self.lk = tf.exp(-0.5*tf.square(self.log_y - log_py_mean)/(1.0*log_py_var))*tf.sqrt(0.5/np.pi/log_py_var)/(1.0*self.y+1e-5)
                self.nnllk = tf.reduce_mean(-1.0*tf.log(self.lk + 1e-5))
        
        elif model_distr_type == 'trunc-normal':
            
            l = 0.0
            r = np.inf
            
            if self.loss_type == 'heter_lk_inv':
                
                # --- loss
                # nnllk: normalized negative log likelihood
                # in the original scale of y
                # [B S]
                lk_src = tf.exp(-0.5*tf.square(self.y - mean_stack)*inv_var_stack)*tf.sqrt(0.5/np.pi*inv_var_stack)
                lk = tf.reduce_sum(tf.multiply(lk_src, self.gates), axis = -1)  
                self.nnllk_loss = tf.reduce_mean(-1.0*tf.log(lk + 1e-5))
                
                # --- evaluation
                
                # [B S]
                a_src = (l - mean_stack)*inv_var_stack
                b_src = (r - mean_stack)*inv_var_stack
                y_norm_src = (y - mean_stack)*inv_var_stack
                
                # lk: likelihood
                self.lk = lk
                self.nnllk = self.nnllk_loss
                
                tmp_var_src = 1.0/(inv_var_stack + 1e-5)
                
                # component mean
                # [B S]
                self.py_mean_src = mean_stack
                
                # mixture mean
                # [B 1]                      [B S]        [B S]
                self.py_mean = tf.reduce_sum(self.py_mean_src*self.gates, 1, keepdims = True)
                
                # component variance
                # [B S]
                self.py_var_src = (tf.exp(tmp_var_src) - 1.0)*tf.exp(2.0*mean_stack + tmp_var_src)
                
                # mixture variance
                # [B 1]
                sq_mean_stack = self.py_var_src + tf.square(self.py_mean_src)
                mix_sq_mean = tf.reduce_sum(tf.multiply(sq_mean_stack, self.gates), 1, keepdims = True)
                self.py_var = mix_sq_mean - tf.square(self.py_mean)
                
        # ----- regularization
        # [S]
        self.regu_var = regu_var 
        self.regu_mean = regu_mean
        self.regu_gate = regu_gate
        
    #   initialize loss and optimization operations for training
    def train_ini(self):
        
        # ----- loss 
        self.monitor = []
        
        # mse
        if self.loss_type == 'mse':
            self.loss = tf.reduce_mean(tf.square(self.y - self.py_mean)) + self.l2*self.regu_mean
            self.monitor = [tf.reduce_mean(tf.square(self.y - self.py_mean)), self.l2*self.regu_mean] 
        # nnllk        
        elif self.loss_type in ['heter_lk', 'heter_lk_inv', 'homo_lk_inv']:
            
            self.loss = self.nnllk_loss
            self.monitor = [self.loss]
            
            if self.bool_regu_mean == True:
                self.loss += ( self.hyper_para_dict["l2_mean"]*self.regu_mean )
                self.monitor.append(self.hyper_para_dict["l2_mean"]*self.regu_mean)
                
            if self.bool_regu_var == True:
                self.loss += (self.hyper_para_dict["l2_var"]*self.regu_var)
                self.monitor.append(self.hyper_para_dict["l2_var"]*self.regu_var)
                
            if self.bool_regu_gate == True:
                self.loss += (self.hyper_para_dict["l2_gate"]*self.regu_gate)
                self.monitor.append((self.hyper_para_dict["l2_gate"]*self.regu_gate))
                
        # self.gates [B S]
        self.monitor.append(tf.slice(self.gates, [0, 0], [3, -1]))
        
        # ----- learning rate set-up
        tf_learning_rate = tf.constant(value = self.lr, 
                                       shape = [], 
                                       dtype = tf.float32)
        global_step = tf.train.get_or_create_global_step()
        
        # -- decay
        if self.optimization_lr_decay == True:
            decay_learning_rate = tf.train.exponential_decay(tf_learning_rate, 
                                                             global_step,
                                                             decay_steps = self.optimization_lr_decay_steps, 
                                                             decay_rate = 0.96, 
                                                             staircase = True)
        else:
            decay_learning_rate = tf_learning_rate
        
        # -- learning rate warm-up
        # ref: https://github.com/google-research/bert/blob/ffbda2a1aafe530525212d13194cc84d92ed0313/optimization.py#L29-L65
        if self.optimization_warmup_step > 0:
            
            global_steps_int = tf.cast(global_step, 
                                       tf.int32)
            warmup_steps_int = tf.constant(self.optimization_warmup_step, 
                                           dtype = tf.int32)
            
            global_steps_float = tf.cast(global_steps_int,
                                         tf.float32)
            warmup_steps_float = tf.cast(warmup_steps_int, 
                                         tf.float32)
            
            warmup_percent_done = global_steps_float / warmup_steps_float
            warmup_learning_rate = tf_learning_rate * warmup_percent_done
                
            is_warmup = tf.cast(global_steps_int < warmup_steps_int, 
                                tf.float32)        
            optimizer_lr = ((1.0 - is_warmup) * decay_learning_rate + is_warmup * warmup_learning_rate)
        
        else:
            optimizer_lr = decay_learning_rate
        
        # ----- optimizer
        
        # -- conventional 
        if self.optimization_method == 'adam':
            train_optimizer = myAdamOptimizer(learning_rate = optimizer_lr)
            
        elif self.optimization_method == 'adam_origin':
            train_optimizer = tf.train.AdamOptimizer(learning_rate = optimizer_lr)
            
        elif self.optimization_method == 'RMSprop':
            train_optimizer = myRMSprop(learning_rate = optimizer_lr)
            
        elif self.optimization_method == 'RMSprop_origin':
            train_optimizer = tf.train.RMSPropOptimizer(learning_rate = optimizer_lr)
            
        elif self.optimization_method == 'sgd':
            train_optimizer = tf.train.MomentumOptimizer(learning_rate = optimizer_lr,
                                                         momentum = 0.9,
                                                         use_nesterov = True)
        elif self.optimization_method == 'adamW':
            # ref.: "Fixing Weight Decay Regularization in Adam", https://arxiv.org/abs/1711.05101
            train_optimizer = tf.contrib.opt.AdamWOptimizer(weight_decay = self.l2,
                                                            learning_rate = optimizer_lr)
        # -- SG-MCMC
        # stochastic gradient Monto-Carlo Markov Chain
        elif self.optimization_method == 'sg_mcmc_adam':
            train_optimizer = sg_mcmc_adam(learning_rate = optimizer_lr)
            
        elif self.optimization_method == 'sg_mcmc_adam_revision':
            train_optimizer = sg_mcmc_adam_revision(learning_rate = optimizer_lr)
            
        elif self.optimization_method == 'sg_mcmc_RMSprop':
            train_optimizer = sg_mcmc_RMSprop(learning_rate = optimizer_lr)
            
        elif self.optimization_method == 'sgld':
            train_optimizer = StochasticGradientLangevinDynamics(learning_rate = optimizer_lr)
            
        else:
            print("\n --- OPTIMIZER ERROR ---- \n")
        
        # -- training operation
        self.train_op = train_optimizer.minimize(self.loss,
                                                 global_step = global_step)
        # -- initialize the graph
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        
    #   training on batch of data
    def train_batch(self, 
                    x, 
                    y,
                    global_step):
        '''
        Argu.:
          global_step: in epoch 
        '''
        data_dict = {}
        data_dict["y:0"] = y
        
        # x: [S, [B T D]]
        for i in range(len(x)):
            data_dict["x" + str(i) + ":0"] = x[i]
        
        if self.model_type == "rnn":
            data_dict["keep_prob:0"] = self.hyper_para_dict['dropout_keep_prob']
            
        # record the global training step 
        self.training_step = global_step
        
        _ = self.sess.run(self.train_op,
                          feed_dict = data_dict)
        return
    
    def inference_ini(self):
        
        # --- inference output and error metric
        
        # RMSE
        self.rmse = tf.sqrt(tf.losses.mean_squared_error(self.y, self.py_mean))
        # MAE
        self.mae = tf.reduce_mean(tf.abs(self.y - self.py_mean))
        # MAPE: based on ground-truth y
        mask = tf.greater(tf.abs(self.y), 1e-5)
        y_mask = tf.boolean_mask(self.y, mask)
        y_hat_mask = tf.boolean_mask(self.py_mean, mask)
        self.mape = tf.reduce_mean(tf.abs((y_mask - y_hat_mask)/(y_mask + 1e-5)))
        
        # --- for model restore and inference
        
        # error metric
        tf.add_to_collection("rmse", self.rmse)
        tf.add_to_collection("mae", self.mae)
        tf.add_to_collection("mape", self.mape)
        tf.add_to_collection("nnllk", self.nnllk)
        
        # monitor metric
        for tmp_idx, tmp_monitor_metric in enumerate(self.monitor):
            tf.add_to_collection(str(tmp_idx), tmp_monitor_metric)
        
        # prediction
        tf.add_to_collection("py_mean", self.py_mean)
        tf.add_to_collection("py_var", self.py_var)
        tf.add_to_collection("py_gate", self.gates)
        tf.add_to_collection("py_mean_src", self.py_mean_src)
        tf.add_to_collection("py_var_src", self.py_var_src)
        tf.add_to_collection("py_lk", self.lk)
        
        # temporary
        tf.add_to_collection("log_py_mean", self.log_py_mean)
        tf.add_to_collection("log_py_var",  self.log_py_var)
                
    # step-wise
    def validation(self,
                   x,
                   y,
                   step,
                   bool_end_of_epoch):
        '''
        Argu.:
          x: [S [B T D]]
          y: [B 1]
        '''
        if bool_end_of_epoch == True:
            
            # data preproc
            data_dict = {}
            data_dict["y:0"] = y
            
            for i in range(len(x)):
                data_dict["x" + str(i) + ":0"] = x[i]
                                
            if self.model_type == "rnn":
                data_dict["keep_prob:0"] = 1.0
            
            # errors           
            rmse, mae, mape, nnllk = self.sess.run([tf.get_collection('rmse')[0],
                                                    tf.get_collection('mae')[0],
                                                    tf.get_collection('mape')[0],
                                                    tf.get_collection('nnllk')[0]],
                                                   feed_dict = data_dict)
            # monitor metric
            monitor_metric = self.sess.run([tf.get_collection(str(tmp_idx))[0] for tmp_idx in range(len(self.monitor))],
                                           feed_dict = data_dict)
            
            # error metric tuple: [rmse, mae, mape, nnllk]
            # monitor tuple: []
            return [rmse, mae, mape, nnllk], monitor_metric
        
        return None, None
        
    # infer given testing data
    def inference(self, 
                  x, 
                  y,
                  bool_py_eval):
        '''
        Argu.:
          x: [S [B T D]]
          y: [B 1]
        '''
        # --
        data_dict = {}
        data_dict['y:0'] = y
        
        for i in range(len(x)):
            data_dict["x" + str(i) + ":0"] = x[i]
            
        if self.model_type == "rnn":
            data_dict["keep_prob:0"] = 1.0
        
        rmse, mae, mape, nnllk = self.sess.run([tf.get_collection('rmse')[0],
                                                tf.get_collection('mae')[0],
                                                tf.get_collection('mape')[0],
                                                tf.get_collection('nnllk')[0]],
                                               feed_dict = data_dict)
        if bool_py_eval == True:
            # [B 1]  [B 1]   [B S]
            py_mean, py_var, py_gate_src, py_mean_src, py_var_src, py_lk, log_py_mean, log_py_var = self.sess.run([tf.get_collection('py_mean')[0],
                                                                                          tf.get_collection('py_var')[0],
                                                                                          tf.get_collection('py_gate')[0],
                                                                                          tf.get_collection('py_mean_src')[0],
                                                                                          tf.get_collection('py_var_src')[0], 
                                                                                          tf.get_collection('py_lk')[0],
                                                                                          tf.get_collection('log_py_mean')[0],
                                                                                          tf.get_collection('log_py_var')[0],
                                                                                         ],
                                                                                         feed_dict = data_dict)
        else:
            py_mean = None
            py_var = None
            py_gate_src = None
            py_mean_src = None
            py_var_src = None
            py_lk = None
            log_py_mean = None 
            log_py_var = None
        
        # error metric tuple [rmse, mae, mape, nnllk], py tuple []
        return [rmse, mae, mape, nnllk], [py_mean, py_var, py_mean_src, py_var_src, py_gate_src, py_lk, log_py_mean, log_py_var]
    
    def model_stored_id(self):
        return self.stored_step_id
    
    def model_saver(self, 
                    path,
                    epoch,
                    step,
                    top_snapshots,
                    bayes_snapshots,
                    early_stop_bool,
                    early_stop_window, 
                    tf_saver):
        # -- early stopping
        # self.log_step_error: [self.training_step, [rmse, mae, mape, nnllk]]
        # -- best snapshots
        if len(top_snapshots) != 0 and epoch in top_snapshots:
            tf_saver.save(self.sess, path)
            return "best_snapshots"
        
        # -- bayesian ensembles
        elif len(bayes_snapshots) != 0 and epoch in bayes_snapshots:            
            tf_saver.save(self.sess, path)
            return "bayeisan_snapshots"
        
        return None
        
    #   restore the model from the files
    def model_restore(self,
                      path_meta, 
                      path_data, 
                      saver):
        saver.restore(self.sess, 
                      path_data)
        return
    
'''
#   collect the optimized variable values
def collect_coeff_values(self, vari_keyword):
        
        return [tf_var.name for tf_var in tf.trainable_variables() if (vari_keyword in tf_var.name)],\
               [tf_var.eval() for tf_var in tf.trainable_variables() if (vari_keyword in tf_var.name)]
'''

from utils_inference import *

def testing(retrain_snapshots,
            retrain_ids,
            xts,
            yts,
            file_path,
            bool_instance_eval,
            loss_type,
            num_src,
            snapshot_features, 
            hpara_dict, 
            para_model_type, 
            para_loss_type):
    
    # ensemble of model snapshots
    infer = ensemble_inference()
    
    with tf.device('/device:GPU:0'):
        
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        
        for tmp_idx, tmp_retrain_id in enumerate(retrain_ids):
            
            for tmp_model_id in retrain_snapshots[tmp_idx]:
                
                # path of the stored models 
                tmp_meta = file_path + para_model_type + '_' + str(tmp_retrain_id) + '_' + str(tmp_model_id) + '.meta'
                tmp_data = file_path + para_model_type + '_' + str(tmp_retrain_id) + '_' + str(tmp_model_id)
        
                # clear graph
                tf.reset_default_graph()
                saver = tf.train.import_meta_graph(tmp_meta, 
                                                   clear_devices = True)
                sess = tf.Session(config = config)
                
                model = mixture_statistic(session = sess, 
                                          loss_type = para_loss_type,
                                          num_src = num_src,
                                          hyper_para_dict = hpara_dict, 
                                          model_type = para_model_type)
                # restore the model
                model.model_restore(tmp_meta, 
                                    tmp_data, 
                                    saver)
                
                # one-shot inference sample
                # error_tuple: [rmse, mae, mape, nnllk],  
                # py_tuple: [py_mean, py_var, py_mean_src, py_var_src, py_gate_src]
                error_tuple, py_tuple = model.inference(xts,
                                                        yts, 
                                                        bool_py_eval = bool_instance_eval)
                if bool_instance_eval == True:
                    # store the samples
                    infer.add_samples(py_mean = py_tuple[0],
                                      py_var = py_tuple[1],
                                      py_mean_src = py_tuple[2],
                                      py_var_src = py_tuple[3],
                                      py_gate_src = py_tuple[4], 
                                      py_lk = py_tuple[5],
                                      log_py_mean = py_tuple[6],
                                      log_py_var = py_tuple[7])
    
    num_snapshots = sum([len(i) for i in retrain_snapshots])
    
    # return: error tuple, prediction tuple
    if num_snapshots == 0:
        return ["None"], ["None"]  
    else:
        # ensemble inference
        if len(snapshot_features) == 0 or num_snapshots == 1:
            return infer.bayesian_inference(yts)
        else:
            return infer.importance_inference(snapshot_features = snapshot_features, 
                                              y = yts)

