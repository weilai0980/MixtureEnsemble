#!/usr/bin/python

import numpy as np
from scipy.stats import truncnorm
from scipy.optimize import fmin_slsqp
from scipy.stats import norm
from sklearn.neighbors.kde import KernelDensity

import tensorflow as tf

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
                 para_train):
        '''
        Argu.:
          session: tensorflow session
        '''
        self.sess = session
        self.para_train = para_train
                
    def network_ini(self, 
                    hyper_para):
        '''
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
        # np.random.seed(1)
        # tf.set_random_seed(1)
        self.hyper_para = hyper_para
        
        # ----- ini
        # placeholders
        # y: [B 1]
        self.y = tf.placeholder(tf.float32,
                                [None, self.para_train['y_dim']],
                                name = 'y')
        # x: [S, [B T D]]
        self.x = []
        for i in range(self.para_train['para_num_source']):
            self.x.append(tf.placeholder(tf.float32,
                                         [None, self.para_train['x_steps'][i], self.para_train['x_dims'][i]],
                                         name = 'x' + str(i)))
        if self.para_train['para_model_type'] == "rnn":
            self.keep_prob = tf.placeholder(tf.float32,
                                            shape = (),
                                            name = 'keep_prob')
        # ---------- tempory
        self.y_normalizer = tf.slice(self.y, [0, 1,], [-1, 1])
        self.y_desea = tf.slice(self.y, [0, 2,], [-1, -1])
        self.y = tf.slice(self.y, [0, 0,], [-1, 1])
        
        # ----- individual models
        '''
            # x: [S, [B T D]]
            self.pre_x = []
            self.cur_x = []
            for i in range(self.num_src):
                self.pre_x.append(tf.slice(self.x[i], [0, 0, 0], [-1, x_steps[i]-1, -1]))
                self.cur_x.append(tf.slice(self.x[i], [0, 1, 0], [-1, x_steps[i]-1, -1]))
        '''
        if self.para_train['para_model_type'] == "linear":
            #[S B]
            tmp_mean, regu_mean, tmp_var, regu_var, tmp_logit, regu_gate = multi_src_predictor_linear(x = self.x,
                                                                                                      n_src = self.para_train['para_num_source'],
                                                                                                      steps = self.para_train['x_steps'],
                                                                                                      dims = self.para_train['x_dims'],
                                                                                                      bool_bias = [self.para_train['para_bool_bias_in_mean'], self.para_train['para_bool_bias_in_var'], self.para_train['para_bool_bias_in_gate']],
                                                                                                      bool_scope_reuse= [False, False, False],
                                                                                                      str_scope = "linear",
                                                                                                      para_share_logit = self.para_train['para_share_type_gate'],
                                                                                                      bool_common_factor = self.para_train['para_add_common_factor'],
                                                                                                      common_factor_dim = 0)
        elif self.para_train['para_model_type'] == "rnn":
            #[S B]
            tmp_mean, regu_mean, tmp_var, regu_var, tmp_logit, regu_gate = multi_src_predictor_rnn(x = self.x,
                                                                                                   n_src = self.para_train['para_num_source'],
                                                                                                   bool_bias = [self.para_train['para_bool_bias_in_mean'], self.para_train['para_bool_bias_in_var'], self.para_train['para_bool_bias_in_gate']],
                                                                                                   bool_scope_reuse = [False, False, False],
                                                                                                   str_scope = "rnn",
                                                                                                   rnn_size_layers = [int(self.hyper_para['rnn_size'])],
                                                                                                   rnn_cell_type = "lstm",
                                                                                                   dropout_keep = self.hyper_para['dropout_keep_prob'],
                                                                                                   dense_num = int(self.hyper_para['dense_num']),
                                                                                                   max_norm_cons = self.hyper_para['max_norm_cons'])
        # ----- individual means and variance
        
        # -- mean
        mean_stack = tf.transpose(tmp_mean, [1, 0])
        
        # -- variance
        if self.para_train['para_var_type'] == "square":
            # square
            inv_var_stack = tf.transpose(tf.square(tmp_var), [1, 0])
            
        elif self.para_train['para_var_type'] == "exp":
            # exp
            inv_var_stack = tf.transpose(tf.exp(tmp_var), [1, 0])
            
        elif self.para_train['para_var_type'] == "logexp":
            # logexp
            inv_var_stack = tf.transpose(tf.log(tf.exp(tmp_var)+1.0), [1, 0])
            
        # ----- gates
        # [B S]
        gate_logits = tf.transpose(tmp_logit, [1, 0])
        # gate probability
        self.gate_src = tf.nn.softmax(gate_logits, axis = -1)
        
        # ----- mixture mean, variance and nllk
        
        if self.para_train['para_distr_type'] == 'normal':
            
            # -- mean
            # component mean
            # [B S] 
            self.py_mean_src = mean_stack
            # mixture mean
            # [B 1]                      [B S]        [B S]
            self.py_mean = tf.reduce_sum(mean_stack * self.gate_src, 1, keepdims = True)
            
            # --
            if self.para_train['para_loss_type'] == 'heter_lk_inv':
                
                # --- loss
                # negative log likelihood
                # [B S]
                lk_src = tf.exp(-0.5*tf.square(self.y - mean_stack)*inv_var_stack)*tf.sqrt(0.5/np.pi*inv_var_stack)
                lk = tf.reduce_sum(tf.multiply(lk_src, self.gate_src) , axis = -1) 
                self.nnllk_loss = tf.reduce_mean(-1.0*tf.log(lk+ 1e-5))
                
                # --- evaluation
                # nnllk
                self.nnllk = self.nnllk_loss
                # component variance
                self.py_var_src = 1.0/(inv_var_stack + 1e-5)
                # variance
                var_plus_sq_mean_src = self.py_var_src + tf.square(self.py_mean_src)
                # [B 1]
                self.py_var = tf.reduce_sum(tf.multiply(var_plus_sq_mean_src, self.gate_src), 1, keepdims = True) - tf.square(self.py_mean)
                
                py_var_src_inv = inv_var_stack
                
                # lk: likelihood
                # in the linear scale
                tmp_lk_src = tf.exp(-0.5*tf.square(self.y - self.py_mean_src)*py_var_src_inv)*tf.sqrt(0.5/np.pi*py_var_src_inv)/(1.0*self.y+1e-5)
                # [B]
                self.lk = tf.reduce_sum(tf.multiply(tmp_lk_src, self.gate_src), axis = -1)
                self.nnllk = tf.reduce_mean(-1.0*tf.log(self.lk + 1e-5))
                
            elif self.para_train['para_loss_type'] == 'mse':
                
                # component variance
                self.py_var_src = tf.constant(1.0, shape = [1, self.num_src])
                # variance
                sq_mean_stack = 1.0 + tf.square(mean_stack)
                # [B 1]
                mix_sq_mean = tf.reduce_sum(tf.multiply(sq_mean_stack, self.gate_src), 1, keepdims = True)
                # [B 1]
                self.py_var = mix_sq_mean - tf.square(self.py_mean)
                
                # negative log likelihood
                # [B S]
                # ? variance of constant 1 
                lk_src = tf.exp(-0.5*tf.square(self.y - mean_stack))/(2.0*np.pi)**0.5
            
                lk = tf.multiply(lk_src, self.gate_src) 
                self.nllk = tf.reduce_sum(-1.0*tf.log(tf.reduce_sum(lk, axis = -1) + 1e-5))
                
        elif self.para_train['para_distr_type'] == 'log_normal_logOpt_linearComb':
            
            self.log_y = tf.math.log(self.y + 1e-5)
            
            if self.para_train['para_loss_type'] == 'heter_lk_inv':
                
                # --- loss
                # nnllk: normalized negative log likelihood
                # [B S]
                # in the log scale
                lk_src = tf.exp(-0.5*tf.square(self.log_y - mean_stack)*inv_var_stack)*tf.sqrt(0.5/np.pi*inv_var_stack)
                lk = tf.multiply(lk_src, self.gate_src) 
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
                self.py_mean = tf.reduce_sum(self.py_mean_src*self.gate_src, 1, keepdims = True)
                
                # component variance
                # [B S]
                self.py_var_src = (tf.exp(log_py_var_src) - 1.0)*tf.exp(2.0*log_py_mean_src + log_py_var_src)
                # mixture variance
                # [B 1]
                var_plus_sq_mean_src = self.py_var_src + tf.square(self.py_mean_src)
                self.py_var = tf.reduce_sum(tf.multiply(var_plus_sq_mean_src, self.gate_src), 1, keepdims = True) - tf.square(self.py_mean)
                
                # lk: likelihood
                # in the linear scale
                tmp_lk_src = tf.exp(-0.5*tf.square(self.log_y - log_py_mean_src)*log_py_var_src_inv)*tf.sqrt(0.5/np.pi*log_py_var_src_inv)/(1.0*self.y+1e-5)
                # [B]
                self.lk = tf.reduce_sum(tf.multiply(tmp_lk_src, self.gate_src), axis = -1)
                self.nnllk = tf.reduce_mean(-1.0*tf.log(self.lk + 1e-5))
                
        elif self.para_train['para_distr_type'] == 'log_normal_linearOpt_linearComb':
            
            self.log_y = tf.math.log(self.y + 1e-5)
            
            if self.para_train['para_loss_type'] == 'heter_lk_inv':
                
                # --- loss
                # nnllk: normalized negative log likelihood
                # [B S]
                # in the log scale 
                lk_src = tf.exp(-0.5*tf.square(self.log_y - mean_stack)*inv_var_stack)*tf.sqrt(0.5/np.pi*inv_var_stack)/(1.0*self.y+1e-5)
                lk = tf.multiply(lk_src, self.gate_src) 
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
                self.py_mean = tf.reduce_sum(self.py_mean_src*self.gate_src, 1, keepdims = True)
                
                # component variance
                # [B S]
                self.py_var_src = (tf.exp(log_py_var_src) - 1.0)*tf.exp(2.0*log_py_mean_src + log_py_var_src)
                # mixture variance
                # [B 1]
                var_plus_sq_mean_src = self.py_var_src + tf.square(self.py_mean_src)
                self.py_var = tf.reduce_sum(tf.multiply(var_plus_sq_mean_src, self.gate_src), 1, keepdims = True) - tf.square(self.py_mean)
                
                # lk: likelihood
                # in the linear scale
                tmp_lk_src = tf.exp(-0.5*tf.square(self.log_y - log_py_mean_src)*log_py_var_src_inv)*tf.sqrt(0.5/np.pi*log_py_var_src_inv)/(1.0*self.y+1e-5)
                # [B]
                self.lk = tf.reduce_sum(tf.multiply(tmp_lk_src, self.gate_src), axis = -1)
                self.nnllk = tf.reduce_mean(-1.0*tf.log(self.lk + 1e-5))
                
        elif self.para_train['para_distr_type'] == 'log_normal_logOpt_logComb':
            
            self.log_y = tf.math.log(self.y + 1e-5)
            
            if self.para_train['para_loss_type'] == 'heter_lk_inv':
                
                # --- loss
                # nnllk: normalized negative log likelihood
                # [B S]
                lk_src = tf.exp(-0.5*tf.square(self.log_y - mean_stack)*inv_var_stack)*tf.sqrt(0.5/np.pi*inv_var_stack)
                lk = tf.multiply(lk_src, self.gate_src) 
                self.nnllk_loss = tf.reduce_mean(-1.0*tf.log(tf.reduce_sum(lk, axis = -1) + 1e-5))
                
                # --- evaluation
                log_py_var_src = 1.0/(inv_var_stack + 1e-5)
                log_py_var_src_inv = inv_var_stack
                log_py_mean_src = mean_stack
                
                # mixture mean of log_y
                # [B 1]                      [B S]        [B S]
                log_py_mean = tf.reduce_sum(log_py_mean_src*self.gate_src, 1, keepdims = True)
                
                # mixture variance of log_y
                # [B 1]
                var_plus_sq_mean_src = log_py_var_src + tf.square(log_py_mean_src)
                log_py_var = tf.reduce_sum(tf.multiply(var_plus_sq_mean_src, self.gate_src), 1, keepdims = True) - tf.square(log_py_mean)
                
                self.py_mean_src = tf.exp(log_py_mean_src + log_py_var_src/2.0)
                self.py_mean = tf.exp(log_py_mean + log_py_var/2.0)
                
                self.py_var_src = (tf.exp(log_py_var_src) - 1.0)*tf.exp(2.0*log_py_mean_src + log_py_var_src)
                self.py_var = (tf.exp(log_py_var) - 1.0)*tf.exp(2.0*log_py_mean + log_py_var)
                
                # lk: likelihood
                # [B]
                self.lk = tf.exp(-0.5*tf.square(self.log_y - log_py_mean)/(1.0*log_py_var))*tf.sqrt(0.5/np.pi/log_py_var)/(1.0*self.y+1e-5)
                self.nnllk = tf.reduce_mean(-1.0*tf.log(self.lk + 1e-5))
                
        # ----- regularization
        # [S]
        self.regu_var = regu_var 
        self.regu_mean = regu_mean
        self.regu_gate = regu_gate
        
        # ----- loss 
        self.monitor = []
        
        # mse
        if self.para_train['para_loss_type'] == 'mse':
            self.loss = tf.reduce_mean(tf.square(self.y - self.py_mean)) + self.l2*self.regu_mean
            self.monitor = [tf.reduce_mean(tf.square(self.y - self.py_mean)), self.l2*self.regu_mean]
            
        # nnllk        
        elif self.para_train['para_loss_type'] in ['heter_lk', 'heter_lk_inv', 'homo_lk_inv']:
            
            self.loss = self.nnllk_loss
            self.monitor = [self.loss]
            
            if self.para_train['para_regu_mean'] == True:
                self.loss += ( self.hyper_para["l2_mean"]*self.regu_mean )
                self.monitor.append(self.hyper_para["l2_mean"]*self.regu_mean)
                
            if self.para_train['para_regu_var'] == True:
                self.loss += (self.hyper_para["l2_var"]*self.regu_var)
                self.monitor.append(self.hyper_para["l2_var"]*self.regu_var)
                
            if self.para_train['para_regu_gate'] == True:
                self.loss += (self.hyper_para["l2_gate"]*self.regu_gate)
                self.monitor.append((self.hyper_para["l2_gate"]*self.regu_gate))
                
        # self.gates [B S]
        #         self.monitor.append(tf.slice(self.gate_src, [0, 0], [3, -1]))
        
    #   initialize loss and optimization operations for training
    def train_ini(self):
        
        # ----- learning rate set-up
        tf_lr_ini = tf.constant(value = self.hyper_para["lr"], 
                                shape = [], 
                                dtype = tf.float32)
        global_step = tf.train.get_or_create_global_step()
        
        # -- decay
        if self.para_train['para_optimizer_lr_decay_epoch'] > 0:
            optimizer_lr = tf.train.exponential_decay(tf_lr_ini, 
                                                             global_step,
                                                             decay_steps = self.para_train['para_optimizer_lr_decay_epoch']*int(np.ceil(self.para_train['tr_num_ins']/int(self.hyper_para["batch_size"]))), 
                                                             decay_rate = 0.96, 
                                                             staircase = True)
        else:
            optimizer_lr = tf_lr_ini
            
        # -- warm-up
        # ref: https://github.com/google-research/bert/blob/ffbda2a1aafe530525212d13194cc84d92ed0313/optimization.py#L29-L65
        
        if self.para_train['para_optimizer_lr_warmup_epoch'] > 0:
            
            global_steps_int = tf.cast(global_step, 
                                       tf.int32)
            warmup_steps_int = tf.constant(self.para_train['para_optimizer_lr_warmup_epoch']*int(np.ceil(self.para_train['tr_num_ins']/int(self.hyper_para["batch_size"]))-1), 
                                           dtype = tf.int32)
            
            global_steps_float = tf.cast(global_steps_int,
                                         tf.float32)
            warmup_steps_float = tf.cast(warmup_steps_int, 
                                         tf.float32)
            
            warmup_percent_done = global_steps_float / warmup_steps_float
            warmup_learning_rate = tf_lr_ini * warmup_percent_done
                
            is_warmup = tf.cast(global_steps_int < warmup_steps_int, 
                                tf.float32)        
            optimizer_lr = ((1.0-is_warmup)*optimizer_lr + is_warmup*warmup_learning_rate)
        
        # ----- optimizer
        
        # -- conventional 
        if   self.para_train['para_optimizer'] == 'adam':
            train_optimizer = myAdamOptimizer(learning_rate = optimizer_lr)
            
        elif self.para_train['para_optimizer'] == 'adam_origin':
            train_optimizer = tf.train.AdamOptimizer(learning_rate = optimizer_lr)
            
        elif self.para_train['para_optimizer'] == 'RMSprop':
            train_optimizer = myRMSprop(learning_rate = optimizer_lr)
            
        elif self.para_train['para_optimizer'] == 'RMSprop_origin':
            train_optimizer = tf.train.RMSPropOptimizer(learning_rate = optimizer_lr)
            
        elif self.para_train['para_optimizer'] == 'sgd':
            train_optimizer = tf.train.MomentumOptimizer(learning_rate = optimizer_lr,
                                                         momentum = 0.9,
                                                         use_nesterov = True)
        elif self.para_train['para_optimizer'] == 'adamW':
            # ref.: "Fixing Weight Decay Regularization in Adam", https://arxiv.org/abs/1711.05101
            train_optimizer = tf.contrib.opt.AdamWOptimizer(weight_decay = self.l2,
                                                            learning_rate = optimizer_lr)
        # -- SG-MCMC
        # stochastic gradient Monto-Carlo Markov Chain
        elif self.para_train['para_optimizer'] == 'sg_mcmc_adam':
            train_optimizer = sg_mcmc_adam(learning_rate = optimizer_lr)
            
        elif self.para_train['para_optimizer'] == 'sg_mcmc_adam_revision':
            train_optimizer = sg_mcmc_adam_revision(learning_rate = optimizer_lr)
            
        elif self.para_train['para_optimizer'] == 'sg_mcmc_RMSprop':
            train_optimizer = sg_mcmc_RMSprop(learning_rate = optimizer_lr)
            
        elif self.para_train['para_optimizer'] == 'sgld':
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
                    y,):
        data_dict = {}
        data_dict["y:0"] = y
        
        # x: [S, [B T D]]
        for i in range(len(x)):
            data_dict["x" + str(i) + ":0"] = x[i]
        if self.para_train['para_model_type'] == "rnn":
            data_dict["keep_prob:0"] = self.hyper_para['dropout_keep_prob']
        
        # update the paramters
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
        tf.add_to_collection("py_gate_src", self.gate_src)
        tf.add_to_collection("py_mean_src", self.py_mean_src)
        tf.add_to_collection("py_var_src", self.py_var_src)
        tf.add_to_collection("py_lk", self.lk)
        
    # infer given testing data
    def inference(self, 
                  x, 
                  y,
                  bool_instance_eval):
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
        if self.para_train['para_model_type'] == "rnn":
            data_dict["keep_prob:0"] = 1.0
            
        # error metric
        rmse, mae, mape, nnllk = self.sess.run([tf.get_collection('rmse')[0],
                                                tf.get_collection('mae')[0],
                                                tf.get_collection('mape')[0],
                                                tf.get_collection('nnllk')[0]],
                                               feed_dict = data_dict)
        # predictions
        if bool_instance_eval == True:
            # [B 1]  [B 1]   [B S]
            py_mean, py_var, py_gate_src, py_mean_src, py_var_src, py_lk = self.sess.run([tf.get_collection('py_mean')[0],
                                                                                          tf.get_collection('py_var')[0],
                                                                                          tf.get_collection('py_gate_src')[0],
                                                                                          tf.get_collection('py_mean_src')[0],
                                                                                          tf.get_collection('py_var_src')[0], 
                                                                                          tf.get_collection('py_lk')[0],],
                                                                                         feed_dict = data_dict)
            # error metric tuple [rmse, mae, mape, nnllk], py tuple []
            return [rmse, mae, mape, nnllk], [py_mean, py_var, py_mean_src, py_var_src, py_gate_src, py_lk], []
        else:
            py_mean = None
            py_var = None
            py_gate_src = None
            py_mean_src = None
            py_var_src = None
            py_lk = None
            
            # monitor metric
            monitor_metric = self.sess.run([tf.get_collection(str(tmp_idx))[0] for tmp_idx in range(len(self.monitor))],
                                       feed_dict = data_dict)
            # error metric tuple [rmse, mae, mape, nnllk], py tuple []
            return [rmse, mae, mape, nnllk], [py_mean, py_var, py_mean_src, py_var_src, py_gate_src, py_lk], monitor_metric
    
    def model_saver(self, 
                    path,
                    epoch,
                    top_snapshots,
                    bayes_snapshots,
                    early_stop_bool,
                    early_stop_window, 
                    tf_saver):
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
