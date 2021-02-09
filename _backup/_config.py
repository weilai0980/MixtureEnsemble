#!/usr/bin/python

# ----- hyper-parameters set-up

# -- model
para_distr_type = "gaussian"
# gaussian, student_t
para_distr_para = []
# gaussian: [] 
# student_t: [nu], nu>=3
para_var_type = "square" # square, exp
# for one-dimensional feature, variance derivation should be re-defined? 
# always positive correlation of the feature to the variance
para_share_type_gate = "no_share"
# no_share, share, mix

# -- data
para_x_src_seperated = True
para_bool_target_seperate = False
# if yes, the last source corresponds to the auto-regressive target variable
para_x_shape_acronym = ["src", "N", "T", "D"]
para_add_common_pattern = False

# -- training and validation
para_model_type = 'rnn'
para_hpara_search = "random" # random, grid 
para_hpara_n_trial = 2

para_n_epoch = 80
para_burn_in_epoch = 20

para_snapshot_type = "epoch_wise"  # batch_wise, epoch_wise
para_snapshot_Bernoulli = 0.001

para_hpara_range = {}

para_hpara_range['grid'] = {}
para_hpara_range['grid']['linear'] = {}
para_hpara_range['grid']['rnn'] = {}

para_hpara_range['grid']['linear']['lr'] = [0.001, ] 
para_hpara_range['grid']['linear']['batch_size'] = [64, 32, 16, 80, ]
para_hpara_range['grid']['linear']['l2'] = [1e-7, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, ]

para_hpara_range['grid']['rnn']['lr'] = [0.001, ]
para_hpara_range['grid']['rnn']['batch_size'] = [10, 20, 30, 40, 50, 60, 70, 80]
para_hpara_range['grid']['rnn']['l2'] = [1e-7, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, ]
para_hpara_range['grid']['rnn']['rnn_size'] =  [16, 32, 64, 128]
para_hpara_range['grid']['rnn']['dense_num'] = [1, 2]

para_hpara_range['random'] = {}
para_hpara_range['random']['linear'] = {}
para_hpara_range['random']['rnn'] = {}

para_hpara_range['random']['linear']['lr'] = [0.001, 0.001]  
para_hpara_range['random']['linear']['batch_size'] = [10, 80]
para_hpara_range['random']['linear']['l2'] = [1e-7, 0.01]

para_hpara_range['random']['rnn']['lr'] = [0.0001, 0.0005]
para_hpara_range['random']['rnn']['batch_size'] = [64, 150]
para_hpara_range['random']['rnn']['l2'] = [1e-7, 0.000001]
para_hpara_range['random']['rnn']['rnn_size'] =  [8, 16]
para_hpara_range['random']['rnn']['dense_num'] = [0, 2]
para_hpara_range['random']['rnn']['dropout_keep_prob'] = [1.0, 1.0]
para_hpara_range['random']['rnn']['max_norm_cons'] = [0.0, 0.0]

# model snapshot sample: epoch_wise or batch_wise
#   epoch_wise: vali. test snapshot numbers are explicited determined 
#   batch_wise: vali. test snapshot numbers are arbitary 
para_val_aggreg_num = max(1, int(0.05*para_n_epoch))
para_test_snapshot_num = para_n_epoch - para_burn_in_epoch

para_early_stop_bool = False
para_early_stop_window = 0

# -- optimization
para_loss_type = "heter_lk_inv"

para_validation_metric = 'nnllk'
para_metric_map = {'rmse':0, 'mae':1, 'mape':2, 'nnllk':3} 

para_optimizer = "adam" # RMSprop, sg_mcmc_RMSprop, adam, sg_mcmc_adam, sgd, adamW 
para_optimizer_lr_decay = True
para_optimizer_lr_decay_epoch = 10 # after the warm-up
para_optimizer_lr_warmup_epoch = int(0.1*para_n_epoch)

# -- regularization
para_regu_mean = True
para_regu_var = True
para_regu_gate = False
para_regu_imbalanced_mean_var = False
para_regu_mean_positive = False
para_regu_global_gate = False  
para_regu_latent_dependence = False
para_regu_weight_on_latent = True

para_bool_bias_in_mean = True
para_bool_bias_in_var = True
para_bool_bias_in_gate = True
para_bool_global_bias = False

para_latent_dependence = "none"
para_latent_prob_type = "none"