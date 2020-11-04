    '''
    if x_src_seperated == True:
        x_list = x
        
    else:
        # shape: [S, [B T D]]
        tmp_x_list = tf.split(x,
                              num_or_size_splits = n_src, 
                              axis = 0)
        x_list = [tf.squeeze(tmp_x, 0) for tmp_x in tmp_x_list]
    '''

# -- top snapshots 
    error_tuple, _ = testing(model_snapshots = [retrain_idx_steps[0][0]], model_retrain_num = 1, xts = src_ts_x, yts = ts_y, file_path = path_model,
                             bool_instance_eval = True, loss_type = para_loss_type, num_src = len(src_ts_x), snapshot_features = [], hpara_dict = best_hpara)
    log_test_performance(path = path_log_error, error_tuple = [error_tuple], ensemble_str = "Top-shots-one-retrain")
    '''
    # -- importance weighted best snapshot steps 
    error_tuple, _ = testing(model_snapshots = top_steps,
                             xts = src_ts_x,
                             yts = ts_y,
                             file_path = path_model,
                             bool_instance_eval = True,
                             loss_type = para_loss_type,
                             num_src = len(src_ts_x), 
                             snapshot_features = top_steps_features, 
                             hpara_dict = best_hpara)
    
    log_test_performance(path = path_log_error,
                         error_tuple = [error_tuple],
                         ensemble_str = "Importance Top-rank")
    '''
    # -- bayesian snapshots
    error_tuple, _ = testing(model_snapshots = [retrain_idx_steps[0][1]], model_retrain_num = 1, xts = src_ts_x, yts = ts_y, file_path = path_model, bool_instance_eval = True, loss_type = para_loss_type, num_src = len(src_ts_x), snapshot_features = [], hpara_dict = best_hpara)
    log_test_performance(path = path_log_error, error_tuple = [error_tuple], ensemble_str = "Bayesian-one-retrain")
    # -- indep steps
    error_tuple, py_tuple = testing(model_snapshots = [tmp_idx_steps[0] for tmp_idx_steps in retrain_idx_steps], model_retrain_num = para_hpara_retrain_num,
                                    xts = src_ts_x,yts = ts_y, file_path = path_model,bool_instance_eval = True, loss_type = para_loss_type,num_src = len(src_ts_x), snapshot_features = [], hpara_dict = best_hpara)
    log_test_performance(path = path_log_error, error_tuple = [error_tuple], ensemble_str = "Bayesian-multi-retrain")
    '''
    # -- importance weighted bayesian steps
    error_tuple, py_tuple = testing(model_snapshots = bayes_steps,
                                    xts = src_ts_x,
                                    yts = ts_y,
                                    file_path = path_model,
                                    bool_instance_eval = True,
                                    loss_type = para_loss_type,
                                    num_src = len(src_ts_x),
                                    snapshot_features = bayes_steps_features,
                                    hpara_dict = best_hpara)
    
    log_test_performance(path = path_log_error,
                         error_tuple = [error_tuple],
                         ensemble_str = "Importance Bayesian")
    '''
    # -- dump predictions on testing data
    pickle.dump(py_tuple, open(path_py, "wb"))
    
# --- results analysis

# -- data uncertatinty
    '''
    fig, ax = plt.subplots(figsize=(15,3));
    ax.plot(range(plot_l, plot_r), vol[plot_l: plot_r], label = 'volatility',  marker='o', alpha=.3, color = 'g');
    ax.set_title("Volatility");
    ax.set_xlabel("Time");
    ax.set_ylabel("Value");
    #ax.set_ylim(20, 120);
    '''
    # -- model uncertainty
    '''
    fig, ax = plt.subplots(figsize=(15,3));
    ax.plot(range(plot_l, plot_r), unc[plot_l: plot_r], label = 'uncertainty',  marker='o', alpha=.3, color = 'b');
    ax.set_title("Uncertainty on predicted mean");
    ax.set_xlabel("Time");
    ax.set_ylabel("Value");
    # ax.set_ylim(0, 50);
    '''
    # -- overview of mean, model uncertainty, and data uncertainty
    '''
    fig, ax = plt.subplots(figsize=(15,10));
    ax.plot(range(plot_l, plot_r), y[plot_l: plot_r], label = 'truth',\
            marker='o', color = 'k', alpha = 0.4);
    ax.plot(range(plot_l, plot_r), mean[plot_l: plot_r], label = 'prediction',\
            marker='o', color = 'b', alpha = 0.4);
    ax.fill_between(range(plot_l, plot_r), mean_low[plot_l: plot_r], mean_up[plot_l: plot_r], 
                color = '#539caf', alpha = 0.4, label = '95% CI')
    ax_twin = ax.twinx()
    ax_twin.plot(range(plot_l, plot_r), vol[plot_l: plot_r], label = 'volatility', \
                 marker='o', alpha=.8, color = 'g');
                 
    #ax2.tick_params(axis='y', labelcolor=color)
    #ax.plot(range(plot_l, plot_r), vol[plot_l: plot_r], label = 'volatility',  marker='o', alpha=.3, color = 'g');
    
    ax.set_title("overview of mean, uncertainty, and volatility");
    ax.set_xlabel("Time");
    ax.set_ylabel("Value");
    ax.legend();
    ax_twin.legend();
    '''
    # -- gate
    '''
    fig, ax = plt.subplots(figsize=(15,5));
    X = range(plot_l, plot_r)
    
    cumu = [0.0 for j in range(plot_l, plot_r)]
    src_dict = {0:"local trans.", 1:"external trans.", 2:"local order book"}
    for tmp_src in range(num_src):
    
    tmp_gate = [i[tmp_src] for i in gate[plot_l: plot_r]]
    tmp_cumu = cumu + np.asarray(tmp_gate)
    
    ax.plot(X, tmp_cumu, alpha=.5, label = src_dict[tmp_src]);
    ax.fill_between(X, cumu, tmp_cumu, alpha=.5);
    
    cumu = tmp_cumu
    ax.legend();
    ax.set_title("Contribution of different sources");
    ax.set_xlabel("Time");
    ax.set_ylabel("Value");
    '''
    
    # -- marginal distribution 
    '''
    fig, ax = plt.subplots();

    ax.hist(y, 2000, label = "Truth");
    ax.set_ylim([0, 200]);
    ax.set_xlim([-10, 500]);
    ax.set_title("True values in testing data");
    
    ax.hist(mean, 1000, label = "Prediction");
    ax.legend()
    '''