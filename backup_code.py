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