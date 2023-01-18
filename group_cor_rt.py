#!/usr/bin/env python3

root_dir = '/home/users/jmumford/RT_sims/'

import sys
import pandas as pd
sys.path.insert(1, root_dir + '/Code')
from simulation_settings import *
from functions import group_2stim_rt_cor

out_file_name = root_dir + '/Output/rt_correlation_output.csv'

rt_diff_s = 0.2
nsim = 5000
nsub = 30

output_cor = group_2stim_rt_cor(n_trials, scan_length, repetition_time, 
              mu_expnorm,lam_expnorm, sigma_expnorm, max_rt, 
              min_rt, event_duration, ISI_min, ISI_max, 
              win_sub_noise_sd,btwn_sub_noise_sd, 
              center_rt, beta_scales_yes, beta_scales_no,  
              rt_diff_s, nsub, nsim)

cor_types = ['group_rtdiff_cor', 'group_rtmn_cor']
data_type = ['blocked', 'random']
models = ['Two stimulus types, no RT', 'Two stimulus types, 2 RT dur only']
scale_type = ['dv_scales_yes', 'dv_scales_no']
beta_cons = ['beta_diff_est', 'beta1_est', 'beta2_est']

cor_types_long = []
data_type_long = []
models_long = []
scale_type_long = []
beta_cons_long = []
rt_cor_val_long = []

for cur_cor_type in cor_types:
    for cur_data_type in data_type:
        for cur_model in models:
            for cur_scale_type in scale_type:
                for cur_beta_cons in beta_cons:
                    rt_cor_loop = output_cor[cur_cor_type][cur_data_type][cur_model]\
                                            [cur_scale_type][cur_beta_cons]
                    nvals = len(rt_cor_loop)
                    cor_types_long.extend([cur_cor_type] * nvals)
                    data_type_long.extend([cur_data_type] * nvals)
                    models_long.extend([cur_model] * nvals)
                    scale_type_long.extend([cur_scale_type] * nvals)
                    beta_cons_long.extend([cur_beta_cons] * nvals)
                    rt_cor_val_long.extend(rt_cor_loop)

all_correlations = pd.DataFrame(list(zip(rt_cor_val_long, cor_types_long,\
                                  data_type_long, models_long, \
                                  scale_type_long, beta_cons_long)),
            columns = ['Correlation','Correlation Type', 'Data Type', 
                       'Model', 'Scale Type', 'Beta Contrast'])

all_correlations.to_csv(out_file_name)