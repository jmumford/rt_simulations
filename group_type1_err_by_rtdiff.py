#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
root_dir = '/home/users/jmumford/RT_sims/'
sys.path.insert(1, root_dir + '/Code')
#from simulation_settings import *
from functions import group_2stim_rt_diff_vec
import json

args_in = sys.argv

simulation_settings = args_in[1]
var_in = __import__(simulation_settings) 
var_in.rt_diff_s_vec = [0, 0.1]

nsim = int(args_in[2])
ISI_min = int(args_in[3])
ISI_max = int(args_in[4])


out_results_name = (f'{root_dir}/Output/group_type1_err_corr_by_rtdiff_output_nsim{nsim}_'
    f'nsub{var_in.nsub}_mu{var_in.mu_expnorm}_btwn_noise{var_in.btwn_sub_noise_sd["dv_scales_yes"]}'
    f'{var_in.btwn_sub_noise_sd["dv_scales_no"]}_isi{ISI_min}_{ISI_max}_new.csv')
out_settings_name = (f'{root_dir}/Output/group_type1_err_corr_by_rtdiff_output_nsim{nsim}'
    f'_nsub{var_in.nsub}_mu{var_in.mu_expnorm}_btwn_noise{var_in.btwn_sub_noise_sd["dv_scales_yes"]}'
    f'{var_in.btwn_sub_noise_sd["dv_scales_no"]}_isi{ISI_min}_{ISI_max}_new.json')


output = group_2stim_rt_diff_vec(var_in.n_trials, var_in.repetition_time, var_in.mu_expnorm,
              var_in.lam_expnorm, var_in.sigma_expnorm, var_in.max_rt, 
              var_in.min_rt, var_in.event_duration, ISI_min, ISI_max, 
              var_in.win_sub_noise_sd,
              var_in.btwn_sub_noise_sd, 
              var_in.center_rt, var_in.beta_scales_yes_type1err, var_in.beta_scales_no_type1err,  
              var_in.rt_diff_s_vec, var_in.nsub, nsim)

data_type = ['blocked', 'random']
models = ['Two stimulus types, no RT', 'Two stimulus types, RT mod',
          'Two stimulus types, RTmod interaction, con main', 
          'Two stimulus types, RTmod interaction, con int',
          'Two stimulus types, 2 RT dur only']
scale_type = ['dv_scales_yes', 'dv_scales_no']


data_type_long = []
models_long = []
scale_type_long = []
rej_rate_beta_diff = []
rej_rate_beta_diff_2step = []
group_cor_betadiff_rtdiff = []
group_cor_betadiff_rtmn = []
rt_diff_long = []

for cur_data_type in data_type:
    for cur_model in models:
        for cur_scale_type in scale_type:
            rej_rate_loop = output['group_rej_rate'][cur_data_type][cur_model]\
                                            [cur_scale_type]['beta_diff_est']
            rej_rate_loop_2step = output['group_rej_rate_2step'][cur_data_type][cur_model]\
                                            [cur_scale_type]['beta_diff_est']
            nvals = len(rej_rate_loop)
            data_type_long.extend([cur_data_type] * nvals)
            models_long.extend([cur_model] * nvals)
            scale_type_long.extend([cur_scale_type] * nvals)
            rej_rate_beta_diff.extend(rej_rate_loop)
            rej_rate_beta_diff_2step.extend(rej_rate_loop_2step)
            group_cor_betadiff_rtdiff.extend(output['group_rtdiff_cor_avg'][cur_data_type][cur_model]\
                                            [cur_scale_type]['beta_diff_est'])
            group_cor_betadiff_rtmn.extend(output['group_rtmn_cor_avg'][cur_data_type][cur_model]\
                                            [cur_scale_type]['beta_diff_est'])
            rt_diff_long.extend(output['rt_diff'] * nvals)


data_long = pd.DataFrame(list(zip(rt_diff_long, data_type_long, models_long, \
                            scale_type_long, rej_rate_beta_diff, rej_rate_beta_diff_2step,
                            group_cor_betadiff_rtdiff, group_cor_betadiff_rtmn)),
            columns = ['RT diff','Data Type', 'Model', 'Scale Type', \
                 'Rejection Rate', 'Rejection Rate (2 step)', 'Correlation (beta diff with rt diff)',
                 'Correlation (beta diff with rt mn)'])
data_long.to_csv(out_results_name)


all_settings = {'nsub': var_in.nsub,
             'mu_expnorm':var_in.mu_expnorm,
             'lam_expnorm': var_in.lam_expnorm,
             'sigma_expnorm': var_in.sigma_expnorm,
             'max_rt': var_in.max_rt,
             'min_rt': var_in.min_rt,
             'event_duration': var_in.event_duration,
             'center_rt': var_in.center_rt,
             'hp_filter': var_in.hp_filter,
             'ISI_min': ISI_min,
             'ISI_max': ISI_max,
             'win_sub_noise_sd': var_in.win_sub_noise_sd,
             'btwn_sub_noise_sd': var_in.btwn_sub_noise_sd,
             'beta_scales_yes': list(var_in.beta_scales_yes_type1err),
             'beta_scale_no': list(var_in.beta_scales_no_type1err),
             'nsim': nsim}
with open(out_settings_name, "w") as outfile:
    json.dump(all_settings, outfile, indent=4)

