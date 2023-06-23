#!/usr/bin/env python3

root_dir = '/home/users/jmumford/RT_sims/'

import sys
import pandas as pd
import numpy as np
sys.path.insert(1, root_dir + '/Code')
#from simulation_settings import *
from functions import group_2stim_beta2_vec
import json

rt_diff_s = 0.8

args_in = sys.argv
simulation_settings = args_in[1]
var_in = __import__(simulation_settings) 
nsim = int(args_in[2])
ISI_min = int(args_in[3])
ISI_max = int(args_in[4])


out_results_name = (f'{root_dir}/Output/revisions/group_power_output_nsim{nsim}_'
    f'nsub{var_in.nsub}_mu{var_in.mu_expnorm}_btwn_noise{var_in.btwn_sub_noise_sd["dv_scales_yes"]}'
    f'{var_in.btwn_sub_noise_sd["dv_scales_no"]}_isi{ISI_min}_{ISI_max}_rtdiff_{rt_diff_s}_revisions.csv')
out_settings_name = (f'{root_dir}/Output/revisions/group_power_output_nsim{nsim}_'
    f'nsub{var_in.nsub}_mu{var_in.mu_expnorm}_settings_btwn_noise'
    f'{var_in.btwn_sub_noise_sd["dv_scales_yes"]}'
    f'{var_in.btwn_sub_noise_sd["dv_scales_no"]}_isi{ISI_min}_{ISI_max}_{rt_diff_s}_revisions.json')


output_power = group_2stim_beta2_vec(var_in.n_trials, var_in.repetition_time, var_in.mu_expnorm,
              var_in.lam_expnorm, var_in.sigma_expnorm, var_in.max_rt, 
              var_in.min_rt, var_in.event_duration, ISI_min, ISI_max, 
              var_in.win_sub_noise_sd,
              var_in.btwn_sub_noise_sd,
              var_in.center_rt, var_in.beta_scales_yes_power,
              var_in.beta_scales_no_power, 
              rt_diff_s, var_in.nsub, nsim)


data_type = ['blocked', 'random']
models = list(output_power['group_rej_rate']['blocked'].keys())
scale_type = ['dv_scales_yes', 'dv_scales_no']
beta_con_type = ['beta_diff_est', 'beta2_est']

beta_diff_vec = {'dv_scales_yes': np.array(var_in.beta_scales_yes_power['beta2']) - np.array(var_in.beta_scales_yes_power['beta1']),
                 'dv_scales_no': np.array(var_in.beta_scales_no_power['beta2']) - np.array(var_in.beta_scales_no_power['beta1'])}


data_type_long = []
models_long = []
scale_type_long = []
beta_con_type_long = []
rej_rate_beta_con = []
rej_rate_beta_con_2step = []
beta_diff_long = []

for cur_data_type in data_type:
    for cur_model in models:
        print(f'{cur_data_type}, {cur_model} starting')
        for cur_scale_type in scale_type:
            for cur_beta_con_type in beta_con_type:
                rej_rate_loop = \
                    output_power['group_rej_rate'][cur_data_type][cur_model]\
                                            [cur_scale_type][cur_beta_con_type]
                rej_rate_loop_2step = \
                    output_power['group_rej_rate_2step'][cur_data_type][cur_model]\
                                            [cur_scale_type][cur_beta_con_type]
                nvals = len(rej_rate_loop)
                data_type_long.extend([cur_data_type] * nvals)
                models_long.extend([cur_model] * nvals)
                scale_type_long.extend([cur_scale_type] * nvals)
                beta_con_type_long.extend([cur_beta_con_type] * nvals) 
                rej_rate_beta_con.extend(rej_rate_loop)
                rej_rate_beta_con_2step.extend(rej_rate_loop_2step)
                beta_diff_long.extend(beta_diff_vec[cur_scale_type])


data_long = pd.DataFrame(list(zip(beta_diff_long, data_type_long, models_long, \
                    scale_type_long, beta_con_type_long, rej_rate_beta_con, rej_rate_beta_con_2step)),
                    columns = ['Beta diff','Data Type', 'Model', 'Scale Type', \
                   'Beta Contrast', 'Rejection Rate', 'Rejection Rate (2 step)'])


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
             'beta_scales_yes_beta1': var_in.beta_scales_yes_power['beta1'],
             'beta_scales_yes_beta2': list(var_in.beta_scales_yes_power['beta2']),
             'beta_scale_no_beta1': var_in.beta_scales_no_power['beta1'],
             'beta_scale_no_beta2': list(var_in.beta_scales_no_power['beta2']),
             'nsim': nsim}
with open(out_settings_name, "w") as outfile:
    json.dump(all_settings, outfile, indent=4)

