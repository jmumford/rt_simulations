
import numpy as np

nsub = 100
n_trials = 80
repetition_time = 1
#Stroop settings
#mu_expnorm = 530
#lam_expnorm = 1 / 160
#sigma_expnorm = 77
mu_grinband_shift = 638
inv_lambda_grinband_shift = 699
sigma_grinband_shift = 103
mu_expnorm = mu_grinband_shift
lam_expnorm = 1 / inv_lambda_grinband_shift
sigma_expnorm = sigma_grinband_shift
max_rt = 8000
min_rt = 50
event_duration = .1  
center_rt=False
hp_filter = True


win_sub_noise_sd={'dv_scales_yes': 1.15, 'dv_scales_no': .09}
btwn_sub_noise_sd={'dv_scales_yes': .65, 'dv_scales_no': .75}

beta_scales_yes_type1err = np.array([.75, .75])
beta_scales_no_type1err = np.array([.85, .85])

beta_scales_yes_power={'beta1': .75, 'beta2': [.75, .8, .9,  .95, 1, 1.025, 1.05, 1.1]}
beta_scales_no_power={'beta1': .85, 'beta2': [.85, .9, .95, 1, 1.05, 1.1, 1.15, 1.2]}

rt_diff_s_vec = [0, .025, 0.05, 0.1, .15, .3, .5, 1, 1.5]

per_shift_mu = np.array([.76, .76])
