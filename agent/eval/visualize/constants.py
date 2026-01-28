# MIT License

# Copyright (c) 2025 ReinFlow Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np
color_dict = {
        'Gaussian': '#228B22',
        'QSM': '#b98c1b',
        'DIPO': '#808080',
        'IDQL': '#000000', 
        'DQL': '#800080',  
        'DRWR': '#1E90B2', 
        'DAWR': "#0984C7", 
        'DPPO': '#F4AD08', 
        'ReinFlow-S (ours)': '#FF2400', 
        'ReinFlow-R (W2=0.01)': "#0FBFE6", 
        'ReinFlow-R (W2=0.1)': "#0F76DD", 
        'ReinFlow-R (W2=1.0)': "#1022E8", 
        'ReinFlow-R (sigma(s,t))': "#FF0000", 
        'ReinFlow-R (sigma(s))': "#990066", 
        'ReinFlow-S (2 steps)': "#3B34FF",
        'ReinFlow-S (1 step)': "#5F1F20",
        'ReinFlow-S, 32 (ours)': '#FF2400', 
        'ReinFlow-S, 64 (ours)': '#FF2400', 
        'ReinFlow-S, 100 (ours)': '#FF2400', 
        'ReinFlow-R (ours)': '#E30DEA', 
        'ReinFlow-beta (ours)': '#1E90FF', 
        'ReinFlow-logitnormal (ours)': '#4B0082', 
        'uniform': '#E30DEA',  
        'beta': "#6E0887",     
        'logitnormal': '#4B0082',
        'FQL': "#0F8EE3", 
        # 新增: score 和 with_score 方法
        'ScoRe-Flow (ours)': "#4335FFD2",  # 橙红色
        'Score-SDE (ours)': '#00CED1',  # 深青色
        'ScoRe-Flow (2 steps)': "#3B34FF",
        'ScoRe-Flow (1 step)': "#FF6B6B",  # 浅红色 - 更明显
        'ScoRe-Flow (4 steps)': "#35FF78D2",  # 蓝紫色 - 与 ScoRe-Flow (ours) 相同
    }


# Time/step ratios for each environment and method (in seconds/step)
time_step_ratios = {
        'hopper':{
            'ReinFlow-R': 11.715,
            'ReinFlow-S': 12.290,
            'DPPO': 99.046,
            'FQL': 4.418,
            'ScoRe-Flow (ours)': 11.720,  
            'Score-SDE (ours)': 11.870,  
        },
        'hopper-d4rl':{
            'ReinFlow-R': 11.715,
            'ReinFlow-S': 12.290,
            'DPPO': 99.046,
            'FQL': 4.418,
            'ScoRe-Flow (ours)': 11.720,  
            'Score-SDE (ours)': 11.870,  
        },
        'walker':{
            'ReinFlow-R': 11.563 ,
            'ReinFlow-S': 13.019 ,
            'DPPO': 101.915 ,
            'FQL': 5.017,
            'ScoRe-Flow (ours)': 11.572,  
            'Score-SDE (ours)': 11.793,  # 13.1
        },
        'walker-d4rl':{
            'ReinFlow-R': 11.563 ,
            'ReinFlow-S': 13.019 ,
            'DPPO': 101.915 ,
            'FQL': 5.017,
            'ScoRe-Flow (ours)': 11.572,  
            'Score-SDE (ours)': 11.793,  
        },
        'ant':{
            'ReinFlow-R': 17.473  ,
            'ReinFlow-S': 17.734  ,
            'DPPO': 102.012  ,
            'FQL': 5.167 ,
            'ScoRe-Flow (ours)': 17.426,  
            'Score-SDE (ours)': 17.723,  
        },
        'ant-d4rl':{
            'ReinFlow-R': 17.473  ,
            'ReinFlow-S': 17.734  ,
            'DPPO': 102.012  ,
            'FQL': 5.167 ,
            'ScoRe-Flow (ours)': 17.426,  
            'Score-SDE (ours)': 17.723,  
        },
        'humanoid':{
            'ReinFlow-R': 30.916   ,
            'ReinFlow-S': 30.529   ,
            'DPPO': 109.566   ,
            'FQL': 5.249  ,
            'ScoRe-Flow (ours)': 30.944 ,
            'Score-SDE (ours)':31.252 ,
        },
        'humanoid-d4rl':{
            'ReinFlow-R': 30.916   ,
            'ReinFlow-S': 30.529   ,
            'DPPO': 109.566   ,
            'FQL': 5.249  ,
            'ScoRe-Flow (ours)': 30.944 ,
            'Score-SDE (ours)':31.252 ,
        },
        'can-img': {
            'ReinFlow-R': 218.061,
            'ReinFlow-S': 218.061,
            'DPPO': 308.933
        },
        'square-img': {
            'ReinFlow-R': 313.206,
            'ReinFlow-S': 313.206,
            'DPPO': 437.830
        },
        'transport-img': {
            'ReinFlow-S': 662.121,
            'DPPO': 701.712
        },
        'kitchen-complete-v0':{
            'ReinFlow-S': 26.537 ,
            'DPPO': 83.158,
            'FQL': 5.249
        },
    }
    

method_name_dict = {
        'hopper':[
            {'original_name': 'ppo_reflow_mlp_ta4_td4', 'display_name': 'ReinFlow-R', 'color': color_dict['ReinFlow-R (ours)']},
            {'original_name': 'ppo_shortcut_mlp_ta4_td4', 'display_name': 'ReinFlow-S', 'color': color_dict['ReinFlow-S (ours)']},
            {'original_name': 'ppo_diffusion_mlp_ta4_td20_tdf10', 'display_name': 'DPPO', 'color': color_dict['DPPO']},
            {'original_name': 'fql_mlp_ta4_td4', 'display_name': 'FQL', 'color': color_dict['FQL']},
            # 新增: score 方法
            {'original_name': 'ppo_reflow_mlp_score_ta4_td4', 'display_name': 'Score-SDE (ours)', 'color': color_dict['Score-SDE (ours)']},
            {'original_name': 'ppo_reflow_mlp_with_score_gammanet_ta4_td4', 'display_name': 'ScoRe-Flow (ours)', 'color': color_dict['ScoRe-Flow (ours)']},
        ],
        'hopper-d4rl':[
            {'original_name': 'ppo_reflow_mlp_ta4_td4_d4rl', 'display_name': 'ReinFlow-R', 'color': color_dict['ReinFlow-R (ours)']},
            {'original_name': 'ppo_shortcut_mlp_ta4_td4_d4rl', 'display_name': 'ReinFlow-S', 'color': color_dict['ReinFlow-S (ours)']},
            {'original_name': 'ppo_diffusion_mlp_ta4_td20_d4rl', 'display_name': 'DPPO', 'color': color_dict['DPPO']},
            {'original_name': 'fql_mlp_ta4_td4', 'display_name': 'FQL', 'color': color_dict['FQL']},
            # 新增: score 方法
            {'original_name': 'ppo_reflow_mlp_score_ta4_td4', 'display_name': 'Score-SDE (ours)', 'color': color_dict['Score-SDE (ours)']},
            {'original_name': 'ppo_reflow_mlp_with_score_gammanet_ta4_td4', 'display_name': 'ScoRe-Flow (ours)', 'color': color_dict['ScoRe-Flow (ours)']},
        ],
        'walker':[
            {'original_name': 'ppo_reflow_mlp_ta4_td4', 'display_name': 'ReinFlow-R', 'color': color_dict['ReinFlow-R (ours)']},
            {'original_name': 'ppo_shortcut_mlp_ta4_td4', 'display_name': 'ReinFlow-S', 'color': color_dict['ReinFlow-S (ours)']},
            {'original_name': 'ppo_diffusion_mlp_ta4_td20_tdf10', 'display_name': 'DPPO', 'color': color_dict['DPPO']},
            {'original_name': 'fql_mlp_ta4_td4', 'display_name': 'FQL', 'color': color_dict['FQL']},
            # 新增: score 方法
            {'original_name': 'ppo_reflow_mlp_score_ta4_td4', 'display_name': 'Score-SDE (ours)', 'color': color_dict['Score-SDE (ours)']},
            {'original_name': 'ppo_reflow_mlp_with_score_gammanet_ta4_td4', 'display_name': 'ScoRe-Flow (ours)', 'color': color_dict['ScoRe-Flow (ours)']},
        ],
        'walker-d4rl':[
            {'original_name': 'ppo_reflow_mlp_ta4_td4_d4rl', 'display_name': 'ReinFlow-R', 'color': color_dict['ReinFlow-R (ours)']},
            {'original_name': 'ppo_shortcut_mlp_ta4_td4_d4rl', 'display_name': 'ReinFlow-S', 'color': color_dict['ReinFlow-S (ours)']},
            {'original_name': 'ppo_diffusion_mlp_ta4_td20_d4rl', 'display_name': 'DPPO', 'color': color_dict['DPPO']},
            {'original_name': 'fql_mlp_ta4_td4', 'display_name': 'FQL', 'color': color_dict['FQL']},
            # 新增: score 方法
            {'original_name': 'ppo_reflow_mlp_score_ta4_td4', 'display_name': 'Score-SDE (ours)', 'color': color_dict['Score-SDE (ours)']},
            {'original_name': 'ppo_reflow_mlp_with_score_gammanet_ta4_td4', 'display_name': 'ScoRe-Flow (ours)', 'color': color_dict['ScoRe-Flow (ours)']},
        ],
        'ant':[
            {'original_name': 'ppo_reflow_mlp_ta4_td4', 'display_name': 'ReinFlow-R', 'color': color_dict['ReinFlow-R (ours)']},
            {'original_name': 'ppo_shortcut_mlp_ta4_td4', 'display_name': 'ReinFlow-S', 'color': color_dict['ReinFlow-S (ours)']},
            {'original_name': 'ppo_diffusion_mlp_ta4_td20_tdf10', 'display_name': 'DPPO', 'color': color_dict['DPPO']},
            {'original_name': 'fql_mlp_ta4_td4', 'display_name': 'FQL', 'color': color_dict['FQL']},
            # 新增: score 方法
            {'original_name': 'ppo_reflow_mlp_score_ta4_td4', 'display_name': 'Score-SDE (ours)', 'color': color_dict['Score-SDE (ours)']},
            {'original_name': 'ppo_reflow_mlp_with_score_gammanet_ta4_td4', 'display_name': 'ScoRe-Flow (ours)', 'color': color_dict['ScoRe-Flow (ours)']},
        ],
        'ant-d4rl':[
            {'original_name': 'ppo_reflow_mlp_ta4_td4_d4rl', 'display_name': 'ReinFlow-R', 'color': color_dict['ReinFlow-R (ours)']},
            {'original_name': 'ppo_shortcut_mlp_ta4_td4_d4rl', 'display_name': 'ReinFlow-S', 'color': color_dict['ReinFlow-S (ours)']},
            {'original_name': 'ppo_diffusion_mlp_ta4_td20_d4rl', 'display_name': 'DPPO', 'color': color_dict['DPPO']},
            {'original_name': 'fql_mlp_ta4_td4', 'display_name': 'FQL', 'color': color_dict['FQL']},
            # 新增: score 方法
            {'original_name': 'ppo_reflow_mlp_score_ta4_td4', 'display_name': 'Score-SDE (ours)', 'color': color_dict['Score-SDE (ours)']},
            {'original_name': 'ppo_reflow_mlp_with_score_gammanet_ta4_td4', 'display_name': 'ScoRe-Flow (ours)', 'color': color_dict['ScoRe-Flow (ours)']},
        ],
        'humanoid':[
            {'original_name': 'ppo_reflow_mlp_ta4_td4', 'display_name': 'ReinFlow-R', 'color': color_dict['ReinFlow-R (ours)']},
            {'original_name': 'ppo_shortcut_mlp_ta4_td4', 'display_name': 'ReinFlow-S', 'color': color_dict['ReinFlow-S (ours)']},
            {'original_name': 'ppo_diffusion_mlp_ta4_td20_tdf10', 'display_name': 'DPPO', 'color': color_dict['DPPO']},
            {'original_name': 'fql_mlp_ta4', 'display_name': 'FQL', 'color': color_dict['FQL']},
            # 新增: score 方法
            {'original_name': 'ppo_reflow_mlp_score_ta4_td4', 'display_name': 'Score-SDE (ours)', 'color': color_dict['Score-SDE (ours)']},
            {'original_name': 'ppo_reflow_mlp_with_score_gammanet_ta4_td4', 'display_name': 'ScoRe-Flow (ours)', 'color': color_dict['ScoRe-Flow (ours)']},
        ],
        'humanoid-d4rl':[
            {'original_name': 'ppo_reflow_mlp_ta4_td4_d4rl', 'display_name': 'ReinFlow-R', 'color': color_dict['ReinFlow-R (ours)']},
            {'original_name': 'ppo_shortcut_mlp_ta4_td4_d4rl', 'display_name': 'ReinFlow-S', 'color': color_dict['ReinFlow-S (ours)']},
            {'original_name': 'ppo_diffusion_mlp_ta4_td20_d4rl', 'display_name': 'DPPO', 'color': color_dict['DPPO']},
            {'original_name': 'fql_mlp_ta4', 'display_name': 'FQL', 'color': color_dict['FQL']},
            # 新增: score 方法
            {'original_name': 'ppo_reflow_mlp_score_ta4_td4', 'display_name': 'Score-SDE (ours)', 'color': color_dict['Score-SDE (ours)']},
            {'original_name': 'ppo_reflow_mlp_with_score_gammanet_ta4_td4', 'display_name': 'ScoRe-Flow (ours)', 'color': color_dict['ScoRe-Flow (ours)']},
        ],
        'humanoid-regularize-compare':[
            # {'original_name': 'ppo_reflow_mlp_ta4_td4', 'display_name': r'Entropy $\alpha=0.03$', 'color': color_dict['ReinFlow-R (ours)']},
            # {'original_name': 'ppo_reflow_Wdiv_regularize_bc0.01_mlp_ta4_td4', 'display_name': r'Wasserstein $\alpha=0.01$', 'color': color_dict['ReinFlow-R (W2=0.01)']},
            # {'original_name': 'ppo_reflow_Wdiv_regularize_bc0.1_mlp_ta4_td4', 'display_name': r'Wasserstein $\alpha=0.1$', 'color': color_dict['ReinFlow-R (W2=0.1)']},
            # {'original_name': 'ppo_reflow_Wdiv_regularize_bc1.0_mlp_ta4_td4', 'display_name': r'Wasserstein $\alpha=1.0$', 'color': color_dict['ReinFlow-R (W2=1.0)']},
            {'original_name': 'ppo_reflow_mlp_ta4_td4', 'display_name': r'$\mathbf{\alpha=0.03}$', 'color': color_dict['ReinFlow-R (ours)']},
            {'original_name': 'ppo_reflow_Wdiv_regularize_bc0.01_mlp_ta4_td4', 'display_name': r'$\mathbf{\beta=0.01}$', 'color': color_dict['ReinFlow-R (W2=0.01)']},
            {'original_name': 'ppo_reflow_Wdiv_regularize_bc0.1_mlp_ta4_td4', 'display_name': r'$\mathbf{\beta=0.1}$', 'color': color_dict['ReinFlow-R (W2=0.1)']},
            {'original_name': 'ppo_reflow_Wdiv_regularize_bc1.0_mlp_ta4_td4', 'display_name': r'$\mathbf{\beta=1.0}$', 'color': color_dict['ReinFlow-R (W2=1.0)']},
        ],
        'gym-state-all': [
            {'original_name': 'qsm_diffusion_mlp_ta4_td20', 'display_name': 'QSM', 'color': color_dict['QSM']},
            {'original_name': 'dipo_diffusion_mlp_ta4_td20', 'display_name': 'DIPO', 'color': color_dict['DIPO']},
            {'original_name': 'idql_diffusion_mlp_ta4_td20', 'display_name': 'IDQL', 'color': color_dict['IDQL']},
            {'original_name': 'dql_diffusion_mlp_ta4_td20', 'display_name': 'DQL', 'color': color_dict['DQL']},
            {'original_name': 'rwr_diffusion_mlp_ta4_td20', 'display_name': 'DRWR', 'color': color_dict['DRWR']},
            {'original_name': 'awr_diffusion_mlp_ta4_td20', 'display_name': 'DAWR', 'color': color_dict['DAWR']},
            {'original_name': 'shortcut_mlp_img_ta4_td1', 'display_name': 'ReinFlow-S (ours)', 'color': color_dict['ReinFlow-S (ours)']},
            {'original_name': 'shortcut_mlp_img_td4_td1_datascale_32', 'display_name': 'ReinFlow-S, 32 (ours)', 'color': color_dict['ReinFlow-S, 32 (ours)']},
            {'original_name': 'shortcut_mlp_img_td4_td1_datascale_64', 'display_name': 'ReinFlow-S, 64 (ours)', 'color': color_dict['ReinFlow-S, 64 (ours)']},
            {'original_name': 'shortcut_mlp_img_td4_td1_datascale_100', 'display_name': 'ReinFlow-S, 100 (ours)', 'color': color_dict['ReinFlow-S, 100 (ours)']},
            {'original_name': 'reflow_mlp_img_ta4_td1', 'display_name': 'ReinFlow-R (ours)', 'color': color_dict['ReinFlow-R (ours)']},
            {'original_name': 'reflow_beta_mlp_img_1', 'display_name': 'ReinFlow-beta (ours)', 'color': color_dict['ReinFlow-beta (ours)']},
            {'original_name': 'reflow_logitnormal_mlp_img_td4_td1', 'display_name': 'ReinFlow-logitnormal (ours)', 'color': color_dict['ReinFlow-logitnormal (ours)']},
            {'original_name': 'diffusion_mlp_img_ta4_td100_tdf5', 'display_name': 'DPPO', 'color': color_dict['DPPO']},
        ],
        'can-img': [
            {'original_name': 'shortcut_mlp_ta4_td1_tdf1', 'display_name': 'ReinFlow-S', 'color': color_dict['ReinFlow-S (ours)']},
            {'original_name': 'flow_mlp_img_ta4_td1_tdf1', 'display_name': 'ReinFlow-R', 'color': color_dict['ReinFlow-R (ours)']},
            {'original_name': 'gaussian_mlp_img_ta4', 'display_name': 'Gaussian', 'color': color_dict['Gaussian']},
            {'original_name': 'diffusion_mlp_img_ta4_td100_tdf5', 'display_name': 'DPPO', 'color': color_dict['DPPO']},
            # 新增: score 方法
            {'original_name': 'flow_mlp_img_score_ta4_td4', 'display_name': 'Score-SDE (ours)', 'color': color_dict['Score-SDE (ours)']},
            {'original_name': 'flow_mlp_img_with_score_gammanet_ta4_td4_tdf1', 'display_name': 'ScoRe-Flow (ours)', 'color': color_dict['ScoRe-Flow (ours)']},
        ],
        'square-img': [
            {'original_name': 'reflow', 'display_name': 'ReinFlow-R', 'color': color_dict['ReinFlow-R (ours)']},
            {'original_name': 'shortcut_mlp_img_td4_td1_datascale_100', 'display_name': 'ReinFlow-S', 'color': color_dict['ReinFlow-S (ours)']},
            {'original_name': 'gaussian_mlp_img_ta4', 'display_name': 'Gaussian', 'color': color_dict['Gaussian']},
            {'original_name': 'diffusion_mlp_img_ta4_td100_tdf5', 'display_name': 'DPPO', 'color': color_dict['DPPO']},
            # 新增: score 方法
            {'original_name': 'flow_mlp_img_score_ta4_td4_gamma1', 'display_name': 'Score-SDE (ours)', 'color': color_dict['Score-SDE (ours)']},
            {'original_name': 'flow_mlp_img_with_score_gammanet_obs_ta4_td2_tdf1', 'display_name': 'ScoRe-Flow (ours)', 'color': color_dict['ScoRe-Flow (ours)']},
        ],
        'square-img-logitbeta': [
            {'original_name': 'reflow_mlp_img_td4_td1_tdf1', 'display_name': 'uniform', 'color': color_dict['uniform']},
            {'original_name': 'reflow_logitnormal_mlp_img_td4_td1', 'display_name': 'logitnormal', 'color': color_dict['logitnormal']},
            {'original_name': 'reflow_mlp_img_ta4_td1_tdf1', 'display_name': 'beta', 'color': color_dict['beta']},
        ],
        'square-img-denoise-steps': [
            {'original_name': 'flow_mlp_img_with_score_gammanet_obs_ta4_td1_tdf1', 'display_name': 'ScoRe-Flow (1 step)', 'color': color_dict['ScoRe-Flow (1 step)']},
            {'original_name': 'flow_mlp_img_with_score_gammanet_obs_ta4_td2_tdf1', 'display_name': 'ScoRe-Flow (2 steps)', 'color': color_dict['ScoRe-Flow (2 steps)']},
            {'original_name': 'flow_mlp_img_with_score_gammanet_ta4_td4_tdf1', 'display_name': 'ScoRe-Flow (4 steps)', 'color': color_dict['ScoRe-Flow (4 steps)']},
        ],
        'transport-img': [
            {'original_name': 'gaussian_mlp_img_ta8', 'display_name': 'Gaussian', 'color': color_dict['Gaussian']},
            {'original_name': 'diffusion_mlp_img_ta8_td100_tdf5', 'display_name': 'DPPO', 'color': color_dict['DPPO']},
            {'original_name': 'shortcut_mlp_img_ta8_td4_tdf4', 'display_name': 'ReinFlow-S', 'color': color_dict['ReinFlow-S (ours)']},
            # 新增: score 方法
            {'original_name': 'shortcut_mlp_img_score_ta8_td4', 'display_name': 'Score-SDE (ours)', 'color': color_dict['Score-SDE (ours)']},
            {'original_name': 'shortcut_mlp_img_with_score_gammanet_obs_ta8_td4_tdf4', 'display_name': 'ScoRe-Flow (ours)', 'color': color_dict['ScoRe-Flow (ours)']},
        ],
        'kitchen-complete-v0': [
            {'original_name': 'fql_mlp_ta4_td4', 'display_name': 'FQL', 'color': color_dict['FQL']},
            {'original_name': 'ppo_diffusion_mlp_ta4_td20_tdf10', 'display_name': 'DPPO', 'color': color_dict['DPPO']},
            {'original_name': 'ppo_shortcut_mlp_ta4_td4_tdf4', 'display_name': 'ReinFlow-S', 'color': color_dict['ReinFlow-S (ours)']},
            # 新增: score 方法
            {'original_name': 'ppo_shortcut_mlp_score_ta4_td4', 'display_name': 'Score-SDE (ours)', 'color': color_dict['Score-SDE (ours)']},
            {'original_name': 'ppo_shortcut_mlp_with_score_gammanet_ta4_td4_tdf4', 'display_name': 'ScoRe-Flow (ours)', 'color': color_dict['ScoRe-Flow (ours)']},
        ],
        'kitchen-complete-v0-denoise_step': [
            {'original_name': 'ppo_shortcut_mlp_ta4_td4_tdf4', 'display_name': 'ReinFlow-S (4 steps)', 'color': color_dict['ReinFlow-S (ours)']},
            {'original_name': 'ppo_shortcut_mlp_ta4_td2_tdf2', 'display_name': 'ReinFlow-S (2 steps)', 'color': color_dict['ReinFlow-S (2 steps)']},
            {'original_name': 'ppo_shortcut_mlp_ta4_td1_tdf1', 'display_name': 'ReinFlow-S (1 step)', 'color': color_dict['ReinFlow-S (1 step)']},
        ],
        'kitchen-partial-v0': [
            {'original_name': 'fql_mlp_ta4_td4', 'display_name': 'FQL', 'color': color_dict['FQL']},
            {'original_name': 'ppo_diffusion_mlp_ta4_td20_tdf10', 'display_name': 'DPPO', 'color': color_dict['DPPO']},
            {'original_name': 'ppo_shortcut_mlp_ta4_td4_tdf4', 'display_name': 'ReinFlow-S', 'color': color_dict['ReinFlow-S (ours)']},
            # 新增: score 方法
            {'original_name': 'ppo_shortcut_mlp_score_ta4_td4_gamma1', 'display_name': 'Score-SDE (ours)', 'color': color_dict['Score-SDE (ours)']},
            {'original_name': 'ppo_shortcut_mlp_with_score_gammanet_obs_ta4_td8_tdf4', 'display_name': 'ScoRe-Flow (ours)', 'color': color_dict['ScoRe-Flow (ours)']},
        ],
        'kitchen-mixed-v0': [
            {'original_name': 'fql_mlp_ta4_td4', 'display_name': 'FQL', 'color': color_dict['FQL']},
            {'original_name': 'ppo_diffusion_mlp_ta4_td20_tdf10', 'display_name': 'DPPO', 'color': color_dict['DPPO']},
            {'original_name': 'ppo_shortcut_mlp_ta4_td4_tdf4', 'display_name': 'ReinFlow-S', 'color': color_dict['ReinFlow-S (ours)']},
            # 新增: score 方法
            {'original_name': 'ppo_shortcut_mlp_score_ta4_td4_gamma1', 'display_name': 'Score-SDE (ours)', 'color': color_dict['Score-SDE (ours)']},
            {'original_name': 'ppo_shortcut_mlp_with_score_gammanet_ta4_td4_tdf4', 'display_name': 'ScoRe-Flow (ours)', 'color': color_dict['ScoRe-Flow (ours)']},
        ],
        'kitchen-partial-v0-denoise_step':[
            {'original_name': 'ppo_shortcut_mlp_ta4_td4_tdf4', 'display_name': 'ReinFlow-S (4 steps)', 'color': color_dict['ReinFlow-S (ours)']},
            {'original_name': 'ppo_shortcut_mlp_ta4_td2_tdf2', 'display_name': 'ReinFlow-S (2 steps)', 'color': color_dict['ReinFlow-S (2 steps)']},
            {'original_name': 'ppo_shortcut_mlp_ta4_td1_tdf1', 'display_name': 'ReinFlow-S (1 step)', 'color': color_dict['ReinFlow-S (1 step)']},
        ],
        'kitchen-partial-v0-sigma_s_t':[
            {'original_name': 'ppo_shortcut_learn_s_mlp_ta4_td4_tdf4', 'display_name': r'ReinFlow-S: $\sigma(s)$', 'color': color_dict['ReinFlow-R (sigma(s,t))']},
            {'original_name': 'ppo_shortcut_mlp_ta4_td4_tdf4', 'display_name': r'ReinFlow-S: $\sigma(s,t)$', 'color': color_dict['ReinFlow-R (sigma(s))']},
        ]
    }

max_n_step_dict={
        'can-img': 100,
        'square-img': 300,
        'square-img-logitbeta': 300,
        'transport-img': 180,
        'hopper': np.float32('inf'),
        'hopper-d4rl': 1_000,
        'walker': 1_000,
        'walker-d4rl': 1_000,
        'ant': 1_000,
        'ant-d4rl': 1_000,
        'humanoid': 2_00,
        'humanoid-d4rl': 2_00,
        'humanoid-regularize-compare': 2_00,
        'kitchen-complete-v0': 1765000,
        'kitchen-complete-v0-denoise_step': np.float32('inf'),
        'kitchen-partial-v0': np.float32('inf'),
        'kitchen-partial-v0-denoise_step': np.float32('inf'),
        'kitchen-partial-v0-sigma_s_t': np.float32('inf'),
        'kitchen-mixed-v0': np.float32('inf'),
        'square-img-denoise-steps': 300,
    }