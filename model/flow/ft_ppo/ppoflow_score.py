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



import torch
from torch import nn
import copy
import torch.nn.functional as F
from torch import Tensor
import logging
log = logging.getLogger(__name__)
from collections import namedtuple
from typing import Tuple
from torch.distributions.normal import Normal
from model.flow.mlp_flow_score import FlowMLP
from model.flow.score_utils import ScoreFunctionMixin
Sample = namedtuple("Sample", "trajectories chains")

class PPOFlow(nn.Module, ScoreFunctionMixin):
    
    """
    PPO with Flow Matching Policy using score-based stochastic sampling.

    核心更新公式 (SDE):
        dxt = [bt(xt) + εt·st(xt)] dt + √(2εt) dWt

    离散形式:
        ak+1 = ak + [bt(ak) + εt·st(ak)]·Δt + √(2εt·Δt)·ε

    其中:
        bt(x) = 速度场 (FlowMLP 输出)
        st(x) = (t * bt(x) - x) / (1 - t) 分数函数
        εt = epsilon_t 噪声系数
    """
    def __init__(self,
                 device,
                 policy,
                 critic,
                 actor_policy_path,
                 act_dim,
                 horizon_steps,
                 act_min,
                 act_max,
                 obs_dim,
                 cond_steps,
                 inference_steps,
                 epsilon_t,                     # 噪声系数 εt
                 randn_clip_value,
                 logprob_min,
                 logprob_max,
                 clip_ploss_coef,
                 clip_ploss_coef_base,
                 clip_ploss_coef_rate,
                 clip_vloss_coef,
                 denoised_clip_value,
                 logprob_debug_sample,
                 logprob_debug_recalculate,
                 epsilon_schedule='constant',   # epsilon schedule: 'constant', 'linear_decay', 'cosine'
                 lamda = 1, # default: 1
                 ):
        
        super().__init__()
        self.device = device
        self.inference_steps = inference_steps          # number of steps for inference.
        # self.ft_denoising_steps = ft_denoising_steps    # could be adjusted
        self.action_dim = act_dim
        self.horizon_steps = horizon_steps
        self.act_dim_total = self.horizon_steps * self.action_dim
        self.act_min = act_min
        self.act_max = act_max
        
        self.obs_dim = obs_dim
        self.cond_steps = cond_steps
        
        # Score-based stochastic sampling parameters
        self.epsilon_t: float = epsilon_t
        self.epsilon_schedule: str = epsilon_schedule

        # prevent extreme values sampled from gaussian
        self.randn_clip_value: float = randn_clip_value

        # logprobability bounds for stability
        self.logprob_min: float = logprob_min
        self.logprob_max: float = logprob_max

        # PPO clipping coefficients
        self.clip_ploss_coef: float = clip_ploss_coef
        self.clip_ploss_coef_base: float = clip_ploss_coef_base
        self.clip_ploss_coef_rate: float = clip_ploss_coef_rate
        self.clip_vloss_coef: float = clip_vloss_coef

        # clip intermediate actions during inference
        self.denoised_clip_value: float = denoised_clip_value
        self.logprob_debug_sample = logprob_debug_sample
        self.logprob_debug_recalculate = logprob_debug_recalculate

        # Load pretrained policy (frozen, for reference)
        self.actor_old: FlowMLP = policy
        self.load_policy(actor_policy_path, use_ema=True)
        for param in self.actor_old.parameters():
            param.requires_grad = False
        self.actor_old.to(self.device)

        # Create fine-tuning policy (trainable copy)
        self.actor_ft: FlowMLP = copy.deepcopy(self.actor_old)
        for param in self.actor_ft.parameters():
            param.requires_grad = True
        self.actor_ft.to(self.device)
        self.lamda = lamda
        logging.info("Cloned policy for fine-tuning (score-based, no noise network)")

        self.critic = critic
        self.critic = self.critic.to(self.device)

        self.report_network_params()
    
    # def init_actor_ft(self, policy_copy):
    #     self.actor_ft = NoisyFlowMLP(policy=policy_copy,
    #                                 denoising_steps=self.inference_steps,
    #                                 learn_explore_noise_from = self.inference_steps - self.ft_denoising_steps,
    #                                 inital_noise_scheduler_type=self.noise_scheduler_type,
    #                                 min_logprob_denoising_std = self.min_logprob_denoising_std,
    #                                 max_logprob_denoising_std = self.max_logprob_denoising_std,
    #                                 learn_explore_time_embedding=self.learn_explore_time_embedding,
    #                                 time_dim_explore=self.time_dim_explore,
    #                                 use_time_independent_noise=self.use_time_independent_noise,
    #                                 device=self.device,
    #                                 noise_hidden_dims=self.noise_hidden_dims,
    #                                 activation_type=self.explore_net_activation_type
    #                                 )
    
    
    # def get_epsilon_at_time(self, t: float, training_progress: float = 0.0) -> float:
    #     """
    #     根据时间步 t 和训练进度计算 epsilon 值。

    #     支持的调度类型:
    #         - 'constant': 固定噪声系数 εt = ε₀
    #         - 'linear_decay': 线性衰减 εt = ε₀ × (1 - t)
    #         - 'cosine': 余弦衰减 εt = ε₀ × 0.5 × (1 + cos(πt))
    #         - 'sqrt_decay': 平方根衰减 εt = ε₀ × √(1 - t)
    #         - 'exponential_decay': 指数衰减 εt = ε₀ × exp(-λt)
    #         - 'warmup_decay': 先增后减 εt = ε₀ × sin(πt)
    #         - 'quadratic_decay': 二次衰减 εt = ε₀ × (1 - t)²
    #         - 'inverse_sqrt': 反平方根 εt = ε₀ / √(1 + t)
    #         - 'adaptive': 使用可学习的 epsilon
    #         - 'training_decay': 随训练进度衰减

    #     Args:
    #         t: 当前时间步 (0 到 1)
    #         training_progress: 训练进度 (0 到 1)，用于某些调度策略

    #     Returns:
    #         epsilon_t: 当前的噪声系数
    #     """
    #     import math
    #     eps_0 = self.epsilon_t
    #     eps_min = getattr(self, 'epsilon_min', 0.01)  # 默认值 0.001

    #     if self.epsilon_schedule == 'constant':
    #         return eps_0
    #     elif self.epsilon_schedule == 'linear_decay':
    #         # εt = ε₀ * (1 - t)
    #         return max(eps_min, eps_0 * (1 - t))
    #     elif self.epsilon_schedule == 'cosine':
    #         # εt = ε₀ * 0.5 * (1 + cos(πt))
    #         return max(eps_min, eps_0 * 0.5 * (1 + math.cos(math.pi * t)))
    #     elif self.epsilon_schedule == 'sqrt_decay':
    #         # εt = ε₀ * sqrt(1 - t)
    #         return max(eps_min, eps_0 * math.sqrt(max(0, 1 - t)))
    #     elif self.epsilon_schedule == 'exponential_decay':
    #         # εt = ε₀ * exp(-λt)
    #         decay_rate = getattr(self, 'epsilon_decay_rate', 2.0)  # 默认值 2.0
    #         return max(eps_min, eps_0 * math.exp(-decay_rate * t))
    #     elif self.epsilon_schedule == 'warmup_decay':
    #         # εt = ε₀ * sin(πt) - 先增后减，t=0.5时最大
    #         return max(eps_min, eps_0 * math.sin(math.pi * t))
    #     elif self.epsilon_schedule == 'quadratic_decay':
    #         # εt = ε₀ * (1 - t)² - 比线性更快衰减
    #         return max(eps_min, eps_0 * (1 - t) ** 2)
    #     elif self.epsilon_schedule == 'inverse_sqrt':
    #         # εt = ε₀ / sqrt(1 + t) - 缓慢衰减
    #         return max(eps_min, eps_0 / math.sqrt(1 + t))
    #     elif self.epsilon_schedule == 'adaptive':
    #         # 使用可学习的 epsilon
    #         adaptive_eps = getattr(self, 'adaptive_epsilon', None)
    #         if adaptive_eps is not None:
    #             return max(eps_min, adaptive_eps.item())
    #         else:
    #             return eps_0  # fallback to constant
    #     elif self.epsilon_schedule == 'training_decay':
    #         # 随训练进度衰减: εt = ε₀ × (1 - 0.5×progress) × (1 - t)
    #         decay_factor = 1 - 0.5 * training_progress
    #         return max(eps_min, eps_0 * decay_factor * (1 - t))
    #     else:
    #         log.warning(f"Unknown epsilon_schedule: {self.epsilon_schedule}, using constant")
    #         return eps_0

    
    def check_gradient_flow(self):
        # print(f"{next(self.actor_ft.policy.parameters()).requires_grad}") #True
        # print(f"{next(self.actor_ft.mlp_logvar.parameters()).requires_grad}")#True
        # print(f"{next(self.actor_ft.time_embedding_explore.parameters()).requires_grad}")#True
        # print(f"{self.actor_ft.logvar_min.requires_grad}")#False
        # print(f"{self.actor_ft.logvar_max.requires_grad}")#False
        print(f"actor_ft requires_grad: {next(self.actor_ft.parameters()).requires_grad}")
    def report_network_params(self):
        logging.info(
            f"Number of network parameters: Total: {sum(p.numel() for p in self.parameters())/1e6} M. Actor:{sum(p.numel() for p in self.actor_old.parameters())/1e6} M. Actor (finetune) : {sum(p.numel() for p in self.actor_ft.parameters())/1e6} M. Critic: {sum(p.numel() for p in self.critic.parameters())/1e6} M"
        )
    
    def load_policy(self, network_path, use_ema=False):
        log.info(f"loading policy from %s" % network_path)
        if network_path:
            print(f"network_path={network_path}, self.device={self.device}")
            model_data = torch.load(network_path, map_location=self.device, weights_only=True)
            actor_network_data = {k.replace("network.", ""): v for k, v in model_data["model"].items()}
            if use_ema:
                ema_actor_network_data = {k.replace("network.", ""): v for k, v in model_data["ema"].items()}
                self.actor_old.load_state_dict(ema_actor_network_data)
                logging.info("Loaded ema actor policy from %s", network_path)
            else:
                self.actor_old.load_state_dict(actor_network_data)
                logging.info("Loaded actor policy from %s", network_path)
            print(f"actor_network_data={actor_network_data.keys()}")
        else:
            logging.warning("No actor policy path provided. Not loading any actor policy. Start from randomly initialized policy.")
    
    @torch.no_grad()
    def sample_first_point(self, B:int)->Tuple[torch.Tensor, torch.Tensor]:
        '''
        B: batchsize
        outputs:
            xt: torch.Tensor of shape `[batchsize, self.horizon_steps, self.action_dim]`
            log_prob: torch.Tensor of shape `[batchsize]`
        '''
        dist = Normal(torch.zeros(B, self.horizon_steps* self.action_dim), 1.0)
        xt= dist.sample()
        log_prob = dist.log_prob(xt).sum(-1).to(self.device)                    # mean() or sum() 
        xt=xt.reshape(B, self.horizon_steps, self.action_dim).to(self.device)
        return xt, log_prob
    
    def get_logprobs(self,
                     cond: dict,
                     x_chain: Tensor,
                     get_entropy=False,
                     normalize_denoising_horizon=False,
                     normalize_act_space_dimension=False,
                     clip_intermediate_actions=True,
                     verbose_entropy_stats=True,
                     debug=True,
                     account_for_initial_stochasticity=False,
                     get_chains_stds=True
                     ):
        '''
        Score-based log probability calculation.

        SDE transition: dxt = [bt(xt) + εt·st(xt)] dt + √(2εt) dWt

        Transition distribution:
            p(xt+1|xt, s) = N(xt+1 | xt + [bt + εt·st]·dt, 2εt·dt)

        inputs:
            x_chain: [B, inference_steps+1, horizon_steps, action_dim]
        outputs:
            logprob: [B]
            entropy_rate_est: [B] (if get_entropy=True)
            noise_std_mean: scalar (if get_chains_stds=True)
        '''
        import numpy as np

        logprob = 0.0
        joint_entropy = 0.0
        entropy_rate_est = 0.0
        logprob_steps = 0

        B = x_chain.shape[0]

        # initial probability: p(x0) = N(0, I)
        init_dist = Normal(
            torch.zeros(B, self.horizon_steps * self.action_dim, device=self.device),
            1.0
        )
        logprob_init = init_dist.log_prob(x_chain[:, 0].reshape(B, -1)).sum(-1)

        if get_entropy:
            entropy_init = init_dist.entropy().sum(-1)
        if account_for_initial_stochasticity:
            logprob += logprob_init
            if get_entropy:
                joint_entropy += entropy_init
            logprob_steps += 1

        # Transition probabilities using score-based SDE
        dt = 1.0 / self.inference_steps
        steps = torch.linspace(0, 1 - dt, self.inference_steps, device=self.device)

        # Storage for noise std (for monitoring)
        noise_std_values = []

        for i in range(self.inference_steps):
            t = steps[i]
            t_batch = t.expand(B)
            xt = x_chain[:, i]  # [B, horizon_steps, action_dim]

            # Get velocity field bt(x)
            vt = self.actor_ft.forward(xt, t_batch, cond)  # [B, horizon_steps, action_dim]

            # Compute score: st(x) = (t * bt(x) - x) / (1 - t)
            st = self.actor_ft.compute_score(xt, vt, t_batch)  # [B, horizon_steps, action_dim]

            # Compute epsilon at this timestep
            # if self.epsilon_schedule == 'constant':
            #     eps_t = self.epsilon_t
            # elif self.epsilon_schedule == 'linear_decay':
            #     eps_t = self.epsilon_t * (1 - t.item())
            # elif self.epsilon_schedule == 'cosine':
            #     eps_t = self.epsilon_t * 0.5 * (1 + np.cos(np.pi * t.item()))
            # else:
            #     eps_t = self.epsilon_t
            eps_t = self.get_epsilon_at_time(t.item())
            
            # Transition mean: xt + [bt + εt·st]·dt
            drift = vt + eps_t * st
            mean = xt + self.lamda * drift * dt
            if clip_intermediate_actions:
                mean = mean.clamp(-self.denoised_clip_value, self.denoised_clip_value)

            # Transition std: √(2εt·dt)
            std = np.sqrt(2 * eps_t * dt)
            noise_std_values.append(std)

            # Transition distribution
            trans_dist = Normal(mean.flatten(-2, -1), std)

            # Log probability of next state
            xt_next = x_chain[:, i + 1].flatten(-2, -1)
            logprob_trans = trans_dist.log_prob(xt_next).sum(-1)
            logprob += logprob_trans

            if get_entropy:
                entropy_trans = trans_dist.entropy().sum(-1)
                joint_entropy += entropy_trans

            logprob_steps += 1

        if self.logprob_debug_recalculate:
            log.info(f"logprob_init={logprob_init.mean().item():.3f}, logprob_total={logprob.mean().item():.3f}")

        if get_entropy:
            entropy_rate_est = joint_entropy / logprob_steps
        if normalize_denoising_horizon:
            logprob = logprob / logprob_steps
        if normalize_act_space_dimension:
            logprob = logprob / self.act_dim_total
            if get_entropy:
                entropy_rate_est = entropy_rate_est / self.act_dim_total

        if verbose_entropy_stats and get_entropy:
            log.info(f"entropy_rate_est={entropy_rate_est.shape} Entropy Percentiles: 10%={entropy_rate_est.quantile(0.1):.2f}, 50%={entropy_rate_est.median():.2f}, 90%={entropy_rate_est.quantile(0.9):.2f}")

        noise_std_mean = torch.tensor(np.mean(noise_std_values), device=self.device)

        if get_entropy:
            if get_chains_stds:
                return logprob, entropy_rate_est, noise_std_mean
            return logprob, entropy_rate_est
        else:
            if get_chains_stds:
                return logprob, noise_std_mean
            return logprob
    
    @torch.no_grad()
    def get_actions(self,
                    cond: dict,
                    eval_mode: bool,
                    save_chains=False,
                    normalize_denoising_horizon=False,
                    normalize_act_space_dimension=False,
                    clip_intermediate_actions=True,
                    account_for_initial_stochasticity=True,
                    ret_logprob=True
                    ):
        '''
        Score-based stochastic action sampling.

        SDE: dxt = [bt(xt) + εt·st(xt)] dt + √(2εt) dWt

        Discrete update:
            ak+1 = ak + [bt(ak) + εt·st(ak)]·Δt + √(2εt·Δt)·ε

        inputs:
            cond: dict with 'state' key, shape (B, To, Do)
            eval_mode: if True, use deterministic sampling (no noise)
            save_chains: whether to return trajectory chain
            ret_logprob: whether to compute log probability
        outputs:
            xt: (B, horizon_steps, action_dim)
            x_chain: (B, inference_steps+1, horizon_steps, action_dim) if save_chains
            logprob: (B,) if ret_logprob
        '''
        import numpy as np

        B = cond["state"].shape[0]
        dt = 1.0 / self.inference_steps
        steps = torch.linspace(0, 1 - dt, self.inference_steps, device=self.device)

        if save_chains:
            x_chain = torch.zeros(
                (B, self.inference_steps + 1, self.horizon_steps, self.action_dim),
                device=self.device
            )
        if ret_logprob:
            log_prob = 0.0
            log_prob_steps = 0
            if self.logprob_debug_sample:
                log_prob_list = []

        # Sample initial point from N(0, I)
        xt, log_prob_init = self.sample_first_point(B)
        if ret_logprob and account_for_initial_stochasticity:
            log_prob += log_prob_init
            log_prob_steps += 1
            if self.logprob_debug_sample:
                log_prob_list.append(log_prob_init.mean().item())

        if save_chains:
            x_chain[:, 0] = xt

        # ----------------------- Score-based SDE integration -----------------------
        for i in range(self.inference_steps):
            t = steps[i]
            t_batch = t.expand(B)

            # 1. Get velocity field bt(x)
            vt = self.actor_ft.forward(xt, t_batch, cond)  # [B, Ta, Da]

            # 2. Compute score: st(x) = (t * bt(x) - x) / (1 - t)
            st = self.actor_ft.compute_score(xt, vt, t_batch)  # [B, Ta, Da]

            # 3. Compute epsilon at this timestep
            # if self.epsilon_schedule == 'constant':
            #     eps_t = self.epsilon_t
            # elif self.epsilon_schedule == 'linear_decay':
            #     eps_t = self.epsilon_t * (1 - t.item())
            # elif self.epsilon_schedule == 'cosine':
            #     eps_t = self.epsilon_t * 0.5 * (1 + np.cos(np.pi * t.item()))
            # else:
            #     eps_t = self.epsilon_t
            eps_t = self.get_epsilon_at_time(t.item()) # TODO: 这里的随机性应该如何保证呢？ 采用调度器就相当于确定了吧

            # 4. Compute drift and diffusion
            # Drift: [bt + εt·st]·Δt
            # lamda = 0.125
            # drift = (vt + lamda * eps_t * st) * dt
            
            # print 更全面的统计信息
            # print("vt: mean={:.3f}, std={:.3f}, abs_mean={:.3f}".format(
            #     vt.mean().item(), vt.std().item(), vt.abs().mean().item()))
            # print("st: mean={:.3f}, std={:.3f}, abs_mean={:.3f}".format(
            #     st.mean().item(), st.std().item(), st.abs().mean().item()))
            # print("eps_t: {:.3f}".format(eps_t))

            # 相对贡献比例
            vt_norm = vt[0].norm().item()
            st_norm = st[0].norm().item()
            ratio_00 = (eps_t * st_norm) / (vt_norm + 1e-8)  # 避免除零
            # print("vt_norm: {:.3f}, st_norm: {:.3f}, st/vt ratio: {:.3f}".format(
            #     vt_norm, st_norm, ratio_00))

            # 更新量统计
            drift = vt + eps_t * st
            # print("drift_norm: {:.3f}, vt_contrib: {:.3f}, st_contrib: {:.3f}".format(
            # drift.norm().item(),
            # vt_norm / (drift[0].norm().item() + 1e-8),
            # (eps_t * st_norm) / (drift[0].norm().item() + 1e-8)))
            # print("----------------------------------------")
            # Diffusion std: √(2εt·Δt)
            diffusion_std = np.sqrt(2 * eps_t * dt)
            # 5. Update: ak+1 = ak + drift + diffusion * noise
            xt_mean = xt + self.lamda * drift * dt
            # print(self.lamda)
            if clip_intermediate_actions:
                xt_mean = xt_mean.clamp(-self.denoised_clip_value, self.denoised_clip_value)

            if not eval_mode:
                # Stochastic update with noise
                noise = torch.randn_like(xt)
                noise = noise.clamp(-self.randn_clip_value, self.randn_clip_value)
                xt = xt_mean + diffusion_std * noise
            else:
                # Deterministic update (no noise)
                xt = xt_mean

            # Clip final action
            if i == self.inference_steps - 1:
                xt = xt.clamp(self.act_min, self.act_max)

            # 6. Compute log probability
            if ret_logprob:
                dist = Normal(xt_mean.flatten(-2, -1), diffusion_std)
                logprob_transition = dist.log_prob(xt.flatten(-2, -1)).sum(-1)
                if self.logprob_debug_sample:
                    log_prob_list.append(logprob_transition.mean().item())
                log_prob += logprob_transition
                log_prob_steps += 1

            # 7. Save chain
            if save_chains:
                x_chain[:, i + 1] = xt

        # Normalize log probability if requested
        if ret_logprob:
            if normalize_denoising_horizon:
                log_prob = log_prob / log_prob_steps
            if normalize_act_space_dimension:
                log_prob = log_prob / self.act_dim_total
            if self.logprob_debug_sample:
                print(f"log_prob_list={log_prob_list}")

        # Return results
        if ret_logprob:
            if save_chains:
                return (xt, x_chain, log_prob)
            return (xt, log_prob)
        else:
            if save_chains:
                return (xt, x_chain)
            return xt
      
    
    def loss(
        self,
        obs,
        chains,
        returns,
        oldvalues,
        advantages,
        oldlogprobs,
        use_bc_loss=False,
        bc_loss_type='W2',
        normalize_denoising_horizon=False,
        normalize_act_space_dimension=False,
        verbose=True,
        clip_intermediate_actions=True,
        account_for_initial_stochasticity=True
    ):
        """
        PPO loss
        obs: dict with key state/rgb; more recent obs at the end
            "state": (B, To, Do)
            "rgb": (B, To, C, H, W)
        chains: (B, K+1, Ta, Da)
        returns: (B, )
        values: (B,)
        advantages: (B,)
        oldlogprobs: (B,)
        use_bc_loss: whether to add BC regularization loss
        normalize_act_space_dimension: whether to normalize logprobs and entropy rates over all horiton steps and action dimensions
        reward_horizon: action horizon that backpropagates gradient, omitted for now.
        Here, B = n_steps x n_envs
        """
        
        newlogprobs, entropy, noise_std = self.get_logprobs(obs, 
                                                            chains, 
                                                            get_entropy=True, 
                                                            normalize_denoising_horizon=normalize_denoising_horizon,
                                                            normalize_act_space_dimension=normalize_act_space_dimension, 
                                                            verbose_entropy_stats=verbose, 
                                                            clip_intermediate_actions=clip_intermediate_actions,
                                                            account_for_initial_stochasticity=account_for_initial_stochasticity)
        if verbose:
            log.info(f"oldlogprobs.min={oldlogprobs.min():5.3f}, max={oldlogprobs.max():5.3f}, std of oldlogprobs={oldlogprobs.std():5.3f}")
            log.info(f"newlogprobs.min={newlogprobs.min():5.3f}, max={newlogprobs.max():5.3f}, std of newlogprobs={newlogprobs.std():5.3f}")
        
        
        newlogprobs = newlogprobs.clamp(min=self.logprob_min, max=self.logprob_max)
        oldlogprobs = oldlogprobs.clamp(min=self.logprob_min, max=self.logprob_max)
        if verbose:
            if oldlogprobs.min() < self.logprob_min: log.info(f"WARNINIG: old logprobs too low, potential policy collapse detected, should encourage exploration.")
            if newlogprobs.min() < self.logprob_min: log.info(f"WARNINIG: new logprobs too low, potential policy collapse detected, should encourage exploration.")
            if newlogprobs.max() > self.logprob_max: log.info(f"WARNINIG: new logprobs too high")
            if oldlogprobs.max() > self.logprob_max: log.info(f"WARNINIG: old logprobs too high")
        # empirically we noticed that when the min of logprobs gets too negative (say, below -3) or when the std gets larger than 0.5 (usually these two events happen simultaneously) t
        # the perfomance drops. 
        # batch normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if verbose:
            with torch.no_grad():
                advantage_stats = {
                    "mean":f"{advantages.mean().item():2.3f}",
                    "std": f"{advantages.std().item():2.3f}",
                    "max": f"{advantages.max().item():2.3f}",
                    "min": f"{advantages.min().item():2.3f}"
                }
                log.info(f"Advantage stats: {advantage_stats}")
                corr = torch.corrcoef(torch.stack([advantages, returns]))[0,1].item()
                log.info(f"Advantage-Reward Correlation: {corr:.2f}")
        
        # Get ratio
        logratio = newlogprobs - oldlogprobs
        ratio = logratio.exp()
        
        # Get kl difference and whether value clipped
        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > self.clip_ploss_coef).float().mean().item()

        # Policy loss
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_ploss_coef, 1 + self.clip_ploss_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalues = self.critic(obs).view(-1)
        v_loss = 0.5 * ((newvalues - returns) ** 2).mean()
        if self.clip_vloss_coef: # better not use. 
            v_clipped = torch.clamp(newvalues, oldvalues -self.clip_vloss_coef, oldvalues + self.clip_vloss_coef)
            v_loss = 0.5 *torch.max((newvalues - returns) ** 2, (v_clipped - returns) ** 2).mean()
        if verbose:
            with torch.no_grad():
                mse = F.mse_loss(newvalues, returns)
                log.info(f"Value/Reward alignment: MSE={mse.item():.3f}")
        
        # Entropy loss
        entropy_loss = -entropy.mean()
        # Monitor policy entropy distribution
        if verbose:
            with torch.no_grad():
                log.info(f"Entropy Percentiles: 10%={entropy.quantile(0.1):.2f}, 50%={entropy.median():.2f}, 90%={entropy.quantile(0.9):.2f}")
        
        # bc loss
        bc_loss = 0.0
        if use_bc_loss:
            if bc_loss_type=='W2':
                # add wasserstein divergence loss via action supervision
                z = torch.zeros((obs['state'].shape[0], self.horizon_steps, self.action_dim), device=self.device)
                a_ω = self.actor_old.sample_action(cond=obs, inference_steps=self.inference_steps, clip_intermediate_actions=True, act_range=[self.act_min, self.act_max], z=z)
                a_θ = self.actor_ft.sample_action(cond=obs, inference_steps=self.inference_steps, clip_intermediate_actions=True, act_range=[self.act_min, self.act_max], z=z)
                bc_loss = F.mse_loss(a_ω.detach(), a_θ)
            else:
                raise NotImplementedError
        return (
            pg_loss,
            entropy_loss,
            v_loss,
            bc_loss,
            clipfrac,
            approx_kl.item(),
            ratio.mean().item(),
            oldlogprobs.min(),
            oldlogprobs.max(),
            oldlogprobs.std(),
            newlogprobs.min(),
            newlogprobs.max(),
            newlogprobs.std(),
            noise_std.item(),
            newvalues.mean().item(),#Q function
        )