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


"""
PPO ShortCut with Score-based Exploration

基于 ppoflow_score.py，使用 ShortCut 的时间步进方式
核心区别：时间步使用 linspace(0, 1-dt, K) 而不是 linspace(0, 1, K+1)

核心更新公式 (SDE):
    dxt = [bt(xt) + εt·st(xt)] dt + √(2εt) dWt

离散形式:
    ak+1 = ak + [bt(ak) + εt·st(ak)]·Δt + √(2εt·Δt)·ε

其中:
    bt(x) = 速度场 (ShortCutFlowMLP 输出)
    st(x) = (t * bt(x) - x) / (1 - t) 分数函数
    εt = epsilon_t 噪声系数
"""

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
from model.flow.mlp_shortcut import ShortCutFlowMLP
from model.flow.score_utils import ScoreFunctionMixin
Sample = namedtuple("Sample", "trajectories chains")

class PPOShortCutScore(nn.Module, ScoreFunctionMixin):
    
    """
    PPO with ShortCut Flow Matching Policy using score-based stochastic sampling.

    核心更新公式 (SDE):
        dxt = [bt(xt) + εt·st(xt)] dt + √(2εt) dWt

    离散形式:
        ak+1 = ak + [bt(ak) + εt·st(ak)]·Δt + √(2εt·Δt)·ε

    其中:
        bt(x) = 速度场 (ShortCutFlowMLP 输出)
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
                 gamma_score = 1.0,
                 load_weights_in_init: bool = True,
                 use_ema=True
                 ):
        
        super().__init__()
        self.gamma_score = gamma_score
        self.device = device
        self.inference_steps = inference_steps          # number of steps for inference.
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
        self.actor_old: ShortCutFlowMLP = policy
        if load_weights_in_init:
            self.load_policy(actor_policy_path, use_ema=use_ema)
        for param in self.actor_old.parameters():
            param.requires_grad = False
        self.actor_old.to(self.device)

        # Create fine-tuning policy (trainable copy)
        self.actor_ft: ShortCutFlowMLP = copy.deepcopy(self.actor_old)
        for param in self.actor_ft.parameters():
            param.requires_grad = True
        self.actor_ft.to(self.device)
        self.lamda = lamda
        logging.info("Cloned ShortCut policy for fine-tuning (score-based, no noise network)")

        self.critic = critic
        self.critic = self.critic.to(self.device)

        self.report_network_params()

    def check_gradient_flow(self):
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

    # def get_epsilon_at_time(self, t: float, training_progress: float = 0.0) -> float:
    #     """
    #     根据时间步 t 计算 epsilon 值
    #     """
    #     import math
    #     eps_0 = self.epsilon_t
    #     eps_min = getattr(self, 'epsilon_min', 0.01)

    #     if self.epsilon_schedule == 'constant':
    #         return eps_0
    #     elif self.epsilon_schedule == 'linear_decay':
    #         return max(eps_min, eps_0 * (1 - t))
    #     elif self.epsilon_schedule == 'cosine':
    #         return max(eps_min, eps_0 * 0.5 * (1 + math.cos(math.pi * t)))
    #     else:
    #         log.warning(f"Unknown epsilon_schedule: {self.epsilon_schedule}, using constant")
    #         return eps_0

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
        log_prob = dist.log_prob(xt).sum(-1).to(self.device)
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
        Score-based log probability calculation for ShortCut.

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
        logprob_steps = 0

        B = x_chain.shape[0]

        # initial probability: p(x0) = N(0, I)
        init_dist = Normal(
            torch.zeros(B, self.horizon_steps * self.action_dim, device=self.device),
            1.0
        )
        logprob_init = init_dist.log_prob(x_chain[:, 0].reshape(B, -1)).sum(-1)

        if account_for_initial_stochasticity:
            logprob += logprob_init
            logprob_steps += 1

        # Transition probabilities using score-based SDE
        # ShortCut 使用 linspace(0, 1-dt, K)
        dt = 1.0 / self.inference_steps
        steps = torch.linspace(0, 1 - dt, self.inference_steps, device=self.device)

        # Storage for noise std (for monitoring)
        noise_std_values = []

        for i in range(self.inference_steps):
            t = steps[i]
            t_batch = t.expand(B)
            d_batch = torch.full((B,), dt, device=self.device)
            xt = x_chain[:, i]  # [B, horizon_steps, action_dim]

            # Get velocity field bt(x)
            vt = self.actor_ft.forward(xt, t_batch, d_batch, cond)  # [B, horizon_steps, action_dim]

            # Compute score: st(x) = (t * bt(x) - x) / (1 - t)
            st = self.compute_score(xt, vt, t_batch)  # [B, horizon_steps, action_dim]

            # Compute epsilon at this timestep
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

            logprob_steps += 1

        if self.logprob_debug_recalculate:
            log.info(f"logprob_init={logprob_init.mean().item():.3f}, logprob_total={logprob.mean().item():.3f}")

        # 使用 -logprob 作为熵的蒙特卡洛估计
        if get_entropy:
            entropy_rate_est = -logprob  # shape: (B,)
        if normalize_denoising_horizon:
            logprob = logprob / logprob_steps
            if get_entropy:
                entropy_rate_est = entropy_rate_est / logprob_steps
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
        Score-based stochastic action sampling for ShortCut.

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
        # ShortCut 使用 linspace(0, 1-dt, K)
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

        # Score-based SDE integration
        for i in range(self.inference_steps):
            t = steps[i]
            t_batch = t.expand(B)
            d_batch = torch.full((B,), dt, device=self.device)

            # 1. Get velocity field bt(x)
            vt = self.actor_ft.forward(xt, t_batch, d_batch, cond)  # [B, Ta, Da]

            # 2. Compute score: st(x) = (t * bt(x) - x) / (1 - t)
            st = self.compute_score(xt, vt, t_batch)  # [B, Ta, Da]

            # 3. Compute epsilon at this timestep
            eps_t = self.get_epsilon_at_time(t.item())

            # 4. Compute drift and diffusion
            drift = vt + eps_t * st
            diffusion_std = np.sqrt(2 * eps_t * dt)

            # 5. Update: ak+1 = ak + drift + diffusion * noise
            xt_mean = xt + self.lamda * drift * dt
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
        PPO loss for ShortCut Score
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
        if self.clip_vloss_coef:
            v_clipped = torch.clamp(newvalues, oldvalues -self.clip_vloss_coef, oldvalues + self.clip_vloss_coef)
            v_loss = 0.5 *torch.max((newvalues - returns) ** 2, (v_clipped - returns) ** 2).mean()
        if verbose:
            with torch.no_grad():
                mse = F.mse_loss(newvalues, returns)
                log.info(f"Value/Reward alignment: MSE={mse.item():.3f}")

        # Entropy loss
        entropy_loss = -entropy.mean()
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
            newvalues.mean().item(),
        )

