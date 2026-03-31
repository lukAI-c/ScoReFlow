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
PPO ShortCut with Learned Epsilon Network (EpsNet)

与 pposhortcut_score.py 的区别:
    - eps_t 不再由固定调度器决定，而是由可学习的 MLP 输出
    - 同一个 α(t) 同时控制 drift 和 diffusion（耦合）

SDE 公式:
    drift  = vt + α(t) · st
    diffusion = √(2 · α(t))

其中 α(t) 由 EpsNet (MLP) 输出:
    α(t) = (1 - t) · MLP(t) · gamma_score
    time_mask(t) = (1 - t)  保证 t→1 时 α→0，防止 score 爆炸

ShortCut 关键区别:
    - 使用 ShortCutFlowMLP (额外接受步长 d 参数)
    - forward(xt, t_batch, d_batch, cond) 而非 forward(xt, t_batch, cond)
"""

import torch
from torch import nn
import copy
import math
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


class PPOShortCutScoreEpsNet(nn.Module, ScoreFunctionMixin):
    """
    PPO with ShortCut Flow Matching Policy using learned epsilon network.

    SDE:
        dxt = [vt + α(t)·st] dt + √(2·α(t)) dWt

    α(t) 由 EpsNet (MLP) 学习，drift 和 diffusion 通过同一个 α(t) 耦合。
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
                 epsilon_t,                     # 基准噪声系数（用于初始化 MLP 输出量级）
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
                 score_clip_value: float = 5.0,
                 # ========== EpsNet 参数 ==========
                 eps_mlp_hidden_dim: int = 32,
                 eps_mlp_type: str = 'mlp',       # 'mlp', 'linear', 'fixed'
                 gamma_score: float = 1.0,
                 eps_max_clamp: float = 2.0,
                 lamda=1,
                 load_weights_in_init: bool = True,
                 use_ema=True
                 ):

        super().__init__()
        self.device = device
        self.inference_steps = inference_steps
        self.action_dim = act_dim
        self.horizon_steps = horizon_steps
        self.act_dim_total = self.horizon_steps * self.action_dim
        self.act_min = act_min
        self.act_max = act_max
        self.obs_dim = obs_dim
        self.cond_steps = cond_steps
        self.score_clip_value = score_clip_value
        self.epsilon_t = epsilon_t
        self.epsilon_schedule = 'learned_mlp'   # 兼容父类 agent 访问此属性

        self.randn_clip_value = randn_clip_value
        self.logprob_min = logprob_min
        self.logprob_max = logprob_max
        self.clip_ploss_coef = clip_ploss_coef
        self.clip_ploss_coef_base = clip_ploss_coef_base
        self.clip_ploss_coef_rate = clip_ploss_coef_rate
        self.clip_vloss_coef = clip_vloss_coef
        self.denoised_clip_value = denoised_clip_value
        self.logprob_debug_sample = logprob_debug_sample
        self.logprob_debug_recalculate = logprob_debug_recalculate

        # ========== EpsNet 参数 ==========
        self.gamma_score = gamma_score
        self.eps_mlp_type = eps_mlp_type
        self.eps_max_clamp = eps_max_clamp

        # ========== 创建 EpsNet ==========
        self._init_eps_net(eps_mlp_hidden_dim)

        # Load pretrained policy (frozen)
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
        logging.info("Cloned ShortCut policy for fine-tuning (with learned EpsNet)")

        self.critic = critic
        self.critic = self.critic.to(self.device)

        self.report_network_params()

    # =====================================================================
    #  EpsNet: Learned ε(t) MLP
    # =====================================================================

    def _init_eps_net(self, hidden_dim: int):
        """
        初始化 EpsNet: t → α(t) > 0

        结构: MLP(t) → Softplus → α_raw
        最终: α(t) = (1 - t) · α_raw · gamma_score
        """
        input_dim = 1

        if self.eps_mlp_type == 'fixed':
            self.eps_net = None
            logging.info(f"EpsNet: FIXED (epsilon_t={self.epsilon_t})")
        elif self.eps_mlp_type == 'linear':
            self.eps_net = nn.Sequential(
                nn.Linear(input_dim, 1),
                nn.Softplus()
            ).to(self.device)
            logging.info("EpsNet: LINEAR")
        elif self.eps_mlp_type == 'mlp':
            self.eps_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
                nn.Softplus()
            ).to(self.device)

            # 初始化使 Softplus 输出 ≈ epsilon_t / gamma_score
            target = self.epsilon_t / max(self.gamma_score, 1e-6)
            init_bias = math.log(max(math.exp(target) - 1, 1e-6))
            nn.init.constant_(self.eps_net[-2].weight, 0)
            nn.init.constant_(self.eps_net[-2].bias, init_bias)

            logging.info(f"EpsNet: MLP (hidden={hidden_dim}, init_target≈{target:.4f})")
        else:
            raise ValueError(f"Unknown eps_mlp_type: {self.eps_mlp_type}")

    def get_learned_epsilon(self, t: Tensor) -> Tensor:
        """
        获取 learned ε(t)

        Args:
            t: [B] 时间步 ∈ [0, 1]
        Returns:
            alpha_t: [B, 1, 1]

        公式: α(t) = (1 - t) · MLP(t) · gamma_score
        """
        B = t.shape[0]
        time_mask = (1.0 - t).view(B, 1)

        if self.eps_net is None:
            alpha_t = torch.full((B, 1), self.epsilon_t, device=self.device) * time_mask
        else:
            t_in = t.view(B, 1)
            alpha_raw = self.eps_net(t_in)       # [B, 1], > 0
            alpha_t = time_mask * alpha_raw * self.gamma_score
            alpha_t = alpha_t.clamp(max=self.gamma_score * self.eps_max_clamp)

        return alpha_t.unsqueeze(-1)             # [B, 1, 1]

    # =====================================================================
    #  Utility
    # =====================================================================

    def get_actor_params(self):
        """actor_ft + eps_net 参数，供 optimizer 使用"""
        params = list(self.actor_ft.parameters())
        if self.eps_net is not None:
            params += list(self.eps_net.parameters())
        return params

    def check_gradient_flow(self):
        print(f"actor_ft requires_grad: {next(self.actor_ft.parameters()).requires_grad}")
        if self.eps_net is not None:
            print(f"eps_net requires_grad: {next(self.eps_net.parameters()).requires_grad}")

    def report_network_params(self):
        eps_net_params = 0
        if self.eps_net is not None:
            eps_net_params = sum(p.numel() for p in self.eps_net.parameters()) / 1e6
        logging.info(
            f"Number of network parameters: Total: {sum(p.numel() for p in self.parameters())/1e6:.4f}M. "
            f"Actor: {sum(p.numel() for p in self.actor_old.parameters())/1e6:.4f}M. "
            f"Actor (finetune): {sum(p.numel() for p in self.actor_ft.parameters())/1e6:.4f}M. "
            f"Critic: {sum(p.numel() for p in self.critic.parameters())/1e6:.4f}M. "
            f"EpsNet: {eps_net_params:.6f}M"
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
        else:
            logging.warning("No actor policy path provided.")

    # =====================================================================
    #  Sampling
    # =====================================================================

    @torch.no_grad()
    def sample_first_point(self, B: int) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = Normal(torch.zeros(B, self.horizon_steps * self.action_dim), 1.0)
        xt = dist.sample()
        log_prob = dist.log_prob(xt).sum(-1).to(self.device)
        xt = xt.reshape(B, self.horizon_steps, self.action_dim).to(self.device)
        return xt, log_prob

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
        """
        ShortCut Score-based stochastic action sampling with learned ε(t).

        SDE: dxt = [vt + α(t)·st] dt + √(2·α(t)) dWt
        ShortCut: forward(xt, t, d, cond) 额外接受步长 d
        """
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

        xt, log_prob_init = self.sample_first_point(B)
        if ret_logprob and account_for_initial_stochasticity:
            log_prob += log_prob_init
            log_prob_steps += 1
            if self.logprob_debug_sample:
                log_prob_list.append(log_prob_init.mean().item())

        if save_chains:
            x_chain[:, 0] = xt

        for i in range(self.inference_steps):
            t = steps[i]
            t_batch = t.expand(B)
            d_batch = torch.full((B,), dt, device=self.device)  # ShortCut 步长

            # 1. Velocity field (ShortCut 多接受 d_batch)
            vt = self.actor_ft.forward(xt, t_batch, d_batch, cond)

            # 2. Score function
            st = self.compute_score(xt, vt, t_batch)
            st = torch.clamp(st, -self.score_clip_value, self.score_clip_value)

            # 3. Learned epsilon
            alpha_t = self.get_learned_epsilon(t_batch)  # [B, 1, 1]

            # 4. Drift: vt + α(t)·st
            drift = vt + alpha_t * st
            xt_mean = xt + self.lamda * drift * dt
            if clip_intermediate_actions:
                xt_mean = xt_mean.clamp(-self.denoised_clip_value, self.denoised_clip_value)

            # 5. Diffusion std: √(2·α(t)·Δt)  —— 与 drift 共享同一个 α(t)
            diffusion_std = torch.sqrt(2 * alpha_t * dt + 1e-8)  # [B, 1, 1]

            # 6. Update
            if not eval_mode:
                noise = torch.randn_like(xt).clamp(-self.randn_clip_value, self.randn_clip_value)
                xt = xt_mean + diffusion_std * noise
            else:
                xt = xt_mean

            if i == self.inference_steps - 1:
                xt = xt.clamp(self.act_min, self.act_max)

            # 7. Log probability
            if ret_logprob:
                dist = Normal(xt_mean.flatten(-2, -1), diffusion_std.expand_as(xt).flatten(-2, -1))
                logprob_transition = dist.log_prob(xt.flatten(-2, -1)).sum(-1)
                if self.logprob_debug_sample:
                    log_prob_list.append(logprob_transition.mean().item())
                log_prob += logprob_transition
                log_prob_steps += 1

            if save_chains:
                x_chain[:, i + 1] = xt

        if ret_logprob:
            if normalize_denoising_horizon:
                log_prob = log_prob / log_prob_steps
            if normalize_act_space_dimension:
                log_prob = log_prob / self.act_dim_total
            if self.logprob_debug_sample:
                print(f"log_prob_list={log_prob_list}")

        if ret_logprob:
            if save_chains:
                return (xt, x_chain, log_prob)
            return (xt, log_prob)
        else:
            if save_chains:
                return (xt, x_chain)
            return xt

    # =====================================================================
    #  Log Probability
    # =====================================================================

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
        """
        Log probability with learned ε(t) for ShortCut.

        Transition: p(x_{t+1}|x_t) = N(x_t + [vt + α(t)·st]·dt,  2·α(t)·dt)
        """
        logprob = 0.0
        logprob_steps = 0

        B = x_chain.shape[0]

        init_dist = Normal(
            torch.zeros(B, self.horizon_steps * self.action_dim, device=self.device),
            1.0
        )
        logprob_init = init_dist.log_prob(x_chain[:, 0].reshape(B, -1)).sum(-1)

        if account_for_initial_stochasticity:
            logprob += logprob_init
            logprob_steps += 1

        dt = 1.0 / self.inference_steps
        steps = torch.linspace(0, 1 - dt, self.inference_steps, device=self.device)
        noise_std_values = []

        for i in range(self.inference_steps):
            t = steps[i]
            t_batch = t.expand(B)
            d_batch = torch.full((B,), dt, device=self.device)
            xt = x_chain[:, i]

            # Velocity field (ShortCut)
            vt = self.actor_ft.forward(xt, t_batch, d_batch, cond)

            # Score function
            st = self.compute_score(xt, vt, t_batch)
            st = torch.clamp(st, -self.score_clip_value, self.score_clip_value)

            # Learned epsilon: 同一个 α(t) 控制 drift 和 diffusion
            alpha_t = self.get_learned_epsilon(t_batch)  # [B, 1, 1]

            # Drift
            drift = vt + alpha_t * st
            mean = xt + self.lamda * drift * dt
            if clip_intermediate_actions:
                mean = mean.clamp(-self.denoised_clip_value, self.denoised_clip_value)

            # Diffusion std: √(2·α(t)·dt)
            std = torch.sqrt(2 * alpha_t * dt + 1e-8)  # [B, 1, 1]
            noise_std_values.append(std.mean().item())

            trans_dist = Normal(
                mean.flatten(-2, -1),
                std.expand_as(xt).flatten(-2, -1)
            )

            xt_next = x_chain[:, i + 1].flatten(-2, -1)
            logprob += trans_dist.log_prob(xt_next).sum(-1)
            logprob_steps += 1

        if self.logprob_debug_recalculate:
            log.info(f"logprob_init={logprob_init.mean().item():.3f}, logprob_total={logprob.mean().item():.3f}")

        if get_entropy:
            entropy_rate_est = -logprob

        if normalize_denoising_horizon:
            logprob = logprob / logprob_steps
            if get_entropy:
                entropy_rate_est = entropy_rate_est / logprob_steps
        if normalize_act_space_dimension:
            logprob = logprob / self.act_dim_total
            if get_entropy:
                entropy_rate_est = entropy_rate_est / self.act_dim_total

        if verbose_entropy_stats and get_entropy:
            log.info(f"Entropy Percentiles: "
                     f"10%={entropy_rate_est.quantile(0.1):.2f}, "
                     f"50%={entropy_rate_est.median():.2f}, "
                     f"90%={entropy_rate_est.quantile(0.9):.2f}")

        noise_std_mean = torch.tensor(sum(noise_std_values) / len(noise_std_values), device=self.device)

        if get_entropy:
            if get_chains_stds:
                return logprob, entropy_rate_est, noise_std_mean
            return logprob, entropy_rate_est
        else:
            if get_chains_stds:
                return logprob, noise_std_mean
            return logprob

    # =====================================================================
    #  PPO Loss
    # =====================================================================

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
        """PPO loss with learned EpsNet for ShortCut."""
        newlogprobs, entropy, noise_std = self.get_logprobs(
            obs, chains,
            get_entropy=True,
            normalize_denoising_horizon=normalize_denoising_horizon,
            normalize_act_space_dimension=normalize_act_space_dimension,
            verbose_entropy_stats=verbose,
            clip_intermediate_actions=clip_intermediate_actions,
            account_for_initial_stochasticity=account_for_initial_stochasticity
        )

        if verbose:
            log.info(f"oldlogprobs.min={oldlogprobs.min():5.3f}, max={oldlogprobs.max():5.3f}, std={oldlogprobs.std():5.3f}")
            log.info(f"newlogprobs.min={newlogprobs.min():5.3f}, max={newlogprobs.max():5.3f}, std={newlogprobs.std():5.3f}")

        newlogprobs = newlogprobs.clamp(min=self.logprob_min, max=self.logprob_max)
        oldlogprobs = oldlogprobs.clamp(min=self.logprob_min, max=self.logprob_max)

        if verbose:
            if oldlogprobs.min() < self.logprob_min:
                log.info("WARNING: old logprobs too low, potential policy collapse")
            if newlogprobs.min() < self.logprob_min:
                log.info("WARNING: new logprobs too low, potential policy collapse")

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if verbose:
            with torch.no_grad():
                log.info(f"Advantage stats: mean={advantages.mean().item():.3f}, std={advantages.std().item():.3f}")
                corr = torch.corrcoef(torch.stack([advantages, returns]))[0, 1].item()
                log.info(f"Advantage-Reward Correlation: {corr:.2f}")

        logratio = newlogprobs - oldlogprobs
        ratio = logratio.exp()

        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > self.clip_ploss_coef).float().mean().item()

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_ploss_coef, 1 + self.clip_ploss_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        newvalues = self.critic(obs).view(-1)
        v_loss = 0.5 * ((newvalues - returns) ** 2).mean()
        if self.clip_vloss_coef:
            v_clipped = torch.clamp(newvalues, oldvalues - self.clip_vloss_coef, oldvalues + self.clip_vloss_coef)
            v_loss = 0.5 * torch.max((newvalues - returns) ** 2, (v_clipped - returns) ** 2).mean()

        entropy_loss = -entropy.mean()

        bc_loss = 0.0
        if use_bc_loss:
            if bc_loss_type == 'W2':
                z = torch.zeros((obs['state'].shape[0], self.horizon_steps, self.action_dim), device=self.device)
                a_old = self.actor_old.sample_action(cond=obs, inference_steps=self.inference_steps,
                                                     clip_intermediate_actions=True,
                                                     act_range=[self.act_min, self.act_max], z=z)
                a_ft = self.actor_ft.sample_action(cond=obs, inference_steps=self.inference_steps,
                                                   clip_intermediate_actions=True,
                                                   act_range=[self.act_min, self.act_max], z=z)
                bc_loss = F.mse_loss(a_old.detach(), a_ft)
            else:
                raise NotImplementedError

        # --- 记录 alpha(t) 曲线用于 wandb ---
        alpha_curve_dict = {}
        if self.eps_net is not None:
            with torch.no_grad():
                dt_log = 1.0 / self.inference_steps
                t_steps = torch.linspace(0, 1 - dt_log, self.inference_steps, device=self.device)
                alpha_vals = self.get_learned_epsilon(t_steps).squeeze().cpu().numpy()
                t_numpy = t_steps.cpu().numpy()
                for step_idx, (t_val, a_val) in enumerate(zip(t_numpy, alpha_vals)):
                    alpha_curve_dict[f"alpha_t/step_{step_idx:02d}_t={t_val:.2f}"] = float(a_val)

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
            alpha_curve_dict,
        )
