# MIT License
# Copyright (c) 2026 ScoRe-Flow Authors
# PPOShortCut with Score-guided drift - Frozen Velocity Field variant
# 冻结速度场 v_t，只学习 variance (noise net) + score scheduler (alpha_t)

"""
Frozen-V variant of PPOShortCutWithScore + GammaNet

与 pposhortcut_with_score_gammanet.py 的唯一区别:
    速度场 v_t 完全冻结，梯度只流向:
    1. explore_noise_net  (方差 / 探索噪声网络)
    2. score_scheduler    (alpha_t 时间调度网络)

核心公式不变:
    drift = vt + alpha_t * st
    xt+1 = xt + drift * dt + nt * epsilon

但 vt 在训练中不再更新，只有 nt 和 alpha_t 被优化。
"""

import torch
from torch import nn
import copy
import torch.nn.functional as F
from torch import Tensor
import logging
import numpy as np
from collections import namedtuple
from typing import Tuple, Optional
from torch.distributions.normal import Normal
from model.flow.mlp_shortcut import ShortCutFlowMLP, NoisyShortCutFlowMLP
from model.flow.score_utils import ScoreFunctionMixin

log = logging.getLogger(__name__)
Sample = namedtuple("Sample", "trajectories chains")


class PPOShortCutWithScoreFrozenV(nn.Module, ScoreFunctionMixin):
    """
    PPOShortCut + Score + GammaNet, with frozen velocity field.

    可训练参数:
        - actor_ft.explore_noise_net  (方差网络)
        - score_scheduler             (alpha_t 网络)
        - critic                      (价值网络)

    冻结参数:
        - actor_old                   (参考策略, 始终冻结)
        - actor_ft.policy             (速度场, 本变体中冻结)
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
                 noise_scheduler_type,
                 inference_steps,
                 ft_denoising_steps,
                 randn_clip_value,
                 min_sampling_denoising_std,
                 min_logprob_denoising_std,
                 logprob_min,
                 logprob_max,
                 clip_ploss_coef,
                 clip_ploss_coef_base,
                 clip_ploss_coef_rate,
                 clip_vloss_coef,
                 denoised_clip_value,
                 max_logprob_denoising_std,
                 time_dim_explore,
                 learn_explore_time_embedding,
                 use_time_independent_noise,
                 noise_hidden_dims,
                 logprob_debug_sample,
                 logprob_debug_recalculate,
                 explore_net_activation_type,
                 # ========== Score 相关参数 ==========
                 gamma_score: float = 1.0,
                 score_clip_value: float = 10.0,
                 # ========== Score Scheduler 参数 ==========
                 use_score_scheduler: bool = True,
                 score_scheduler_hidden_dim: int = 16,
                 score_scheduler_type: str = 'mlp',
                 # ========== 固定 alpha 消融选项 ==========
                 alpha_constant: float = None,  # None=使用 scheduler; 0.0/1.0=固定常数 alpha
                 ):

        super().__init__()
        self.device = device
        self.inference_steps = inference_steps
        self.ft_denoising_steps = ft_denoising_steps
        self.action_dim = act_dim
        self.horizon_steps = horizon_steps
        self.act_dim_total = self.horizon_steps * self.action_dim
        self.act_min = act_min
        self.act_max = act_max
        self.obs_dim = obs_dim
        self.cond_steps = cond_steps
        self.noise_scheduler_type = noise_scheduler_type
        self.randn_clip_value = randn_clip_value
        self.min_sampling_denoising_std = min_sampling_denoising_std
        self.min_logprob_denoising_std = min_logprob_denoising_std
        self.max_logprob_denoising_std = max_logprob_denoising_std
        self.logprob_min = logprob_min
        self.logprob_max = logprob_max
        self.clip_ploss_coef = clip_ploss_coef
        self.clip_ploss_coef_base = clip_ploss_coef_base
        self.clip_ploss_coef_rate = clip_ploss_coef_rate
        self.clip_vloss_coef = clip_vloss_coef
        self.denoised_clip_value = denoised_clip_value
        self.logprob_debug_sample = logprob_debug_sample
        self.logprob_debug_recalculate = logprob_debug_recalculate
        self.learn_explore_time_embedding = learn_explore_time_embedding
        self.time_dim_explore = time_dim_explore
        self.use_time_independent_noise = use_time_independent_noise
        self.noise_hidden_dims = noise_hidden_dims
        self.explore_net_activation_type = explore_net_activation_type

        # ========== Score 参数 ==========
        self.gamma_score = gamma_score
        self.score_clip_value = score_clip_value
        self.use_score_scheduler = use_score_scheduler
        self.score_scheduler_type = score_scheduler_type
        self.alpha_constant = alpha_constant

        if alpha_constant is not None:
            logging.info(f"alpha_constant={alpha_constant}: score runs at fixed α, scheduler disabled")

        # ========== 创建 Score Scheduler 网络 ==========
        self._init_score_scheduler(score_scheduler_hidden_dim)

        # 加载预训练 policy
        self.actor_old: ShortCutFlowMLP = policy
        self.load_policy(actor_policy_path, use_ema=True)
        for param in self.actor_old.parameters():
            param.requires_grad = False
        self.actor_old.to(self.device)

        # 创建可训练副本
        policy_copy = copy.deepcopy(self.actor_old)
        for param in policy_copy.parameters():
            param.requires_grad = True

        self.init_actor_ft(policy_copy)

        # ★ 核心修改: 冻结速度场 v_t
        for param in self.actor_ft.policy.parameters():
            param.requires_grad = False
        logging.info("Velocity field FROZEN. Only training: noise_net + score_scheduler")

        self.critic = critic
        self.critic = self.critic.to(self.device)

        self.report_network_params()

    def _init_score_scheduler(self, hidden_dim: int):
        """初始化自适应 Score Scheduler (仅使用时间 t)"""
        input_dim = 1

        if not self.use_score_scheduler or self.score_scheduler_type == 'fixed':
            self.score_scheduler = None
            logging.info(f"Score scheduler: FIXED (gamma_score={self.gamma_score})")
        elif self.score_scheduler_type == 'linear':
            self.score_scheduler = nn.Sequential(
                nn.Linear(input_dim, 1),
                nn.Softplus()
            ).to(self.device)
            logging.info(f"Score scheduler: LINEAR (input={input_dim})")
        elif self.score_scheduler_type == 'mlp':
            self.score_scheduler = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
                nn.Softplus()
            ).to(self.device)
            nn.init.constant_(self.score_scheduler[-2].weight, 0)
            nn.init.constant_(self.score_scheduler[-2].bias, -2.0)
            logging.info(f"Score scheduler: Adaptive MLP (input={input_dim}, hidden={hidden_dim})")
        else:
            raise ValueError(f"Unknown score_scheduler_type: {self.score_scheduler_type}")

    def get_alpha_t(self, t: Tensor) -> Tensor:
        """获取时间依赖的 Score 系数 alpha_t

        alpha_constant 不为 None 时: 直接返回固定常数，忽略 time_mask 和 scheduler
            alpha_constant=0.0 → 关闭 score 引导（纯 NoisyShortCut）
            alpha_constant=1.0 → 常数强度 score（验证 time_mask 的必要性）
        alpha_constant 为 None 时: 使用 time_mask * scheduler (默认行为)
        """
        B = t.shape[0]

        # 固定常数模式
        if self.alpha_constant is not None:
            return torch.full((B, 1, 1), self.alpha_constant, device=self.device)

        # 正常 scheduler 模式
        time_mask = (1.0 - t).view(B, 1)
        if self.score_scheduler is None:
            alpha_t = torch.full((B, 1), self.gamma_score, device=self.device) * time_mask
        else:
            t_in = t.view(B, 1)
            learned_gain = self.score_scheduler(t_in)
            alpha_t = time_mask * learned_gain * self.gamma_score
            alpha_t = alpha_t.clamp(max=self.gamma_score * 2.0)

        return alpha_t.unsqueeze(-1)  # [B, 1, 1]

    def init_actor_ft(self, policy_copy):
        self.actor_ft = NoisyShortCutFlowMLP(
            policy=policy_copy,
            denoising_steps=self.inference_steps,
            learn_explore_noise_from=self.inference_steps - self.ft_denoising_steps,
            inital_noise_scheduler_type=self.noise_scheduler_type,
            min_logprob_denoising_std=self.min_logprob_denoising_std,
            max_logprob_denoising_std=self.max_logprob_denoising_std,
            learn_explore_time_embedding=self.learn_explore_time_embedding,
            time_dim_explore=self.time_dim_explore,
            use_time_independent_noise=self.use_time_independent_noise,
            device=self.device,
            noise_hidden_dims=self.noise_hidden_dims,
            activation_type=self.explore_net_activation_type,
        )
        self.actor_ft = self.actor_ft.to(self.device)

    # ========== ★ 核心新增: 只返回可训练参数 ==========

    def get_actor_params(self):
        """
        只返回 variance (noise net) + score scheduler 的参数.
        速度场 actor_ft.policy 已冻结, 不包含在内.
        """
        params = list(self.actor_ft.explore_noise_net.parameters())
        if hasattr(self.actor_ft, 'time_embedding_explore'):
            params += list(self.actor_ft.time_embedding_explore.parameters())
        if self.score_scheduler is not None:
            params += list(self.score_scheduler.parameters())
        return params

    def get_trainable_params(self):
        """Get all trainable parameters (actor trainable parts + critic)"""
        params = self.get_actor_params()
        params += list(self.critic.parameters())
        return params

    def get_critic_params(self):
        """Get critic parameters"""
        return self.critic.parameters()

    def report_network_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_v_params = sum(p.numel() for p in self.actor_ft.policy.parameters())
        noise_net_params = sum(p.numel() for p in self.actor_ft.explore_noise_net.parameters())
        scheduler_params = sum(p.numel() for p in self.score_scheduler.parameters()) if self.score_scheduler else 0

        logging.info(
            f"PPOShortCutWithScoreFrozenV - Network params:\n"
            f"  Total: {total_params/1e6:.2f}M, Trainable: {trainable_params/1e6:.4f}M\n"
            f"  Actor_old (frozen): {sum(p.numel() for p in self.actor_old.parameters())/1e6:.2f}M\n"
            f"  Velocity field (frozen): {frozen_v_params/1e6:.2f}M\n"
            f"  Noise net (trainable): {noise_net_params}\n"
            f"  ScoreScheduler (trainable): {scheduler_params}\n"
            f"  Critic (trainable): {sum(p.numel() for p in self.critic.parameters())/1e6:.2f}M"
        )
        logging.info(f"Score params: gamma_score={self.gamma_score}, score_clip_value={self.score_clip_value}, "
                     f"scheduler_type={self.score_scheduler_type}, use_scheduler={self.use_score_scheduler}")

    def load_policy(self, network_path, use_ema=False):
        log.info(f"Loading policy from {network_path}")
        if network_path:
            model_data = torch.load(network_path, map_location=self.device, weights_only=True)
            actor_network_data = {k.replace("network.", ""): v for k, v in model_data["model"].items()}
            if use_ema and "ema" in model_data:
                ema_data = {k.replace("network.", ""): v for k, v in model_data["ema"].items()}
                self.actor_old.load_state_dict(ema_data)
                logging.info("Loaded EMA actor policy")
            elif use_ema:
                logging.warning("No 'ema' key found, falling back to 'model'")
                self.actor_old.load_state_dict(actor_network_data)
            else:
                self.actor_old.load_state_dict(actor_network_data)
        else:
            logging.warning("No actor policy path provided.")

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
        """采样动作，使用 Score 增强的 drift (解耦控制)"""
        B = cond["state"].shape[0]
        dt = 1.0 / self.inference_steps
        steps = torch.linspace(0, 1 - dt, self.inference_steps).repeat(B, 1).to(self.device)

        if save_chains:
            x_chain = torch.zeros((B, self.inference_steps + 1, self.horizon_steps, self.action_dim), device=self.device)
        if ret_logprob:
            log_prob = 0.0
            log_prob_steps = 0

        xt = torch.randn((B, self.horizon_steps, self.action_dim), device=self.device)

        if ret_logprob and account_for_initial_stochasticity:
            init_dist = Normal(torch.zeros_like(xt), 1.0)
            log_prob_init = init_dist.log_prob(xt).sum(dim=(-2, -1))
            log_prob += log_prob_init
            log_prob_steps += 1

        if save_chains:
            x_chain[:, 0] = xt

        for i in range(self.inference_steps):
            t = steps[:, i]
            d = torch.full((B,), dt, device=self.device)

            vt, nt = self.actor_ft.forward(xt, t, d, cond, learn_exploration_noise=False, step=i)

            st = self.compute_score(xt, vt, t)
            st = torch.clamp(st, -self.score_clip_value, self.score_clip_value)

            alpha_t = self.get_alpha_t(t)
            drift = vt + alpha_t * st

            xt_mean = xt + drift * dt
            if clip_intermediate_actions:
                xt_mean = xt_mean.clamp(-self.denoised_clip_value, self.denoised_clip_value)

            std = nt.unsqueeze(-1).reshape(xt.shape)
            std = torch.clamp(std, min=self.min_sampling_denoising_std)
            dist = Normal(xt_mean, std)

            if not eval_mode:
                xt = dist.sample().clamp_(
                    dist.loc - self.randn_clip_value * dist.scale,
                    dist.loc + self.randn_clip_value * dist.scale
                ).to(self.device)
            else:
                xt = xt_mean

            if i == self.inference_steps - 1:
                xt = xt.clamp(self.act_min, self.act_max)

            if ret_logprob:
                logprob_transition = dist.log_prob(xt).sum(dim=(-2, -1))
                log_prob += logprob_transition
                log_prob_steps += 1

            if save_chains:
                x_chain[:, i + 1] = xt

        if ret_logprob:
            if normalize_denoising_horizon:
                log_prob = log_prob / log_prob_steps
            if normalize_act_space_dimension:
                log_prob = log_prob / self.act_dim_total

            if save_chains:
                return (xt, x_chain, log_prob)
            return (xt, log_prob)
        else:
            if save_chains:
                return (xt, x_chain)
            return xt

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
        """计算 log 概率，使用 Score 增强的 drift (解耦控制)"""
        B = x_chain.shape[0]
        logprob = 0.0
        joint_entropy = 0.0
        logprob_steps = 0

        chains_prev = x_chain[:, :-1, :, :].flatten(-2, -1)
        chains_next = x_chain[:, 1:, :, :].flatten(-2, -1)
        chains_stds = torch.zeros_like(chains_prev, device=self.device)

        init_dist = Normal(torch.zeros(B, self.horizon_steps * self.action_dim, device=self.device), 1.0)
        logprob_init = init_dist.log_prob(x_chain[:, 0].reshape(B, -1)).sum(-1)

        if get_entropy:
            entropy_init = init_dist.entropy().sum(-1)

        if account_for_initial_stochasticity:
            logprob += logprob_init
            if get_entropy:
                joint_entropy += entropy_init
            logprob_steps += 1

        dt = 1.0 / self.inference_steps
        steps = torch.linspace(0, 1 - dt, self.inference_steps).repeat(B, 1).to(self.device)

        chains_mean = torch.zeros_like(chains_prev, device=self.device)

        for i in range(self.inference_steps):
            t = steps[:, i]
            d = torch.full((B,), dt, device=self.device)
            xt = x_chain[:, i]

            vt, nt = self.actor_ft.forward(xt, t, d, cond, learn_exploration_noise=True, step=i)

            st = self.compute_score(xt, vt, t)
            st = torch.clamp(st, -self.score_clip_value, self.score_clip_value)

            alpha_t = self.get_alpha_t(t)
            drift = vt + alpha_t * st

            mean_next = xt + drift * dt
            chains_mean[:, i] = mean_next.flatten(-2, -1)

            std = torch.clamp(nt, min=self.min_logprob_denoising_std, max=self.max_logprob_denoising_std)
            chains_stds[:, i] = std

            logprob_steps += 1

        if clip_intermediate_actions:
            chains_mean = chains_mean.clamp(-self.denoised_clip_value, self.denoised_clip_value)

        chains_dist = Normal(chains_mean, chains_stds)

        logprob_trans = chains_dist.log_prob(chains_next).sum(-1)
        if get_entropy:
            entropy_trans = chains_dist.entropy().sum(-1)

        logprob += logprob_trans.sum(-1)

        if get_entropy:
            joint_entropy += entropy_trans.sum(-1)
            entropy_rate_est = joint_entropy / logprob_steps

        if normalize_denoising_horizon:
            logprob = logprob / logprob_steps
        if normalize_act_space_dimension:
            logprob = logprob / self.act_dim_total
            if get_entropy:
                entropy_rate_est = entropy_rate_est / self.act_dim_total

        if verbose_entropy_stats and get_entropy:
            log.info(f"Entropy Percentiles: 10%={entropy_rate_est.quantile(0.1):.2f}, "
                     f"50%={entropy_rate_est.median():.2f}, 90%={entropy_rate_est.quantile(0.9):.2f}")

        if get_entropy:
            if get_chains_stds:
                return logprob, entropy_rate_est, chains_stds.mean()
            return logprob, entropy_rate_est
        else:
            if get_chains_stds:
                return logprob, chains_stds.mean()
            return logprob

    def train(self):
        self.actor_ft.train()
        self.critic.train()
        if self.score_scheduler is not None:
            self.score_scheduler.train()

    def eval(self):
        self.actor_ft.eval()
        self.critic.eval()
        if self.score_scheduler is not None:
            self.score_scheduler.eval()

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
        """PPO loss with score-enhanced exploration (解耦控制)"""
        newlogprobs, entropy, noise_std = self.get_logprobs(
            obs,
            chains,
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

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if verbose:
            with torch.no_grad():
                log.info(f"Advantage stats: mean={advantages.mean().item():.3f}, std={advantages.std().item():.3f}")

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
                a_new = self.actor_ft.policy.sample_action(cond=obs, inference_steps=self.inference_steps,
                                                           clip_intermediate_actions=True,
                                                           act_range=[self.act_min, self.act_max], z=z)
                bc_loss = F.mse_loss(a_old.detach(), a_new)
            else:
                raise NotImplementedError

        # alpha_t curve logging
        alpha_curve_dict = {}
        if self.score_scheduler is not None:
            with torch.no_grad():
                dt_log = 1.0 / self.inference_steps
                t_exact_steps = torch.linspace(0, 1 - dt_log, self.inference_steps, device=self.device)
                alpha_vals = self.get_alpha_t(t_exact_steps).squeeze().cpu().numpy()
                t_numpy = t_exact_steps.cpu().numpy()
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
