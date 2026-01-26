# MIT License
# Copyright (c) 2026 ScoRe-Flow Authors
# PPOShortCut with Score-guided drift (combining learnable noise + score guidance)

"""
结合 ShortCut + 可学习噪声 + Score 导向的 PPOFlow

核心公式:
    原始 ShortCut: xt+1 = xt + vt·dt + nt·ε

    结合 Score:    xt+1 = xt + (vt + γ·nt·st)·dt + nt·ε

其中:
    vt = 速度场 (ShortCutFlowMLP 输出)
    nt = 学习到的噪声标准差 (NoisyShortCutFlowMLP 输出)
    st = (t·vt - xt) / (1 - t)  分数函数
    γ = gamma_score  Score 系数

关键创新:
    用学习的 nt 来动态控制 Score 强度，而不是固定的 epsilon_t
    - nt 大 → Score 导向强 + 探索多
    - nt 小 → Score 导向弱 + 探索少
    自适应平衡！
"""

import torch
from torch import nn
import copy
import torch.nn.functional as F
from torch import Tensor
import logging
import numpy as np
from collections import namedtuple
from typing import Tuple
from torch.distributions.normal import Normal
from model.flow.mlp_shortcut import ShortCutFlowMLP, NoisyShortCutFlowMLP
from model.flow.score_utils import ScoreFunctionMixin

log = logging.getLogger(__name__)
Sample = namedtuple("Sample", "trajectories chains")


class PPOShortCutWithScore(nn.Module, ScoreFunctionMixin):
    """
    PPOShortCut + Score 导向
    
    保持 ShortCut 的可学习噪声网络，同时在 drift 中加入 score 项
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
                 gamma_score: float = 1.0,         # Score 系数最大值
                 score_clip_value: float = 10.0,   # Score 裁剪值
                 # ========== Score Scheduler 参数 ==========
                 use_score_scheduler: bool = True,    # 是否使用可学习的时间调度
                 score_scheduler_hidden_dim: int = 16,  # Schedule 网络隐藏层维度
                 score_scheduler_type: str = 'mlp',   # 'mlp', 'linear', 'fixed'
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
        logging.info("Cloned ShortCut policy for fine-tuning (with score guidance)")

        self.critic = critic
        self.critic = self.critic.to(self.device)

        self.report_network_params()

    def _init_score_scheduler(self, hidden_dim: int):
        """
        初始化基于时间的 Score Schedule 网络
        """
        if not self.use_score_scheduler or self.score_scheduler_type == 'fixed':
            self.score_scheduler = None
            logging.info(f"Score scheduler: FIXED (gamma_score={self.gamma_score})")
        elif self.score_scheduler_type == 'linear':
            self.score_scheduler = nn.Sequential(
                nn.Linear(1, 1),
                nn.Sigmoid()
            ).to(self.device)
            logging.info(f"Score scheduler: LINEAR (learnable)")
        elif self.score_scheduler_type == 'mlp':
            self.score_scheduler = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ).to(self.device)
            logging.info(f"Score scheduler: MLP (hidden_dim={hidden_dim})")
        else:
            raise ValueError(f"Unknown score_scheduler_type: {self.score_scheduler_type}")

    def get_alpha_t(self, t: Tensor) -> Tensor:
        """
        获取时间依赖的 Score 系数 alpha_t
        """
        if self.score_scheduler is None:
            alpha_t = torch.full((t.shape[0], 1, 1), self.gamma_score, device=self.device)
        else:
            t_input = t.unsqueeze(-1)  # [B, 1]
            alpha_t = self.score_scheduler(t_input) * self.gamma_score  # [B, 1]
            alpha_t = alpha_t.unsqueeze(-1)  # [B, 1, 1]
        return alpha_t

    def init_actor_ft(self, policy_copy):
        """初始化带噪声网络的 actor"""
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
            activation_type=self.explore_net_activation_type
        )

    def report_network_params(self):
        total_params = sum(p.numel() for p in self.parameters()) / 1e6
        actor_params = sum(p.numel() for p in self.actor_old.parameters()) / 1e6
        actor_ft_params = sum(p.numel() for p in self.actor_ft.parameters()) / 1e6
        critic_params = sum(p.numel() for p in self.critic.parameters()) / 1e6

        logging.info(
            f"Number of network parameters: Total: {total_params:.2f}M. "
            f"Actor: {actor_params:.2f}M. Actor (finetune): {actor_ft_params:.2f}M. "
            f"Critic: {critic_params:.2f}M"
        )

    def load_policy(self, network_path, use_ema=False):
        log.info(f"loading policy from %s" % network_path)
        if network_path:
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
        采样动作，使用 Score 增强的 drift (ShortCut版本)

        核心修改: drift = vt + alpha_t * nt * st
        其中 nt 是学习到的噪声标准差，alpha_t 是时间依赖的score系数
        """
        B = cond["state"].shape[0]
        dt = 1.0 / self.inference_steps

        # 初始化
        xt, log_prob = self.sample_first_point(B)
        log_prob_steps = 1 if account_for_initial_stochasticity else 0
        log_prob = log_prob if account_for_initial_stochasticity else 0.0

        # ShortCut 使用 linspace(0, 1-dt, K)
        steps = torch.linspace(0, 1 - dt, self.inference_steps, device=self.device)
        steps = steps.unsqueeze(0).expand(B, -1)

        if save_chains:
            x_chain = torch.zeros((B, self.inference_steps + 1, self.horizon_steps, self.action_dim), device=self.device)
            x_chain[:, 0] = xt

        if self.logprob_debug_sample and ret_logprob:
            log_prob_list = []
            if account_for_initial_stochasticity:
                log_prob_list.append(log_prob.mean().item())

        # 迭代采样
        for i in range(self.inference_steps):
            t = steps[:, i]
            d = torch.full((B,), dt, device=self.device)

            # 1. 获取速度场和噪声标准差
            vt, nt = self.actor_ft.forward(xt, t, d, cond, learn_exploration_noise=False, step=i)

            # 2. 计算 score: st = (t·vt - xt) / (1 - t)
            st = self.compute_score(xt, vt, t)
            st = torch.clamp(st, -self.score_clip_value, self.score_clip_value)

            # 3. 获取时间依赖的 score 系数
            alpha_t = self.get_alpha_t(t)  # [B, 1, 1]

            # 4. 计算增强的 drift: vt + alpha_t * nt * st
            # nt: [B, horizon*action_dim] -> reshape to [B, horizon, action_dim]
            nt_reshaped = nt.reshape(B, self.horizon_steps, self.action_dim)
            # alpha_t: [B, 1, 1] 会自动广播到 [B, horizon, action_dim]
            drift = vt + alpha_t * nt_reshaped * st

            # 5. 更新位置
            xt = xt + drift * dt
            if clip_intermediate_actions:
                xt = xt.clamp(-self.denoised_clip_value, self.denoised_clip_value)

            # 6. 添加噪声
            std = nt.unsqueeze(-1).reshape(xt.shape)
            std = torch.clamp(std, min=self.min_sampling_denoising_std)
            dist = Normal(xt, std)

            if not eval_mode:
                xt = dist.sample().clamp_(
                    dist.loc - self.randn_clip_value * dist.scale,
                    dist.loc + self.randn_clip_value * dist.scale
                ).to(self.device)

            # 7. 最后一步裁剪到动作范围
            if i == self.inference_steps - 1:
                xt = xt.clamp_(self.act_min, self.act_max)

            # 8. 计算 log probability
            if ret_logprob:
                logprob_transition = dist.log_prob(xt).sum(dim=(-2, -1)).to(self.device)
                if self.logprob_debug_sample:
                    log_prob_list.append(logprob_transition.mean().item())
                log_prob += logprob_transition
                log_prob_steps += 1

            if save_chains:
                x_chain[:, i + 1] = xt

        # 归一化
        if ret_logprob:
            if normalize_denoising_horizon:
                log_prob = log_prob / log_prob_steps
            if normalize_act_space_dimension:
                log_prob = log_prob / self.act_dim_total
            if self.logprob_debug_sample:
                print(f"log_prob_list={log_prob_list}")

        # 返回逻辑与原始 PPOShortCut 保持一致
        if ret_logprob:
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
                     get_entropy=True,
                     normalize_denoising_horizon=False,
                     normalize_act_space_dimension=False,
                     clip_intermediate_actions=True,
                     verbose_entropy_stats=True,
                     debug=True,
                     account_for_initial_stochasticity=False,
                     get_chains_stds=True
                     ):
        """
        计算 log 概率，使用 Score 增强的 drift (ShortCut版本)
        """
        B = x_chain.shape[0]
        dt = 1.0 / self.inference_steps

        logprob = 0.0
        joint_entropy = 0.0
        logprob_steps = 0

        chains_prev = x_chain[:, :-1, :, :].flatten(-2, -1)
        chains_next = x_chain[:, 1:, :, :].flatten(-2, -1)
        chains_stds = torch.zeros_like(chains_prev, device=self.device)

        # 初始概率
        init_dist = Normal(torch.zeros(B, self.horizon_steps * self.action_dim, device=self.device), 1.0)
        logprob_init = init_dist.log_prob(x_chain[:, 0].reshape(B, -1)).sum(-1)
        if get_entropy:
            entropy_init = init_dist.entropy().sum(-1)

        if account_for_initial_stochasticity:
            logprob += logprob_init
            if get_entropy:
                joint_entropy += entropy_init
            logprob_steps += 1

        # ShortCut 使用 linspace(0, 1-dt, K)
        steps = torch.linspace(0, 1 - dt, self.inference_steps, device=self.device)

        chains_vel = torch.zeros_like(chains_prev, device=self.device)

        for i in range(self.inference_steps):
            t = steps[i]
            t_batch = t.expand(B)
            d = torch.full((B,), dt, device=self.device)
            xt = x_chain[:, i]

            # 获取速度场和噪声
            vt, nt = self.actor_ft.forward(xt, t_batch, d, cond, learn_exploration_noise=True, step=i)

            # 计算 score
            st = self.compute_score(xt, vt, t_batch)
            st = torch.clamp(st, -self.score_clip_value, self.score_clip_value)

            # 获取时间依赖的 score 系数
            alpha_t = self.get_alpha_t(t_batch)  # [B, 1, 1]

            # 计算增强的 drift
            nt_reshaped = nt.reshape(B, self.horizon_steps, self.action_dim)
            # alpha_t: [B, 1, 1] 会自动广播到 [B, horizon, action_dim]
            drift = vt + alpha_t * nt_reshaped * st

            chains_vel[:, i] = drift.flatten(-2, -1)
            chains_stds[:, i] = nt
            logprob_steps += 1

        # 计算转移概率
        chains_mean = (chains_prev + chains_vel * dt)
        if clip_intermediate_actions:
            chains_mean = chains_mean.clamp(-self.denoised_clip_value, self.denoised_clip_value)

        chains_dist = Normal(chains_mean, chains_stds)

        logprob_trans = chains_dist.log_prob(chains_next).sum(-1)
        if get_entropy:
            entropy_trans = chains_dist.entropy().sum(-1)

        logprob += logprob_trans.sum(-1)
        if self.logprob_debug_recalculate:
            log.info(f"logprob_init={logprob_init.mean().item()}, logprob_trans={logprob_trans.mean().item()}")

        if get_entropy:
            joint_entropy += entropy_trans.sum(-1)

        if get_entropy:
            entropy_rate_est = joint_entropy / logprob_steps
        if normalize_denoising_horizon:
            logprob = logprob / logprob_steps
        if normalize_act_space_dimension:
            logprob = logprob / self.act_dim_total
            if get_entropy:
                entropy_rate_est = entropy_rate_est / self.act_dim_total
        if verbose_entropy_stats and get_entropy:
            log.info(f"Entropy Percentiles: 10%={entropy_rate_est.quantile(0.1):.2f}, 50%={entropy_rate_est.median():.2f}, 90%={entropy_rate_est.quantile(0.9):.2f}")

        if get_entropy:
            if get_chains_stds:
                return logprob, entropy_rate_est, chains_stds.mean()
            return logprob, entropy_rate_est
        else:
            if get_chains_stds:
                return logprob, chains_stds.mean()
            return logprob

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
        PPO loss - 与原始 PPOShortCut 保持一致的签名
        """
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

        # batch normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if verbose:
            with torch.no_grad():
                log.info(f"Advantage stats: mean={advantages.mean().item():.3f}, std={advantages.std().item():.3f}")

        # Get ratio
        logratio = newlogprobs - oldlogprobs
        ratio = logratio.exp()

        # Get kl difference and clipfrac
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
            v_clipped = torch.clamp(newvalues, oldvalues - self.clip_vloss_coef, oldvalues + self.clip_vloss_coef)
            v_loss = 0.5 * torch.max((newvalues - returns) ** 2, (v_clipped - returns) ** 2).mean()

        # Entropy loss
        entropy_loss = -entropy.mean()

        # BC loss
        bc_loss = 0.0
        if use_bc_loss:
            if bc_loss_type == 'W2':
                z = torch.zeros((obs['state'].shape[0], self.horizon_steps, self.action_dim), device=self.device)
                a_ω = self.actor_old.sample_action(cond=obs, inference_steps=self.inference_steps,
                                                   clip_intermediate_actions=True,
                                                   act_range=[self.act_min, self.act_max], z=z)
                a_θ = self.actor_ft.policy.sample_action(cond=obs, inference_steps=self.inference_steps,
                                                         clip_intermediate_actions=True,
                                                         act_range=[self.act_min, self.act_max], z=z)
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

