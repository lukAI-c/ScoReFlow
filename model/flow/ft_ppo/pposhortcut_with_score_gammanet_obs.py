# MIT License
# Copyright (c) 2025 ReinFlow Authors
# PPOShortCut with Score-guided drift + Learnable Gamma Network (Observation-Dependent)

"""
PPOShortCut + Score 导向 + 可学习的时间调度 (GammaNet) + Observation-Dependent

核心公式:
    drift = vt + alpha_t(t, obs) * st
    xt+1 = xt + drift * dt + nt * ε

其中:
    vt = 速度场 (ShortCutFlowMLP 输出)
    st = (t·vt - xt) / (1 - t)  分数函数
    alpha_t = time_mask * learned_gain(t, obs) * gamma_score  可学习的 Score 系数（依赖观测）
    nt = 学习到的噪声标准差 (NoisyShortCutFlowMLP 输出)

关键创新:
    1. 解耦控制: alpha_t 独立控制 Score 强度，不与 nt 耦合
    2. 时间调度: alpha_t 由网络学习，自适应调整不同时间步的 Score 强度
    3. 观测依赖: alpha_t 依赖于当前观测 obs，实现状态自适应的 Score 控制
    4. 物理约束: 当 t → 1 时，alpha_t 自动衰减为 0，防止数值爆炸
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


class PPOShortCutWithScoreGammaNetObs(nn.Module, ScoreFunctionMixin):
    """
    PPOShortCut + Score 导向 + 可学习的 Gamma 网络 (Observation-Dependent)
    
    使用可学习的网络来调度 Score 强度，实现解耦控制，并依赖于观测状态
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

        # ========== 创建 Score Scheduler 网络 (Observation-Dependent) ==========
        # self._init_score_scheduler(score_scheduler_hidden_dim)
        self.score_scheduler_hidden_dim = score_scheduler_hidden_dim  # 保存参数，延迟初始化
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
        logging.info("Cloned ShortCut policy for fine-tuning (with obs-dependent score guidance)")
        
        # ========== 创建 Score Scheduler 网络 (Observation-Dependent) ==========
        # 注意：必须在 init_actor_ft 之后，因为需要访问 self.actor_ft.policy.cond_enc_dim
        self._init_score_scheduler(self.score_scheduler_hidden_dim)
        
        self.critic = critic
        self.critic = self.critic.to(self.device)

        self.report_network_params()

    def _init_score_scheduler(self, hidden_dim: int):
        """
        初始化自适应 Score Scheduler (Observation-Dependent)

        关键设计:
        1. 输入维度: 1 (时间) + obs_dim (条件)
        2. 激活函数: SiLU (Swish) - 比 ReLU 在连续控制更好
        3. 输出激活: Softplus - 保证输出 > 0，且比 Sigmoid 梯度更好
        4. 初始化: 最后一层接近 0，让训练初期 alpha 较小，保持稳定
        """
        # 输入维度: 1 (时间) + cond_enc_dim (观测嵌入)
        # 关键改进: 复用 Actor 的 cond_emb，而非直接使用原始观测
        self.cond_enc_dim = self.actor_ft.policy.cond_enc_dim
        input_dim = 1 + self.cond_enc_dim
        if not self.use_score_scheduler or self.score_scheduler_type == 'fixed':
            # 不使用调度，固定使用 gamma_score
            self.score_scheduler = None
            logging.info(f"Score scheduler: FIXED (gamma_score={self.gamma_score})")
        elif self.score_scheduler_type == 'linear':
            # 简单的线性映射（仅用于消融实验）
            self.score_scheduler = nn.Sequential(
                nn.Linear(input_dim, 1),
                nn.Softplus()
            ).to(self.device)
            logging.info(f"Score scheduler: LINEAR (input={input_dim})")
        elif self.score_scheduler_type == 'mlp':
            # 自适应 MLP 网络
            self.score_scheduler = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),  # SiLU (Swish) 通常比 ReLU 在连续控制更好
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
                nn.Softplus()  # 使用 Softplus 保证输出 > 0，且比 Sigmoid 梯度更好
            ).to(self.device)

            # 智能初始化：让最后一层输出接近 0，训练初期 alpha 较小
            # 这样可以保持训练稳定性
            nn.init.constant_(self.score_scheduler[-2].weight, 0)
            nn.init.constant_(self.score_scheduler[-2].bias, -2.0)  # Softplus(-2) ≈ 0.13

            logging.info(f"Score scheduler: MLP (input={input_dim}, hidden={hidden_dim})")
        else:
            raise ValueError(f"Unknown score_scheduler_type: {self.score_scheduler_type}")

    def get_alpha_t(self, t: Tensor, cond: dict = None) -> Tensor:
        """
        获取时间依赖的 Score 系数 alpha_t (带 Time Decay Mask + Observation-Dependent)

        Args:
            t: [B] 时间步 ∈ [0, 1]
            cond: dict, 包含 'state' 或其他 embedding
        Returns:
            alpha_t: [B, 1, 1] Score 系数，用于 drift = vt + alpha_t * st

        关键设计:
        1. 物理硬约束 (Hard Constraint): 当 t → 1 时，强制衰减为 0
           防止 score (1/(1-t)) 爆炸
        2. 学习增益 (Learnable Gain): 网络只需要学习 "倍率"
           而不需要处理 "归零" 这种难任务
        3. 观测依赖: 根据当前观测 obs 自适应调整 Score 强度
        4. 组合: time_mask * learned_gain(t, obs) * gamma_score
        """
        B = t.shape[0]

        # 1. 物理硬约束 (Hard Constraint)
        # 当 t -> 1 时，强制衰减为 0，防止 score (1/(1-t)) 爆炸
        # 使用 (1-t) 作为 mask
        time_mask = (1.0 - t).view(B, 1)  # [B, 1]

        # if self.score_scheduler is None:
        #     # Fixed 模式也要乘 time_mask 保证数值稳定！
        #     alpha_t = torch.full((B, 1), self.gamma_score, device=self.device) * time_mask
        # else:
        #     # 2. 准备输入
        #     t_in = t.view(B, 1)  # [B, 1]

        #     # 提取条件特征 (这里简化处理，直接取 obs 的平均或部分特征)
        #     # 实际项目中，你可以把 Actor 的 obs_embedding 传进来
        #     if cond is not None and 'state' in cond:
        #         obs = cond['state']  # [B, obs_dim] or [B, T, obs_dim]
        #         # 如果 obs 是序列 [B, T, D]，可以取 mean pooling
        #         if obs.dim() > 2:
        #             obs = obs.mean(dim=1)
        #     else:
        #         # 如果没有条件，用零向量，并报警告
        #         obs = torch.zeros(B, self.obs_dim, device=self.device)
        #         logging.warning(
        #             "get_alpha_t: cond is None or missing 'state' key. "
        #             "Using zero vector for obs. "
        #             "Expected cond to contain 'state' with shape [B, obs_dim] or [B, T, obs_dim]."
        #         )
        
        if self.score_scheduler is None:
            # Fixed 模式也要乘 time_mask 保证数值稳定！
            alpha_t = torch.full((B, 1), self.gamma_score, device=self.device) * time_mask
        else:
            # 2. 准备输入
            t_in = t.view(B, 1)  # [B, 1]

            # 提取条件嵌入 (复用 Actor 的 cond_emb)
            # 关键改进: 使用预训练的观测编码器，而非原始观测
            if cond is not None and 'state' in cond:
                state = cond['state']  # [B, obs_dim] or [B, T, obs_dim]
                # 如果 state 是序列 [B, T, D]，flatten 为 [B, T*D]
                state = state.view(B, -1)  # [B, cond_dim]

                # 使用 Actor 的 cond_emb 编码器
                if hasattr(self.actor_ft.policy, 'cond_emb'):
                    cond_emb = self.actor_ft.policy.cond_emb(state)  # [B, cond_enc_dim]
                else:
                    # 如果没有 cond_emb（例如 cond_mlp_dims=None），直接使用 state
                    cond_emb = state
            else:
                # 如果没有条件，用零向量，并报警告
                cond_emb = torch.zeros(B, self.cond_enc_dim, device=self.device)
                logging.warning(
                    "get_alpha_t: cond is None or missing 'state' key. "
                    "Using zero vector for cond_emb. "
                    "Expected cond to contain 'state' with shape [B, obs_dim] or [B, T, obs_dim]."
                )

            # inp = torch.cat([t_in, obs], dim=-1)  # [B, 1 + obs_dim]
            inp = torch.cat([t_in, cond_emb], dim=-1)  # [B, 1 + cond_enc_dim]
            # 3. 网络预测 (Learnable Gain)
            # 输出范围 (0, +inf)，代表相对于基准的放缩倍率
            learned_gain = self.score_scheduler(inp)  # [B, 1]

            # 4. 组合: 硬约束 * 学习到的增益 * 全局系数
            # 这样网络只需要学习 "倍率"，而不需要处理 "归零" 这种难任务
            alpha_t = time_mask * learned_gain * self.gamma_score

            # 5. 可选：截断最大值，防止偶尔预测出极大值导致梯度爆炸
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

    def report_network_params(self):
        """Report number of parameters in each network"""
        scheduler_params = 0
        if self.score_scheduler is not None:
            scheduler_params = sum(p.numel() for p in self.score_scheduler.parameters()) / 1e6

        log.info(
            f"Number of network parameters: Total: {sum(p.numel() for p in self.parameters())/1e6:.2f}M. "
            f"Actor (old): {sum(p.numel() for p in self.actor_old.parameters())/1e6:.2f}M. "
            f"Actor (finetune): {sum(p.numel() for p in self.actor_ft.parameters())/1e6:.2f}M. "
            f"Critic: {sum(p.numel() for p in self.critic.parameters())/1e6:.2f}M. "
            f"ScoreScheduler (obs-dependent): {scheduler_params:.4f}M"
        )
        logging.info(f"Score params: gamma_score={self.gamma_score}, score_clip_value={self.score_clip_value}, "
                     f"scheduler_type={self.score_scheduler_type}, use_scheduler={self.use_score_scheduler}, "
                     f"obs_dependent=True")

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
        """Sample initial point from N(0, I)"""
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
        采样动作，使用 Score 增强的 drift (解耦控制 + Observation-Dependent)

        核心公式: drift = vt + alpha_t(t, obs) * st
        其中 alpha_t 由 score_scheduler 独立控制，不与 nt 耦合，且依赖于观测 obs
        """
        B = cond["state"].shape[0]
        dt = 1.0 / self.inference_steps
        steps = torch.linspace(0, 1-dt, self.inference_steps).repeat(B, 1).to(self.device)

        if save_chains:
            x_chain = torch.zeros((B, self.inference_steps+1, self.horizon_steps, self.action_dim), device=self.device)
        if ret_logprob:
            log_prob = 0.0
            log_prob_steps = 0

        # Sample initial point from N(0, I)
        xt = torch.randn((B, self.horizon_steps, self.action_dim), device=self.device)

        if ret_logprob and account_for_initial_stochasticity:
            init_dist = Normal(torch.zeros_like(xt), 1.0)
            log_prob_init = init_dist.log_prob(xt).sum(dim=(-2,-1))
            log_prob += log_prob_init
            log_prob_steps += 1

        if save_chains:
            x_chain[:, 0] = xt

        # Iterative denoising with score-enhanced exploration (解耦控制 + obs-dependent)
        for i in range(self.inference_steps):
            t = steps[:, i]
            d = torch.full((B,), dt, device=self.device)

            # Get velocity and noise std from policy
            vt, nt = self.actor_ft.forward(xt, t, d, cond, learn_exploration_noise=False, step=i)

            # Compute score function: st = (t * vt - xt) / (1 - t)
            st = self.compute_score(xt, vt, t)

            # Clip score to prevent instability
            st = torch.clamp(st, -self.score_clip_value, self.score_clip_value)

            # ========== 核心修改：解耦的 Score 控制 (Observation-Dependent) ==========
            # 获取时间依赖的 alpha_t (独立于 nt，依赖于 obs)
            alpha_t = self.get_alpha_t(t, cond)  # [B, 1, 1]  ← 传入 cond

            # drift = vt + alpha_t * st
            # 关键：alpha_t 独立控制 Score 强度，不与 nt 耦合，且根据 obs 自适应
            drift = vt + alpha_t * st

            # Update position (mean of transition distribution)
            xt_mean = xt + drift * dt

            if clip_intermediate_actions:
                xt_mean = xt_mean.clamp(-self.denoised_clip_value, self.denoised_clip_value)

            # Transition distribution with learned noise std
            std = nt.unsqueeze(-1).reshape(xt.shape)
            std = torch.clamp(std, min=self.min_sampling_denoising_std)
            dist = Normal(xt_mean, std)

            if not eval_mode:
                # Sample from the distribution
                xt = dist.sample().clamp_(
                    dist.loc - self.randn_clip_value * dist.scale,
                    dist.loc + self.randn_clip_value * dist.scale
                ).to(self.device)
            else:
                # Use mean in eval mode
                xt = xt_mean

            # Final clipping
            if i == self.inference_steps - 1:
                xt = xt.clamp(self.act_min, self.act_max)

            # Compute log probability
            if ret_logprob:
                logprob_transition = dist.log_prob(xt).sum(dim=(-2,-1))
                log_prob += logprob_transition
                log_prob_steps += 1

            if save_chains:
                x_chain[:, i+1] = xt

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
        """
        Compute log probabilities for given action chains using score-enhanced dynamics (解耦控制 + obs-dependent)
        """
        B = x_chain.shape[0]
        logprob = 0.0
        joint_entropy = 0.0
        logprob_steps = 0

        chains_prev = x_chain[:, :-1, :, :].flatten(-2, -1)
        chains_next = x_chain[:, 1:, :, :].flatten(-2, -1)
        chains_stds = torch.zeros_like(chains_prev, device=self.device)

        # Initial probability
        init_dist = Normal(torch.zeros(B, self.horizon_steps * self.action_dim, device=self.device), 1.0)
        logprob_init = init_dist.log_prob(x_chain[:, 0].reshape(B, -1)).sum(-1)

        if get_entropy:
            entropy_init = init_dist.entropy().sum(-1)

        if account_for_initial_stochasticity:
            logprob += logprob_init
            if get_entropy:
                joint_entropy += entropy_init
            logprob_steps += 1

        # Transition probabilities
        dt = 1.0 / self.inference_steps
        steps = torch.linspace(0, 1-dt, self.inference_steps).repeat(B, 1).to(self.device)

        chains_mean = torch.zeros_like(chains_prev, device=self.device)

        for i in range(self.inference_steps):
            t = steps[:, i]
            d = torch.full((B,), dt, device=self.device)
            xt = x_chain[:, i]

            # Get velocity and noise std
            vt, nt = self.actor_ft.forward(xt, t, d, cond, learn_exploration_noise=True, step=i)

            # Compute score
            st = self.compute_score(xt, vt, t)
            st = torch.clamp(st, -self.score_clip_value, self.score_clip_value)

            # ========== 核心修改：解耦的 Score 控制 (Observation-Dependent) ==========
            # 获取时间依赖的 alpha_t (独立于 nt，依赖于 obs)
            alpha_t = self.get_alpha_t(t, cond)  # [B, 1, 1]  ← 传入 cond

            # drift = vt + alpha_t * st
            # 关键：alpha_t 独立控制 Score 强度，不与 nt 耦合，且根据 obs 自适应
            # CRITICAL: This MUST match the drift in get_actions!
            drift = vt + alpha_t * st

            # Mean of next state
            mean_next = xt + drift * dt
            chains_mean[:, i] = mean_next.flatten(-2, -1)

            # Std of transition (same as in get_actions)
            std = torch.clamp(nt, min=self.min_logprob_denoising_std, max=self.max_logprob_denoising_std)
            chains_stds[:, i] = std

            logprob_steps += 1

        if clip_intermediate_actions:
            chains_mean = chains_mean.clamp(-self.denoised_clip_value, self.denoised_clip_value)

        # Transition distribution
        chains_dist = Normal(chains_mean, chains_stds)

        # Log probability and entropy
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

    def get_values(self, cond: dict):
        """Get value estimates from critic"""
        return self.critic(cond)

    def get_policy_loss_coef(self, epoch):
        """Get policy loss coefficient with exponential decay"""
        return self.clip_ploss_coef_base * np.exp(-self.clip_ploss_coef_rate * epoch)

    def get_trainable_params(self):
        """Get trainable parameters (actor_ft + critic + score_scheduler)"""
        params = list(self.actor_ft.parameters()) + list(self.critic.parameters())
        if self.score_scheduler is not None:
            params += list(self.score_scheduler.parameters())
        return params

    def get_actor_params(self):
        """Get actor parameters (actor_ft + score_scheduler)"""
        params = list(self.actor_ft.parameters())
        if self.score_scheduler is not None:
            params += list(self.score_scheduler.parameters())
        return params

    def get_critic_params(self):
        """Get critic parameters"""
        return self.critic.parameters()

    def train(self):
        """Set all networks to training mode"""
        self.actor_ft.train()
        self.critic.train()
        if self.score_scheduler is not None:
            self.score_scheduler.train()

    def eval(self):
        """Set all networks to evaluation mode"""
        self.actor_ft.eval()
        self.critic.eval()
        if self.score_scheduler is not None:
            self.score_scheduler.eval()

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'actor_ft': self.actor_ft.state_dict(),
            'critic': self.critic.state_dict(),
        }
        if self.score_scheduler is not None:
            checkpoint['score_scheduler'] = self.score_scheduler.state_dict()
        torch.save(checkpoint, path)
        log.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_ft.load_state_dict(checkpoint['actor_ft'])
        self.critic.load_state_dict(checkpoint['critic'])
        if 'score_scheduler' in checkpoint and self.score_scheduler is not None:
            self.score_scheduler.load_state_dict(checkpoint['score_scheduler'])
        log.info(f"Loaded checkpoint from {path}")

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
        PPO loss with score-enhanced exploration (解耦控制 + obs-dependent)
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

