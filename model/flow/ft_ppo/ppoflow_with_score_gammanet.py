# MIT License
# Copyright (c) 2025 ReinFlow Authors
# PPOFlow with Score-guided drift (combining learnable noise + score guidance)
# 可学习score t 的score-reinflow

"""
结合 ReinFlow 可学习噪声 + Score 导向的 PPOFlow

核心公式:
    原始 ReinFlow: xt+1 = xt + vt·dt + nt·ε

    结合 Score:    xt+1 = xt + (vt + γ·nt·st)·dt + nt·ε

其中:
    vt = 速度场 (FlowMLP 输出)
    nt = 学习到的噪声标准差 (NoisyFlowMLP 输出)
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
from model.flow.mlp_flow import FlowMLP, NoisyFlowMLP
from model.flow.score_utils import ScoreFunctionMixin

log = logging.getLogger(__name__)
Sample = namedtuple("Sample", "trajectories chains")


class PPOFlowWithScore(nn.Module, ScoreFunctionMixin):
    """
    PPOFlow + Score 导向
    
    保持 ReinFlow 的可学习噪声网络，同时在 drift 中加入 score 项
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
        # 只输入时间 t，输出 alpha_t ∈ [0, gamma_score]
        # 学习 "在什么时间点该用多少 Score 导向"
        self._init_score_scheduler(score_scheduler_hidden_dim)
        
        # 加载预训练 policy
        self.actor_old: FlowMLP = policy
        self.load_policy(actor_policy_path, use_ema=True)
        for param in self.actor_old.parameters():
            param.requires_grad = False
        self.actor_old.to(self.device)
        
        # 创建可训练副本
        policy_copy = copy.deepcopy(self.actor_old)
        for param in policy_copy.parameters():
            param.requires_grad = True
        
        self.init_actor_ft(policy_copy)
        logging.info("Cloned policy for fine-tuning (with score guidance)")
        
        self.critic = critic
        self.critic = self.critic.to(self.device)

        self.report_network_params()

    def _init_score_scheduler(self, hidden_dim: int):
        """
        初始化自适应 Score Scheduler (仅使用时间 t)

        输入: t (时间)
        输出: alpha_t (Score 系数的增益倍率)

        关键设计:
        1. 输入维度: 1 (仅时间 t)
        2. 激活函数: SiLU (Swish) - 比 ReLU 在连续控制更好
        3. 输出激活: Softplus - 保证输出 > 0，且比 Sigmoid 梯度更好
        4. 初始化: 最后一层接近 0，让训练初期 alpha 较小，保持稳定
        """
        # 输入维度: 仅时间 t
        input_dim = 1

        # 不需要状态压缩层
        self.state_projector = None

        # ========== 创建 Score Scheduler 网络 ==========
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

            # 初始化最后一层为接近 0 的小值，让训练初期 alpha 较小，保持稳定
            nn.init.constant_(self.score_scheduler[-2].weight, 0)
            nn.init.constant_(self.score_scheduler[-2].bias, -2.0)

            logging.info(f"Score scheduler: Adaptive MLP (input={input_dim}, hidden={hidden_dim})")
        else:
            raise ValueError(f"Unknown score_scheduler_type: {self.score_scheduler_type}")

    # def get_alpha_t(self, t: Tensor) -> Tensor:
    #     """
    #     获取时间依赖的 Score 系数 alpha_t (带 Time Decay Mask)

    #     Args:
    #         t: [B] 时间步 ∈ [0, 1]
    #         cond: dict, 包含 'state' 或其他 embedding
    #     Returns:
    #         alpha_t: [B, 1, 1] Score 系数，用于 drift = vt + alpha_t * st

    #     关键设计:
    #     1. 物理硬约束 (Hard Constraint): 当 t → 1 时，强制衰减为 0
    #        防止 score (1/(1-t)) 爆炸
    #     2. 学习增益 (Learnable Gain): 网络只需要学习 "倍率"
    #        而不需要处理 "归零" 这种难任务
    #     3. 组合: time_mask * learned_gain * gamma_score
    #     """
    #     if self.score_scheduler is None:
    #         # 固定值
    #         alpha_t = torch.full((t.shape[0], 1, 1), self.gamma_score, device=self.device)
    #     else:
    #         # 动态调度
    #         time_input = t.view(-1, 1)  # [B, 1]
    #         alpha_raw = self.score_scheduler(time_input)  # [B, 1], 范围 [0, 1]
    #         alpha_t = alpha_raw * self.gamma_score  # [B, 1], 范围 [0, gamma_score]
    #         alpha_t = alpha_t.unsqueeze(-1)  # [B, 1, 1] 用于广播
    #     return alpha_t
    def get_alpha_t(self, t: Tensor) -> Tensor:
        """
        获取时间依赖的 Score 系数 alpha_t (仅使用时间 t)

        Args:
            t: [B] 时间步 ∈ [0, 1]
        Returns:
            alpha_t: [B, 1, 1] Score 系数，用于 drift = vt + alpha_t * st

        关键设计:
        1. 物理硬约束 (Hard Constraint): 当 t → 1 时，强制衰减为 0
           防止 score (1/(1-t)) 爆炸
        2. 学习增益 (Learnable Gain): 网络只需要学习 "倍率"
           而不需要处理 "归零" 这种难任务
        3. 组合: time_mask * learned_gain * gamma_score
        """
        B = t.shape[0]

        # 1. 物理硬约束 (Hard Constraint)
        # 当 t -> 1 时，强制衰减为 0，防止 score (1/(1-t)) 爆炸
        # 使用 (1-t) 作为 mask
        time_mask = (1.0 - t).view(B, 1)  # [B, 1]

        if self.score_scheduler is None:
            # Fixed 模式也要乘 time_mask 保证数值稳定！
            alpha_t = torch.full((B, 1), self.gamma_score, device=self.device) * time_mask
        else:
            # 2. 准备输入 (仅时间 t)
            t_in = t.view(B, 1)  # [B, 1]

            # 3. 网络预测 (Learnable Gain)
            # 输出范围 (0, +inf)，代表相对于基准的放缩倍率
            learned_gain = self.score_scheduler(t_in)  # [B, 1]

            # 4. 组合: 硬约束 * 学习到的增益 * 全局系数
            # 这样网络只需要学习 "倍率"，而不需要处理 "归零" 这种难任务
            alpha_t = time_mask * learned_gain * self.gamma_score

            # 5. 可选：截断最大值，防止偶尔预测出极大值导致梯度爆炸
            alpha_t = alpha_t.clamp(max=self.gamma_score * 2.0)

        return alpha_t.unsqueeze(-1)  # [B, 1, 1]
        

    def init_actor_ft(self, policy_copy):
        self.actor_ft = NoisyFlowMLP(
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
        scheduler_params = sum(p.numel() for p in self.score_scheduler.parameters()) if self.score_scheduler else 0
        logging.info(
            f"PPOFlowWithScore - Network params: Total: {sum(p.numel() for p in self.parameters())/1e6:.2f}M, "
            f"Actor: {sum(p.numel() for p in self.actor_old.parameters())/1e6:.2f}M, "
            f"Actor_ft: {sum(p.numel() for p in self.actor_ft.parameters())/1e6:.2f}M, "
            f"Critic: {sum(p.numel() for p in self.critic.parameters())/1e6:.2f}M, "
            f"ScoreScheduler: {scheduler_params}"
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
        """
        采样动作，使用 Score 增强的 drift

        核心修改: drift = vt + gamma_score * nt * st
        其中 nt 是学习到的噪声标准差
        """
        B = cond["state"].shape[0]
        dt = 1.0 / self.inference_steps

        # 初始化
        xt, log_prob = self.sample_first_point(B)
        log_prob_steps = 1 if account_for_initial_stochasticity else 0
        log_prob = log_prob if account_for_initial_stochasticity else 0.0

        steps = torch.linspace(0, 1 - dt, self.inference_steps, device=self.device)
        steps = steps.unsqueeze(0).expand(B, -1)

        if save_chains:
            x_chain = torch.zeros(B, self.inference_steps + 1, self.horizon_steps, self.action_dim, device=self.device)
            x_chain[:, 0] = xt

        log_prob_list = [] if self.logprob_debug_sample else None

        for i in range(self.inference_steps):
            t = steps[:, i]

            # 获取速度场和噪声标准差
            vt, nt = self.actor_ft.forward(xt, t, cond, learn_exploration_noise=False, step=i)

            # 处理噪声标准差 (用于采样噪声)
            std = nt.unsqueeze(-1).reshape(xt.shape)  # [B, Ta, Da]
            std = torch.clamp(std, min=self.min_sampling_denoising_std)

            # ========== 核心修改：解耦的 Score 控制 ==========
            # 计算 score 函数
            st = self.compute_score(xt, vt, t)
            st = torch.clamp(st, -self.score_clip_value, self.score_clip_value)

            # 获取时间依赖的 alpha_t (独立于 std 和 cond)
            alpha_t = self.get_alpha_t(t)  # [B, 1, 1]

            # drift = vt + alpha_t * st
            # 关键：alpha_t 独立控制 Score 强度，不与 std 耦合
            # 这样 Score 的影响由 alpha_t 单独决定，而不受噪声网络的 std 影响
            drift = vt + alpha_t * st

            # 更新位置 (mean of transition distribution)
            xt_mean = xt + drift * dt

            if clip_intermediate_actions:
                xt_mean = xt_mean.clamp(-self.denoised_clip_value, self.denoised_clip_value)

            # 转移分布 (噪声由 std 控制，与 Score 解耦)
            dist = Normal(xt_mean, std)

            if not eval_mode:
                # 从分布中采样
                xt = dist.sample().clamp_(
                    dist.loc - self.randn_clip_value * dist.scale,
                    dist.loc + self.randn_clip_value * dist.scale
                ).to(self.device)
            else:
                # 评估模式使用均值
                xt = xt_mean

            if i == self.inference_steps - 1:
                xt = xt.clamp_(self.act_min, self.act_max)

            if ret_logprob:
                logprob_transition = dist.log_prob(xt).sum(dim=(-2, -1)).to(self.device)
                if self.logprob_debug_sample:
                    log_prob_list.append(logprob_transition.mean().item())
                log_prob += logprob_transition
                log_prob_steps += 1

            if save_chains:
                x_chain[:, i + 1] = xt

        if ret_logprob:
            if normalize_denoising_horizon:
                log_prob /= log_prob_steps
            if normalize_act_space_dimension:
                log_prob /= self.act_dim_total
            if self.logprob_debug_sample:
                log.info(f"logprob_list={log_prob_list}")

        # 返回逻辑与原始 PPOFlow 保持一致
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
        计算 log 概率，使用 Score 增强的 drift
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

        chains_vel = torch.zeros_like(chains_prev, device=self.device)
        steps = torch.linspace(0, 1 - dt, self.inference_steps, device=self.device)

        for i in range(self.inference_steps):
            t = steps[i].expand(B)
            xt = x_chain[:, i]

            # 获取速度场和噪声
            vt, nt = self.actor_ft.forward(xt, t, cond, True, i)

            # ========== 核心修改：解耦的 Score 控制 ==========
            # 计算 score 函数
            st = self.compute_score(xt, vt, t)
            st = torch.clamp(st, -self.score_clip_value, self.score_clip_value)

            # 获取时间依赖的 alpha_t (独立于 std 和 cond)
            alpha_t = self.get_alpha_t(t)  # [B, 1, 1]

            # drift = vt + alpha_t * st
            # 关键：alpha_t 独立控制 Score 强度，不与 nt 耦合
            # CRITICAL: This MUST match the drift in get_actions!
            drift = vt + alpha_t * st

            chains_vel[:, i] = drift.flatten(-2, -1)
            chains_stds[:, i] = nt
            logprob_steps += 1

        chains_mean = (chains_prev + chains_vel * dt)
        if clip_intermediate_actions:
            chains_mean = chains_mean.clamp(-self.denoised_clip_value, self.denoised_clip_value)

        chains_dist = Normal(chains_mean, chains_stds)
        logprob_trans = chains_dist.log_prob(chains_next).sum(-1)

        if get_entropy:
            entropy_trans = chains_dist.entropy().sum(-1)

        logprob += logprob_trans.sum(-1)
        if get_entropy:
            joint_entropy += entropy_trans.sum(-1)

        if normalize_denoising_horizon:
            logprob /= logprob_steps
            if get_entropy:
                joint_entropy /= logprob_steps
        if normalize_act_space_dimension:
            logprob /= self.act_dim_total
            if get_entropy:
                joint_entropy /= self.act_dim_total

        entropy_rate_est = joint_entropy / (self.inference_steps + 1) if get_entropy else 0.0

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
        PPO loss - 与原始 PPOFlow 保持一致的签名
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

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Get ratio
        logratio = newlogprobs - oldlogprobs
        ratio = logratio.exp()

        # Get kl difference and clip fraction
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

        # BC loss (保持兼容性)
        bc_loss = 0.0
        if use_bc_loss:
            if bc_loss_type == 'W2':
                z = torch.zeros((obs['state'].shape[0], self.horizon_steps, self.action_dim), device=self.device)
                a_old = self.actor_old.sample_action(cond=obs, inference_steps=self.inference_steps,
                                                     clip_intermediate_actions=True, act_range=[self.act_min, self.act_max], z=z)
                a_new = self.actor_ft.policy.sample_action(cond=obs, inference_steps=self.inference_steps,
                                                           clip_intermediate_actions=True, act_range=[self.act_min, self.act_max], z=z)
                bc_loss = F.mse_loss(a_old.detach(), a_new)

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

