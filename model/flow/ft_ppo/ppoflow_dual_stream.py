# MIT License
# Copyright (c) 2025 ReinFlow Authors - Dual-Stream Score Editing

"""
Dual-Stream Score Editing for Score-based SDE with CFG-style Reward Guidance.

核心创新：
1. 保守流 (Conservative/Prior Stream): 锚定数据流形，提供 v_uncond
2. 激进流 (Optimistic/Reward Stream): 探索高奖励区域，提供 v_cond  
3. CFG 引导: ∇r ≈ v_cond - v_uncond (隐式奖励梯度)

数学基础:
    贝叶斯公式: ∇log p(x|c) = ∇log p(x) + ∇log p(c|x)
    CFG 实现:  v_guided = v_uncond + w * (v_cond - v_uncond)
                        = (1-w) * v_uncond + w * v_cond

SDE 方程:
    dxt = [bt(xt) + εt·st(xt) + w·(v_cond - v_uncond)] dt + √(2εt) dWt

优势:
    1. 流形内引导: CFG 差值在数据流形切空间内
    2. 保留多模态: 先验流提供多样性基础
    3. 训练稳定: 无需二阶导数，只需两次前向传播
"""

import torch
from torch import nn, Tensor
import copy
import torch.nn.functional as F
import logging
import math
import numpy as np
from collections import namedtuple
from typing import Tuple, Optional
from torch.distributions.normal import Normal

from model.flow.mlp_flow_score import FlowMLP
from model.flow.score_utils import ScoreFunctionMixin

log = logging.getLogger(__name__)
Sample = namedtuple("Sample", "trajectories chains")


class DualStreamPPOFlow(nn.Module, ScoreFunctionMixin):
    """
    Dual-Stream Score Editing PPO Flow.
    
    架构:
        - actor_prior: 保守流 (冻结), 提供 v_uncond
        - actor_reward: 激进流 (可训练), 提供 v_cond
        - CFG guidance weight: w (可配置)
    
    推理时的 SDE:
        v_guided = v_prior + w * (v_reward - v_prior)
        dxt = [v_guided + εt·st] dt + √(2εt) dWt
    """
    
    def __init__(self,
                 device,
                 policy,                      # FlowMLP 网络结构
                 critic,                      # 价值网络
                 actor_policy_path: str,      # 预训练策略路径
                 act_dim: int,
                 horizon_steps: int,
                 act_min: float,
                 act_max: float,
                 obs_dim: int,
                 cond_steps: int,
                 inference_steps: int,        # 流采样步数
                 epsilon_t: float,            # 噪声系数 εt
                 # Dual-Stream 特定参数
                 cfg_weight: float = 1.5,     # CFG 引导权重 w
                 cfg_weight_schedule: str = 'constant',  # CFG 权重调度
                 freeze_prior: bool = True,   # 是否冻结保守流
                 # 逐步解冻参数
                 unfreeze_prior_schedule: str = 'none',  # 解冻策略: none, linear, threshold, layerwise
                 unfreeze_start_progress: float = 0.5,   # 开始解冻的训练进度 (0-1)
                 unfreeze_end_progress: float = 1.0,     # 完成解冻的训练进度 (0-1)
                 unfreeze_prior_lr_scale: float = 0.1,   # 解冻后 prior 学习率缩放
                 # PPO 参数
                 randn_clip_value: float = 3.0,
                 logprob_min: float = -1.0,
                 logprob_max: float = 1.0,
                 clip_ploss_coef: float = 0.01,
                 clip_ploss_coef_base: float = 0.01,
                 clip_ploss_coef_rate: float = 3.0,
                 clip_vloss_coef: float = 0.0,
                 denoised_clip_value: float = 1.0,
                 logprob_debug_sample: bool = False,
                 logprob_debug_recalculate: bool = False,
                 epsilon_schedule: str = 'linear_decay',
                 lamda: float = 1.0,
                 ):
        super().__init__()
        self.device = device
        self.inference_steps = inference_steps
        self.action_dim = act_dim
        self.horizon_steps = horizon_steps
        self.act_dim_total = horizon_steps * act_dim
        self.act_min = act_min
        self.act_max = act_max
        self.obs_dim = obs_dim
        self.cond_steps = cond_steps
        
        # Score-based SDE parameters
        self.epsilon_t: float = epsilon_t
        self.epsilon_schedule: str = epsilon_schedule
        self.lamda = lamda
        
        # Dual-Stream CFG parameters
        self.cfg_weight = cfg_weight
        self.cfg_weight_schedule = cfg_weight_schedule
        self.freeze_prior = freeze_prior

        # 逐步解冻参数
        self.unfreeze_prior_schedule = unfreeze_prior_schedule
        self.unfreeze_start_progress = unfreeze_start_progress
        self.unfreeze_prior_lr_scale = unfreeze_prior_lr_scale
        self.unfreeze_end_progress = unfreeze_end_progress
        self.prior_unfrozen_layers = []  # 记录已解冻的层
        
        # PPO parameters
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
        
        # ============== 双流架构 ==============
        # 1. 保守流 (Prior Stream) - 锚定数据流形
        self.actor_prior: FlowMLP = policy
        self._load_policy(actor_policy_path, self.actor_prior, use_ema=True)
        if self.freeze_prior:
            for param in self.actor_prior.parameters():
                param.requires_grad = False
            log.info("Prior stream frozen (anchoring data manifold)")
        self.actor_prior.to(self.device)
        
        # 2. 激进流 (Reward Stream) - 探索高奖励区域
        self.actor_reward: FlowMLP = copy.deepcopy(self.actor_prior)
        for param in self.actor_reward.parameters():
            param.requires_grad = True
        self.actor_reward.to(self.device)
        log.info("Reward stream initialized from prior (trainable)")
        
        # 3. 为了兼容现有代码，actor_ft 指向 actor_reward
        self.actor_ft = self.actor_reward  # 别名，用于兼容现有训练流程
        self.actor_old = self.actor_prior  # 别名，用于兼容现有训练流程
        
        # Critic 网络
        self.critic = critic.to(self.device)

        self._report_network_params()

    def _load_policy(self, network_path: str, network: FlowMLP, use_ema: bool = True):
        """加载预训练策略权重"""
        if network_path:
            log.info(f"Loading policy from {network_path}")
            model_data = torch.load(network_path, map_location=self.device, weights_only=True)
            if use_ema and "ema" in model_data:
                weights = {k.replace("network.", ""): v for k, v in model_data["ema"].items()}
                log.info("Loaded EMA weights for prior stream")
            else:
                weights = {k.replace("network.", ""): v for k, v in model_data["model"].items()}
                log.info("Loaded model weights for prior stream")
            network.load_state_dict(weights)
        else:
            log.warning("No policy path provided, using random initialization")

    def _report_network_params(self):
        """报告网络参数量"""
        total = sum(p.numel() for p in self.parameters()) / 1e6
        prior = sum(p.numel() for p in self.actor_prior.parameters()) / 1e6
        reward = sum(p.numel() for p in self.actor_reward.parameters()) / 1e6
        critic = sum(p.numel() for p in self.critic.parameters()) / 1e6
        log.info(f"Dual-Stream params: Total={total:.2f}M, Prior={prior:.2f}M, Reward={reward:.2f}M, Critic={critic:.2f}M")

    # ===================== 逐步解冻保守流 =====================

    def get_prior_layer_names(self) -> list:
        """获取 prior 网络的所有层名称 (从输出层到输入层排序，用于逐层解冻)"""
        layer_names = []
        for name, _ in self.actor_prior.named_parameters():
            # 提取层名 (去掉 .weight / .bias)
            layer_name = '.'.join(name.split('.')[:-1]) if '.' in name else name
            if layer_name not in layer_names:
                layer_names.append(layer_name)
        # 反转顺序：从输出层开始解冻
        return list(reversed(layer_names))

    def unfreeze_prior_layers(self, layer_names: list):
        """解冻指定的层"""
        for name, param in self.actor_prior.named_parameters():
            layer_name = '.'.join(name.split('.')[:-1]) if '.' in name else name
            if layer_name in layer_names and layer_name not in self.prior_unfrozen_layers:
                param.requires_grad = True
                self.prior_unfrozen_layers.append(layer_name)
                log.info(f"Unfroze prior layer: {layer_name}")

    def update_prior_freeze_state(self, training_progress: float) -> dict:
        """
        根据训练进度更新保守流的冻结状态

        Args:
            training_progress: 训练进度 (0-1)

        Returns:
            dict: 解冻状态信息
        """
        if self.unfreeze_prior_schedule == 'none' or not self.freeze_prior:
            return {'unfrozen': False, 'layers': [], 'ratio': 0.0}

        # 检查是否达到解冻开始阈值
        if training_progress < self.unfreeze_start_progress:
            return {'unfrozen': False, 'layers': [], 'ratio': 0.0}

        # 计算解冻进度 (从 unfreeze_start_progress 到 1.0)
        # unfreeze_progress = (training_progress - self.unfreeze_start_progress) / (1.0 - self.unfreeze_start_progress)
        # unfreeze_progress = min(1.0, max(0.0, unfreeze_progress))
        # 速度控制: end - start 越小，解冻越快
        unfreeze_duration = max(0.01, self.unfreeze_end_progress - self.unfreeze_start_progress)
        unfreeze_progress = (training_progress - self.unfreeze_start_progress) / unfreeze_duration

        all_layers = self.get_prior_layer_names()
        n_layers = len(all_layers)

        if self.unfreeze_prior_schedule == 'threshold':
            # 阈值策略：达到阈值后一次性解冻所有层
            if training_progress >= self.unfreeze_start_progress:
                self.unfreeze_prior_layers(all_layers)
                return {'unfrozen': True, 'layers': all_layers, 'ratio': 1.0}

        elif self.unfreeze_prior_schedule == 'linear':
            # 线性策略：根据进度逐步解冻所有层 (但每层同时解冻)
            for name, param in self.actor_prior.named_parameters():
                if not param.requires_grad:
                    param.requires_grad = True
            # 通过学习率缩放控制解冻程度
            effective_lr_scale = self.unfreeze_prior_lr_scale * unfreeze_progress
            return {'unfrozen': True, 'layers': all_layers, 'ratio': unfreeze_progress, 'lr_scale': effective_lr_scale}

        elif self.unfreeze_prior_schedule == 'layerwise':
            # 逐层策略：从输出层到输入层逐步解冻
            n_layers_to_unfreeze = int(n_layers * unfreeze_progress)
            layers_to_unfreeze = all_layers[:n_layers_to_unfreeze]
            self.unfreeze_prior_layers(layers_to_unfreeze)
            return {
                'unfrozen': n_layers_to_unfreeze > 0,
                'layers': self.prior_unfrozen_layers.copy(),
                'ratio': n_layers_to_unfreeze / n_layers if n_layers > 0 else 0.0
            }

        return {'unfrozen': False, 'layers': [], 'ratio': 0.0}

    def get_prior_trainable_params(self):
        """获取 prior 网络中可训练的参数 (用于优化器)"""
        return [p for p in self.actor_prior.parameters() if p.requires_grad]

    def get_prior_freeze_status(self) -> dict:
        """获取 prior 网络的冻结状态统计"""
        total_params = sum(p.numel() for p in self.actor_prior.parameters())
        frozen_params = sum(p.numel() for p in self.actor_prior.parameters() if not p.requires_grad)
        trainable_params = total_params - frozen_params
        return {
            'total': total_params,
            'frozen': frozen_params,
            'trainable': trainable_params,
            'frozen_ratio': frozen_params / total_params if total_params > 0 else 0.0,
            'unfrozen_layers': self.prior_unfrozen_layers.copy()
        }

    def get_cfg_weight_at_time(self, t: float, training_progress: float = 0.0) -> float:
        """根据时间步和训练进度获取 CFG 权重"""
        w = self.cfg_weight

        if self.cfg_weight_schedule == 'constant':
            return w
        elif self.cfg_weight_schedule == 'linear_increase':
            # 随时间增加引导强度: t=0 时最小, t=1 时最大
            return w * (0.5 + 0.5 * t)
        elif self.cfg_weight_schedule == 'linear_decrease':
            # 随时间减少引导强度: t=0 时最大, t=1 时最小
            return w * (1.5 - 0.5 * t)
        elif self.cfg_weight_schedule == 'cosine':
            # 余弦调度
            return w * (0.5 + 0.5 * math.cos(math.pi * t))
        elif self.cfg_weight_schedule == 'training_adaptive':
            # 随训练进度增加引导强度
            return w * (0.5 + 0.5 * training_progress)
        else:
            return w

    # ===================== Dual-Stream 核心: CFG 引导 =====================

    def get_guided_velocity(
        self,
        xt: Tensor,
        t_batch: Tensor,
        cond: dict,
        cfg_weight: float = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        计算 CFG 引导的速度场

        数学原理:
            v_guided = v_prior + w * (v_reward - v_prior)
                     = (1-w) * v_prior + w * v_reward

        当 w > 1 时，实现外推效果（更强的奖励引导）

        Args:
            xt: 当前状态 [B, horizon_steps, action_dim]
            t_batch: 时间步 [B]
            cond: 条件信息
            cfg_weight: CFG 权重，如果为 None 则使用默认值

        Returns:
            v_guided: 引导后的速度 [B, horizon_steps, action_dim]
            v_prior: 保守流速度 [B, horizon_steps, action_dim]
            v_reward: 激进流速度 [B, horizon_steps, action_dim]
        """
        if cfg_weight is None:
            cfg_weight = self.cfg_weight

        # 保守流 (Prior): 锚定数据流形
        with torch.no_grad() if self.freeze_prior else torch.enable_grad():
            v_prior = self.actor_prior(xt, t_batch, cond)

        # 激进流 (Reward): 探索高奖励区域
        v_reward = self.actor_reward(xt, t_batch, cond)

        # CFG 引导: v_guided = v_prior + w * (v_reward - v_prior)
        v_guided = v_prior + cfg_weight * (v_reward - v_prior)

        return v_guided, v_prior, v_reward

    def get_guided_score(
        self,
        xt: Tensor,
        v_guided: Tensor,
        t_batch: Tensor
    ) -> Tensor:
        """计算引导后的分数函数"""
        return self.compute_score(xt, v_guided, t_batch)

    # ===================== 采样 =====================

    @torch.no_grad()
    def sample_first_point(self, B: int) -> Tuple[Tensor, Tensor]:
        """采样初始点 x0 ~ N(0, I)"""
        dist = Normal(torch.zeros(B, self.act_dim_total, device=self.device), 1.0)
        xt = dist.sample()
        log_prob = dist.log_prob(xt).sum(-1)
        xt = xt.reshape(B, self.horizon_steps, self.action_dim)
        return xt, log_prob

    @torch.no_grad()
    def get_actions(
        self,
        cond: dict,
        eval_mode: bool = False,
        save_chains: bool = False,
        normalize_denoising_horizon: bool = False,
        normalize_act_space_dimension: bool = False,
        clip_intermediate_actions: bool = True,
        account_for_initial_stochasticity: bool = True,
        ret_logprob: bool = True,
        training_progress: float = 0.0
    ):
        """
        Dual-Stream CFG 引导的随机采样

        SDE: dxt = [v_guided + εt·st] dt + √(2εt) dWt
        其中: v_guided = v_prior + w * (v_reward - v_prior)

        Args:
            cond: 条件信息 {"state": [B, cond_steps, obs_dim]}
            eval_mode: 评估模式下使用确定性采样
            save_chains: 是否保存轨迹链
            ret_logprob: 是否返回 log 概率
            training_progress: 训练进度 [0, 1]

        Returns:
            xt: 最终动作 [B, horizon_steps, action_dim]
            x_chain: 轨迹链 (如果 save_chains=True)
            log_prob: log 概率 (如果 ret_logprob=True)
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

        # 采样初始点
        xt, log_prob_init = self.sample_first_point(B)
        if ret_logprob and account_for_initial_stochasticity:
            log_prob += log_prob_init
            log_prob_steps += 1
            if self.logprob_debug_sample:
                log_prob_list.append(log_prob_init.mean().item())

        if save_chains:
            x_chain[:, 0] = xt

        # Dual-Stream SDE 积分
        for i in range(self.inference_steps):
            t = steps[i]
            t_batch = t.expand(B)

            # 获取 CFG 权重
            cfg_w = self.get_cfg_weight_at_time(t.item(), training_progress)

            # 1. 获取 CFG 引导的速度场
            v_guided, v_prior, v_reward = self.get_guided_velocity(xt, t_batch, cond, cfg_w)

            # 2. 计算分数函数
            st = self.get_guided_score(xt, v_guided, t_batch)

            # 3. 获取 epsilon
            eps_t = self.get_epsilon_at_time(t.item())

            # 4. SDE 更新: drift = v_guided + εt·st
            drift = v_guided + eps_t * st
            diffusion_std = np.sqrt(2 * eps_t * dt)

            # 5. 更新均值
            xt_mean = xt + self.lamda * drift * dt
            if clip_intermediate_actions:
                xt_mean = xt_mean.clamp(-self.denoised_clip_value, self.denoised_clip_value)

            # 6. 随机/确定性更新
            if not eval_mode:
                noise = torch.randn_like(xt)
                noise = noise.clamp(-self.randn_clip_value, self.randn_clip_value)
                xt = xt_mean + diffusion_std * noise
            else:
                xt = xt_mean

            # 最终裁剪
            if i == self.inference_steps - 1:
                xt = xt.clamp(self.act_min, self.act_max)

            # 7. 计算 log 概率
            if ret_logprob:
                dist = Normal(xt_mean.flatten(-2, -1), diffusion_std)
                logprob_trans = dist.log_prob(xt.flatten(-2, -1)).sum(-1)
                if self.logprob_debug_sample:
                    log_prob_list.append(logprob_trans.mean().item())
                log_prob += logprob_trans
                log_prob_steps += 1

            if save_chains:
                x_chain[:, i + 1] = xt

        # 归一化 log 概率
        if ret_logprob:
            if normalize_denoising_horizon:
                log_prob = log_prob / log_prob_steps
            if normalize_act_space_dimension:
                log_prob = log_prob / self.act_dim_total

        # 返回结果
        if ret_logprob:
            if save_chains:
                return (xt, x_chain, log_prob)
            return (xt, log_prob)
        else:
            if save_chains:
                return (xt, x_chain)
            return xt

    def get_logprobs(
        self,
        cond: dict,
        x_chain: Tensor,
        get_entropy: bool = False,
        normalize_denoising_horizon: bool = False,
        normalize_act_space_dimension: bool = False,
        clip_intermediate_actions: bool = True,
        verbose_entropy_stats: bool = True,
        debug: bool = True,
        account_for_initial_stochasticity: bool = False,
        get_chains_stds: bool = True,
        training_progress: float = 0.0
    ):
        """
        Dual-Stream CFG 引导的 log 概率计算

        Args:
            x_chain: 轨迹链 [B, inference_steps+1, horizon_steps, action_dim]

        Returns:
            logprob: [B]
            entropy_rate_est: [B] (如果 get_entropy=True)
            noise_std_mean: scalar (如果 get_chains_stds=True)
        """
        logprob = 0.0
        joint_entropy = 0.0
        entropy_rate_est = 0.0
        logprob_steps = 0

        B = x_chain.shape[0]

        # 初始概率
        init_dist = Normal(
            torch.zeros(B, self.act_dim_total, device=self.device), 1.0
        )
        logprob_init = init_dist.log_prob(x_chain[:, 0].reshape(B, -1)).sum(-1)

        if get_entropy:
            entropy_init = init_dist.entropy().sum(-1)
        if account_for_initial_stochasticity:
            logprob += logprob_init
            if get_entropy:
                joint_entropy += entropy_init
            logprob_steps += 1

        dt = 1.0 / self.inference_steps
        steps = torch.linspace(0, 1 - dt, self.inference_steps, device=self.device)
        noise_std_values = []

        for i in range(self.inference_steps):
            t = steps[i]
            t_batch = t.expand(B)
            xt = x_chain[:, i]

            # 获取 CFG 权重
            cfg_w = self.get_cfg_weight_at_time(t.item(), training_progress)

            # 获取 CFG 引导的速度场
            v_guided, _, _ = self.get_guided_velocity(xt, t_batch, cond, cfg_w)

            # 计算分数函数
            st = self.get_guided_score(xt, v_guided, t_batch)

            # 获取 epsilon
            eps_t = self.get_epsilon_at_time(t.item())

            # 转移均值
            drift = v_guided + eps_t * st
            mean = xt + self.lamda * drift * dt
            if clip_intermediate_actions:
                mean = mean.clamp(-self.denoised_clip_value, self.denoised_clip_value)

            # 转移标准差
            std = np.sqrt(2 * eps_t * dt)
            noise_std_values.append(std)

            # 转移分布
            trans_dist = Normal(mean.flatten(-2, -1), std)

            # 下一状态的 log 概率
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
            log.info(f"Entropy Percentiles: 10%={entropy_rate_est.quantile(0.1):.2f}, 50%={entropy_rate_est.median():.2f}, 90%={entropy_rate_est.quantile(0.9):.2f}")

        noise_std_mean = torch.tensor(np.mean(noise_std_values), device=self.device)

        if get_entropy:
            if get_chains_stds:
                return logprob, entropy_rate_est, noise_std_mean
            return logprob, entropy_rate_est
        else:
            if get_chains_stds:
                return logprob, noise_std_mean
            return logprob

    # ===================== PPO 损失计算 =====================

    def loss(
        self,
        obs,
        chains,
        returns,
        oldvalues,
        advantages,
        oldlogprobs,
        use_bc_loss: bool = False,
        bc_loss_type: str = 'W2',
        normalize_denoising_horizon: bool = False,
        normalize_act_space_dimension: bool = False,
        verbose: bool = True,
        clip_intermediate_actions: bool = True,
        account_for_initial_stochasticity: bool = True,
        training_progress: float = 0.0
    ):
        """
        PPO 损失计算，使用 Dual-Stream CFG 引导

        Args:
            obs: 观测 {"state": [B, cond_steps, obs_dim]}
            chains: 轨迹链 [B, K+1, horizon_steps, action_dim]
            returns: 回报 [B]
            oldvalues: 旧价值 [B]
            advantages: 优势 [B]
            oldlogprobs: 旧 log 概率 [B]
        """
        # 计算新的 log 概率和熵
        newlogprobs, entropy, noise_std = self.get_logprobs(
            obs, chains,
            get_entropy=True,
            normalize_denoising_horizon=normalize_denoising_horizon,
            normalize_act_space_dimension=normalize_act_space_dimension,
            verbose_entropy_stats=verbose,
            clip_intermediate_actions=clip_intermediate_actions,
            account_for_initial_stochasticity=account_for_initial_stochasticity,
            training_progress=training_progress
        )

        if verbose:
            log.info(f"oldlogprobs: min={oldlogprobs.min():.3f}, max={oldlogprobs.max():.3f}, std={oldlogprobs.std():.3f}")
            log.info(f"newlogprobs: min={newlogprobs.min():.3f}, max={newlogprobs.max():.3f}, std={newlogprobs.std():.3f}")

        # 裁剪 log 概率
        newlogprobs = newlogprobs.clamp(min=self.logprob_min, max=self.logprob_max)
        oldlogprobs = oldlogprobs.clamp(min=self.logprob_min, max=self.logprob_max)

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if verbose:
            with torch.no_grad():
                log.info(f"Advantage stats: mean={advantages.mean():.3f}, std={advantages.std():.3f}")

        # 计算比率
        logratio = newlogprobs - oldlogprobs
        ratio = logratio.exp()

        # 近似 KL 和 clip 比例
        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > self.clip_ploss_coef).float().mean().item()

        # 策略损失
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_ploss_coef, 1 + self.clip_ploss_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # 价值损失
        newvalues = self.critic(obs).view(-1)
        v_loss = 0.5 * ((newvalues - returns) ** 2).mean()
        if self.clip_vloss_coef:
            v_clipped = torch.clamp(newvalues, oldvalues - self.clip_vloss_coef, oldvalues + self.clip_vloss_coef)
            v_loss = 0.5 * torch.max((newvalues - returns) ** 2, (v_clipped - returns) ** 2).mean()

        # 熵损失
        entropy_loss = -entropy.mean()

        # BC 正则化损失
        bc_loss = torch.tensor(0.0, device=self.device)
        if use_bc_loss and bc_loss_type == 'W2':
            z = torch.zeros((obs['state'].shape[0], self.horizon_steps, self.action_dim), device=self.device)
            with torch.no_grad():
                a_prior = self.actor_prior.sample_action(
                    cond=obs, inference_steps=self.inference_steps,
                    clip_intermediate_actions=True, act_range=[self.act_min, self.act_max], z=z
                )
            a_reward = self.actor_reward.sample_action(
                cond=obs, inference_steps=self.inference_steps,
                clip_intermediate_actions=True, act_range=[self.act_min, self.act_max], z=z
            )
            bc_loss = F.mse_loss(a_prior.detach(), a_reward)

        if verbose:
            with torch.no_grad():
                log.info(f"Value/Reward MSE: {F.mse_loss(newvalues, returns).item():.3f}")
                log.info(f"CFG weight: {self.cfg_weight:.3f}")

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

