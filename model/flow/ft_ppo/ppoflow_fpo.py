# MIT License
# Copyright (c) 2025 ReinFlow Authors + FPO Integration

"""
PPOFlow with FPO (Flow Policy Optimization) style policy ratio computation.

核心改动（相比原始 ReinFlow）：
1. 不再计算显式的 log probability
2. 使用 CFM (Conditional Flow Matching) 损失变化来近似策略比率
3. 在采样时保存 (eps, t, initial_cfm_loss) 用于后续损失计算

策略比率计算方式:
- ReinFlow: ρ = exp(log π_new - log π_old)  需要计算整条链的似然
- FPO:      ρ = exp(CFM_loss_old - CFM_loss_new)  直接用 CFM 损失近似

参考文献:
- Flow Policy Optimization (FPO): playground/src/flow_policy/fpo.py
"""

import torch
from torch import nn, Tensor
import copy
import torch.nn.functional as F
import logging
from collections import namedtuple
from typing import Tuple, Optional
from dataclasses import dataclass

from model.flow.mlp_flow import FlowMLP, VisionFlowMLP

log = logging.getLogger(__name__)
Sample = namedtuple("Sample", "trajectories chains")


@dataclass
class FPOActionInfo:
    """
    FPO 模式下采样时保存的信息
    
    与 ReinFlow 原始方法的区别：
    - ReinFlow 保存: (chain, log_prob) 其中 chain 形状为 [B, K+1, H, A]
    - FPO 保存: (action, loss_eps, loss_t, initial_cfm_loss) 更节省内存
    
    Attributes:
        action: 最终采样的动作 [B, horizon_steps, action_dim]
        loss_eps: 用于 CFM 损失计算的噪声样本 [B, n_samples, horizon_steps * action_dim]
        loss_t: 用于 CFM 损失计算的时间点 [B, n_samples, 1]
        initial_cfm_loss: 采样时旧策略的 CFM 损失 [B, n_samples]
    """
    action: Tensor
    loss_eps: Tensor
    loss_t: Tensor
    initial_cfm_loss: Tensor


class PPOFlowFPO(nn.Module):
    """
    PPOFlow with FPO-style policy ratio computation.
    
    核心创新：使用 CFM 损失变化代替显式似然比计算策略更新。
    
    数学原理:
        传统方法: ρ = π_θ(a|s) / π_θ_old(a|s)
        FPO 方法: ρ ≈ exp(L_CFM_old - L_CFM_new)
        
        其中 CFM 损失:
        L_CFM = E_{t,ε} [||v_θ(x_t, t, s) - (a - ε)||^2]
        x_t = t * a + (1-t) * ε  (Optimal Transport 插值)
    """
    
    def __init__(self,
                 device,
                 policy,                      # FlowMLP 或 VisionFlowMLP
                 critic,                      # 价值网络
                 actor_policy_path: str,      # 预训练策略路径
                 act_dim: int,
                 horizon_steps: int,
                 act_min: float,
                 act_max: float,
                 obs_dim: int,
                 cond_steps: int,
                 inference_steps: int,        # 流采样步数
                 n_samples_per_action: int = 8,  # CFM 损失的采样数
                 clipping_epsilon: float = 0.2,  # PPO clip 参数
                 clip_vloss_coef: float = 0.0,
                 average_losses_before_exp: bool = True,  # 是否先平均再取 exp
                 denoised_clip_value: float = 1.0,
                 use_ema: bool = True,
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
        
        # FPO 特定参数
        self.n_samples_per_action = n_samples_per_action
        self.clipping_epsilon = clipping_epsilon
        self.clip_vloss_coef = clip_vloss_coef
        self.average_losses_before_exp = average_losses_before_exp
        self.denoised_clip_value = denoised_clip_value
        
        self.ft_denoising_steps = inference_steps  # 用 inference_steps 代替
        self.noise_scheduler_type = "fixed"  # FPO 不使用噪声调度
        
        
        # 加载预训练策略（frozen，用于 BC 正则化）
        self.actor_old: FlowMLP = policy
        self._load_policy(actor_policy_path, use_ema=use_ema)
        for param in self.actor_old.parameters():
            param.requires_grad = False
        self.actor_old.to(self.device)
        
        # 创建可训练的策略副本
        self.actor_ft = copy.deepcopy(self.actor_old)
        for param in self.actor_ft.parameters():
            param.requires_grad = True
        self.actor_ft.to(self.device)
        
        # Critic 网络
        self.critic = critic.to(self.device)
        
        self._report_network_params()

    def _load_policy(self, network_path: str, use_ema: bool = True):
        """加载预训练策略权重"""
        if network_path:
            log.info(f"Loading policy from {network_path}")
            model_data = torch.load(network_path, map_location=self.device, weights_only=True)
            if use_ema and "ema" in model_data:
                weights = {k.replace("network.", ""): v for k, v in model_data["ema"].items()}
                log.info("Loaded EMA weights")
            else:
                weights = {k.replace("network.", ""): v for k, v in model_data["model"].items()}
            self.actor_old.load_state_dict(weights)
        else:
            log.warning("No policy path provided, using random initialization")

    def _report_network_params(self):
        """报告网络参数量"""
        total = sum(p.numel() for p in self.parameters()) / 1e6
        actor = sum(p.numel() for p in self.actor_ft.parameters()) / 1e6
        critic = sum(p.numel() for p in self.critic.parameters()) / 1e6
        log.info(f"Network params: Total={total:.2f}M, Actor={actor:.2f}M, Critic={critic:.2f}M")

    # ===================== CFM 损失计算（FPO 核心） =====================

    def _compute_cfm_loss(
        self,
        cond: dict,
        action: Tensor,      # [B, horizon_steps * action_dim]
        eps: Tensor,         # [B, n_samples, horizon_steps * action_dim]
        t: Tensor,           # [B, n_samples, 1]
    ) -> Tensor:
        """
        计算 Conditional Flow Matching 损失

        这是 FPO 的核心创新：通过 CFM 损失变化近似策略比率

        数学原理 (适配 ReinFlow 的约定):
            ReinFlow 的约定:
            - x1 = 真实动作 (目标)
            - x0 = 噪声 (eps)
            - xt = t * x1 + (1-t) * x0 = t * action + (1-t) * eps
              - 当 t=0 时，xt = eps (噪声起点)
              - 当 t=1 时，xt = action (动作终点)
            - v = x1 - x0 = action - eps (速度方向：从噪声到动作)

        Args:
            cond: 条件信息字典 {"state": [B, cond_steps, obs_dim]}
            action: 展平的动作 x1 [B, D] 其中 D = horizon_steps * action_dim
            eps: 噪声样本 x0 ~ N(0,I) [B, n_samples, D]
            t: 时间点 [B, n_samples, 1]

        Returns:
            cfm_loss: [B, n_samples] 每个样本的 CFM 损失
        """
        B, n_samples, D = eps.shape

        # 扩展 action: [B, D] -> [B, n_samples, D]
        action_expanded = action.unsqueeze(1).expand(-1, n_samples, -1)

        # OT 插值 (ReinFlow 约定): xt = t * x1 + (1-t) * x0 = t * action + (1-t) * eps
        # t=0: 噪声起点, t=1: 动作终点
        x_t = t * action_expanded + (1 - t) * eps  # [B, n_samples, D]

        # 目标速度 (ReinFlow 约定): v = x1 - x0 = action - eps
        target_velocity = action_expanded - eps  # [B, n_samples, D]

        # 准备网络输入：需要 reshape 来匹配网络期望的形状
        # x_t: [B * n_samples, horizon_steps, action_dim]
        x_t_reshaped = x_t.reshape(B * n_samples, self.horizon_steps, self.action_dim)

        # t: [B * n_samples]
        t_reshaped = t.reshape(B * n_samples)

        # 扩展 cond 到 [B * n_samples, ...]
        cond_expanded = {}
        for k, v in cond.items():
            # v: [B, ...] -> [B, n_samples, ...] -> [B * n_samples, ...]
            # 使用 -1 保持原始维度大小，而不是 1
            expand_shape = [-1, n_samples] + [-1] * (len(v.shape) - 1)
            v_expanded = v.unsqueeze(1).expand(*expand_shape)
            cond_expanded[k] = v_expanded.reshape(B * n_samples, *v.shape[1:])

        # 前向传播获取预测速度
        v_pred = self.actor_ft(x_t_reshaped, t_reshaped, cond_expanded)
        # 网络返回 (B * n_samples, horizon_steps, action_dim)
        # 需要先 flatten 再 reshape
        v_pred = v_pred.view(B * n_samples, -1)  # [B * n_samples, D]
        v_pred = v_pred.reshape(B, n_samples, D)  # [B, n_samples, D]

        # CFM 损失: ||v_pred - u||^2, 在动作维度上取均值
        # 使用 mean 而不是 sum，使损失量级与维度无关（参考原始 FPO）
        cfm_loss = ((v_pred - target_velocity) ** 2).mean(dim=-1)  # [B, n_samples]

        return cfm_loss

    # ===================== 动作采样 =====================

    @torch.no_grad()
    def get_actions(
        self,
        cond: dict,
        eval_mode: bool = False,
        clip_intermediate_actions: bool = True,
    ) -> Tuple[Tensor, FPOActionInfo]:
        """
        FPO 风格的动作采样

        与原始 ReinFlow 的关键区别：
        1. 不需要计算 log probability
        2. 使用纯 Euler 积分（不添加探索噪声）
        3. 保存 (eps, t, initial_cfm_loss) 用于后续策略更新

        Args:
            cond: 条件信息 {"state": [B, cond_steps, obs_dim]}
            eval_mode: 是否为评估模式（当前未使用差异化逻辑）
            clip_intermediate_actions: 是否裁剪中间动作

        Returns:
            action: 最终动作 [B, horizon_steps, action_dim]
            action_info: FPOActionInfo 包含 CFM 损失计算所需信息
        """
        B = cond["state"].shape[0]
        dt = 1.0 / self.inference_steps

        # 1. 从标准正态分布采样初始噪声
        xt = torch.randn(B, self.horizon_steps, self.action_dim, device=self.device)

        # 2. Euler 积分：从 t=0 到 t=1（噪声到动作）
        steps = torch.linspace(0, 1 - dt, self.inference_steps, device=self.device)
        for i in range(self.inference_steps):
            t = steps[i].expand(B)  # [B]
            vt = self.actor_ft(xt, t, cond)  # [B, horizon_steps, action_dim]
            xt = xt + vt * dt

            if clip_intermediate_actions:
                xt = xt.clamp(-self.denoised_clip_value, self.denoised_clip_value)

        # 最终裁剪到动作范围
        action = xt.clamp(self.act_min, self.act_max)

        # 3. 为 CFM 损失采样 (eps, t)
        action_flat = action.flatten(-2, -1)  # [B, D]
        loss_eps = torch.randn(
            B, self.n_samples_per_action, self.act_dim_total, device=self.device
        )
        loss_t = torch.rand(
            B, self.n_samples_per_action, 1, device=self.device
        )

        # 4. 计算初始 CFM 损失（使用当前策略）
        initial_cfm_loss = self._compute_cfm_loss(cond, action_flat, loss_eps, loss_t)

        return action, FPOActionInfo(
            action=action,
            loss_eps=loss_eps,
            loss_t=loss_t,
            initial_cfm_loss=initial_cfm_loss
        )

    @torch.no_grad()
    def get_actions_eval(
        self,
        cond: dict,
        clip_intermediate_actions: bool = True,
    ) -> Tensor:
        """
        评估模式下的动作采样（不保存 FPO 信息）

        Args:
            cond: 条件信息
            clip_intermediate_actions: 是否裁剪中间动作

        Returns:
            action: 最终动作 [B, horizon_steps, action_dim]
        """
        B = cond["state"].shape[0]
        dt = 1.0 / self.inference_steps

        xt = torch.randn(B, self.horizon_steps, self.action_dim, device=self.device)
        steps = torch.linspace(0, 1 - dt, self.inference_steps, device=self.device)

        for i in range(self.inference_steps):
            t = steps[i].expand(B)
            vt = self.actor_ft(xt, t, cond)
            xt = xt + vt * dt
            if clip_intermediate_actions:
                xt = xt.clamp(-self.denoised_clip_value, self.denoised_clip_value)

        return xt.clamp(self.act_min, self.act_max)

    @torch.no_grad()
    def get_actions_pretrained(
        self,
        cond: dict,
        clip_intermediate_actions: bool = True,
    ) -> Tensor:
        """
        使用预训练模型 (actor_old) 采样动作，用于验证预训练模型是否工作

        这个方法用于调试：如果预训练模型本身就不能完成任务，
        那么 FPO 微调也不会成功。
        """
        B = cond["state"].shape[0]
        dt = 1.0 / self.inference_steps

        xt = torch.randn(B, self.horizon_steps, self.action_dim, device=self.device)
        steps = torch.linspace(0, 1 - dt, self.inference_steps, device=self.device)

        for i in range(self.inference_steps):
            t = steps[i].expand(B)
            vt = self.actor_old(xt, t, cond)  # 使用 actor_old 而不是 actor_ft
            xt = xt + vt * dt
            if clip_intermediate_actions:
                xt = xt.clamp(-self.denoised_clip_value, self.denoised_clip_value)

        return xt.clamp(self.act_min, self.act_max)

    # ===================== FPO 损失计算 =====================

    def loss(
        self,
        obs: dict,
        actions: Tensor,           # [B, horizon_steps, action_dim]
        loss_eps: Tensor,          # [B, n_samples, D]
        loss_t: Tensor,            # [B, n_samples, 1]
        initial_cfm_loss: Tensor,  # [B, n_samples]
        returns: Tensor,           # [B]
        oldvalues: Tensor,         # [B]
        advantages: Tensor,        # [B]
        use_bc_loss: bool = False,
        bc_coeff: float = 0.1,
        verbose: bool = True,
    ) -> Tuple:
        """
        FPO 风格的 PPO 损失计算

        核心区别：使用 CFM 损失变化计算策略比率，而非显式似然

        策略比率计算:
            ρ = exp(CFM_loss_old - CFM_loss_new)

            直觉：如果新策略更好地拟合动作（损失更低），
            则 new_loss < old_loss，所以 ρ > 1

        Args:
            obs: 观测字典 {"state": [B, cond_steps, obs_dim]}
            actions: 动作 [B, horizon_steps, action_dim]
            loss_eps: CFM 损失用的噪声 [B, n_samples, D]
            loss_t: CFM 损失用的时间 [B, n_samples, 1]
            initial_cfm_loss: 初始 CFM 损失 [B, n_samples]
            returns: 回报 [B]
            oldvalues: 旧价值估计 [B]
            advantages: 优势 [B]
            use_bc_loss: 是否使用 BC 正则化
            bc_coeff: BC 损失系数
            verbose: 是否打印详细日志

        Returns:
            Tuple: (pg_loss, v_loss, bc_loss, clipfrac, approx_kl, ratio_mean, ...)
        """
        B = actions.shape[0]
        action_flat = actions.flatten(-2, -1).detach()  # [B, D]

        # 确保 obs 是字典格式（_compute_cfm_loss 期望字典）
        if isinstance(obs, dict):
            cond = obs
        else:
            cond = {"state": obs}

        # 1. 计算当前策略下的 CFM 损失（使用相同的 eps, t）
        new_cfm_loss = self._compute_cfm_loss(cond, action_flat, loss_eps, loss_t)

        # 2. 计算策略比率 ρ = exp(old_loss - new_loss)
        # 参考原始 FPO: 直接使用损失差，不额外 scaling
        if self.average_losses_before_exp:
            # 方案1: 先平均损失，再取指数（更稳定，原始 FPO 默认使用）
            old_mean = initial_cfm_loss.mean(dim=-1)  # [B]
            new_mean = new_cfm_loss.mean(dim=-1)      # [B]
            diff = old_mean - new_mean
            # 直接取 exp，不需要严格 clamp（CFM loss 用 mean 后量级较小）
            rho = torch.exp(diff)  # [B]
        else:
            # 方案2: 逐样本计算比率再平均（更精确但可能不稳定）
            diff = initial_cfm_loss - new_cfm_loss
            # 仅在 average=False 时 clamp 防止指数爆炸
            diff_clipped = torch.clamp(diff, -3.0, 3.0)
            rho = torch.exp(diff_clipped).mean(dim=-1)  # [B]

        # 3. 优势标准化
        advantages_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 4. PPO Clipped Surrogate Loss
        # L^CLIP = E[min(ρ*A, clip(ρ, 1-ε, 1+ε)*A)]
        pg_loss1 = -advantages_norm * rho
        pg_loss2 = -advantages_norm * torch.clamp(
            rho, 1 - self.clipping_epsilon, 1 + self.clipping_epsilon
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # 5. 价值函数损失
        newvalues = self.critic(obs).view(-1)
        v_loss = 0.5 * ((newvalues - returns) ** 2).mean()

        if self.clip_vloss_coef > 0:
            v_clipped = torch.clamp(
                newvalues,
                oldvalues - self.clip_vloss_coef,
                oldvalues + self.clip_vloss_coef
            )
            v_loss = 0.5 * torch.max(
                (newvalues - returns) ** 2,
                (v_clipped - returns) ** 2
            ).mean()

        # 6. BC 正则化损失（可选）
        bc_loss = torch.tensor(0.0, device=self.device)
        if use_bc_loss:
            # 使用相同初始噪声，比较新旧策略的输出
            z = torch.zeros(B, self.horizon_steps, self.action_dim, device=self.device)
            with torch.no_grad():
                a_old = self.actor_old.sample_action(
                    cond=obs,
                    inference_steps=self.inference_steps,
                    clip_intermediate_actions=True,
                    act_range=[self.act_min, self.act_max],
                    z=z
                )
            a_new = self.actor_ft.sample_action(
                cond=obs,
                inference_steps=self.inference_steps,
                clip_intermediate_actions=True,
                act_range=[self.act_min, self.act_max],
                z=z
            )
            bc_loss = F.mse_loss(a_old, a_new)

        # 7. 计算训练指标
        with torch.no_grad():
            # 近似 KL 散度
            approx_kl = ((rho - 1) - torch.log(rho + 1e-8)).mean()
            # Clip 比例
            clipfrac = ((rho - 1.0).abs() > self.clipping_epsilon).float().mean().item()
            # CFM 损失统计
            cfm_diff = initial_cfm_loss.mean() - new_cfm_loss.mean()

        if verbose:
            # log.info(f"CFM loss diff: {cfm_diff.item():.4f}, "
            #          f"ratio: {rho.mean().item():.3f} [{rho.min().item():.3f}, {rho.max().item():.3f}], "
            #          f"clipfrac: {clipfrac:.3f}")
            # log.info(f"Value MSE: {F.mse_loss(newvalues, returns).item():.4f}")
            log.info(f"CFM loss: old={initial_cfm_loss.mean().item():.4f}, new={new_cfm_loss.mean().item():.4f}, diff={cfm_diff.item():.4f}")
            log.info(f"Ratio: mean={rho.mean().item():.3f}, min={rho.min().item():.3f}, max={rho.max().item():.3f}, clipfrac={clipfrac:.3f}")
            log.info(f"Advantage: mean={advantages_norm.mean().item():.4f}, std={advantages_norm.std().item():.4f}")
            log.info(f"PG loss: {pg_loss.item():.4f}, Value loss: {v_loss.item():.4f}")

        return (
            pg_loss,
            v_loss,
            bc_loss,
            clipfrac,
            approx_kl.item(),
            rho.mean().item(),
            rho.min().item(),
            rho.max().item(),
            new_cfm_loss.mean().item(),
            newvalues.mean().item(),
        )

