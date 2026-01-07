# MIT License
# Copyright (c) 2025 ReinFlow Authors - QR-DQN Extension

"""
PPOFlow with Quantile Regression Critic (QR-DQN)

核心优势 (相比 C51):
    1. 无需预设 v_min/v_max，自动适应任意奖励范围
    2. 更鲁棒地处理异常奖励
    3. 分位数直接输出价值，无需 Softmax

参考:
    - Distributional Reinforcement Learning with Quantile Regression (AAAI 2018)
    - IQN: Implicit Quantile Networks for Distributional Reinforcement Learning

分位数表示:
    θ(s) = {θ_i}_{i=1}^{N}  # N 个分位数值
    V(s) = E[θ(s)] = (1/N) Σ θ_i
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import logging
import numpy as np
from typing import Tuple, Optional
from collections import OrderedDict

from model.flow.ft_ppo.ppoflow_score import PPOFlow

log = logging.getLogger(__name__)


class PPOFlowQRDQN(PPOFlow):
    """
    PPO with Flow Matching Policy and Quantile Regression Critic (QR-DQN)
    
    继承自 PPOFlow (score-based)，使用分位数回归价值网络。
    
    核心修改 (相比 C51):
        1. 去掉 Softmax，直接输出分位数值
        2. Value Loss 使用 Quantile Huber Loss
        3. 不再需要 v_min/v_max/reward_scale
    """
    
    def __init__(
        self,
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
        epsilon_t,
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
        epsilon_schedule='constant',
        lamda=1,
        gamma_score=1.0,
        # ========== QR-DQN 参数 ==========
        use_distributional_critic: bool = True,
        n_quantiles: int = 51,  # 分位数数量 (类似 n_atoms)
        huber_kappa: float = 1.0,  # Huber Loss 阈值
        # 以下参数保留但不使用 (为了兼容 C51 配置)
        v_min: float = 0.0,
        v_max: float = 1.0,
        reward_scale: float = 1.0,
        n_atoms: int = 51,  # 兼容性别名
    ):
        # 保存 QR-DQN 参数
        self.use_distributional_critic = use_distributional_critic
        self.n_quantiles = n_quantiles if n_quantiles else n_atoms  # 兼容 n_atoms 参数
        self.huber_kappa = huber_kappa
        self.gamma_score = gamma_score
        
        # 不再需要这些 C51 参数
        self.v_min = None
        self.v_max = None
        self.reward_scale = 1.0  # QR-DQN 不需要归一化
        
        # 调用父类初始化
        super().__init__(
            device=device,
            policy=policy,
            critic=critic,
            actor_policy_path=actor_policy_path,
            act_dim=act_dim,
            horizon_steps=horizon_steps,
            act_min=act_min,
            act_max=act_max,
            obs_dim=obs_dim,
            cond_steps=cond_steps,
            inference_steps=inference_steps,
            epsilon_t=epsilon_t,
            randn_clip_value=randn_clip_value,
            logprob_min=logprob_min,
            logprob_max=logprob_max,
            clip_ploss_coef=clip_ploss_coef,
            clip_ploss_coef_base=clip_ploss_coef_base,
            clip_ploss_coef_rate=clip_ploss_coef_rate,
            clip_vloss_coef=clip_vloss_coef,
            denoised_clip_value=denoised_clip_value,
            logprob_debug_sample=logprob_debug_sample,
            logprob_debug_recalculate=logprob_debug_recalculate,
            epsilon_schedule=epsilon_schedule,
            lamda=lamda,
        )

        # QR-DQN 设置
        if self.use_distributional_critic:
            self._setup_qr_critic()
            log.info(f"QR-DQN Critic: n_quantiles={self.n_quantiles}, "
                     f"huber_kappa={self.huber_kappa}")

    def _setup_qr_critic(self):
        """设置 QR-DQN Critic"""
        # 注册固定的分位点 tau: [0.5/N, 1.5/N, ..., (N-0.5)/N]
        tau = (torch.arange(self.n_quantiles, device=self.device).float() + 0.5) / self.n_quantiles
        self.register_buffer("tau", tau)

        # 修改 Critic 输出层
        self._modify_critic_output()

    def _modify_critic_output(self):
        """
        修改 Critic 输出层为 n_quantiles 维

        与 C51 不同: 不需要特殊的偏置初始化，因为输出是直接的价值
        """
        if hasattr(self.critic, 'Q1'):
            mlp = self.critic.Q1
            last_linear_layer = None
            last_linear_idx = -1
            is_mlp_sequential = False

            if hasattr(mlp, 'layers'):  # ResidualMLP
                for idx in range(len(mlp.layers) - 1, -1, -1):
                    if isinstance(mlp.layers[idx], nn.Linear):
                        last_linear_layer = mlp.layers[idx]
                        last_linear_idx = idx
                        break
            elif hasattr(mlp, 'moduleList'):  # MLP
                for idx in range(len(mlp.moduleList) - 1, -1, -1):
                    module = mlp.moduleList[idx]
                    if isinstance(module, nn.Sequential):
                        for name, sub_module in module.named_children():
                            if isinstance(sub_module, nn.Linear):
                                last_linear_layer = sub_module
                                last_linear_idx = idx
                                is_mlp_sequential = True
                                break
                        if last_linear_layer is not None:
                            break
                    elif isinstance(module, nn.Linear):
                        last_linear_layer = module
                        last_linear_idx = idx
                        break

            if last_linear_layer is not None:
                in_features = last_linear_layer.in_features
                new_layer = nn.Linear(in_features, self.n_quantiles).to(self.device)

                # 初始化: 权重小，偏置从 0 开始
                with torch.no_grad():
                    new_layer.weight.data.normal_(0, 0.01)
                    new_layer.bias.data.fill_(0)

                # 替换层
                if hasattr(mlp, 'layers'):
                    mlp.layers[last_linear_idx] = new_layer
                elif hasattr(mlp, 'moduleList'):
                    if is_mlp_sequential:
                        mlp.moduleList[last_linear_idx] = nn.Sequential(
                            OrderedDict([('linear_1', new_layer)])
                        )
                    else:
                        mlp.moduleList[last_linear_idx] = new_layer

                log.info(f"QR-DQN: Modified Critic output: {in_features} -> {self.n_quantiles}")
            else:
                log.error("Could not find Linear layer in Critic!")
        else:
            log.error("Critic does not have Q1 attribute!")

    def get_value(self, obs) -> Tensor:
        """
        获取标量价值用于 PPO GAE 计算

        QR-DQN: 价值 = 所有分位数的均值
        """
        if self.use_distributional_critic:
            quantiles = self.get_quantiles(obs)  # (B, N)
            return quantiles.mean(dim=-1)  # (B,)
        else:
            return self.critic(obs).view(-1)

    def get_quantiles(self, obs) -> Tensor:
        """
        获取分位数预测

        Returns:
            quantiles: (B, n_quantiles) 每个分位数的价值预测
        """
        # QR-DQN: 直接输出，不需要 Softmax！
        quantiles = self.critic(obs)  # (B, n_quantiles)
        return quantiles

    def get_value_distribution(self, obs) -> Tuple[Tensor, Tensor]:
        """
        兼容接口: 返回分位数和期望值

        Returns:
            quantiles: (B, n_quantiles) 分位数值
            value: (B,) 期望值 (均值)
        """
        quantiles = self.get_quantiles(obs)
        value = quantiles.mean(dim=-1)
        return quantiles, value

    def compute_distributional_value_loss(
        self,
        obs,
        returns: Tensor,
        oldvalues: Optional[Tensor] = None,
        verbose: bool = True
    ) -> Tuple[Tensor, Tensor]:
        """
        计算 QR-DQN Loss (Quantile Huber Loss)

        核心公式:
            L = E_τ[ |τ - I(δ < 0)| × ρ_κ(δ) ]
            其中 δ = target - θ(s), ρ_κ 是 Huber Loss

        Args:
            obs: 观测
            returns: (B,) 真实回报 (不需要归一化!)

        Returns:
            v_loss: Quantile Huber Loss
            newvalues: (B,) 期望价值
        """
        # 1. 获取当前分位数预测 (B, N)
        current_quantiles = self.get_quantiles(obs)

        # 2. Target: 将标量 returns 扩展为 (B, 1)
        target = returns.unsqueeze(1)  # (B, 1)

        # 3. 计算误差: target - current (B, N)
        diff = target - current_quantiles  # (B, N)

        # 4. Huber Loss
        abs_diff = diff.abs()
        huber_loss = torch.where(
            abs_diff < self.huber_kappa,
            0.5 * diff.pow(2),
            self.huber_kappa * (abs_diff - 0.5 * self.huber_kappa)
        )

        # 5. Quantile Regression 加权
        # tau: (N,) -> (1, N)
        tau = self.tau.unsqueeze(0)  # (1, N)

        # |τ - I(diff < 0)|: 低估时用 τ 加权，高估时用 (1-τ) 加权
        # 这使得高分位数（τ 大）更关注低估，低分位数更关注高估
        quantile_weight = torch.abs(tau - (diff.detach() < 0).float())

        # 6. 最终 Loss
        element_wise_loss = quantile_weight * huber_loss
        v_loss = element_wise_loss.mean(dim=-1).mean()  # 先对分位数求均值，再对 batch 求均值

        # 7. 期望价值 = 分位数均值
        newvalues = current_quantiles.mean(dim=-1)

        if verbose and self.training:
            with torch.no_grad():
                # 计算分位数统计
                q_min = current_quantiles.min(dim=-1)[0].mean()
                q_max = current_quantiles.max(dim=-1)[0].mean()
                q_std = current_quantiles.std(dim=-1).mean()
                log.info(f"QR-DQN: v_loss={v_loss.item():.4f}, "
                        f"ret=[{returns.min():.1f}, {returns.max():.1f}], "
                        f"pred={newvalues.mean():.1f}, q_range=[{q_min:.1f}, {q_max:.1f}], "
                        f"q_std={q_std:.1f}")

        return v_loss, newvalues

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
        PPO loss with QR-DQN Critic

        主要修改:
            - Value Loss: MSE -> Quantile Huber Loss
            - 不再需要 reward_scale 归一化
        """
        # =================== Policy Loss (与父类相同) ===================
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
            log.info(f"oldlogprobs: min={oldlogprobs.min():5.3f}, max={oldlogprobs.max():5.3f}")
            log.info(f"newlogprobs: min={newlogprobs.min():5.3f}, max={newlogprobs.max():5.3f}")

        newlogprobs = newlogprobs.clamp(min=self.logprob_min, max=self.logprob_max)
        oldlogprobs = oldlogprobs.clamp(min=self.logprob_min, max=self.logprob_max)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if verbose:
            with torch.no_grad():
                log.info(f"Advantage: mean={advantages.mean():.3f}, std={advantages.std():.3f}")
                corr = torch.corrcoef(torch.stack([advantages, returns]))[0, 1].item()
                log.info(f"Advantage-Reward Correlation: {corr:.2f}")

        # Get ratio
        logratio = newlogprobs - oldlogprobs
        ratio = logratio.exp()

        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > self.clip_ploss_coef).float().mean().item()

        # Policy loss (PPO clip)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_ploss_coef, 1 + self.clip_ploss_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # =================== Value Loss (QR-DQN) ===================
        if self.use_distributional_critic:
            v_loss, newvalues = self.compute_distributional_value_loss(obs, returns, oldvalues, verbose)
        else:
            newvalues = self.critic(obs).view(-1)
            v_loss = 0.5 * ((newvalues - returns) ** 2).mean()

        if verbose:
            with torch.no_grad():
                mse = F.mse_loss(newvalues, returns)
                log.info(f"Value/Reward alignment: MSE={mse.item():.3f}")

        # Entropy loss
        entropy_loss = -entropy.mean()

        # BC loss
        bc_loss = 0.0
        if use_bc_loss:
            if bc_loss_type == 'W2':
                z = torch.zeros((obs['state'].shape[0], self.horizon_steps, self.action_dim),
                               device=self.device)
                a_ω = self.actor_old.sample_action(
                    cond=obs, inference_steps=self.inference_steps,
                    clip_intermediate_actions=True, act_range=[self.act_min, self.act_max], z=z
                )
                a_θ = self.actor_ft.sample_action(
                    cond=obs, inference_steps=self.inference_steps,
                    clip_intermediate_actions=True, act_range=[self.act_min, self.act_max], z=z
                )
                bc_loss = F.mse_loss(a_ω.detach(), a_θ)

        return (
            pg_loss, entropy_loss, v_loss, bc_loss,
            clipfrac, approx_kl.item(), ratio.mean().item(),
            oldlogprobs.min(), oldlogprobs.max(), oldlogprobs.std(),
            newlogprobs.min(), newlogprobs.max(), newlogprobs.std(),
            noise_std.item(), newvalues.mean().item(),
        )

    def report_network_params(self):
        """报告网络参数数量"""
        total = sum(p.numel() for p in self.parameters()) / 1e6
        actor = sum(p.numel() for p in self.actor_old.parameters()) / 1e6
        critic = sum(p.numel() for p in self.critic.parameters()) / 1e6

        log.info(f"Network: Total={total:.2f}M, Actor={actor:.2f}M, Critic={critic:.2f}M")
        log.info(f"QR-DQN Critic: n_quantiles={self.n_quantiles}, huber_kappa={self.huber_kappa}")

    @torch.no_grad()
    def plot_value_distribution(
        self,
        obs,
        target_returns: Optional[Tensor] = None,
        num_samples: int = 4,
        step: int = 0,
    ):
        """
        绘制 QR-DQN 的分位数分布图

        与 C51 不同: X 轴是分位点 τ，Y 轴是预测的价值
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import io
        from PIL import Image

        quantiles, values = self.get_value_distribution(obs)
        quantiles = quantiles.cpu().numpy()
        values = values.cpu().numpy()
        tau = self.tau.cpu().numpy()

        if target_returns is not None:
            target_returns = target_returns.cpu().numpy()

        B = min(quantiles.shape[0], num_samples)

        fig, axes = plt.subplots(1, B, figsize=(4 * B, 4), squeeze=False)
        axes = axes.flatten()

        for i in range(B):
            ax = axes[i]
            q = quantiles[i]
            expected_val = values[i]

            # 绘制分位数曲线
            ax.plot(tau, q, 'b-o', markersize=3, linewidth=1.5, label='Quantiles')

            # 标出期望值 (均值)
            ax.axhline(y=expected_val, color='green', linestyle='--', linewidth=2,
                      label=f'E[V]={expected_val:.1f}')

            # 标出真实值
            if target_returns is not None and i < len(target_returns):
                real_ret = target_returns[i]
                ax.axhline(y=real_ret, color='red', linestyle='-', linewidth=2,
                          label=f'True={real_ret:.1f}')

            # 填充分位数范围
            ax.fill_between(tau, q.min(), q, alpha=0.3, color='blue')

            # 统计信息
            q_std = np.std(q)
            q_range = q.max() - q.min()
            ax.set_title(f'S{i} | σ={q_std:.1f} | range={q_range:.1f}')

            if i == 0:
                ax.set_ylabel('Value')
            ax.set_xlabel('Quantile τ')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'QR-DQN Quantiles (Step {step})', fontsize=12)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf).copy()
        buf.close()
        plt.close(fig)

        return image

