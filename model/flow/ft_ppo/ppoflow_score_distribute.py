# MIT License
# Copyright (c) 2025 ReinFlow Authors - Distributional RL Extension

"""
PPOFlow with Distributional Critic (C51)

核心创新:
    1. 使用 C51 分布式 Critic 替代标量 Critic
    2. 输出价值分布 p(z|s) 而非单一标量 V(s)
    3. 更鲁棒地处理稀疏奖励和任务进度评估

参考:
    - GR-RL: 使用分布式 RL 处理机器人操作任务
    - C51: A Distributional Perspective on Reinforcement Learning

价值分布:
    Z(s) = {z_i}_{i=1}^{N_atoms}
    V(s) = E[Z(s)] = Σ p_i × z_i
"""
import torch
from torch import nn
import copy
import torch.nn.functional as F
from torch import Tensor
import logging
import numpy as np
log = logging.getLogger(__name__)
from collections import namedtuple
from typing import Tuple, Optional
from torch.distributions.normal import Normal
from model.flow.mlp_flow_score import FlowMLP
from model.flow.score_utils import ScoreFunctionMixin
from model.flow.ft_ppo.ppoflow_score import PPOFlow
from collections import OrderedDict
# from model.flow.ft_ppo.meanstd import RunningMeanStd

Sample = namedtuple("Sample", "trajectories chains")


class PPOFlowDistributional(PPOFlow):
    """
    PPO with Flow Matching Policy and Distributional Critic (C51)
    
    继承自 PPOFlow (score-based)，使用分布式价值网络。
    
    核心修改:
        1. Critic 输出价值分布而非标量
        2. Value Loss 使用交叉熵而非 MSE
        3. GAE 计算使用分布期望值
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
        gamma_score = 1.0,
        # ========== 分布式 RL 参数 ==========
        use_distributional_critic: bool = True,
        n_atoms: int = 51,
        v_min: float = 0.0,
        v_max: float = 1.0,
        reward_scale: float = 500, # 关键参数！根据你的任务最大回报设置
    ):
        # 保存分布式参数（在父类初始化之前）
        self.use_distributional_critic = use_distributional_critic
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.gamma_score = gamma_score
        
        self.reward_scale = float(reward_scale)

        
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
        # self.ret_rms = RunningMeanStd(shape=()).to(device)

        # 分布式 Critic 设置
        if self.use_distributional_critic:
            self._setup_distributional_critic()
            log.info(f"Distributional Critic (C51): n_atoms={n_atoms}, "
                     f"v_range=[{v_min}, {v_max}], reward_scale={self.reward_scale}")

    def _setup_distributional_critic(self):
        """设置分布式 Critic"""
        # 注册 atoms 为 buffer (确保在正确的设备上)
        self.register_buffer(
            "atoms",
            torch.linspace(self.v_min, self.v_max, self.n_atoms, device=self.device)
        )

        # 修改 Critic 输出层
        self._modify_critic_output()

    # def _modify_critic_output(self):
    #     """修改 Critic 的最后一层输出维度为 n_atoms"""
    #     if hasattr(self.critic, 'Q1'):
    #         mlp = self.critic.Q1
    #         if hasattr(mlp, 'layers'):
    #             # 获取最后一层
    #             last_layer = mlp.layers[-1]
    #             if isinstance(last_layer, nn.Linear):
    #                 in_features = last_layer.in_features
    #                 # 替换为新的输出层
    #                 mlp.layers[-1] = nn.Linear(in_features, self.n_atoms).to(self.device)
    #                 log.info(f"Modified Critic output layer: {in_features} -> {self.n_atoms}")
    #             else:
    #                 log.warning(f"Last layer is not Linear: {type(last_layer)}")
    def _modify_critic_output(self):
        """
        修改 Critic 输出层，并进行偏置初始化

        关键点：
            1. 输出层从 1 维改为 n_atoms 维
            2. 偏置初始化：让初始分布集中在 v_min 附近（低价值）
            3. 避免初始 Advantage 过大导致训练不稳定

        支持的 Critic 结构：
            - CriticObs: Q1 是 MLP 或 ResidualMLP
            - ViTCritic: 继承自 CriticObs
        """
        if hasattr(self.critic, 'Q1'):
            mlp = self.critic.Q1

            # 查找最后的 Linear 层
            last_linear_layer = None
            last_linear_idx = -1
            is_mlp_sequential = False  # MLP 使用 Sequential 包装

            if hasattr(mlp, 'layers'):  # ResidualMLP - layers 直接包含 Linear
                for idx in range(len(mlp.layers) - 1, -1, -1):
                    if isinstance(mlp.layers[idx], nn.Linear):
                        last_linear_layer = mlp.layers[idx]
                        last_linear_idx = idx
                        break
            elif hasattr(mlp, 'moduleList'):  # MLP - moduleList 包含 Sequential
                # MLP 的结构: moduleList[i] = Sequential([('linear_1', Linear), ...])
                # 找最后一个包含 Linear 的 Sequential
                for idx in range(len(mlp.moduleList) - 1, -1, -1):
                    module = mlp.moduleList[idx]
                    if isinstance(module, nn.Sequential):
                        # 在 Sequential 中查找 Linear
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

                # 1. 创建新的输出层
                new_layer = nn.Linear(in_features, self.n_atoms).to(self.device)

                # 2. 【关键】偏置初始化：让分布集中在 v_min 附近
                with torch.no_grad():
                    new_layer.weight.data.normal_(0, 0.01)
                    new_layer.bias.data.fill_(0)
                    # slope = torch.linspace(10.0, -10.0, self.n_atoms, device=self.device)
                    # new_layer.bias.data += slope

                # 替换层
                if hasattr(mlp, 'layers'):
                    mlp.layers[last_linear_idx] = new_layer
                    log.info(f"Replaced ResidualMLP.layers[{last_linear_idx}]")
                elif hasattr(mlp, 'moduleList'):
                    if is_mlp_sequential:
                        # MLP: 替换 Sequential 中的 Linear，保持其他层（如 activation）
                        # 直接替换整个 Sequential 为新的 Linear
                        mlp.moduleList[last_linear_idx] = nn.Sequential(
                            OrderedDict([('linear_1', new_layer)])
                        )
                        log.info(f"Replaced MLP.moduleList[{last_linear_idx}] (Sequential)")
                    else:
                        mlp.moduleList[last_linear_idx] = new_layer
                        log.info(f"Replaced MLP.moduleList[{last_linear_idx}]")

                # 验证初始预测值
                with torch.no_grad():
                    test_logits = new_layer.bias.data.unsqueeze(0)
                    test_probs = F.softmax(test_logits, dim=-1)
                    initial_value = torch.sum(test_probs * self.atoms).item()

                log.info(f"Modified Critic output layer: {in_features} -> {self.n_atoms}")
                log.info(f"Bias initialization: initial predicted V(s) ≈ {initial_value:.3f} (target: ~{self.v_min})")
            else:
                log.error("Could not find Linear layer in Critic Q1! C51 will NOT work!")
        else:
            log.error("Critic does not have Q1 attribute! C51 will NOT work!")
    
    
    # def _modify_critic_output(self):
    #     """
    #     修改 Critic 输出层，并进行偏置初始化

    #     关键点：
    #         1. 输出层从 1 维改为 n_atoms 维
    #         2. 偏置初始化：让初始分布集中在 v_min 附近（低价值）
    #         3. 避免初始 Advantage 过大导致训练不稳定
    #     """
    #     if hasattr(self.critic, 'Q1'):
    #         mlp = self.critic.Q1
    #         if hasattr(mlp, 'layers'):
    #             last_layer = mlp.layers[-1]
    #             if isinstance(last_layer, nn.Linear):
    #                 in_features = last_layer.in_features

    #                 # 1. 创建新的输出层
    #                 new_layer = nn.Linear(in_features, self.n_atoms).to(self.device)

    #                 # 2. 【关键】偏置初始化：让分布集中在 index=0 (v_min) 附近
    #                 # 设置 bias 为递减斜坡，Softmax 后前面概率大、后面概率小
    #                 with torch.no_grad():
    #                     # 权重初始化：很小的值，减少初始方差
    #                     new_layer.weight.data.normal_(0, 0.01)
    #                     new_layer.bias.data.fill_(0)

    #                     # 创建斜坡：从 10 到 -10，让 Softmax 后分布集中在低价值区域
    #                     # 这样初始预测 V(s) ≈ v_min，Advantage 接近 0
    #                     slope = torch.linspace(10.0, -10.0, self.n_atoms, device=self.device)
    #                     new_layer.bias.data += slope

    #                 mlp.layers[-1] = new_layer

    #                 # 验证初始预测值
    #                 with torch.no_grad():
    #                     test_logits = new_layer.bias.data.unsqueeze(0)  # (1, n_atoms)
    #                     test_probs = F.softmax(test_logits, dim=-1)
    #                     initial_value = torch.sum(test_probs * self.atoms).item()

    #                 log.info(f"Modified Critic output layer: {in_features} -> {self.n_atoms}")
    #                 log.info(f"Bias initialization: initial predicted V(s) ≈ {initial_value:.3f} (target: ~{self.v_min})")
    #             else:
    #                 log.warning(f"Last layer is not Linear: {type(last_layer)}")

    def get_value(self, obs) -> Tensor:
        """
        [修正] 获取标量价值用于 PPO GAE 计算。
        必须返回真实尺度 (Real Scale) 的值！
        """
        if self.use_distributional_critic:
            # 1. 获取归一化尺度下的期望 (0.0 ~ 1.0)
            _, value_norm = self.get_value_distribution(obs)
            
            # 2. [关键] 放大回真实尺度 (e.g., 0 ~ 100)
            return value_norm * self.reward_scale
        else:
            return self.critic(obs).view(-1)

    def get_value_distribution(self, obs) -> Tuple[Tensor, Tensor]:
        """
        获取完整的价值分布

        Returns:
            probs: (B, n_atoms) 概率分布
            value: (B,) 期望值
        """
        logits = self.critic(obs)  # (B, n_atoms)
        probs = F.softmax(logits, dim=-1)  # (B, n_atoms)
        value = torch.sum(probs * self.atoms, dim=-1)  # (B,)
        return probs, value

    def compute_distributional_value_loss(
        self,
        obs,
        returns: Tensor,
        oldvalues: Optional[Tensor] = None,
        verbose: bool = True
    ) -> Tuple[Tensor, Tensor]:
        """
        [修正] 计算 C51 Loss
        输入 returns 是真实尺度，需要除以 reward_scale 后再给 C51 学习。
        """
        # 1. 获取 Critic 输出 (归一化尺度)
        probs, newvalues_norm = self.get_value_distribution(obs)

        # 2. [关键] 将真实回报 (returns) 归一化到 [0, 1]
        #    并截断防止越界
        returns_norm = returns / self.reward_scale
        returns_norm = torch.clamp(returns_norm, self.v_min, self.v_max)
        
        # 3. 投影到分布 (Target Distribution)
        target_dist = self._project_to_distribution(returns_norm)

        # 4. 计算 Cross-Entropy Loss
        logits = self.critic(obs)
        log_probs = F.log_softmax(logits, dim=-1)
        v_loss = -torch.sum(target_dist * log_probs, dim=-1).mean()

        # 5. [关键] 还原用于日志记录的真实价值
        real_newvalues = newvalues_norm * self.reward_scale

        if verbose and self.training: # 减少一些日志打印频率
             log.info(f"C51: v_loss={v_loss.item():.4f}, "
                      f"ret_real=[{returns.min():.1f}, {returns.max():.1f}], "
                      f"pred_real_mean={real_newvalues.mean():.1f}")

        return v_loss, real_newvalues
    
    # def compute_distributional_value_loss(
    #     self,
    #     obs,
    #     returns: Tensor,
    #     oldvalues: Optional[Tensor] = None,
    #     verbose: bool = True
    # ) -> Tuple[Tensor, Tensor]:
    #     """
    #     计算 C51 分布式价值损失

    #     Args:
    #         obs: 观测
    #         returns: (B,) 目标回报 (标量)
    #         oldvalues: (B,) 旧价值估计 (用于日志)
    #         verbose: 是否打印详细日志

    #     Returns:
    #         v_loss: 交叉熵损失
    #         newvalues: (B,) 新价值估计 (期望值)
    #     """
    #     # 更新统计量 (仅在训练模式下)
    #     if self.training:
    #         # 确保 returns 不需要梯度，只用于统计更新
    #         with torch.no_grad():
    #             self.ret_rms.update(returns)
        
    #     # 1. 获取 Critic 输出分布
    #     probs, newvalues = self.get_value_distribution(obs)  # (B, n_atoms), (B,)

    #     # 2. 将标量 returns 投影到分布
    #     # 公式: (x - min) / (max - min)
    #     # Clamp 确保不会因为意外的 reward 导致越界
    #     target_returns_norm = self.ret_rms.normalize(returns)
        
    #     target_dist = self._project_to_distribution(target_returns_norm)  # (B, n_atoms)

    #     # 3. 计算交叉熵损失: -Σ m_i × log(p_i)
    #     # 使用 log_softmax 更数值稳定
    #     logits = self.critic(obs)
    #     print(f"Critic output shape: {self.critic(obs).shape}")
    #     log_probs = F.log_softmax(logits, dim=-1)  # (B, n_atoms)
    #     v_loss = -torch.sum(target_dist * log_probs, dim=-1).mean()
    #     log.info(f"C51 Debug: v_loss={v_loss.item():.6f}, returns_range=[{returns.min():.2f}, {returns.max():.2f}], "
    #                     f"v_min={self.v_min}, v_max={self.v_max}")
        
        
    #     real_newvalues = self.ret_rms.denormalize(newvalues)

    #     if verbose:
    #         with torch.no_grad():
    #             # 计算分布统计信息
    #             entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
    #             log.info(f"Value Distribution: mean={newvalues.mean():.3f}, "
    #                     f"std={newvalues.std():.3f}, entropy={entropy:.3f}")
                

    #     return v_loss, real_newvalues

    def _project_to_distribution(self, returns: Tensor) -> Tensor:
        """
        将标量回报投影到分布 (C51 核心算法)

        Args:
            returns: (B,) 目标回报

        Returns:
            target_dist: (B, n_atoms) 目标分布
        """
        B = returns.shape[0]

        # Clip returns 到支撑范围
        target_z = returns.clamp(self.v_min, self.v_max)

        # 计算在 atoms 中的浮点索引
        bj = (target_z - self.v_min) / self.delta_z  # (B,)

        # 获取上下界索引
        l = bj.floor().long()  # (B,)
        u = bj.ceil().long()   # (B,)

        # 边界处理
        l = torch.clamp(l, min=0, max=self.n_atoms - 1)
        u = torch.clamp(u, min=0, max=self.n_atoms - 1)

        # 构建目标分布
        target_dist = torch.zeros(B, self.n_atoms, device=returns.device)

        # 投影: 将概率质量分配给最近的两个桶
        # m_l += (u - bj), m_u += (bj - l)
        offset = torch.arange(B, device=returns.device) * self.n_atoms

        # 处理 l == u 的情况 (target 正好落在某个 atom 上)
        eq_mask = (l == u)

        # 对于 l != u 的情况，分配概率
        target_dist.view(-1).scatter_add_(
            0,
            (l + offset).view(-1),
            (u.float() - bj).view(-1)
        )
        target_dist.view(-1).scatter_add_(
            0,
            (u + offset).view(-1),
            (bj - l.float()).view(-1)
        )

        # 对于 l == u 的情况，全部概率给这个 atom
        target_dist.view(-1).scatter_(
            0,
            (l[eq_mask] + offset[eq_mask]).view(-1),
            torch.ones(eq_mask.sum(), device=returns.device)
        )

        return target_dist

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
        PPO loss with Distributional Critic (C51)

        主要修改:
            - Value Loss: MSE -> Cross-Entropy (C51)
            - newvalues: 分布期望值
        """
        # =================== Policy Loss (与父类相同) ===================
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
            log.info(f"oldlogprobs: min={oldlogprobs.min():5.3f}, max={oldlogprobs.max():5.3f}, std={oldlogprobs.std():5.3f}")
            log.info(f"newlogprobs: min={newlogprobs.min():5.3f}, max={newlogprobs.max():5.3f}, std={newlogprobs.std():5.3f}")

        # Clamp log probabilities
        newlogprobs = newlogprobs.clamp(min=self.logprob_min, max=self.logprob_max)
        oldlogprobs = oldlogprobs.clamp(min=self.logprob_min, max=self.logprob_max)

        if verbose:
            if oldlogprobs.min() < self.logprob_min:
                log.info(f"WARNING: old logprobs too low, potential policy collapse")
            if newlogprobs.min() < self.logprob_min:
                log.info(f"WARNING: new logprobs too low, potential policy collapse")

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if verbose:
            with torch.no_grad():
                advantage_stats = {
                    "mean": f"{advantages.mean().item():2.3f}",
                    "std": f"{advantages.std().item():2.3f}",
                    "max": f"{advantages.max().item():2.3f}",
                    "min": f"{advantages.min().item():2.3f}"
                }
                log.info(f"Advantage stats: {advantage_stats}")
                corr = torch.corrcoef(torch.stack([advantages, returns]))[0, 1].item()
                log.info(f"Advantage-Reward Correlation: {corr:.2f}")

        # Get ratio
        logratio = newlogprobs - oldlogprobs
        ratio = logratio.exp()

        # KL and clip fraction
        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > self.clip_ploss_coef).float().mean().item()

        # Policy loss (PPO clip)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_ploss_coef, 1 + self.clip_ploss_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # =================== Value Loss (C51 分布式) ===================
        if self.use_distributional_critic:
            v_loss, newvalues = self.compute_distributional_value_loss(
                obs, returns, oldvalues, verbose
            )
        else:
            # 回退到标量 Critic
            newvalues = self.critic(obs).view(-1)
            v_loss = 0.5 * ((newvalues - returns) ** 2).mean()
            if self.clip_vloss_coef:
                v_clipped = torch.clamp(newvalues, oldvalues - self.clip_vloss_coef,
                                       oldvalues + self.clip_vloss_coef)
                v_loss = 0.5 * torch.max((newvalues - returns) ** 2,
                                         (v_clipped - returns) ** 2).mean()

        if verbose:
            with torch.no_grad():
                mse = F.mse_loss(newvalues, returns)
                log.info(f"Value/Reward alignment: MSE={mse.item():.3f}")

        # Entropy loss
        entropy_loss = -entropy.mean()
        if verbose:
            with torch.no_grad():
                log.info(f"Entropy Percentiles: 10%={entropy.quantile(0.1):.2f}, "
                        f"50%={entropy.median():.2f}, 90%={entropy.quantile(0.9):.2f}")

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
            newvalues.mean().item(),  # Q function (期望值)
        )

    def report_network_params(self):
        """报告网络参数数量"""
        total = sum(p.numel() for p in self.parameters()) / 1e6
        actor = sum(p.numel() for p in self.actor_old.parameters()) / 1e6
        actor_ft = sum(p.numel() for p in self.actor_ft.parameters()) / 1e6
        critic = sum(p.numel() for p in self.critic.parameters()) / 1e6

        log.info(f"Network parameters: Total={total:.2f}M, Actor={actor:.2f}M, "
                f"Actor_FT={actor_ft:.2f}M, Critic={critic:.2f}M")
        log.info(f"Distributional Critic: n_atoms={self.n_atoms}, "
                f"v_range=[{self.v_min}, {self.v_max}]")
    @torch.no_grad()
    def plot_value_distribution(
        self,
        obs,
        target_returns: Optional[Tensor] = None,
        num_samples: int = 4,
        step: int = 0,
    ):
        """
        绘制 Critic 的价值分布图 (修正版：显示真实尺度)
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import io
        from PIL import Image

        # 1. 获取归一化分布和归一化期望
        probs, values_norm = self.get_value_distribution(obs)  # values_norm 是 0~1
        
        # 2. 【关键修正】转换到真实尺度 (Real Scale) 用于绘图
        # atoms 原本是 0~1，现在变成 0~7000
        atoms_real = self.atoms.cpu().numpy() * self.reward_scale 
        probs = probs.cpu().numpy()
        # 期望值也还原
        values_real = values_norm.cpu().numpy() * self.reward_scale 

        if target_returns is not None:
            target_returns = target_returns.cpu().numpy()

        B = min(probs.shape[0], num_samples)

        # 创建子图
        fig, axes = plt.subplots(1, B, figsize=(4 * B, 4), squeeze=False)
        axes = axes.flatten()

        # 计算柱子的宽度 (根据真实尺度调整)
        width = (atoms_real[-1] - atoms_real[0]) / len(atoms_real) * 0.8

        for i in range(B):
            ax = axes[i]
            prob = probs[i]
            expected_val = values_real[i] # 使用真实尺度

            # 绘制柱状图 (X轴是真实分数)
            ax.bar(atoms_real, prob, width=width, color='skyblue', edgecolor='navy', alpha=0.7)

            # 标出期望值
            ax.axvline(x=expected_val, color='green', linestyle='--', linewidth=2,
                      label=f'Pred: {expected_val:.0f}') # 取整显示更清爽

            # 标出真实值 (如果有)
            if target_returns is not None and i < len(target_returns):
                real_ret = target_returns[i]
                ax.axvline(x=real_ret, color='red', linestyle='-', linewidth=2,
                          label=f'True: {real_ret:.0f}')

            # 计算熵 (熵与尺度无关，用概率算即可)
            entropy = -np.sum(prob * np.log(prob + 1e-8))
            
            # 计算标准差 (需要用真实尺度)
            std = np.sqrt(np.sum(prob * (atoms_real - expected_val) ** 2))

            ax.set_title(f'S{i} | H={entropy:.2f} | σ={std:.0f}')
            
            # X轴标签改为 Real Reward
            if i == 0: ax.set_ylabel('Probability')
            ax.set_xlabel('Reward Value')
            
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # 设置 X 轴范围 (真实尺度)
            real_min = self.v_min * self.reward_scale
            real_max = self.v_max * self.reward_scale
            margin = 0.1 * (real_max - real_min)
            ax.set_xlim(real_min - margin, real_max + margin)

        fig.suptitle(f'C51 Dist (Step {step}) | Scale={self.reward_scale}', fontsize=12)
        plt.tight_layout()

        # 转为 Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf).copy()
        buf.close()
        plt.close(fig)

        return image
