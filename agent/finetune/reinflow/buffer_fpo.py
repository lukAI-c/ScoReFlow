# MIT License
# Copyright (c) 2025 ReinFlow Authors + FPO Integration

"""
FPO 风格的 PPO Buffer

设计思路：
- 尽可能复用原始 PPOFlowBuffer 的逻辑
- 只替换存储内容：用 (eps, t, cfm_loss) 替换 (chain, logprob)
- 保持相同的 GAE 计算、统计更新等逻辑

与原始 PPOFlowBuffer 的区别:
- 原始: 存储 chains_trajs [n_steps, n_envs, K+1, H, A] 和 logprobs_trajs [n_steps, n_envs]
- FPO:  存储 loss_eps, loss_t, initial_cfm_loss

为什么不能直接复用 PPOFlowBuffer:
1. 存储字段不同: chains → (eps, t, cfm_loss)
2. add() 方法签名不同
3. make_dataset() 返回结构不同

但我们复用了 PPOBuffer 基类的:
- GAE 计算 (update_adv_returns)
- 统计更新 (update)
- 基础属性初始化
"""

import numpy as np
import torch
import logging
from agent.finetune.reinflow.buffer import PPOFlowBuffer

log = logging.getLogger(__name__)


class PPOFlowFPOBuffer(PPOFlowBuffer):
    """
    FPO 风格的 PPO Buffer（CPU 版本）

    继承自 PPOFlowBuffer，复用其 GAE 计算和统计逻辑，
    只替换存储内容为 CFM 损失计算所需的信息。
    """

    def __init__(self,
                 n_steps: int,
                 n_envs: int,
                 n_samples_per_action: int,  # FPO 特定参数（替代 n_ft_denoising_steps）
                 horizon_steps: int,
                 act_steps: int,
                 action_dim: int,
                 n_cond_step: int,
                 obs_dim: int,
                 save_full_observation: bool,
                 furniture_sparse_reward: bool,
                 best_reward_threshold_for_success: float,
                 reward_scale_running: bool,
                 gamma: float,
                 gae_lambda: float,
                 reward_scale_const: float,
                 device):
        # 调用 PPOFlowBuffer 的父类 PPOBuffer 的 __init__
        # 注意：我们跳过 PPOFlowBuffer.__init__ 因为我们不需要 ft_denoising_steps
        from agent.finetune.reinflow.buffer import PPOBuffer
        PPOBuffer.__init__(
            self,
            n_steps, n_envs, horizon_steps, act_steps, action_dim,
            n_cond_step, obs_dim, save_full_observation,
            furniture_sparse_reward, best_reward_threshold_for_success,
            reward_scale_running, gamma, gae_lambda, reward_scale_const, device
        )
        # FPO 特定参数
        self.n_samples_per_action = n_samples_per_action
        self.act_dim_total = horizon_steps * action_dim

    def reset(self):
        """
        重置 buffer，分配内存

        与 PPOFlowBuffer.reset() 的区别:
        - 不存储 chains_trajs，改为存储 loss_eps, loss_t, initial_cfm_loss
        - 不需要 logprobs_trajs
        """
        # 观测存储结构：支持字典（图像+状态）或单一维度（仅状态）
        if isinstance(self.obs_dim, dict):
            # 图像观测模式：存储多个模态
            self.obs_trajs = {
                k: np.zeros((self.n_steps, self.n_envs, self.n_cond_step, *self.obs_dim[k]))
                for k in self.obs_dim
            }
        else:
            # 仅状态观测模式
            self.obs_trajs = {
                "state": np.zeros((self.n_steps, self.n_envs, self.n_cond_step, self.obs_dim))
            }

        # 存储最终动作（替代 chains_trajs 中的最后一个）
        self.actions_trajs = np.zeros(
            (self.n_steps, self.n_envs, self.horizon_steps, self.action_dim)
        )

        # FPO 特定：CFM 损失计算所需信息
        self.loss_eps_trajs = np.zeros(
            (self.n_steps, self.n_envs, self.n_samples_per_action, self.act_dim_total)
        )
        self.loss_t_trajs = np.zeros(
            (self.n_steps, self.n_envs, self.n_samples_per_action, 1)
        )
        self.initial_cfm_loss_trajs = np.zeros(
            (self.n_steps, self.n_envs, self.n_samples_per_action)
        )

        # 奖励和终止信息（与基类相同）
        self.reward_trajs = np.zeros((self.n_steps, self.n_envs))
        self.terminated_trajs = np.zeros((self.n_steps, self.n_envs))
        self.firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
        self.value_trajs = np.empty((self.n_steps, self.n_envs))

    def add(self, step: int,
            obs_venv: dict,  # 改为接收完整的观测字典
            action_venv: np.ndarray,
            loss_eps_venv: np.ndarray,
            loss_t_venv: np.ndarray,
            initial_cfm_loss_venv: np.ndarray,
            reward_venv: np.ndarray,
            terminated_venv: np.ndarray,
            truncated_venv: np.ndarray,
            value_venv: np.ndarray,
            # 保留向后兼容性
            state_venv: np.ndarray = None):
        """
        添加一个时间步的数据

        与 PPOFlowBuffer.add() 的区别:
        - 接收 (action, loss_eps, loss_t, initial_cfm_loss) 而非 (chains, logprob)
        - 支持多模态观测（图像+状态）

        Args:
            obs_venv: 观测字典，包含 "state" 和可选的 "rgb" 等键
            state_venv: (已弃用) 为了向后兼容保留，如果提供则覆盖 obs_venv["state"]
        """
        done_venv = terminated_venv | truncated_venv

        # 存储观测（支持多模态）
        if state_venv is not None:
            # 向后兼容：如果提供了 state_venv，使用它
            self.obs_trajs["state"][step] = state_venv
        else:
            # 新方式：从 obs_venv 字典中提取所有模态
            for k in self.obs_trajs:
                if k in obs_venv:
                    self.obs_trajs[k][step] = obs_venv[k]

        self.actions_trajs[step] = action_venv
        self.loss_eps_trajs[step] = loss_eps_venv
        self.loss_t_trajs[step] = loss_t_venv
        self.initial_cfm_loss_trajs[step] = initial_cfm_loss_venv
        self.reward_trajs[step] = reward_venv
        self.terminated_trajs[step] = terminated_venv
        self.firsts_trajs[step + 1] = done_venv
        self.value_trajs[step] = value_venv

    def make_dataset(self):
        """
        将 buffer 转换为训练用的 tensor 数据集

        返回元组结构:
        (obs, actions, loss_eps, loss_t, initial_cfm_loss, returns, values, advantages)

        与 PPOFlowBuffer 返回 (obs, chains, returns, values, advantages, logprobs) 不同

        obs 可以是:
        - 字典 (图像观测模式): {"state": tensor, "rgb": tensor}
        - tensor (仅状态模式): state tensor
        """
        # 构建观测：支持多模态
        if isinstance(self.obs_dim, dict):
            # 图像观测模式：返回字典
            obs = {
                k: torch.tensor(self.obs_trajs[k], device=self.device).float().flatten(0, 1)
                for k in self.obs_trajs
            }
        else:
            # 仅状态模式：返回 tensor
            obs = torch.tensor(
                self.obs_trajs["state"], device=self.device
            ).float().flatten(0, 1)

        actions = torch.tensor(
            self.actions_trajs, device=self.device
        ).float().flatten(0, 1)

        loss_eps = torch.tensor(
            self.loss_eps_trajs, device=self.device
        ).float().flatten(0, 1)

        loss_t = torch.tensor(
            self.loss_t_trajs, device=self.device
        ).float().flatten(0, 1)

        initial_cfm_loss = torch.tensor(
            self.initial_cfm_loss_trajs, device=self.device
        ).float().flatten(0, 1)

        returns = torch.tensor(
            self.returns_trajs, device=self.device
        ).float().flatten(0, 1)

        values = torch.tensor(
            self.value_trajs, device=self.device
        ).float().flatten(0, 1)

        advantages = torch.tensor(
            self.advantages_trajs, device=self.device
        ).float().flatten(0, 1)

        return (obs, actions, loss_eps, loss_t, initial_cfm_loss,
                returns, values, advantages)

    @torch.no_grad()
    def update_adv_returns(self, obs_venv: dict, critic: torch.nn.Module, device='cpu'):
        """
        更新优势估计和回报（CPU 版本）

        使用 GAE (Generalized Advantage Estimation)

        覆盖基类方法以支持图像观测（rgb 键）
        """
        # 构建观测字典，包含所有可用的键（state 和可选的 rgb）
        obs_venv_ts = {}

        # 处理 state 键
        if "state" in obs_venv:
            obs_venv_ts["state"] = torch.from_numpy(obs_venv["state"]).float().to(self.device)

        # 处理 rgb 键（如果存在）
        if "rgb" in obs_venv:
            obs_venv_ts["rgb"] = torch.from_numpy(obs_venv["rgb"]).float().to(self.device)

        self.advantages_trajs = np.zeros((self.n_steps, self.n_envs))

        lastgaelam = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                nextvalues = critic.forward(obs_venv_ts).reshape(1, -1)
                nextvalues = nextvalues.cpu().numpy()
            else:
                nextvalues = self.value_trajs[t + 1]

            non_terminal = 1.0 - self.terminated_trajs[t]
            delta = (
                self.reward_trajs[t] * self.reward_scale_const
                + self.gamma * nextvalues * non_terminal
                - self.value_trajs[t]
            )
            self.advantages_trajs[t] = lastgaelam = (
                delta + self.gamma * self.gae_lambda * non_terminal * lastgaelam
            )

        self.returns_trajs = self.advantages_trajs + self.value_trajs

    # 注意：update() 方法从 PPOBuffer 基类继承
    # 无需重写，因为它只是调用 normalize_reward() 和 update_adv_returns()


class PPOFlowFPOBufferGPU(PPOFlowFPOBuffer):
    """
    FPO 风格的 PPO Buffer（GPU 版本）

    继承自 PPOFlowFPOBuffer，将数据直接存储在 GPU 上以减少传输开销。

    注意：这个类存在是为了优化性能，但功能上与 CPU 版本等价。
    如果 GPU 内存不足，可以使用 CPU 版本。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self):
        """重置 buffer，在 GPU 上分配内存"""
        # 观测存储结构：支持字典（图像+状态）或单一维度（仅状态）
        if isinstance(self.obs_dim, dict):
            # 图像观测模式：存储多个模态
            self.obs_trajs = {
                k: torch.zeros(
                    (self.n_steps, self.n_envs, self.n_cond_step, *self.obs_dim[k]),
                    dtype=torch.float32, device=self.device
                )
                for k in self.obs_dim
            }
        else:
            # 仅状态观测模式
            self.obs_trajs = {
                "state": torch.zeros(
                    (self.n_steps, self.n_envs, self.n_cond_step, self.obs_dim),
                    dtype=torch.float32, device=self.device
                )
            }

        self.actions_trajs = torch.zeros(
            (self.n_steps, self.n_envs, self.horizon_steps, self.action_dim),
            dtype=torch.float32, device=self.device
        )

        self.loss_eps_trajs = torch.zeros(
            (self.n_steps, self.n_envs, self.n_samples_per_action, self.act_dim_total),
            dtype=torch.float32, device=self.device
        )

        self.loss_t_trajs = torch.zeros(
            (self.n_steps, self.n_envs, self.n_samples_per_action, 1),
            dtype=torch.float32, device=self.device
        )

        self.initial_cfm_loss_trajs = torch.zeros(
            (self.n_steps, self.n_envs, self.n_samples_per_action),
            dtype=torch.float32, device=self.device
        )

        self.reward_trajs = torch.zeros(
            (self.n_steps, self.n_envs), dtype=torch.float32, device=self.device
        )
        self.terminated_trajs = torch.zeros(
            (self.n_steps, self.n_envs), dtype=torch.float32, device=self.device
        )
        self.firsts_trajs = torch.zeros(
            (self.n_steps + 1, self.n_envs), dtype=torch.float32, device=self.device
        )
        self.value_trajs = torch.zeros(
            (self.n_steps, self.n_envs), dtype=torch.float32, device=self.device
        )

    def add(self, step: int,
            obs_venv,  # 改为接收完整的观测字典
            action_venv,
            loss_eps_venv,
            loss_t_venv,
            initial_cfm_loss_venv,
            reward_venv,
            terminated_venv,
            truncated_venv,
            value_venv,
            # 保留向后兼容性
            state_venv=None):
        """
        添加一个时间步的数据（GPU 版本，自动处理类型转换）

        Args:
            obs_venv: 观测字典，包含 "state" 和可选的 "rgb" 等键
            state_venv: (已弃用) 为了向后兼容保留，如果提供则覆盖 obs_venv["state"]
        """
        # 将 numpy 数据转换为 tensor
        if isinstance(action_venv, np.ndarray):
            action_venv = torch.from_numpy(action_venv).float().to(self.device)
        if isinstance(loss_eps_venv, np.ndarray):
            loss_eps_venv = torch.from_numpy(loss_eps_venv).float().to(self.device)
        if isinstance(loss_t_venv, np.ndarray):
            loss_t_venv = torch.from_numpy(loss_t_venv).float().to(self.device)
        if isinstance(initial_cfm_loss_venv, np.ndarray):
            initial_cfm_loss_venv = torch.from_numpy(initial_cfm_loss_venv).float().to(self.device)
        if isinstance(reward_venv, np.ndarray):
            reward_venv = torch.from_numpy(reward_venv).float().to(self.device)
        if isinstance(terminated_venv, np.ndarray):
            terminated_venv = torch.from_numpy(terminated_venv).float().to(self.device)
        if isinstance(truncated_venv, np.ndarray):
            truncated_venv = torch.from_numpy(truncated_venv).float().to(self.device)
        if isinstance(value_venv, np.ndarray):
            value_venv = torch.from_numpy(value_venv).float().to(self.device)

        done_venv = terminated_venv.bool() | truncated_venv.bool()

        # 存储观测（支持多模态）
        if state_venv is not None:
            # 向后兼容：如果提供了 state_venv，使用它
            if isinstance(state_venv, np.ndarray):
                state_venv = torch.from_numpy(state_venv).float().to(self.device)
            self.obs_trajs["state"][step] = state_venv
        else:
            # 新方式：从 obs_venv 字典中提取所有模态
            for k in self.obs_trajs:
                if k in obs_venv:
                    obs_k = obs_venv[k]
                    if isinstance(obs_k, np.ndarray):
                        obs_k = torch.from_numpy(obs_k).float().to(self.device)
                    self.obs_trajs[k][step] = obs_k

        self.actions_trajs[step] = action_venv
        self.loss_eps_trajs[step] = loss_eps_venv
        self.loss_t_trajs[step] = loss_t_venv
        self.initial_cfm_loss_trajs[step] = initial_cfm_loss_venv
        self.reward_trajs[step] = reward_venv
        self.terminated_trajs[step] = terminated_venv
        self.firsts_trajs[step + 1] = done_venv.float()
        self.value_trajs[step] = value_venv

    def make_dataset(self):
        """
        将 buffer 转换为训练用的 tensor 数据集（GPU 版本，无需转换）

        obs 可以是:
        - 字典 (图像观测模式): {"state": tensor, "rgb": tensor}
        - tensor (仅状态模式): state tensor
        """
        # 构建观测：支持多模态
        if isinstance(self.obs_dim, dict):
            # 图像观测模式：返回字典
            obs = {
                k: self.obs_trajs[k].flatten(0, 1)
                for k in self.obs_trajs
            }
        else:
            # 仅状态模式：返回 tensor
            obs = self.obs_trajs["state"].flatten(0, 1)

        actions = self.actions_trajs.flatten(0, 1)
        loss_eps = self.loss_eps_trajs.flatten(0, 1)
        loss_t = self.loss_t_trajs.flatten(0, 1)
        initial_cfm_loss = self.initial_cfm_loss_trajs.flatten(0, 1)
        returns = self.returns_trajs.flatten(0, 1)
        values = self.value_trajs.flatten(0, 1)
        advantages = self.advantages_trajs.flatten(0, 1)

        return (obs, actions, loss_eps, loss_t, initial_cfm_loss,
                returns, values, advantages)

    @torch.no_grad()
    def normalize_reward(self):
        """
        归一化奖励（GPU 版本）

        注意：running_reward_scaler 需要 numpy array，
        所以需要在 CPU 上执行后再转回 GPU
        """
        if self.reward_scale_running:
            # 将 GPU tensor 转换为 numpy
            reward_np = self.reward_trajs.cpu().numpy()
            firsts_np = self.firsts_trajs[:-1].cpu().numpy()

            # 在 CPU 上执行归一化
            reward_trajs_transpose = self.running_reward_scaler(
                reward=reward_np.T, first=firsts_np.T
            )

            # 转回 GPU
            self.reward_trajs = torch.from_numpy(
                reward_trajs_transpose.T
            ).float().to(self.device)

    @torch.no_grad()
    def update(self, obs_venv: dict, critic: torch.nn.Module):
        """
        更新 buffer（GPU 版本）

        覆盖父类方法以正确处理 GPU tensor
        """
        self.normalize_reward()
        self.update_adv_returns(obs_venv, critic)

    @torch.no_grad()
    def update_adv_returns(self, obs_venv: dict, critic: torch.nn.Module):
        """
        更新优势估计和回报（GPU 版本）

        使用 GAE (Generalized Advantage Estimation)

        注意：这个方法与 PPOBuffer.update_adv_returns 逻辑相同，
        但针对 GPU tensor 优化，避免 CPU-GPU 数据传输。
        """
        # 构建观测字典，包含所有可用的键（state 和可选的 rgb）
        obs_venv_ts = {}

        # 处理 state 键
        if "state" in obs_venv:
            obs_venv_ts["state"] = (
                torch.from_numpy(obs_venv["state"]).float().to(self.device)
                if isinstance(obs_venv["state"], np.ndarray)
                else obs_venv["state"].to(self.device)
            )

        # 处理 rgb 键（如果存在）
        if "rgb" in obs_venv:
            obs_venv_ts["rgb"] = (
                torch.from_numpy(obs_venv["rgb"]).float().to(self.device)
                if isinstance(obs_venv["rgb"], np.ndarray)
                else obs_venv["rgb"].to(self.device)
            )

        self.advantages_trajs = torch.zeros(
            self.n_steps, self.n_envs, device=self.device
        )

        lastgaelam = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                nextvalues = critic.forward(obs_venv_ts).reshape(1, -1)
            else:
                nextvalues = self.value_trajs[t + 1]

            non_terminal = 1.0 - self.terminated_trajs[t]
            delta = (
                self.reward_trajs[t] * self.reward_scale_const
                + self.gamma * nextvalues * non_terminal
                - self.value_trajs[t]
            )
            self.advantages_trajs[t] = lastgaelam = (
                delta + self.gamma * self.gae_lambda * non_terminal * lastgaelam
            )

        self.returns_trajs = self.advantages_trajs + self.value_trajs

    @torch.no_grad()
    def summarize_episode_reward(self):
        """
        汇总 episode 奖励统计（GPU 版本）

        从 PPODiffusionBufferGPU.summarize_episode_reward 复制，
        主要改动是在访问 GPU tensor 前调用 .cpu().numpy()
        """
        episodes_start_end = []
        # 将 GPU tensor 转换为 numpy 用于处理
        firsts_trajs_np = self.firsts_trajs.cpu().numpy()

        for env_ind in range(self.n_envs):
            env_steps = np.where(firsts_trajs_np[:, env_ind] == 1)[0]
            for i in range(len(env_steps) - 1):
                start = env_steps[i]
                end = env_steps[i + 1]
                if end - start > 1:
                    episodes_start_end.append((env_ind, start, end - 1))

        if len(episodes_start_end) > 0:
            # 使用 numpy 处理 reward 数据
            reward_trajs_split = [
                self.reward_trajs[start:end + 1, env_ind].cpu().numpy()
                for env_ind, start, end in episodes_start_end
            ]
            self.num_episode_finished = len(reward_trajs_split)

            # 计算 episode 奖励
            episode_reward = np.array([np.sum(reward_traj) for reward_traj in reward_trajs_split])

            if self.furniture_sparse_reward:
                episode_best_reward = episode_reward
            else:
                episode_best_reward = np.array([
                    np.max(reward_traj) / self.act_steps
                    for reward_traj in reward_trajs_split
                ])

            # 计算统计指标
            self.avg_episode_reward = np.mean(episode_reward)
            self.avg_best_reward = np.mean(episode_best_reward)
            self.success_rate = np.mean(
                episode_best_reward >= self.best_reward_threshold_for_success
            )

            self.std_episode_reward = np.std(episode_reward)
            self.std_best_reward = np.std(episode_best_reward)
            self.std_success_rate = np.std(
                episode_best_reward >= self.best_reward_threshold_for_success
            )

            episode_lengths = np.array([
                end - start + 1 for _, start, end in episodes_start_end
            ]) * self.act_steps
            self.avg_episode_length = np.mean(episode_lengths)
            self.std_episode_length = np.std(episode_lengths)

        else:
            self.num_episode_finished = 0
            self.avg_episode_reward = 0
            self.avg_best_reward = 0
            self.success_rate = 0
            self.avg_episode_length = 0.0
            self.std_episode_reward = 0
            self.std_best_reward = 0
            self.std_success_rate = 0
            self.std_episode_length = 0.0
