# Copyright (c) 2026 ScoRe-Flow Authors
"""
GRPO Buffer: Group Relative Policy Optimization buffer.

核心区别: 去掉 critic / GAE, 用 episode-level group-relative advantages 替代.
对于每个 iteration 中收集到的所有完成 episode, 计算:
    A_i = (R_i - mean(R_all)) / (std(R_all) + eps)
然后将 episode 级 advantage 广播到该 episode 内的每个 timestep.
"""

import logging
import numpy as np
import torch

from agent.finetune.reinflow.buffer import PPOFlowBuffer, PPOFlowImgBuffer, PPOFlowImgBufferGPU

log = logging.getLogger(__name__)


class _GRPOAdvMixin:
    """GRPO group-relative advantage 计算逻辑, 可被 state 和 image buffer 复用."""

    @torch.no_grad()
    def update_grpo_advantages(self):
        """
        GRPO 优势估计:
        1. 识别所有完成的 episode (通过 firsts_trajs)
        2. 每个 episode 计算折扣累积回报 R_i = sum(gamma^t * r_t)
        3. 跨所有 episode 归一化: A_i = (R_i - mean(R)) / (std(R) + eps)
        4. 将 episode-level advantage 广播到 timestep 级
        5. 未完成的 episode 使用 partial return 同样参与归一化
        """
        # 兼容 numpy 和 torch tensor
        if isinstance(self.firsts_trajs, torch.Tensor):
            firsts_np = self.firsts_trajs.cpu().numpy()
            reward_np = self.reward_trajs.cpu().numpy()
            use_torch = True
        else:
            firsts_np = self.firsts_trajs
            reward_np = self.reward_trajs
            use_torch = False

        adv_np = np.zeros((self.n_steps, self.n_envs))

        # ── Step 1: 检测 episode 边界 ──────────────────────────────────────
        episodes = []  # [(env_ind, start, end_exclusive)]
        for env_ind in range(self.n_envs):
            env_steps = np.where(firsts_np[:, env_ind] == 1)[0]
            for i in range(len(env_steps) - 1):
                start = env_steps[i]
                end = env_steps[i + 1]
                if end - start > 1:
                    episodes.append((env_ind, start, end))
            # 未完成的尾部 episode
            if len(env_steps) > 0:
                last_start = env_steps[-1]
                if last_start < self.n_steps:
                    episodes.append((env_ind, last_start, self.n_steps))

        if len(episodes) == 0:
            log.warning("[GRPO] No episodes found! Setting advantages to 0.")
            if use_torch:
                self.advantages_trajs = torch.zeros(self.n_steps, self.n_envs, device=self.device)
                self.returns_trajs = self.advantages_trajs.clone()
            else:
                self.advantages_trajs = adv_np
                self.returns_trajs = adv_np.copy()
            return

        # ── Step 2: 计算每个 episode 的折扣累积回报 ────────────────────────
        episode_returns = []
        for env_ind, start, end in episodes:
            rewards = reward_np[start:end, env_ind]
            discounted_return = 0.0
            for t in reversed(range(len(rewards))):
                discounted_return = rewards[t] * self.reward_scale_const + self.gamma * discounted_return
            episode_returns.append(discounted_return)

        episode_returns = np.array(episode_returns)

        # ── Step 3: 组内归一化 ─────────────────────────────────────────────
        mean_return = np.mean(episode_returns)
        std_return = np.std(episode_returns) + 1e-8
        normalized_advantages = (episode_returns - mean_return) / std_return

        # ── Step 4: 广播到 timestep 级 ─────────────────────────────────────
        for (env_ind, start, end), adv in zip(episodes, normalized_advantages):
            adv_np[start:end, env_ind] = adv

        if use_torch:
            self.advantages_trajs = torch.from_numpy(adv_np).float().to(self.device)
            self.returns_trajs = self.advantages_trajs.clone()
            self.value_trajs = torch.zeros_like(self.advantages_trajs)
        else:
            self.advantages_trajs = adv_np
            self.returns_trajs = adv_np.copy()
            self.value_trajs[:] = 0.0

        log.info(f"[GRPO] {len(episodes)} episodes, "
                 f"return mean={mean_return:.3f}, std={std_return:.3f}, "
                 f"advantage range=[{normalized_advantages.min():.3f}, {normalized_advantages.max():.3f}]")


# ── State-based GRPO buffer ──────────────────────────────────────────────

class GRPOFlowBuffer(_GRPOAdvMixin, PPOFlowBuffer):
    """State-based GRPO buffer (gym tasks)."""

    @torch.no_grad()
    def update(self, obs_venv, critic=None, device='cpu'):
        self.normalize_reward()
        self.update_grpo_advantages()


# ── Image-based GRPO buffers ─────────────────────────────────────────────

class GRPOFlowImgBuffer(_GRPOAdvMixin, PPOFlowImgBuffer):
    """Image-based GRPO buffer (CPU version)."""

    @torch.no_grad()
    def update_img(self, obs_venv, model):
        """保留 value/logprob 计算 (augmented), 替换 GAE 为 GRPO advantages."""
        self.normalize_reward()
        self.update_value_logprob(model)
        self.update_grpo_advantages()


class GRPOFlowImgBufferGPU(_GRPOAdvMixin, PPOFlowImgBufferGPU):
    """Image-based GRPO buffer (GPU version)."""

    @torch.no_grad()
    def update_img(self, obs_venv, model):
        """保留 value/logprob 计算 (augmented), 替换 GAE 为 GRPO advantages."""
        self.normalize_reward()
        self.update_value_logprob(model)
        self.update_grpo_advantages()
