# Copyright (c) 2026 ScoRe-Flow Authors
"""
GRPOFlow / GRPOShortCut — Plain GRPO (无 Score) 版本

与 PPOFlow / PPOShortCut 的区别:
    1. loss() 去掉 value loss, 加 KL penalty
    2. returns / oldvalues 被忽略

保留全部: actor_old, actor_ft (NoisyFlowMLP/NoisyShortCutFlowMLP),
          get_actions, get_logprobs, noise scheduling 等
"""

import torch
import torch.nn.functional as F
import logging

from model.flow.ft_ppo.ppoflow import PPOFlow
from model.flow.ft_ppo.pposhortcut import PPOShortCut

log = logging.getLogger(__name__)


class _GRPOPlainLossMixin:
    """Plain GRPO loss (无 score), 可被 ReFlow 和 ShortCut 复用."""

    def grpo_loss(
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
            log.info(f"oldlogprobs.min={oldlogprobs.min():5.3f}, max={oldlogprobs.max():5.3f}, std={oldlogprobs.std():5.3f}")
            log.info(f"newlogprobs.min={newlogprobs.min():5.3f}, max={newlogprobs.max():5.3f}, std={newlogprobs.std():5.3f}")

        newlogprobs = newlogprobs.clamp(min=self.logprob_min, max=self.logprob_max)
        oldlogprobs = oldlogprobs.clamp(min=self.logprob_min, max=self.logprob_max)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Ratio & KL
        logratio = newlogprobs - oldlogprobs
        ratio = logratio.exp()

        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > self.clip_ploss_coef).float().mean().item()

        # Clipped surrogate policy loss
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_ploss_coef, 1 + self.clip_ploss_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # KL penalty (替代 value loss)
        kl_loss = approx_kl * self.kl_coef

        # Entropy loss
        entropy_loss = -entropy.mean()

        # BC loss
        bc_loss = 0.0
        if use_bc_loss:
            if bc_loss_type == 'W2':
                z = torch.zeros((obs['state'].shape[0], self.horizon_steps, self.action_dim), device=self.device)
                a_old = self.actor_old.sample_action(
                    cond=obs, inference_steps=self.inference_steps,
                    clip_intermediate_actions=True, act_range=[self.act_min, self.act_max], z=z)
                a_new = self.actor_ft.policy.sample_action(
                    cond=obs, inference_steps=self.inference_steps,
                    clip_intermediate_actions=True, act_range=[self.act_min, self.act_max], z=z)
                bc_loss = F.mse_loss(a_old.detach(), a_new)

        return (
            pg_loss,
            entropy_loss,
            kl_loss,        # 位置 2: 原 v_loss → kl_loss
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
            0.0,            # 位置 14: 原 newvalues.mean() → 0
        )


# ── ReFlow 版本 ──────────────────────────────────────────────────────────

class GRPOFlow(_GRPOPlainLossMixin, PPOFlow):
    """GRPO + ReFlow (无 score). 用于 FlowMLP 策略."""

    def __init__(self, kl_coef=0.04, **kwargs):
        PPOFlow.__init__(self, **kwargs)
        self.kl_coef = kl_coef
        log.info(f"[GRPOFlow] kl_coef={kl_coef}, critic kept as dummy.")

    def loss(self, *args, **kwargs):
        return self.grpo_loss(*args, **kwargs)


# ── ShortCut 版本 ────────────────────────────────────────────────────────

class GRPOShortCut(_GRPOPlainLossMixin, PPOShortCut):
    """GRPO + ShortCut (无 score). 用于 ShortCutFlowMLP 策略."""

    def __init__(self, kl_coef=0.04, **kwargs):
        PPOShortCut.__init__(self, **kwargs)
        self.kl_coef = kl_coef
        log.info(f"[GRPOShortCut] kl_coef={kl_coef}, critic kept as dummy.")

    def loss(self, *args, **kwargs):
        return self.grpo_loss(*args, **kwargs)
