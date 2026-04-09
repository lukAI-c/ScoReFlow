# MIT License
# Copyright (c) 2026 ScoRe-Flow Authors

"""
Training agent for Frozen-V variants (PPOFlowWithScoreFrozenV / PPOShortCutWithScoreFrozenV).

与 train_ppo_flow_agent.py 的唯一区别:
    agent_update() 处理 model.loss() 返回的 16 个值 (末尾多一个 alpha_curve_dict)
"""

import logging
import numpy as np
import torch

from agent.finetune.reinflow.train_ppo_flow_agent import TrainPPOFlowAgent

log = logging.getLogger(__name__)


class TrainPPOFlowAgentFrozenV(TrainPPOFlowAgent):
    """
    训练 agent，适配冻结速度场的 Frozen-V 模型。

    model.loss() 返回 16 个值:
        pg_loss, entropy_loss, v_loss, bc_loss,
        clipfrac, approx_kl, ratio,
        oldlogprobs_min, oldlogprobs_max, oldlogprobs_std,
        newlogprobs_min, newlogprobs_max, newlogprobs_std,
        noise_std, newvalues_mean,
        alpha_curve_dict          ← 新增
    """

    def agent_update(self, verbose=True):
        clipfracs_list = []
        noise_std_list = []

        generator = (self.minibatch_generator() if not self.repeat_samples
                     else self.minibatch_generator_repeat())

        for update_epoch, batch_id, minibatch in generator:

            (
                pg_loss, entropy_loss, v_loss, bc_loss,
                clipfrac, approx_kl, ratio,
                oldlogprob_min, oldlogprob_max, oldlogprob_std,
                newlogprob_min, newlogprob_max, newlogprob_std,
                noise_std, Q_values,
                alpha_curve_dict,
            ) = self.model.loss(
                *minibatch,
                use_bc_loss=self.use_bc_loss,
                bc_loss_type=self.bc_loss_type,
                normalize_denoising_horizon=self.normalize_denoising_horizon,
                normalize_act_space_dimension=self.normalize_act_space_dim,
                verbose=verbose,
                clip_intermediate_actions=self.clip_intermediate_actions,
                account_for_initial_stochasticity=self.account_for_initial_stochasticity,
            )

            self.approx_kl = approx_kl
            if verbose:
                log.info(
                    f"update_epoch={update_epoch}/{self.update_epochs}, "
                    f"batch_id={batch_id}/{max(1, self.total_steps // self.batch_size)}, "
                    f"ratio={ratio:.3f}, clipfrac={clipfrac:.3f}, approx_kl={self.approx_kl:.2e}"
                )

            if (update_epoch == 0 and batch_id == 0
                    and np.abs(ratio - 1.00) > self.initial_ratio_error_threshold):
                raise ValueError(
                    f"ratio={ratio} not 1.00 at epoch=0 batch=0. "
                    f"Check for bugs unrelated to hyperparameters!"
                )

            if self.target_kl and self.lr_schedule == 'adaptive_kl':
                self.update_lr_adaptive_kl(self.approx_kl)

            loss = (pg_loss
                    + entropy_loss * self.ent_coef
                    + v_loss * self.vf_coef
                    + bc_loss * self.bc_coeff)

            clipfracs_list.append(clipfrac)
            noise_std_list.append(noise_std)

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()

            actor_norm = torch.nn.utils.clip_grad_norm_(
                self.model.actor_ft.parameters(), max_norm=float('inf'))
            actor_old_norm = torch.nn.utils.clip_grad_norm_(
                self.model.actor_old.parameters(), max_norm=float('inf'))
            critic_norm = torch.nn.utils.clip_grad_norm_(
                self.model.critic.parameters(), max_norm=float('inf'))

            if verbose:
                log.info(
                    f"before clipping: actor_norm={actor_norm:.2e}, "
                    f"critic_norm={critic_norm:.2e}, actor_old_norm={actor_old_norm:.2e}"
                )

            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            if self.itr >= self.n_critic_warmup_itr:
                if (self.itr - self.n_critic_warmup_itr) % self.actor_update_freq == 0:
                    for _ in range(self.actor_update_epoch):
                        if self.max_grad_norm:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.actor_ft.parameters(), self.max_grad_norm)
                        self.actor_optimizer.step()

        clip_fracs = np.mean(clipfracs_list)
        noise_stds = np.mean(noise_std_list)

        self.train_ret_dict = {
            "loss": loss,
            "pg loss": pg_loss,
            "value loss": v_loss,
            "entropy_loss": entropy_loss,
            "bc_loss": bc_loss,
            "approx kl": self.approx_kl,
            "ratio": ratio,
            "clipfrac": clip_fracs,
            "explained variance": self.explained_var,
            "old_logprob_min": oldlogprob_min,
            "old_logprob_max": oldlogprob_max,
            "old_logprob_std": oldlogprob_std,
            "new_logprob_min": newlogprob_min,
            "new_logprob_max": newlogprob_max,
            "new_logprob_std": newlogprob_std,
            "actor_norm": actor_norm,
            "critic_norm": critic_norm,
            "actor lr": self.actor_optimizer.param_groups[0]["lr"],
            "critic lr": self.critic_optimizer.param_groups[0]["lr"],
            "noise_std": noise_stds,
            "Q_values": Q_values,
            "gamma_score": self.model.gamma_score,
            # alpha_t 曲线 (来自最后一个 minibatch)
            **alpha_curve_dict,
        }
