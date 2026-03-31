# MIT License
# Copyright (c) 2026 ScoRe-Flow Authors

"""
PPO Flow Agent with Learned Epsilon Network (EpsNet)

继承自 score-based agent，处理 PPOFlowEpsNet 的 16 元素返回值
（多了 alpha_curve_dict 用于 wandb 记录学到的 α(t) 曲线）
"""

import os
import logging
import numpy as np
import torch

from agent.finetune.reinflow.train_ppo_flow_agent_score import TrainPPOFlowAgent
from agent.finetune.reinflow.train_ppo_flow_img_agent_score import TrainPPOImgFlowAgent
from model.flow.ft_ppo.ppoflow_score_mlp import PPOFlowEpsNet

log = logging.getLogger(__name__)


class TrainPPOFlowEpsNetAgent(TrainPPOFlowAgent):
    """
    Gym (state-only) agent for PPOFlowEpsNet.

    覆盖 __init__ 和 agent_update 以:
    1. 跳过 epsilon_schedule 属性（EpsNet 没有这个）
    2. 处理 loss() 返回的 16 个值（含 alpha_curve_dict）
    3. 把 alpha(t) 曲线 log 到 wandb
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model: PPOFlowEpsNet
        log.info(f"EpsNet agent initialized: eps_mlp_type={self.model.eps_mlp_type}, "
                 f"gamma_score={self.model.gamma_score}")

    def agent_update(self, verbose=True):
        clipfracs_list = []
        noise_std_list = []
        alpha_curve = {}

        for update_epoch, batch_id, minibatch in (
            self.minibatch_generator() if not self.repeat_samples
            else self.minibatch_generator_repeat()
        ):
            self.model: PPOFlowEpsNet

            (
                pg_loss, entropy_loss, v_loss, bc_loss,
                clipfrac, approx_kl, ratio,
                oldlogprob_min, oldlogprob_max, oldlogprob_std,
                newlogprob_min, newlogprob_max, newlogprob_std,
                noise_std, Q_values, alpha_curve_dict
            ) = self.model.loss(
                *minibatch,
                use_bc_loss=self.use_bc_loss,
                bc_loss_type=self.bc_loss_type,
                normalize_denoising_horizon=self.normalize_denoising_horizon,
                normalize_act_space_dimension=self.normalize_act_space_dim,
                verbose=verbose,
                clip_intermediate_actions=self.clip_intermediate_actions,
                account_for_initial_stochasticity=self.account_for_initial_stochasticity
            )

            alpha_curve = alpha_curve_dict
            self.approx_kl = approx_kl
            if verbose:
                log.info(f"update_epoch={update_epoch}/{self.update_epochs}, "
                         f"batch_id={batch_id}/{max(1, self.total_steps // self.batch_size)}, "
                         f"ratio={ratio:.3f}, clipfrac={clipfrac:.3f}, approx_kl={self.approx_kl:.2e}")

            if update_epoch == 0 and batch_id == 0 and np.abs(ratio - 1.00) > self.initial_ratio_error_threshold:
                raise ValueError(f"ratio={ratio} not 1.00 at epoch=0 batch=0!")

            if self.target_kl and self.lr_schedule == 'adaptive_kl':
                self.update_lr_adaptive_kl(self.approx_kl)

            loss = pg_loss + entropy_loss * self.ent_coef + v_loss * self.vf_coef + bc_loss * self.bc_coeff

            clipfracs_list.append(clipfrac)
            noise_std_list.append(noise_std)

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            loss.backward()

            actor_norm = torch.nn.utils.clip_grad_norm_(self.model.actor_ft.parameters(), max_norm=float('inf'))
            actor_old_norm = torch.nn.utils.clip_grad_norm_(self.model.actor_old.parameters(), max_norm=float('inf'))
            critic_norm = torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), max_norm=float('inf'))

            # EpsNet gradient norm
            eps_net_norm = 0.0
            if self.model.eps_net is not None:
                eps_net_norm = torch.nn.utils.clip_grad_norm_(self.model.eps_net.parameters(), max_norm=float('inf'))

            if verbose:
                log.info(f"before clipping: actor_norm={actor_norm:.2e}, critic_norm={critic_norm:.2e}, "
                         f"eps_net_norm={eps_net_norm:.2e}")

            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            if self.itr >= self.n_critic_warmup_itr:
                if (self.itr - self.n_critic_warmup_itr) % self.actor_update_freq == 0:
                    for _ in range(self.actor_update_epoch):
                        if self.max_grad_norm:
                            # actor_ft + eps_net params are already in actor_optimizer via get_actor_params()
                            torch.nn.utils.clip_grad_norm_(self.model.get_actor_params(), self.max_grad_norm)
                        self.actor_optimizer.step()

        clip_fracs = np.mean(clipfracs_list)
        score_stds = np.mean(noise_std_list)
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
            "eps_net_norm": eps_net_norm,
            "actor lr": self.actor_optimizer.param_groups[0]["lr"],
            "critic lr": self.critic_optimizer.param_groups[0]["lr"],
            "epsilon_t": self.model.epsilon_t,
            "gamma_score": self.model.gamma_score,
            "score_std": score_stds,
            "Q_values": Q_values,
        }
        # Merge alpha(t) curve into train_ret_dict for wandb logging
        if alpha_curve:
            self.train_ret_dict.update(alpha_curve)


class TrainPPOImgFlowEpsNetAgent(TrainPPOImgFlowAgent):
    """
    Image-based agent for PPOFlowEpsNet (robomimic etc.)

    继承自 TrainPPOImgFlowAgent，覆盖 agent_update 处理 16 元素返回值
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model: PPOFlowEpsNet
        log.info(f"EpsNet img agent initialized: eps_mlp_type={self.model.eps_mlp_type}, "
                 f"gamma_score={self.model.gamma_score}")

    def agent_update(self, verbose=True):
        clipfracs_list = []
        noise_std_list = []
        alpha_curve = {}
        actor_norm = 0.0
        critic_norm = 0.0

        for update_epoch, batch_id, minibatch in (
            self.minibatch_generator() if not self.repeat_samples
            else self.minibatch_generator_repeat()
        ):
            self.model: PPOFlowEpsNet

            (
                pg_loss, entropy_loss, v_loss, bc_loss,
                clipfrac, approx_kl, ratio,
                oldlogprob_min, oldlogprob_max, oldlogprob_std,
                newlogprob_min, newlogprob_max, newlogprob_std,
                noise_std, newQ_values, alpha_curve_dict
            ) = self.model.loss(
                *minibatch,
                use_bc_loss=self.use_bc_loss,
                bc_loss_type=self.bc_loss_type,
                normalize_denoising_horizon=self.normalize_denoising_horizon,
                normalize_act_space_dimension=self.normalize_act_space_dim,
                verbose=verbose,
                clip_intermediate_actions=self.clip_intermediate_actions,
                account_for_initial_stochasticity=self.account_for_initial_stochasticity
            )

            alpha_curve = alpha_curve_dict
            self.approx_kl = approx_kl
            if verbose:
                log.info(f"update_epoch={update_epoch}/{self.update_epochs}, "
                         f"batch_id={batch_id}/{max(1, self.total_steps // self.batch_size)}, "
                         f"ratio={ratio:.3f}, clipfrac={clipfrac:.3f}, approx_kl={self.approx_kl:.2e}")

            if update_epoch == 0 and batch_id == 0 and np.abs(ratio - 1.00) > self.initial_ratio_error_threshold:
                log.info(f"Warning: ratio={ratio} not 1.00 at epoch=0 batch=0!")

            if self.target_kl and self.lr_schedule == 'adaptive_kl':
                self.update_lr_adaptive_kl(self.approx_kl)

            loss = pg_loss + entropy_loss * self.ent_coef + v_loss * self.vf_coef + bc_loss * self.bc_coeff

            clipfracs_list.append(clipfrac)
            noise_std_list.append(noise_std)

            loss.backward()

            # Gradient accumulation (same as parent TrainPPOImgFlowAgent)
            if (batch_id + 1) % self.grad_accumulate == 0:
                actor_norm = torch.nn.utils.clip_grad_norm_(self.model.actor_ft.parameters(), max_norm=float('inf'))
                critic_norm = torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), max_norm=float('inf'))

                eps_net_norm = 0.0
                if self.model.eps_net is not None:
                    eps_net_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.eps_net.parameters(), max_norm=float('inf'))

                if verbose:
                    log.info(f"before clipping: actor_norm={actor_norm:.2e}, "
                             f"critic_norm={critic_norm:.2e}, eps_net_norm={eps_net_norm:.2e}")

                # Update actor (after critic warmup)
                if self.itr >= self.n_critic_warmup_itr:
                    if self.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.get_actor_params(), self.max_grad_norm)
                    self.actor_optimizer.step()

                # Update critic
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                log.info(f"run grad update at batch {batch_id}")

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
            "gamma_score": self.model.gamma_score,
            "Q_values": self.Q_values,
        }
        if alpha_curve:
            self.train_ret_dict.update(alpha_curve)
