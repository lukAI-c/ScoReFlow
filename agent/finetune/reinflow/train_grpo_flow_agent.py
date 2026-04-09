# Copyright (c) 2026 ScoRe-Flow Authors
"""
GRPO Flow Agent — Critic-Free training agent.

核心改动 vs TrainPPOFlowAgent:
    1. 去掉 rollout 中的 critic value 计算 (get_value)
    2. buffer.update() 不传 critic → 使用 GRPO group-relative advantages
    3. agent_update() 中去掉 critic optimizer 更新, loss = pg + ent + kl
    4. save_model() 中不保存 critic 相关 state
"""

import os
import logging
import numpy as np
import torch

from agent.finetune.reinflow.train_ppo_flow_agent import TrainPPOFlowAgent
from agent.finetune.reinflow.buffer_grpo import GRPOFlowBuffer

log = logging.getLogger(__name__)


class TrainGRPOFlowAgent(TrainPPOFlowAgent):
    """
    Critic-free GRPO agent for flow matching policy fine-tuning.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # GRPO 特有超参
        self.kl_coef = cfg.train.get("kl_coef", 0.04)
        log.info(f"[GRPO] Critic-free mode. kl_coef={self.kl_coef}")

    # ── 使用 GRPO Buffer ──────────────────────────────────────────────────
    def init_buffer(self):
        self.buffer = GRPOFlowBuffer(
            n_steps=self.n_steps,
            n_envs=self.n_envs,
            n_ft_denoising_steps=self.inference_steps,
            horizon_steps=self.horizon_steps,
            act_steps=self.act_steps,
            action_dim=self.action_dim,
            n_cond_step=self.n_cond_step,
            obs_dim=self.obs_dim,
            save_full_observation=self.save_full_observations,
            furniture_sparse_reward=self.furniture_sparse_reward,
            best_reward_threshold_for_success=self.best_reward_threshold_for_success,
            reward_scale_running=self.reward_scale_running,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            reward_scale_const=self.reward_scale_const,
            device=self.device,
        )

    # ── 覆盖 run: 去掉 critic value 计算 ─────────────────────────────────
    def run(self):
        self.init_buffer()
        self.prepare_run()
        self.buffer.reset()
        if self.resume:
            self.resume_training()
        while self.itr < self.n_train_itr:
            self.prepare_video_path()
            self.set_model_mode()
            self.reset_env()
            self.buffer.update_full_obs()
            for step in range(self.n_steps):

                with torch.no_grad():
                    cond = {
                        "state": torch.tensor(self.prev_obs_venv["state"],
                                              device=self.device, dtype=torch.float32)
                    }
                    # ★ GRPO: 不计算 value, 传 0
                    value_venv = np.zeros(self.n_envs)
                    action_samples, chains_venv, logprob_venv = self.get_samples_logprobs(
                        cond=cond,
                        normalize_denoising_horizon=self.normalize_denoising_horizon,
                        normalize_act_space_dimension=self.normalize_act_space_dim,
                        clip_intermediate_actions=self.clip_intermediate_actions,
                        account_for_initial_stochasticity=self.account_for_initial_stochasticity
                    )

                action_venv = action_samples[:, :self.act_steps]
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)

                self.buffer.save_full_obs(info_venv)
                self.buffer.add(step, self.prev_obs_venv["state"], chains_venv,
                                reward_venv, terminated_venv, truncated_venv,
                                value_venv, logprob_venv)

                self.prev_obs_venv = obs_venv
                self.cnt_train_step += self.n_envs * self.act_steps if not self.eval_mode else 0

            self.buffer.summarize_episode_reward()

            if not self.eval_mode:
                # ★ GRPO: buffer.update 不需要 critic
                self.buffer.update(obs_venv)
                self.agent_update(verbose=self.verbose)

            self.log()
            self.update_lr()
            self.adjust_finetune_schedule()
            self.save_model()
            self.itr += 1

    # ── 覆盖 agent_update: 去掉 critic, 使用 KL penalty ──────────────────
    def agent_update(self, verbose=True):
        clipfracs_list = []
        noise_std_list = []

        generator = (self.minibatch_generator() if not self.repeat_samples
                     else self.minibatch_generator_repeat())

        for update_epoch, batch_id, minibatch in generator:

            pg_loss, entropy_loss, kl_loss, bc_loss, \
                clipfrac, approx_kl, ratio, \
                oldlogprob_min, oldlogprob_max, oldlogprob_std, \
                newlogprob_min, newlogprob_max, newlogprob_std, \
                noise_std, Q_values = self.model.loss(
                    *minibatch,
                    use_bc_loss=self.use_bc_loss,
                    bc_loss_type=self.bc_loss_type,
                    normalize_denoising_horizon=self.normalize_denoising_horizon,
                    normalize_act_space_dimension=self.normalize_act_space_dim,
                    verbose=verbose,
                    clip_intermediate_actions=self.clip_intermediate_actions,
                    account_for_initial_stochasticity=self.account_for_initial_stochasticity
                )

            self.approx_kl = approx_kl
            if verbose:
                log.info(f"update_epoch={update_epoch}/{self.update_epochs}, "
                         f"batch_id={batch_id}/{max(1, self.total_steps // self.batch_size)}, "
                         f"ratio={ratio:.3f}, clipfrac={clipfrac:.3f}, "
                         f"approx_kl={self.approx_kl:.2e}")

            if (update_epoch == 0 and batch_id == 0
                    and np.abs(ratio - 1.00) > self.initial_ratio_error_threshold):
                msg = (f"ratio={ratio} not 1.00 at epoch=0 batch=0. "
                       f"Check for bugs unrelated to hyperparameters!")
                if np.abs(ratio - 1.00) > 0.01:
                    raise ValueError(msg)
                else:
                    log.warning(msg)

            if self.target_kl and self.lr_schedule == 'adaptive_kl':
                self.update_lr_adaptive_kl(self.approx_kl)

            # ★ GRPO loss: pg + entropy + KL penalty + BC (无 value loss)
            loss = (pg_loss
                    + entropy_loss * self.ent_coef
                    + kl_loss
                    + bc_loss * self.bc_coeff)

            clipfracs_list.append(clipfrac)
            noise_std_list.append(noise_std)

            # ★ 只更新 actor, 不更新 critic
            self.actor_optimizer.zero_grad()
            loss.backward()

            actor_norm = torch.nn.utils.clip_grad_norm_(
                self.model.actor_ft.parameters(), max_norm=float('inf'))

            if verbose:
                log.info(f"before clipping: actor_norm={actor_norm:.2e}")

            # ★ 无 critic warmup, actor 始终更新
            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.actor_ft.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

        clip_fracs = np.mean(clipfracs_list)
        noise_stds = np.mean(noise_std_list)

        self.train_ret_dict = {
            "loss": loss,
            "pg loss": pg_loss,
            "kl_loss": kl_loss,
            "entropy_loss": entropy_loss,
            "bc_loss": bc_loss,
            "approx kl": self.approx_kl,
            "ratio": ratio,
            "clipfrac": clip_fracs,
            "explained variance": 0.0,  # GRPO 无 critic, 无 explained variance
            "old_logprob_min": oldlogprob_min,
            "old_logprob_max": oldlogprob_max,
            "old_logprob_std": oldlogprob_std,
            "new_logprob_min": newlogprob_min,
            "new_logprob_max": newlogprob_max,
            "new_logprob_std": newlogprob_std,
            "actor_norm": actor_norm,
            "actor lr": self.actor_optimizer.param_groups[0]["lr"],
            "min_logprob_noise_std": self.model.min_logprob_denoising_std,
            "min_sampling_noise_std": self.model.min_sampling_denoising_std,
            "noise_std": noise_stds,
        }

    # ── 覆盖 update_lr: 只更新 actor scheduler ──────────────────────────
    def update_lr(self):
        self.actor_lr_scheduler.step()
        log.info(f"learning rate updated. actor_lr={self.actor_optimizer.param_groups[0]['lr']:.2e}")

    # ── 覆盖 save_model: 不保存 critic ───────────────────────────────────
    def save_model(self, only_save_policy_network=False):
        policy_network_state_dict = {
            'network.' + key: value
            for key, value in self.model.actor_ft.policy.state_dict().items()
        }

        data = {
            "itr": self.itr,
            "cnt_train_steps": self.cnt_train_step,
            "model": self.model.state_dict(),
            "policy": policy_network_state_dict,
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_scheduler": self.actor_lr_scheduler.state_dict(),
        }

        save_path = os.path.join(self.checkpoint_dir, "last.pt")
        torch.save(data, save_path)

        if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
            save_path = os.path.join(self.checkpoint_dir, f"state_{self.itr}.pt")
            torch.save(data, save_path)
            log.info(f"\n Saved model at itr={self.itr} to {save_path}\n ")

        if self.is_best_so_far:
            save_path = os.path.join(self.checkpoint_dir, "best.pt")
            torch.save(data, save_path)
            log.info(f"\n Saved best model (reward={self.current_best_reward:4.3f}) to {save_path}\n ")
            self.is_best_so_far = False
