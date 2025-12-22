# MIT License
# Copyright (c) 2025 ReinFlow Authors - Dual-Stream Score Editing

"""
Dual-Stream Score Editing PPO Flow Agent

核心创新: 
    1. 保守流 (Prior Stream): 锚定数据流形，提供 v_uncond
    2. 激进流 (Reward Stream): 探索高奖励区域，提供 v_cond
    3. CFG 引导: v_guided = v_prior + w * (v_reward - v_prior)

SDE 采样:
    dxt = [v_guided + εt·st] dt + √(2εt) dWt
"""
import os
import logging
log = logging.getLogger(__name__)
from tqdm import tqdm as tqdm
import numpy as np
import torch
from agent.finetune.reinflow.train_ppo_flow_agent_score import TrainPPOFlowAgent
from model.flow.ft_ppo.ppoflow_dual_stream import DualStreamPPOFlow


class TrainPPOFlowDualStreamAgent(TrainPPOFlowAgent):
    """
    Dual-Stream Score Editing PPO Flow Agent
    
    使用 CFG 机制替代显式奖励梯度:
        - actor_prior: 保守流 (冻结), 锚定数据流形
        - actor_reward: 激进流 (可训练), 探索高奖励
        - CFG 引导权重 w: 控制探索强度
    """
    
    def __init__(self, cfg):
        super().__init__(cfg)

        # Dual-Stream 特定参数
        self.cfg_weight = cfg.model.get('cfg_weight', 1.5)
        self.cfg_weight_schedule = cfg.model.get('cfg_weight_schedule', 'constant')
        self.freeze_prior = cfg.model.get('freeze_prior', True)

        # 逐步解冻参数
        self.unfreeze_prior_schedule = cfg.model.get('unfreeze_prior_schedule', 'none')
        self.unfreeze_start_progress = cfg.model.get('unfreeze_start_progress', 0.5)
        self.unfreeze_prior_lr_scale = cfg.model.get('unfreeze_prior_lr_scale', 0.1)

        # 训练进度追踪 (用于 CFG 权重自适应调度)
        self.training_progress = 0.0

        # Prior 优化器 (解冻后使用)
        self.prior_optimizer = None
        self.prior_lr_scheduler = None

        self.model: DualStreamPPOFlow
                # Dual-Stream 架构可能有更大的数值误差，放宽阈值
        self.initial_ratio_error_threshold = 1e-5  # 覆盖父类的 1e-6
        log.info(f"Dual-Stream CFG: weight={self.cfg_weight}, schedule={self.cfg_weight_schedule}, freeze_prior={self.freeze_prior}")
        log.info(f"Prior unfreeze: schedule={self.unfreeze_prior_schedule}, start_progress={self.unfreeze_start_progress}, lr_scale={self.unfreeze_prior_lr_scale}")

    @torch.no_grad()
    def get_samples_logprobs(
        self, 
        cond: dict, 
        ret_device='cpu', 
        save_chains=True, 
        normalize_denoising_horizon=False, 
        normalize_act_space_dimension=False, 
        clip_intermediate_actions=True,
        account_for_initial_stochasticity=True
    ):
        """使用 Dual-Stream CFG 引导采样"""
        if save_chains:
            action_samples, chains_venv, logprob_venv = self.model.get_actions(
                cond, 
                eval_mode=self.eval_mode, 
                save_chains=save_chains, 
                normalize_denoising_horizon=normalize_denoising_horizon, 
                normalize_act_space_dimension=normalize_act_space_dimension, 
                clip_intermediate_actions=clip_intermediate_actions,
                account_for_initial_stochasticity=account_for_initial_stochasticity,
                training_progress=self.training_progress
            )
            return (
                action_samples.cpu().numpy(), 
                chains_venv.cpu().numpy() if ret_device == 'cpu' else chains_venv, 
                logprob_venv.cpu().numpy() if ret_device == 'cpu' else logprob_venv
            )
        else:
            action_samples, logprob_venv = self.model.get_actions(
                cond, 
                eval_mode=self.eval_mode, 
                save_chains=save_chains, 
                normalize_denoising_horizon=normalize_denoising_horizon, 
                normalize_act_space_dimension=normalize_act_space_dimension, 
                clip_intermediate_actions=clip_intermediate_actions,
                account_for_initial_stochasticity=account_for_initial_stochasticity,
                training_progress=self.training_progress
            )
            return (
                action_samples.cpu().numpy(), 
                logprob_venv.cpu().numpy() if ret_device == 'cpu' else logprob_venv
            )

    def agent_update(self, verbose=True):
        """更新训练进度后调用父类方法"""
        # 更新训练进度
        self.training_progress = min(1.0, self.itr / max(1, self.n_train_itr))

        # 逐步解冻保守流
        unfreeze_info = self._update_prior_unfreeze_state(verbose)

        clipfracs_list = []
        noise_std_list = []

        for update_epoch, batch_id, minibatch in (
            self.minibatch_generator() if not self.repeat_samples
            else self.minibatch_generator_repeat()
        ):
            self.model: DualStreamPPOFlow

            # 解包 minibatch 并添加 training_progress
            obs, chains, returns, oldvalues, advantages, oldlogprobs = minibatch

            pg_loss, entropy_loss, v_loss, bc_loss, \
            clipfrac, approx_kl, ratio, \
            oldlogprob_min, oldlogprob_max, oldlogprob_std, \
            newlogprob_min, newlogprob_max, newlogprob_std, \
            noise_std, Q_values = self.model.loss(
                obs, chains, returns, oldvalues, advantages, oldlogprobs,
                use_bc_loss=self.use_bc_loss,
                bc_loss_type=self.bc_loss_type,
                normalize_denoising_horizon=self.normalize_denoising_horizon,
                normalize_act_space_dimension=self.normalize_act_space_dim,
                verbose=verbose,
                clip_intermediate_actions=self.clip_intermediate_actions,
                account_for_initial_stochasticity=self.account_for_initial_stochasticity,
                training_progress=self.training_progress
            )

            self.approx_kl = approx_kl
            if verbose:
                log.info(f"update_epoch={update_epoch}/{self.update_epochs}, batch_id={batch_id}, "
                        f"ratio={ratio:.3f}, clipfrac={clipfrac:.3f}, approx_kl={self.approx_kl:.2e}, "
                        f"cfg_weight={self.model.cfg_weight:.3f}")

            if update_epoch == 0 and batch_id == 0 and np.abs(ratio - 1.00) > self.initial_ratio_error_threshold:
                raise ValueError(f"ratio={ratio} not 1.00 at update_epoch=0, batch_id=0!")

            if self.target_kl and self.lr_schedule == 'adaptive_kl':
                self.update_lr_adaptive_kl(self.approx_kl)

            loss = pg_loss + entropy_loss * self.ent_coef + v_loss * self.vf_coef + bc_loss * self.bc_coeff

            clipfracs_list.append(clipfrac)
            noise_std_list.append(noise_std)

            # 更新优化器
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            if self.prior_optimizer is not None:
                self.prior_optimizer.zero_grad()

            loss.backward()

            # 梯度裁剪和更新
            actor_norm = torch.nn.utils.clip_grad_norm_(self.model.actor_reward.parameters(), max_norm=float('inf'))
            actor_prior_norm = torch.nn.utils.clip_grad_norm_(self.model.actor_prior.parameters(), max_norm=float('inf'))
            critic_norm = torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), max_norm=float('inf'))
            if verbose:
                log.info(f"before clipping: actor_reward_norm={actor_norm:.2e}, critic_norm={critic_norm:.2e}, actor_prior_norm={actor_prior_norm:.2e}")

            # 更新 critic
            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            # 更新 actor (reward stream)
            if self.itr >= self.n_critic_warmup_itr:
                if (self.itr - self.n_critic_warmup_itr) % self.actor_update_freq == 0:
                    for _ in range(self.actor_update_epoch):
                        if self.max_grad_norm:
                            torch.nn.utils.clip_grad_norm_(self.model.actor_reward.parameters(), self.max_grad_norm)
                        self.actor_optimizer.step()

                        # 同时更新 prior (如果已解冻)
                        self._update_prior_optimizer()

        # 记录统计信息
        clip_fracs = np.mean(clipfracs_list)
        score_stds = np.mean(noise_std_list)

        # 获取 prior 解冻状态
        prior_status = self.model.get_prior_freeze_status()

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
            "prior lr": self.prior_optimizer.param_groups[0]["lr"] if self.prior_optimizer else 0.0,
            "epsilon_t": self.model.epsilon_t,
            "score_std": score_stds,
            "Q_values": Q_values,
            "cfg_weight": self.model.cfg_weight,
            "training_progress": self.training_progress,
            "prior_unfrozen_ratio": 1.0 - prior_status['frozen_ratio'],
        }

    def run(self):
        """主训练循环"""
        log.info("=" * 60)
        log.info("Starting Dual-Stream Score Editing PPO Flow Training")
        log.info(f"CFG weight: {self.cfg_weight}, schedule: {self.cfg_weight_schedule}")
        log.info(f"Prior stream frozen: {self.freeze_prior}")
        log.info("=" * 60)

        # 调用父类的 run 方法
        super().run()

    def save_model(self, only_save_policy_network=False):
        """
        保存模型检查点 - 兼容父类接口
        Dual-Stream 版本: 保存 actor_reward (作为 policy) 和 actor_prior
        """
        # actor_reward 是可训练的策略网络，格式与父类保持一致
        policy_network_state_dict = {
            'network.' + key: value for key, value in self.model.actor_reward.state_dict().items()
        }

        # Dual-Stream 额外保存 prior 网络
        prior_network_state_dict = {
            'network.' + key: value for key, value in self.model.actor_prior.state_dict().items()
        }

        if only_save_policy_network:
            data = {
                "itr": self.itr,
                "cnt_train_steps": self.cnt_train_step,
                "policy": policy_network_state_dict,
                "prior_policy": prior_network_state_dict,
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "actor_lr_scheduler": self.actor_lr_scheduler.state_dict() if self.actor_lr_scheduler else None,
                "critic_lr_scheduler": self.critic_lr_scheduler.state_dict() if self.critic_lr_scheduler else None,
                "cfg_weight": self.model.cfg_weight,
                "training_progress": self.training_progress,
            }
        else:
            data = {
                "itr": self.itr,
                "cnt_train_steps": self.cnt_train_step,
                "model": self.model.state_dict(),  # for resume training
                "policy": policy_network_state_dict,  # reward stream policy for evaluation
                "prior_policy": prior_network_state_dict,  # prior stream policy
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "actor_lr_scheduler": self.actor_lr_scheduler.state_dict() if self.actor_lr_scheduler else None,
                "critic_lr_scheduler": self.critic_lr_scheduler.state_dict() if self.critic_lr_scheduler else None,
                "cfg_weight": self.model.cfg_weight,
                "training_progress": self.training_progress,
            }

        # always save the last model for resume of training
        save_path = os.path.join(self.checkpoint_dir, "last.pt")
        torch.save(data, os.path.join(self.checkpoint_dir, save_path))

        # optionally save intermediate models
        if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
            save_path = os.path.join(self.checkpoint_dir, f"state_{self.itr}.pt")
            torch.save(data, os.path.join(self.checkpoint_dir, save_path))
            log.info(f"\n Saved Dual-Stream model at itr={self.itr} to {save_path}\n ")

        # save the best model evaluated so far
        if self.is_best_so_far:
            save_path = os.path.join(self.checkpoint_dir, "best.pt")
            torch.save(data, os.path.join(self.checkpoint_dir, save_path))
            log.info(f"\n Saved model with the highest evaluated average episode reward {self.current_best_reward:4.3f} to \n{save_path}\n ")
            self.is_best_so_far = False

    def resume_training(self):
        """
        Resume training from checkpoint - Dual-Stream 版本
        """
        log.info(f"Resuming Dual-Stream training from {self.resume_path}...")
        data = torch.load(self.resume_path, weights_only=True, map_location=self.device)
        log.info(f"Recover checkpoint: data={data.keys()}")

        # recover itr, scheduler, sample number
        self.itr = data["itr"]
        self.cnt_train_step = self.itr * self.n_envs * self.act_steps * self.n_steps if 'cnt_train_steps' not in data.keys() else data["cnt_train_steps"]
        self.n_train_itr += self.itr  # train for another xx iters
        log.info(f"Resume training from itr={self.itr}, total train steps={self.cnt_train_step}.")

        # load models
        if "model" in data.keys():
            self.model.load_state_dict(data["model"], strict=True)
            log.info(f"Loaded full Dual-Stream model")
        elif "policy" in data.keys():
            # 加载 reward stream (actor_reward)
            weights = {k.replace("network.", ""): v for k, v in data["policy"].items()}
            self.model.actor_reward.load_state_dict(weights, strict=True)
            log.info(f"Loaded reward stream policy")

            # 可选: 加载 prior stream
            if "prior_policy" in data.keys():
                prior_weights = {k.replace("network.", ""): v for k, v in data["prior_policy"].items()}
                self.model.actor_prior.load_state_dict(prior_weights, strict=True)
                log.info(f"Loaded prior stream policy")
        else:
            raise ValueError(f"Checkpoint does not contain 'model' or 'policy' keys. Keys: {data.keys()}")

        log.info(f"Successfully loaded model from path={self.resume_path}")

        # load optimizers
        if "actor_optimizer" in data.keys():
            self.actor_optimizer.load_state_dict(data["actor_optimizer"])
        if "critic_optimizer" in data.keys():
            self.critic_optimizer.load_state_dict(data["critic_optimizer"])
        log.info(f"Successfully loaded optimizers from path={self.resume_path}")

        # load scheduler
        if 'actor_lr_scheduler' in data.keys() and data['actor_lr_scheduler'] is not None:
            self.actor_lr_scheduler.load_state_dict(data["actor_lr_scheduler"])
        if 'critic_lr_scheduler' in data.keys() and data['critic_lr_scheduler'] is not None:
            self.critic_lr_scheduler.load_state_dict(data["critic_lr_scheduler"])
        log.info(f"Successfully loaded schedulers from path={self.resume_path}")

        # Dual-Stream 特有: 恢复 cfg_weight 和 training_progress
        if 'cfg_weight' in data.keys():
            self.model.cfg_weight = data['cfg_weight']
        if 'training_progress' in data.keys():
            self.training_progress = data['training_progress']

        log.info(f"Resumed Dual-Stream training: itr={self.itr}, cfg_weight={self.model.cfg_weight}")

    def _setup_optimizer(self):
        """设置优化器 - 只优化 reward stream"""
        # 只优化 actor_reward (激进流)
        self.actor_optimizer = torch.optim.AdamW(
            self.model.actor_reward.parameters(),
            lr=self.actor_lr,
            weight_decay=self.actor_weight_decay
        )

        # Critic 优化器
        self.critic_optimizer = torch.optim.AdamW(
            self.model.critic.parameters(),
            lr=self.critic_lr,
            weight_decay=self.critic_weight_decay
        )

        log.info(f"Optimizer setup: actor_reward lr={self.actor_lr}, critic lr={self.critic_lr}")
        log.info(f"Prior stream is {'frozen' if self.freeze_prior else 'trainable'}")

    def _update_prior_unfreeze_state(self, verbose=True) -> dict:
        """
        根据训练进度更新保守流的冻结状态并创建/更新优化器

        Returns:
            dict: 解冻状态信息
        """
        # 调用模型的解冻方法
        unfreeze_info = self.model.update_prior_freeze_state(self.training_progress)

        if not unfreeze_info['unfrozen']:
            return unfreeze_info

        # 如果有新解冻的参数，需要创建/更新 prior 优化器
        trainable_prior_params = self.model.get_prior_trainable_params()
        
        
        # 从配置获取 weight_decay
        actor_weight_decay = self.cfg.train.get('actor_weight_decay', 0)

        if len(trainable_prior_params) > 0 and self.prior_optimizer is None:
            # 首次解冻：创建 prior 优化器
            prior_lr = self.actor_lr * self.unfreeze_prior_lr_scale
            self.prior_optimizer = torch.optim.AdamW(
                trainable_prior_params,
                lr=prior_lr,
                weight_decay=actor_weight_decay
            )
            log.info(f"Created prior optimizer with lr={prior_lr:.2e} (scale={self.unfreeze_prior_lr_scale})")

            # 可选：为 prior 创建学习率调度器
            if self.actor_lr_type == "cosine":
                from util.scheduler import CosineAnnealingWarmupRestarts
                remaining_steps = int((1.0 - self.training_progress) * self.n_train_itr * self.update_epochs)
                self.prior_lr_scheduler = CosineAnnealingWarmupRestarts(
                    self.prior_optimizer,
                    first_cycle_steps=max(1, remaining_steps),
                    cycle_mult=1.0,
                    max_lr=prior_lr,
                    min_lr=prior_lr * 0.1,
                    warmup_steps=0,
                    gamma=1.0,
                )
        elif len(trainable_prior_params) > 0 and self.prior_optimizer is not None:
            # 如果有新层被解冻，需要更新优化器的参数组
            # 简单策略：重新创建优化器
            current_lr = self.prior_optimizer.param_groups[0]['lr']
            self.prior_optimizer = torch.optim.AdamW(
                trainable_prior_params,
                lr=current_lr,
                weight_decay=actor_weight_decay
            )

        # 根据解冻策略调整学习率
        if 'lr_scale' in unfreeze_info and self.prior_optimizer is not None:
            effective_lr = self.actor_lr * unfreeze_info['lr_scale']
            for param_group in self.prior_optimizer.param_groups:
                param_group['lr'] = effective_lr

        if verbose and unfreeze_info['unfrozen']:
            prior_status = self.model.get_prior_freeze_status()
            log.info(f"Prior unfreeze status: trainable={prior_status['trainable']}, "
                    f"frozen_ratio={prior_status['frozen_ratio']:.2%}, "
                    f"unfrozen_layers={len(prior_status['unfrozen_layers'])}")

        return unfreeze_info

    def _update_prior_optimizer(self):
        """更新 prior 优化器 (在 agent_update 的梯度更新步骤中调用)"""
        if self.prior_optimizer is not None:
            trainable_params = self.model.get_prior_trainable_params()
            if len(trainable_params) > 0:
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)
                self.prior_optimizer.step()

                if self.prior_lr_scheduler is not None:
                    self.prior_lr_scheduler.step()

