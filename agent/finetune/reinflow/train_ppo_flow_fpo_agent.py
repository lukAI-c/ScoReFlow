# MIT License
# Copyright (c) 2025 ReinFlow Authors + FPO Integration

"""
FPO 风格的 PPO Flow 训练脚本

核心改动（相比 train_ppo_flow_agent.py）:
1. 使用 PPOFlowFPO 模型代替 PPOFlow
2. 使用 PPOFlowFPOBuffer 代替 PPOFlowBuffer
3. 采样时保存 (eps, t, initial_cfm_loss) 而非 (chain, logprob)
4. 更新时使用 CFM 损失变化计算策略比率
"""

import logging
import numpy as np
import torch

from agent.finetune.reinflow.train_ppo_flow_agent import TrainPPOFlowAgent
from model.flow.ft_ppo.ppoflow_fpo import PPOFlowFPO, FPOActionInfo
from agent.finetune.reinflow.buffer_fpo import PPOFlowFPOBuffer, PPOFlowFPOBufferGPU

log = logging.getLogger(__name__)


class TrainPPOFlowFPOAgent(TrainPPOFlowAgent):
    """
    FPO 风格的 PPO Flow 训练 Agent

    继承自 TrainPPOFlowAgent，复用其环境管理、学习率调度等逻辑，
    只替换采样和更新逻辑为 FPO 风格。

    与原始 TrainPPOFlowAgent 的主要区别:
    - 不需要计算 log probability
    - 使用 CFM 损失比率作为策略更新的依据
    - 更简洁的采样逻辑
    - 不需要 noise scheduler 相关配置
    """

    def __init__(self, cfg):
        # 跳过 TrainPPOFlowAgent.__init__，直接调用 TrainPPOAgent.__init__
        # 因为 TrainPPOFlowAgent 的 noise scheduler 逻辑需要 FPO 不使用的配置项
        from agent.finetune.reinflow.train_ppo_agent import TrainPPOAgent
        TrainPPOAgent.__init__(self, cfg)

        # 从 TrainPPOFlowAgent.__init__ 复制必要的初始化
        self.skip_initial_eval = cfg.get('skip_initial_eval', False)
        self.reward_horizon = cfg.get("reward_horizon", self.act_steps)
        self.inference_steps = self.model.inference_steps
        self.ft_denoising_steps = self.model.ft_denoising_steps
        self.repeat_samples = cfg.train.get("repeat_samples", False)

        self.normalize_act_space_dim = True
        self.normalize_denoising_horizon = True
        self.lr_schedule = cfg.train.lr_schedule
        self.clip_intermediate_actions = cfg.train.get("clip_intermediate_actions", True)
        self.account_for_initial_stochasticity = cfg.train.get('account_for_initial_stochasticity', True)
        if self.lr_schedule not in ["fixed", "adaptive_kl"]:
            raise ValueError("lr_schedule should be 'fixed' or 'adaptive_kl'")
        self.actor_lr = cfg.train.actor_lr
        self.critic_lr = cfg.train.critic_lr

        # FPO 不需要 noise scheduler，跳过相关配置
        self.initial_ratio_error_threshold = 1e-6

        # FPO 特定配置
        self.n_samples_per_action = cfg.train.get("n_samples_per_action", 8)
        self.average_losses_before_exp = cfg.train.get("average_losses_before_exp", True)
        self.buffer_on_gpu = cfg.train.get("buffer_on_gpu", False)

        # 模型类型检查
        self.model: PPOFlowFPO

        log.info(f"FPO Agent initialized with n_samples_per_action={self.n_samples_per_action}")

    def init_buffer(self):
        """初始化 FPO 专用 Buffer（覆盖父类方法）"""
        buffer_cls = PPOFlowFPOBufferGPU if self.buffer_on_gpu else PPOFlowFPOBuffer

        self.buffer = buffer_cls(
            n_steps=self.n_steps,
            n_envs=self.n_envs,
            n_samples_per_action=self.n_samples_per_action,
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
            device=self.device
        )
        log.info(f"Initialized FPO buffer: {buffer_cls.__name__}")

    def run(self):
        """
        主训练循环（FPO 版本）

        覆盖父类的 run() 方法，因为 FPO 的采样和更新逻辑与原始 ReinFlow 不同：
        - 不需要 get_samples_logprobs，改用 get_actions
        - buffer.add 参数不同
        - agent_update 使用 CFM 损失比率
        """
        self.init_buffer()
        self.prepare_run()
        self.buffer.reset()

        if self.resume:
            self.resume_training()

        while self.itr < self.n_train_itr:
            self.prepare_video_path()
            self.set_model_mode()
            # self.reset_env()
            # GPU buffer 需要传入设备参数
            buffer_device = self.device if self.buffer_on_gpu else 'cpu'
            self.reset_env(buffer_device=buffer_device)
            # FPO 风格的轨迹收集
            for step in range(self.n_steps):
                with torch.no_grad():
                    cond = {
                        "state": torch.tensor(
                            self.prev_obs_venv["state"],
                            device=self.device,
                            dtype=torch.float32
                        )
                    }
                    value_venv = self.get_value(cond=cond)

                    # FPO 采样：返回 (action, FPOActionInfo)
                    action, action_info = self.model.get_actions(
                        cond=cond,
                        eval_mode=self.eval_mode,
                        clip_intermediate_actions=self.clip_intermediate_actions
                    )

                # 执行动作
                action_venv = action[:, :self.act_steps]
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(
                    action_venv.cpu().numpy() if isinstance(action_venv, torch.Tensor) else action_venv
                )

                # 存储到 FPO buffer
                self.buffer.add(
                    step=step,
                    state_venv=self.prev_obs_venv["state"],
                    action_venv=action.cpu().numpy() if isinstance(action, torch.Tensor) else action,
                    loss_eps_venv=action_info.loss_eps.cpu().numpy(),
                    loss_t_venv=action_info.loss_t.cpu().numpy(),
                    initial_cfm_loss_venv=action_info.initial_cfm_loss.cpu().numpy(),
                    reward_venv=reward_venv,
                    terminated_venv=terminated_venv,
                    truncated_venv=truncated_venv,
                    value_venv=value_venv
                )

                self.prev_obs_venv = obs_venv
                self.cnt_train_step += self.n_envs * self.act_steps if not self.eval_mode else 0

            self.buffer.summarize_episode_reward()

            if not self.eval_mode:
                self.buffer.update(obs_venv, self.model.critic)
                self.agent_update(verbose=self.verbose)

            self.log()
            self.update_lr()
            self.save_model()
            self.itr += 1

    def agent_update(self, verbose: bool = True):
        """
        FPO 风格的策略更新（覆盖父类方法）

        与原始 ReinFlow 的区别:
        - 不需要重新计算 log probability
        - 使用 CFM 损失变化作为策略比率
        """
        # 获取数据集
        dataset = self.buffer.make_dataset()
        (obs, actions, loss_eps, loss_t, initial_cfm_loss,
         returns, values, advantages) = dataset

        total_steps = obs.shape[0]
        indices = np.arange(total_steps)
        
                # 调试: 打印数据集统计信息
        if verbose:
            log.info(f"Dataset stats: steps={total_steps}, "
                     f"initial_cfm_loss=[{initial_cfm_loss.mean().item():.4f}, "
                     f"std={initial_cfm_loss.std().item():.4f}], "
                     f"advantages=[{advantages.mean().item():.4f}, "
                     f"std={advantages.std().item():.4f}]")

        # 用于记录指标
        early_stop = False

        # 多轮更新
        for update_epoch in range(self.update_epochs):
            if early_stop:
                break
            np.random.shuffle(indices)

            # Mini-batch 更新
            for start in range(0, total_steps, self.batch_size):
                end = min(start + self.batch_size, total_steps)
                batch_idx = indices[start:end]

                # 构建 mini-batch
                batch_obs = {"state": obs[batch_idx]}
                batch_actions = actions[batch_idx]
                batch_loss_eps = loss_eps[batch_idx]
                batch_loss_t = loss_t[batch_idx]
                batch_initial_cfm_loss = initial_cfm_loss[batch_idx]
                batch_returns = returns[batch_idx]
                batch_values = values[batch_idx]
                batch_advantages = advantages[batch_idx]

                # 计算损失
                (pg_loss, v_loss, bc_loss, clipfrac, approx_kl,
                 ratio_mean, ratio_min, ratio_max, cfm_loss_mean,
                 value_mean) = self.model.loss(
                    obs=batch_obs,
                    actions=batch_actions,
                    loss_eps=batch_loss_eps,
                    loss_t=batch_loss_t,
                    initial_cfm_loss=batch_initial_cfm_loss,
                    returns=batch_returns,
                    oldvalues=batch_values,
                    advantages=batch_advantages,
                    use_bc_loss=getattr(self, 'use_bc_loss', False),
                    verbose=verbose and (update_epoch == 0 and start == 0)
                )

                # 总损失
                total_loss = pg_loss + v_loss * self.vf_coef
                if getattr(self, 'use_bc_loss', False):
                    total_loss += bc_loss * getattr(self, 'bc_coeff', 0.1)

                # 反向传播 - 使用父类的 actor_optimizer 和 critic_optimizer
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()

                # 梯度裁剪
                max_grad_norm = getattr(self, 'max_grad_norm', 0.5)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.actor_ft.parameters(), max_grad_norm
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.model.critic.parameters(), max_grad_norm
                    )

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                # 早停检查 - 但不立即返回，先记录指标
                if self.target_kl and approx_kl > self.target_kl * 1.5:
                    log.info(f"Early stopping at epoch {update_epoch} due to KL divergence "
                             f"(approx_kl={approx_kl:.4f} > target={self.target_kl * 1.5:.4f})")
                    early_stop = True
                    break

        # 记录最终指标
        self.pg_loss = pg_loss.item() if torch.is_tensor(pg_loss) else pg_loss
        self.v_loss = v_loss.item() if torch.is_tensor(v_loss) else v_loss
        self.approx_kl = approx_kl
        self.clipfrac = clipfrac
        
            # 计算 explained variance
        values_flat = self.buffer.value_trajs.flatten() if isinstance(
            self.buffer.value_trajs, np.ndarray
        ) else self.buffer.value_trajs.flatten().cpu().numpy()
        returns_flat = self.buffer.returns_trajs.flatten() if isinstance(
            self.buffer.returns_trajs, np.ndarray
        ) else self.buffer.returns_trajs.flatten().cpu().numpy()
        var_y = np.var(returns_flat)
        explained_var = np.nan if var_y == 0 else 1 - np.var(returns_flat - values_flat) / var_y

        # 设置 train_ret_dict 供 log() 方法使用
        total_loss = self.pg_loss + self.v_loss * self.vf_coef
        self.train_ret_dict = {
            "loss": total_loss,
            "pg loss": self.pg_loss,
            "value loss": self.v_loss,
            "approx kl": self.approx_kl,
            "clip_frac": self.clipfrac,
            "explained variance": explained_var,
        }

    def save_model(self, only_save_policy_network=False):
        """
        保存模型到磁盘（FPO 版本，覆盖父类方法）

        与父类的区别：FPO 的 actor_ft 直接是 FlowMLP，没有 .policy 属性
        """
        import os

        # FPO 的 actor_ft 直接是 FlowMLP，不需要 .policy
        policy_network_state_dict = {
            'network.' + key: value
            for key, value in self.model.actor_ft.state_dict().items()
        }

        if only_save_policy_network:
            data = {
                "itr": self.itr,
                "cnt_train_steps": self.cnt_train_step,
                "policy": policy_network_state_dict,
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "actor_lr_scheduler": self.actor_lr_scheduler.state_dict(),
                "critic_lr_scheduler": self.critic_lr_scheduler.state_dict(),
            }
        else:
            data = {
                "itr": self.itr,
                "cnt_train_steps": self.cnt_train_step,
                "model": self.model.state_dict(),
                "policy": policy_network_state_dict,
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "actor_lr_scheduler": self.actor_lr_scheduler.state_dict(),
                "critic_lr_scheduler": self.critic_lr_scheduler.state_dict(),
            }

        # 保存最新模型
        save_path = os.path.join(self.checkpoint_dir, "last.pt")
        torch.save(data, save_path)

        # 定期保存中间模型
        if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
            save_path = os.path.join(self.checkpoint_dir, f"state_{self.itr}.pt")
            torch.save(data, save_path)
            log.info(f"\n Saved model at itr={self.itr} to {save_path}\n ")

        # 保存最佳模型
        if self.is_best_so_far:
            save_path = os.path.join(self.checkpoint_dir, "best.pt")
            torch.save(data, save_path)
            log.info(f"\n Saved best model with reward {self.current_best_reward:.3f} to {save_path}\n ")
            self.is_best_so_far = False

    # 以下方法使用父类的实现，无需覆盖：
    # - prepare_run(): 使用父类的优化器和学习率调度设置
    # - log(): 使用父类的日志记录
    # - update_lr(): 使用父类的学习率更新逻辑
