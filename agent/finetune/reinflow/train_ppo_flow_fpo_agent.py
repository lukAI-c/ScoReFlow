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

        # 图像观测配置
        self.use_image_obs = cfg.env.get('use_image_obs', False)

        # 设置观测维度（图像模式下使用字典）
        if self.use_image_obs:
            shape_meta = cfg.shape_meta
            self.obs_dims = {k: shape_meta.obs[k]["shape"] for k in shape_meta.obs}
            log.info(f"Image observation mode enabled. obs_dims={self.obs_dims}")
        else:
            # 非图像模式，obs_dim 已经在父类中设置
            log.info(f"State-only observation mode. obs_dim={self.obs_dim}")

        # 模型类型检查
        self.model: PPOFlowFPO
        self.options_venv = [{"timestep": 0} for _ in range(self.n_envs)]

        log.info(f"FPO Agent initialized with n_samples_per_action={self.n_samples_per_action}")
    
    def _process_obs(self, obs):
        """
        处理观测格式，确保键名符合模型期望
        
        robomimic_image wrapper 返回的观测键名可能是:
        - 'agentview_image': 图像观测
        - 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos': 低维状态
        
        模型（ViTCritic）期望的键名是:
        - 'rgb': 图像观测
        - 'state': 合并后的低维状态
        
        Args:
            obs: 原始观测，可能是字典或 numpy 数组
            
        Returns:
            处理后的观测字典，键名为 'rgb' 和 'state'
        """
        if not isinstance(obs, dict):
            return obs
        
        if not self.use_image_obs:
            # 如果不使用图像观测，直接返回
            return obs
        
        # 处理图像观测：重命名键
        processed_obs = {}
        
        # 处理图像键
        if 'agentview_image' in obs:
            processed_obs['rgb'] = obs['agentview_image']
        elif 'rgb' in obs:
            processed_obs['rgb'] = obs['rgb']
        else:
            raise KeyError(f"No image key found in observation. Available keys: {list(obs.keys())}")
        
        # 处理状态键：合并多个 lowdim 键
        state_keys = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
        
        if any(k in obs for k in state_keys):
            # 合并所有存在的状态键
            state_parts = [obs[k] for k in state_keys if k in obs]
            if state_parts:
                processed_obs['state'] = np.concatenate(state_parts, axis=-1)
        elif 'state' in obs:
            # 如果已经有合并好的 state 键
            processed_obs['state'] = obs['state']
        else:
            raise KeyError(f"No state keys found in observation. Available keys: {list(obs.keys())}")
        
        return processed_obs
    
    def reset_env_all(self, options_venv=None, verbose=False, **kwargs):
        """
        FPO 版本的 reset_env_all，处理返回格式兼容性问题
        
        覆盖父类方法以确保返回格式符合底层 async_vector_env 的预期
        """
        if options_venv is None:
            options_venv = self.options_venv

        # 调用底层 reset_arg
        results = self.venv.reset_arg(options_list=options_venv)
        
        # 处理返回值格式
        # gym 环境的 reset 可能返回 (obs, info) 或直接返回 obs
        if isinstance(results, tuple) and len(results) == 2:
            obs_venv, infos = results
        else:
            obs_venv = results
            infos = None
        
        # 如果 obs_venv 是列表且元素是元组，解包
        if isinstance(obs_venv, list) and len(obs_venv) > 0:
            if isinstance(obs_venv[0], tuple):
                obs_venv = [obs[0] for obs in obs_venv]
            
            # 转换为字典格式（stack 各个环境的观测）
            if isinstance(obs_venv[0], dict):
                obs_venv = {
                    key: np.stack([obs[key] for obs in obs_venv])
                    for key in obs_venv[0].keys()
                }
        
        if verbose:
            for index in range(self.n_envs):
                log.info(f"<-- Reset environment {index} with options {options_venv[index]}")
        
        return obs_venv
    def init_buffer(self):
        """初始化 FPO 专用 Buffer（覆盖父类方法）"""
        buffer_cls = PPOFlowFPOBufferGPU if self.buffer_on_gpu else PPOFlowFPOBuffer

        # 根据是否使用图像观测选择正确的 obs_dim 参数
        obs_dim_param = self.obs_dims if self.use_image_obs else self.obs_dim

        self.buffer = buffer_cls(
            n_steps=self.n_steps,
            n_envs=self.n_envs,
            n_samples_per_action=self.n_samples_per_action,
            horizon_steps=self.horizon_steps,
            act_steps=self.act_steps,
            action_dim=self.action_dim,
            n_cond_step=self.n_cond_step,
            obs_dim=obs_dim_param,
            save_full_observation=self.save_full_observations,
            furniture_sparse_reward=self.furniture_sparse_reward,
            best_reward_threshold_for_success=self.best_reward_threshold_for_success,
            reward_scale_running=self.reward_scale_running,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            reward_scale_const=self.reward_scale_const,
            device=self.device
        )
        log.info(f"Initialized FPO buffer: {buffer_cls.__name__} with obs_dim={type(obs_dim_param).__name__}")

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
                    # 处理观测格式
                    processed_obs = self._process_obs(self.prev_obs_venv)
                    
                    # 构建条件输入
                    if self.use_image_obs:
                        cond = {
                            "rgb": torch.tensor(
                                processed_obs["rgb"],
                                device=self.device,
                                dtype=torch.float32
                            ),
                            "state": torch.tensor(
                                processed_obs["state"],
                                device=self.device,
                                dtype=torch.float32
                            )
                        }
                    else:
                        cond = {
                            "state": torch.tensor(
                                processed_obs["state"],
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

                # 存储到 FPO buffer（使用处理后的观测）
                processed_prev_obs = self._process_obs(self.prev_obs_venv)
                self.buffer.add(
                    step=step,
                    obs_venv=processed_obs,
                    state_venv=processed_prev_obs["state"],
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
            # 在rollout结束后，更新done_venv以便下一次迭代正确设置firsts_trajs[0]
            # 这对于正确检测episode边界很重要
            # done_venv = terminated_venv | truncated_venv
            # self.done_venv = done_venv.reshape(1, -1) if done_venv.ndim == 1 else done_venv

            self.buffer.summarize_episode_reward()

            if not self.eval_mode:
                # 处理最终观测格式，确保包含所有必要的键
                processed_final_obs = self._process_obs(obs_venv)
                self.buffer.update(processed_final_obs, self.model.critic)
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

        # total_steps = obs.shape[0]
                # 获取总步数（obs 可能是字典或 tensor）
        if isinstance(obs, dict):
            # 图像观测模式：使用任意一个键的第一个维度
            first_key = next(iter(obs.keys()))
            total_steps = obs[first_key].shape[0]
        else:
            # 状态观测模式
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
                # batch_obs = {"state": obs[batch_idx]}
                                # 构建 batch 数据
                if isinstance(obs, dict):
                    batch_obs = {k: v[batch_idx] for k, v in obs.items()}
                else:
                    batch_obs = obs[batch_idx]
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
                # 添加详细的 CFM 调试信息
                log.info(f"CFM Debug - Loss: {cfm_loss_mean:.6f} | "
                        f"Ratio: mean={ratio_mean:.6f}, min={ratio_min:.6f}, max={ratio_max:.6f}")
                # 总损失
                total_loss = cfm_loss_mean + v_loss * self.vf_coef
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
            torch.save(data, os.path.join(self.checkpoint_dir, save_path))
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
