# MIT License
# Copyright (c) 2025 ReinFlow Authors - QR-DQN Extension

"""
PPO Flow Agent with Image Input and QR-DQN Critic

用于 robomimic square 等图像输入任务的 QR-DQN 训练

核心优势 (相比 C51):
    - 无需预设 v_min/v_max
    - 自动适应任意奖励范围
    - 更鲁棒的价值估计

运行示例:
    python script/run.py --config-dir=cfg/robomimic/finetune/square \
        --config-name=ft_ppo_reflow_mlp_img_score_qrdqn
"""

from tqdm import tqdm
import torch
import numpy as np
import logging
import os
log = logging.getLogger(__name__)

from agent.finetune.reinflow.train_ppo_flow_img_agent_score import TrainPPOImgFlowAgent
from model.flow.ft_ppo.ppoflow_score_qrdqn import PPOFlowQRDQN
from agent.finetune.reinflow.buffer_distribute import PPOFlowImgBufferDistribute
from model.common.modules import RandomShiftsAug


class TrainPPOImgFlowAgentQRDQN(TrainPPOImgFlowAgent):
    """
    图像输入 + QR-DQN Critic 的 PPO Flow Agent
    
    适用于:
        - robomimic square/can/lift 等任务
        - 图像观测 + 低维状态
        - 稀疏奖励任务
    
    核心修改 (相比 C51):
        1. 使用 PPOFlowQRDQN 模型
        2. 不需要 reward_scale 归一化
        3. 分位数回归 Loss
    """
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # 类型提示
        self.model: PPOFlowQRDQN
        
        # 检查 QR-DQN 配置
        if hasattr(self.model, 'use_distributional_critic'):
            log.info(f"QR-DQN Critic: enabled={self.model.use_distributional_critic}")
            if self.model.use_distributional_critic:
                log.info(f"  n_quantiles={self.model.n_quantiles}, "
                        f"huber_kappa={self.model.huber_kappa}")
        
        # 创建分布图保存目录
        self.dist_plot_dir = os.path.join(self.logdir, "qrdqn_distributions")
        os.makedirs(self.dist_plot_dir, exist_ok=True)
        log.info(f"QR-DQN distribution plots will be saved to: {self.dist_plot_dir}")

    def init_buffer(self):
        """使用支持分布式 Critic 的图像 Buffer"""
        log.info(f"Initializing PPOFlowImgBufferDistribute for QR-DQN on {self.device}")

        log_prob_cfg_dict = {
            'normalize_denoising_horizon': self.normalize_denoising_horizon,
            'normalize_act_space_dimension': self.normalize_act_space_dim,
            'clip_intermediate_actions': self.clip_intermediate_actions,
            'account_for_initial_stochasticity': self.account_for_initial_stochasticity
        }

        self.buffer = PPOFlowImgBufferDistribute(
            n_steps=self.n_steps,
            n_envs=self.n_envs,
            n_ft_denoising_steps=self.inference_steps,
            horizon_steps=self.horizon_steps,
            act_steps=self.act_steps,
            action_dim=self.action_dim,
            n_cond_step=self.n_cond_step,
            obs_dim=self.obs_dims,
            save_full_observation=self.save_full_observations,
            furniture_sparse_reward=self.furniture_sparse_reward,
            best_reward_threshold_for_success=self.best_reward_threshold_for_success,
            reward_scale_running=self.reward_scale_running,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            reward_scale_const=self.reward_scale_const,
            aug=self.aug if self.augment else None,
            fix_nextvalue_augment_bug=self.fix_nextvalue_augment_bug,
            device=self.device,
            log_prob_cfg_dict=log_prob_cfg_dict
        )
        log.info("Using PPOFlowImgBufferDistribute for QR-DQN support")

    def get_value(self, cond: dict, device='cpu'):
        """
        获取价值估计，支持 QR-DQN Critic

        对于 QR-DQN，返回所有分位数的均值
        """
        if hasattr(self.model, 'get_value'):
            value = self.model.get_value(cond)
        else:
            value = self.model.critic.forward(cond).view(-1)

        if device == 'cpu':
            return value.cpu().numpy().flatten()
        else:
            return value.float().to(self.device)

    def run(self):
        """主训练循环"""
        self.init_buffer()
        self.prepare_run()
        self.buffer.reset()

        if self.resume:
            self.resume_training()

        while self.itr < self.n_train_itr:
            self.prepare_video_path()
            self.set_model_mode()
            self.reset_env(buffer_device=self.buffer_device)
            self.buffer.update_full_obs()

            for step in tqdm(range(self.n_steps)) if self.verbose else range(self.n_steps):
                if not self.verbose and step % 100 == 0:
                    print(f"Processed {step} of {self.n_steps}")

                with torch.no_grad():
                    cond = {
                        key: torch.from_numpy(self.prev_obs_venv[key])
                        .float()
                        .to(self.device)
                        for key in self.obs_dims
                    }

                    action_samples, chains_venv = self.get_samples(
                        cond=cond,
                        ret_device=self.buffer_device,
                        normalize_denoising_horizon=self.normalize_denoising_horizon,
                        normalize_act_space_dimension=self.normalize_act_space_dim,
                        clip_intermediate_actions=self.clip_intermediate_actions,
                        account_for_initial_stochasticity=self.account_for_initial_stochasticity
                    )

                action_venv = action_samples[:, :self.act_steps]
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)

                self.buffer.add(step, self.prev_obs_venv, chains_venv, reward_venv, terminated_venv, truncated_venv)

                self.prev_obs_venv = obs_venv
                self.cnt_train_step += self.n_envs * self.act_steps if not self.eval_mode else 0

            self.buffer.summarize_episode_reward()

            if not self.eval_mode:
                self.buffer.update_img(obs_venv, self.model)
                self.agent_update(verbose=self.verbose)

            self.log()
            self.update_lr()
            self.adjust_finetune_schedule()
            self.save_model()

            # 每隔一个 iteration 保存分布图
            if self.itr % 2 == 0:
                self.save_distribution_plot()

            self.itr += 1

            # Early stopping
            if self.use_early_stop and (self.buffer.success_rate < 0.05 or self.buffer.avg_episode_reward < 2.0):
                log.info(f"Finetuning failed. success_rate={self.buffer.success_rate*100:.2f}%")
                exit()

            self.clear_cache()
            self.inspect_memory()

    def save_distribution_plot(self):
        """保存 QR-DQN 分位数分布图"""
        if not hasattr(self.model, 'use_distributional_critic') or not self.model.use_distributional_critic:
            return

        try:
            if hasattr(self.buffer, 'obs_trajs') and self.buffer.obs_trajs is not None:
                obs = {
                    "state": self.buffer.obs_trajs["state"][:4, 0].to(self.device),
                }
                if "rgb" in self.buffer.obs_trajs:
                    obs["rgb"] = self.buffer.obs_trajs["rgb"][:4, 0].to(self.device)

                target_returns = None
                if hasattr(self.buffer, 'returns_trajs') and self.buffer.returns_trajs is not None:
                    target_returns = self.buffer.returns_trajs[0, :4].to(self.device)

                image = self.model.plot_value_distribution(
                    obs=obs,
                    target_returns=target_returns,
                    num_samples=4,
                    step=self.itr
                )

                save_path = os.path.join(self.dist_plot_dir, f"qr_iter_{self.itr:05d}.png")
                image.save(save_path)
                log.info(f"Saved QR-DQN distribution plot to: {save_path}")

        except Exception as e:
            log.warning(f"Failed to save distribution plot: {e}")

