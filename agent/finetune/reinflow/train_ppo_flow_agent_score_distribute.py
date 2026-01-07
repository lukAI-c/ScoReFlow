# MIT License
# Copyright (c) 2025 ReinFlow Authors - Distributional RL Extension

"""
PPO Flow Agent with Distributional Critic (C51) Support

继承自 TrainPPOFlowAgent (score-based)，主要修改：
    1. 使用 PPOFlowBufferDistribute 替代 PPOFlowBuffer
    2. get_value() 方法使用 model.get_value() 支持分布式 Critic
    3. buffer.update() 传入整个 model 而非 critic
"""

import os
import logging
log = logging.getLogger(__name__)
from tqdm import tqdm as tqdm
import numpy as np
import torch

from agent.finetune.reinflow.train_ppo_flow_agent_score import TrainPPOFlowAgent
from model.flow.ft_ppo.ppoflow_score_distribute import PPOFlowDistributional
from agent.finetune.reinflow.buffer_distribute import PPOFlowBufferDistribute


class TrainPPOFlowAgentDistribute(TrainPPOFlowAgent):
    """
    Score-based PPO Flow Agent with Distributional Critic (C51)
    
    核心修改:
        1. Buffer: 使用 PPOFlowBufferDistribute
        2. get_value(): 使用 model.get_value() 支持 C51
        3. buffer.update(): 传入整个 model
    
    分布式 Critic 优势:
        - 输出价值分布 p(z|s) 而非单一标量
        - 更好地处理稀疏奖励和多模态回报
        - 使用交叉熵损失替代 MSE
    """
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # 类型提示
        self.model: PPOFlowDistributional
        
        # 检查是否使用分布式 Critic
        if hasattr(self.model, 'use_distributional_critic'):
            log.info(f"Distributional Critic (C51): enabled={self.model.use_distributional_critic}")
            if self.model.use_distributional_critic:
                log.info(f"  n_atoms={self.model.n_atoms}, "
                        f"v_range=[{self.model.v_min}, {self.model.v_max}]")
    
    def init_buffer(self):
        """使用支持分布式 Critic 的 Buffer"""
        self.buffer = PPOFlowBufferDistribute(
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
        log.info("Using PPOFlowBufferDistribute for distributional critic support")
    
    def get_value(self, cond: dict, device='cpu'):
        """
        获取价值估计，支持分布式 Critic (C51)
        
        对于分布式 Critic，返回分布的期望值 E[Z(s)]
        """
        # 使用 model.get_value() 支持分布式 Critic
        if hasattr(self.model, 'get_value'):
            value = self.model.get_value(cond)
        else:
            # 回退到标量 Critic
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
            self.reset_env()
            self.buffer.update_full_obs()
            
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
                    action_samples, chains_venv, logprob_venv = self.get_samples_logprobs(
                        cond=cond,
                        normalize_denoising_horizon=self.normalize_denoising_horizon,
                        normalize_act_space_dimension=self.normalize_act_space_dim,
                        clip_intermediate_actions=self.clip_intermediate_actions,
                        account_for_initial_stochasticity=self.account_for_initial_stochasticity
                    )
                
                # Apply multi-step action
                action_venv = action_samples[:, :self.act_steps]
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)
                
                self.buffer.save_full_obs(info_venv)
                self.buffer.add(
                    step, 
                    self.prev_obs_venv["state"], 
                    chains_venv, 
                    reward_venv, 
                    terminated_venv, 
                    truncated_venv, 
                    value_venv, 
                    logprob_venv
                )
                
                self.prev_obs_venv = obs_venv
                self.cnt_train_step += self.n_envs * self.act_steps if not self.eval_mode else 0
            
            self.buffer.summarize_episode_reward()
            
            if not self.eval_mode:
                # 核心修改：传入整个 model 而非 critic，支持分布式 Critic
                self.buffer.update(obs_venv, self.model)
                self.agent_update(verbose=self.verbose)

            self.log()
            self.update_lr()
            self.adjust_finetune_schedule()
            self.save_model()
            self.itr += 1

