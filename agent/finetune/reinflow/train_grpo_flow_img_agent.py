# Copyright (c) 2026 ScoRe-Flow Authors
"""
GRPO Image-based Flow Agent for robomimic tasks (square, transport).

继承关系:
    TrainGRPOFlowImgAgent
        └── TrainGRPOFlowAgent      (提供 agent_update: critic-free GRPO loss)
        └── TrainPPOImgFlowAgent    (提供 run, init_buffer, get_samples 等图像逻辑)

MRO 保证:
    - run()         来自 TrainPPOImgFlowAgent (处理 rgb + state 的 rollout)
    - agent_update  来自 TrainGRPOFlowAgent (critic-free, KL penalty)
    - init_buffer   在此类中覆盖, 使用 GRPOFlowImgBuffer
    - update_lr     来自 TrainGRPOFlowAgent (只更新 actor)
    - save_model    来自 TrainGRPOFlowAgent (不保存 critic)
"""

import logging

from agent.finetune.reinflow.train_ppo_flow_img_agent import TrainPPOImgFlowAgent
from agent.finetune.reinflow.train_grpo_flow_agent import TrainGRPOFlowAgent
from agent.finetune.reinflow.buffer_grpo import GRPOFlowImgBuffer, GRPOFlowImgBufferGPU

log = logging.getLogger(__name__)


class TrainGRPOFlowImgAgent(TrainGRPOFlowAgent, TrainPPOImgFlowAgent):
    """
    GRPO + Image input agent.
    - run / get_samples / rollout: from TrainPPOImgFlowAgent (handles rgb obs)
    - agent_update / update_lr / save_model: from TrainGRPOFlowAgent (critic-free)
    - init_buffer: overridden here to use GRPO image buffers
    """

    def __init__(self, cfg):
        # TrainPPOImgFlowAgent.__init__ → chains up MRO to initialize everything
        TrainPPOImgFlowAgent.__init__(self, cfg)
        self.kl_coef = cfg.train.get("kl_coef", 0.04)
        log.info(f"[GRPO-Img] Critic-free mode. kl_coef={self.kl_coef}")

    # ── 使用 GRPO Image Buffer ───────────────────────────────────────────
    def init_buffer(self):
        log.info(f"[GRPO-Img] init_buffer, buffer_device={self.buffer_device}")
        log_prob_cfg_dict = {
            'normalize_denoising_horizon': self.normalize_denoising_horizon,
            'normalize_act_space_dimension': self.normalize_act_space_dim,
            'clip_intermediate_actions': self.clip_intermediate_actions,
            'account_for_initial_stochasticity': self.account_for_initial_stochasticity
        }

        if self.buffer_device == 'cpu':
            self.buffer = GRPOFlowImgBuffer(
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
                aug=self.aug,
                fix_nextvalue_augment_bug=self.fix_nextvalue_augment_bug,
                device=self.device,
                log_prob_cfg_dict=log_prob_cfg_dict
            )
        else:
            self.buffer = GRPOFlowImgBufferGPU(
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
                aug=self.aug,
                fix_nextvalue_augment_bug=self.fix_nextvalue_augment_bug,
                device=self.device,
                log_prob_cfg_dict=log_prob_cfg_dict
            )
        log.info(f"[GRPO-Img] Created buffer: {self.buffer.__class__.__name__} on {self.buffer_device}")

    # ── run() 不覆盖, 使用 TrainPPOImgFlowAgent.run() ───────────────────
    # MRO: TrainGRPOFlowAgent.run() 会被解析到, 但它是 state-only 的.
    # 我们需要显式委托给 TrainPPOImgFlowAgent.run().
    def run(self):
        return TrainPPOImgFlowAgent.run(self)
