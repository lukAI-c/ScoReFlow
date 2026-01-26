# MIT License

# Copyright (c) 2026 ScoRe-Flow Authors

"""
PPO ShortCut Agent with Score GammaNet and Stein Diversity Loss

专门用于 PPOShortCutWithScoreGammaNet 模型的 Agent，支持:
1. Score-enhanced exploration with learnable gamma network
2. Stein Diversity Loss (斯坦因斥力项) 用于跳出局部最优

运行示例:
    python script/run.py --config-dir=cfg/gym/finetune/kitchen-complete-v0 \
        --config-name=ft_ppo_shortcut_mlp_with_score_gammanet

核心特性:
    - diversity_coef: Stein 多样性损失系数 (推荐 0.1 ~ 1.0)
    - diversity_kernel: 核函数类型 ('rbf', 'exp', 'linear')
    - diversity_bandwidth: RBF 核带宽 (-1 表示自适应 median heuristic)
"""

import logging
import numpy as np
import torch

from agent.finetune.reinflow.train_ppo_flow_agent import TrainPPOFlowAgent
from model.flow.ft_ppo.pposhortcut_with_score_gammanet_dloss import PPOShortCutWithScoreGammaNet

log = logging.getLogger(__name__)


class TrainPPOShortCutGammaNetAgent(TrainPPOFlowAgent):
    """
    PPO ShortCut Agent with Score GammaNet and Stein Diversity Loss
    
    专门处理 PPOShortCutWithScoreGammaNet 模型的 16 元素返回值:
    (pg_loss, entropy_loss, v_loss, bc_loss, diversity_loss, 
     clipfrac, approx_kl, ratio, ...)
    
    新增功能:
    - Stein Diversity Loss: 在 batch 内施加"静电斥力"，防止动作聚集
    - 可学习的 Score Scheduler (GammaNet)
    """
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # 验证模型类型
        if isinstance(self.model, PPOShortCutWithScoreGammaNet):
            log.info("Using PPOShortCutWithScoreGammaNet (score-enhanced with learnable gamma)")
            log.info(f"  - diversity_coef: {self.model.diversity_coef}")
            log.info(f"  - diversity_kernel: {self.model.diversity_kernel}")
            log.info(f"  - diversity_bandwidth: {self.model.diversity_bandwidth}")
            log.info(f"  - gamma_score: {self.model.gamma_score}")
            log.info(f"  - score_scheduler_type: {self.model.score_scheduler_type}")
        else:
            log.warning(f"Unexpected model type: {type(self.model)}. "
                       f"Expected PPOShortCutWithScoreGammaNet.")
    
    def agent_update(self, verbose=True):
        """
        覆盖父类的 agent_update 方法，处理包含 diversity_loss 的 16 元素返回值
        """
        clipfracs_list = []
        noise_std_list = []
        diversity_loss_list = []
        
        generator = (self.minibatch_generator() if not self.repeat_samples 
                     else self.minibatch_generator_repeat())
        
        for update_epoch, batch_id, minibatch in generator:
            self.model: PPOShortCutWithScoreGammaNet
            
            # 调用 model.loss，返回 16 个值（包含 diversity_loss）
            (pg_loss, entropy_loss, v_loss, bc_loss, diversity_loss,
             clipfrac, approx_kl, ratio,
             oldlogprob_min, oldlogprob_max, oldlogprob_std,
             newlogprob_min, newlogprob_max, newlogprob_std,
             noise_std, Q_values) = self.model.loss(
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
            
            if update_epoch == 0 and batch_id == 0 and np.abs(ratio - 1.00) > self.initial_ratio_error_threshold:
                raise ValueError(f"ratio={ratio} not 1.00 when update_epoch==0 and batch_id==0, "
                               f"there must be some bugs in your code!")
            
            if self.target_kl and self.lr_schedule == 'adaptive_kl':
                self.update_lr_adaptive_kl(self.approx_kl)
            
            # 计算总损失，包含 Stein Diversity Loss
            loss = (pg_loss 
                    + entropy_loss * self.ent_coef 
                    + v_loss * self.vf_coef 
                    + bc_loss * self.bc_coeff
                    + diversity_loss * self.model.diversity_coef)
            
            clipfracs_list.append(clipfrac)
            noise_std_list.append(noise_std)
            div_loss_val = diversity_loss.item() if hasattr(diversity_loss, 'item') else float(diversity_loss)
            diversity_loss_list.append(div_loss_val)
            
            # 更新 policy 和 critic
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            
            loss.backward()
            
            # 梯度调试
            actor_norm = torch.nn.utils.clip_grad_norm_(
                self.model.actor_ft.parameters(), max_norm=float('inf'))
            actor_old_norm = torch.nn.utils.clip_grad_norm_(
                self.model.actor_old.parameters(), max_norm=float('inf'))
            critic_norm = torch.nn.utils.clip_grad_norm_(
                self.model.critic.parameters(), max_norm=float('inf'))
            
            if verbose:
                log.info(f"before clipping: actor_norm={actor_norm:.2e}, "
                        f"critic_norm={critic_norm:.2e}, actor_old_norm={actor_old_norm:.2e}")

            # 更新 critic (始终更新)
            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            # 更新 actor (在 critic warmup 后)
            if self.itr >= self.n_critic_warmup_itr:
                if (self.itr - self.n_critic_warmup_itr) % self.actor_update_freq == 0:
                    for _ in range(self.actor_update_epoch):
                        if self.max_grad_norm:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.actor_ft.parameters(), self.max_grad_norm)
                        self.actor_optimizer.step()

        # 汇总统计
        clip_fracs = np.mean(clipfracs_list)
        noise_stds = np.mean(noise_std_list)
        diversity_losses = np.mean(diversity_loss_list) if diversity_loss_list else 0.0

        self.train_ret_dict = {
            "loss": loss,
            "pg loss": pg_loss,
            "value loss": v_loss,
            "entropy_loss": entropy_loss,
            "bc_loss": bc_loss,
            "diversity_loss": diversity_losses,  # Stein 多样性损失
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
            "min_logprob_noise_std": self.model.min_logprob_denoising_std,
            "min_sampling_noise_std": self.model.min_sampling_denoising_std,
            "noise_std": noise_stds,
            "Q_values": Q_values,
            # GammaNet 相关指标
            "diversity_coef": self.model.diversity_coef,
            "gamma_score": self.model.gamma_score,
        }


# ==================== 图像输入版本 ====================

class TrainPPOShortCutGammaNetImgAgent(TrainPPOShortCutGammaNetAgent):
    """
    图像输入的 PPO ShortCut GammaNet Agent

    用于 robomimic 等需要视觉输入的任务
    继承自 TrainPPOShortCutGammaNetAgent，复用其 agent_update 逻辑
    """

    def __init__(self, cfg):
        # 导入图像相关的 buffer 和增强
        from agent.finetune.reinflow.buffer import PPOFlowImgBuffer, PPOFlowImgBufferGPU
        from model.common.modules import RandomShiftsAug

        super().__init__(cfg)

        # 图像增强 (如果需要)
        self.use_augmentation = cfg.train.get("use_augmentation", False)
        if self.use_augmentation:
            self.aug = RandomShiftsAug(pad=4)
            log.info("Using RandomShiftsAug for image augmentation")

        log.info("Initialized TrainPPOShortCutGammaNetImgAgent for image input tasks")

