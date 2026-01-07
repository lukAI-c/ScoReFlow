# MIT License
# Copyright (c) 2025 ReinFlow Authors - Dual-Stream Evaluation

"""
Evaluate Dual-Stream Score Editing PPO Flow policy.

专门用于评估使用 CFG 引导的 Dual-Stream 模型：
    - actor_prior: 保守流 (锚定数据流形)
    - actor_reward: 激进流 (探索高奖励)
    - CFG 引导: v_guided = v_prior + w * (v_reward - v_prior)
"""
import logging
import torch
import torch.nn as nn

log = logging.getLogger(__name__)
from agent.eval.eval_agent_base import EvalAgent
from model.flow.ft_ppo.ppoflow_dual_stream import DualStreamPPOFlow
from util.timer import Timer


class EvalDualStreamAgent(EvalAgent):
    """Dual-Stream Score Editing PPO Flow 评估 Agent"""
    
    def __init__(self, cfg):
        super().__init__(cfg)
        # Overload settings
        self.load_ema = cfg.get('load_ema', False)
        self.clip_intermediate_actions = True
        self.record_video = cfg.get('record_video', False)  # 从配置文件读取，默认不录制视频
        self.record_env_index = 0
        self.frame_width = 640
        self.frame_height = 480
        self.render_onscreen = cfg.get('render_onscreen', False)
        self.denoising_steps = cfg.get("denoising_step_list", [4])
        self.denoising_steps_trained = cfg.get("denoising_steps", 4)
        self.plot_scale = 'standard'
        
        # Dual-Stream specific
        self.cfg_weight = cfg.get("cfg_weight", 1.5)
        self.eval_mode = cfg.get("eval_mode", True)  # 评估时使用确定性采样
        
        
        log.info(f"DualStream Eval: load_ema={self.load_ema}, cfg_weight={self.cfg_weight}, eval_mode={self.eval_mode}")
    
    def load_model_for_eval(self):
        """
        加载 Dual-Stream 模型权重

        支持三种加载方式：
        1. 完整的 DualStreamPPOFlow checkpoint (包含 actor_prior, actor_reward, critic)
        2. EMA checkpoint (从训练中保存的 EMA 状态)
        3. 单独的 policy checkpoint (只有 actor_reward)
        """
        data = torch.load(self.base_policy_path, weights_only=True, map_location=self.device)
        self.model: DualStreamPPOFlow

        log.info(f"Loading Dual-Stream model from {self.base_policy_path}")
        log.info(f"Checkpoint keys: {list(data.keys())}")

        # 打印一些 sample keys 用于调试
        if 'model' in data:
            sample_keys = list(data['model'].keys())[:10]
            log.info(f"Sample model state keys: {sample_keys}")

        # 优先加载 EMA 权重
        if self.load_ema and 'ema' in data:
            ema_state = data["ema"]
            log.info(f"Found EMA state with {len(ema_state)} keys")

            # 检查是否是完整的 DualStream checkpoint (检查更多可能的前缀)
            has_dual_stream = any(k.startswith(('actor_prior.', 'actor_reward.', 'actor_ft.', 'actor_old.'))
                                 for k in ema_state.keys())

            if has_dual_stream:
                # 完整的 DualStream checkpoint - 直接加载
                missing, unexpected = self.model.load_state_dict(ema_state, strict=False)
                log.info(f"Loaded complete DualStream EMA checkpoint")
                if missing:
                    log.warning(f"Missing keys: {missing[:5]}... (total {len(missing)})")
                if unexpected:
                    log.warning(f"Unexpected keys: {unexpected[:5]}... (total {len(unexpected)})")
            else:
                # 单个 FlowMLP checkpoint - 加载到 actor_reward
                weights = {k.replace("network.", ""): v for k, v in ema_state.items()}
                self.model.actor_reward.load_state_dict(weights, strict=False)
                log.info(f"Loaded EMA weights to actor_reward ({len(weights)} params)")

        # 否则加载 model 权重
        elif 'model' in data:
            model_state = data["model"]
            log.info(f"Found model state with {len(model_state)} keys")

            # 检查是否是完整的 DualStream checkpoint (检查更多可能的前缀)
            has_dual_stream = any(k.startswith(('actor_prior.', 'actor_reward.', 'actor_ft.', 'actor_old.'))
                                 for k in model_state.keys())

            if has_dual_stream:
                # 完整的 DualStream checkpoint
                missing, unexpected = self.model.load_state_dict(model_state, strict=False)
                log.info(f"Loaded complete DualStream model checkpoint")
                if missing:
                    log.warning(f"Missing keys: {missing[:5]}... (total {len(missing)})")
                if unexpected:
                    log.warning(f"Unexpected keys: {unexpected[:5]}... (total {len(unexpected)})")
            else:
                # 单个 FlowMLP checkpoint - 加载到 actor_reward
                weights = {k.replace("network.", ""): v for k, v in model_state.items()}
                self.model.actor_reward.load_state_dict(weights, strict=False)
                log.info(f"Loaded model weights to actor_reward ({len(weights)} params)")

        # 直接的 state_dict
        else:
            log.warning(f"Unknown checkpoint format. Available keys: {list(data.keys())}")
            # 尝试直接加载
            missing, unexpected = self.model.load_state_dict(data, strict=False)
            log.info(f"Loaded state_dict directly")

        # 更新模型的 cfg_weight (也从 checkpoint 中恢复如果有的话)
        if 'cfg_weight' in data:
            saved_cfg_weight = data['cfg_weight']
            log.info(f"Checkpoint cfg_weight: {saved_cfg_weight}, config cfg_weight: {self.cfg_weight}")

        if hasattr(self.model, 'cfg_weight'):
            self.model.cfg_weight = self.cfg_weight
            log.info(f"Set model cfg_weight to {self.cfg_weight}")

        # 验证加载是否成功 - 检查 actor_prior 和 actor_reward 权重是否不同
        with torch.no_grad():
            prior_sample = list(self.model.actor_prior.parameters())[0].flatten()[:5]
            reward_sample = list(self.model.actor_reward.parameters())[0].flatten()[:5]
            weights_diff = (prior_sample - reward_sample).abs().mean().item()
            log.info(f"Weight diff between actor_prior and actor_reward: {weights_diff:.6f}")
            if weights_diff < 1e-6:
                log.warning("actor_prior and actor_reward have identical weights! CFG guidance may not work properly.")
    
    def infer(self, cond: dict, num_denoising_steps: int):
        """使用 Dual-Stream CFG 引导采样"""
        self.model: DualStreamPPOFlow
        timer = Timer()
        
        # 临时设置推理步数
        original_steps = self.model.inference_steps
        self.model.inference_steps = num_denoising_steps
        
        # 使用 get_actions 进行采样 (需要保存 chains 来构造 Sample)
        trajectories, chains = self.model.get_actions(
            cond=cond,
            eval_mode=self.eval_mode,
            save_chains=True,  # 必须设为 True 来获取 chains
            ret_logprob=False,
            clip_intermediate_actions=self.clip_intermediate_actions,
            training_progress=1.0  # 评估时设为1.0
        )
        
        # 恢复原始设置
        self.model.inference_steps = original_steps
        
        duration = timer()
        
        # 构造 Sample namedtuple (兼容父类接口)
        from collections import namedtuple
        Sample = namedtuple("Sample", "trajectories chains")
        samples = Sample(trajectories=trajectories, chains=chains)
        
        return samples, duration

