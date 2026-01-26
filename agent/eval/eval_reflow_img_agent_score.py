# MIT License
# Copyright (c) 2025 ReinFlow Authors - Score-based Image Evaluation

"""
Evaluate Score-based PPO Flow policy with image inputs.

基于 Score 函数的 SDE 采样评估 (图像输入版本):
    SDE: dxt = [bt(xt) + εt·st(xt)] dt + √(2εt) dWt
    其中 st(x) = (t * bt(x) - x) / (1 - t) 为分数函数

用于 robomimic 等图像输入任务的评估
"""
import logging
import torch

log = logging.getLogger(__name__)
from agent.eval.eval_agent_img_base import EvalImgAgent
from model.flow.ft_ppo.ppoflow_score import PPOFlow
from util.timer import Timer
from collections import namedtuple

Sample = namedtuple("Sample", "trajectories chains")


class EvalImgScoreAgent(EvalImgAgent):
    """Score-based PPO Flow 图像输入评估 Agent"""
    
    def __init__(self, cfg):
        super().__init__(cfg)
        # Overload settings for image-based evaluation
        self.load_ema = cfg.get('load_ema', False)
        self.clip_intermediate_actions = True
        self.record_video = cfg.get('record_video', False)
        self.record_env_index = 0
        self.frame_width = 640
        self.frame_height = 480
        self.render_onscreen = cfg.get('render_onscreen', False)
        self.denoising_steps = cfg.get("denoising_step_list", [4])
        self.denoising_steps_trained = cfg.get("denoising_steps", 4)
        self.plot_scale = 'standard'
        
        # Score-based specific parameters
        self.eval_mode = cfg.get("eval_mode", True)  # 评估时默认使用确定性采样
        self.epsilon_t = cfg.get("epsilon_t", 0.0)  # 评估时默认无噪声
        self.epsilon_schedule = cfg.get("epsilon_schedule", "constant")
        # 支持 gamma_score (新配置) 和 gamma (旧配置) 两种命名
        self.gamma = cfg.get("gamma_score", cfg.get("gamma", 1.0))

        log.info(f"Score Img Eval: load_ema={self.load_ema}, eval_mode={self.eval_mode}, "
                 f"epsilon_t={self.epsilon_t}, epsilon_schedule={self.epsilon_schedule}, gamma={self.gamma}")
    
    def load_model_for_eval(self):
        """
        加载 Score-based PPO Flow 模型权重

        支持的 checkpoint 格式：
        1. 完整的 PPOFlow checkpoint (包含 actor_old, actor_ft, critic)
        2. EMA checkpoint
        3. 单独的 policy checkpoint (只有 network)
        """
        data = torch.load(self.base_policy_path, weights_only=False, map_location=self.device)
        self.model: PPOFlow

        log.info(f"Loading Score-based model from {self.base_policy_path}")
        log.info(f"Checkpoint keys: {list(data.keys())}")

        # 确定使用哪个 state_dict
        if self.load_ema and 'ema' in data:
            state_dict = data["ema"]
            log.info(f"Using EMA state with {len(state_dict)} keys")
        elif 'model' in data:
            state_dict = data["model"]
            log.info(f"Using model state with {len(state_dict)} keys")
        else:
            state_dict = data
            log.warning(f"Unknown checkpoint format, using data directly")

        # 打印一些 sample keys 用于调试
        sample_keys = list(state_dict.keys())[:10]
        log.info(f"Sample state keys: {sample_keys}")

        # 检查 checkpoint 类型
        has_actor_ft = any(k.startswith('actor_ft.') for k in state_dict.keys())
        has_actor_old = any(k.startswith('actor_old.') for k in state_dict.keys())

        if has_actor_ft or has_actor_old:
            # 完整的 PPOFlow checkpoint - 直接加载
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            log.info(f"Loaded complete PPOFlow checkpoint")
            if missing:
                log.warning(f"Missing keys: {missing[:5]}... (total {len(missing)})")
            if unexpected:
                log.warning(f"Unexpected keys: {unexpected[:5]}... (total {len(unexpected)})")
        else:
            # 单个 FlowMLP checkpoint (预训练模型) - 加载到 actor_old 和 actor_ft
            weights = {k.replace("network.", ""): v for k, v in state_dict.items()}
            self.model.actor_old.load_state_dict(weights, strict=False)
            self.model.actor_ft.load_state_dict(weights, strict=False)
            log.info(f"Loaded pretrained weights to actor_old and actor_ft ({len(weights)} params)")

        # 更新模型的 Score 参数
        if hasattr(self.model, 'epsilon_t'):
            self.model.epsilon_t = self.epsilon_t
        if hasattr(self.model, 'epsilon_schedule'):
            self.model.epsilon_schedule = self.epsilon_schedule
        if hasattr(self.model, 'gamma'):
            self.model.gamma = self.gamma

        log.info(f"Model Score params: epsilon_t={self.model.epsilon_t}, "
                 f"epsilon_schedule={self.model.epsilon_schedule}, gamma={getattr(self.model, 'gamma', 'N/A')}")
    
    def infer(self, cond: dict, num_denoising_steps: int):
        """使用 Score-based SDE 采样"""
        self.model: PPOFlow
        timer = Timer()
        
        # 临时设置推理步数
        original_steps = self.model.inference_steps
        self.model.inference_steps = num_denoising_steps
        
        # 使用 get_actions 进行采样
        result = self.model.get_actions(
            cond=cond,
            eval_mode=self.eval_mode,
            save_chains=True,
            ret_logprob=False,
            clip_intermediate_actions=self.clip_intermediate_actions
        )
        
        # 恢复原始设置
        self.model.inference_steps = original_steps
        
        duration = timer()
        
        # 解析返回值并构造 Sample
        trajectories, chains = result
        samples = Sample(trajectories=trajectories, chains=chains)
        
        return samples, duration

