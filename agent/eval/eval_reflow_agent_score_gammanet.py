# MIT License
# Copyright (c) 2025 ReinFlow Authors - Score GammaNet Evaluation

"""
Evaluate ScoRe-Flow (with_score_gammanet) policy for gym-state tasks.

ScoRe-Flow 采样评估 (状态输入版本):
    drift = vt + gamma_score * nt * st
    其中:
        vt = 速度场 (velocity field)
        nt = 学习到的噪声网络 (learned noise network)  
        st = (t * vt - xt) / (1 - t) 是 score 函数

用于 gym locomotion 等状态输入任务的评估
"""
import logging
import torch

log = logging.getLogger(__name__)
from agent.eval.eval_agent_base import EvalAgent
from util.timer import Timer
from collections import namedtuple

Sample = namedtuple("Sample", "trajectories chains")


class EvalScoreGammaNetAgent(EvalAgent):
    """ScoRe-Flow (with_score_gammanet) 状态输入评估 Agent"""
    
    def __init__(self, cfg):
        super().__init__(cfg)
        # Overload settings
        self.load_ema = cfg.get('load_ema', False)
        self.clip_intermediate_actions = True
        self.record_video = cfg.get('record_video', False)
        self.record_env_index = 0
        self.frame_width = cfg.get('frame_width', 640)
        self.frame_height = cfg.get('frame_height', 480)
        self.render_onscreen = cfg.get('render_onscreen', False)
        self.denoising_steps = cfg.get("denoising_step_list", [4])
        self.denoising_steps_trained = cfg.get("ft_denoising_steps", 4)
        self.plot_scale = 'standard'
        
        # ScoRe-Flow specific parameters
        self.eval_mode = cfg.get("eval_mode", True)  # 评估时默认使用确定性采样
        self.gamma_score = cfg.get("gamma_score", 1.0)
        self.min_std = cfg.get("min_std", 0.10)
        self.max_std = cfg.get("max_std", 0.24)

        log.info(f"Score GammaNet Eval: load_ema={self.load_ema}, eval_mode={self.eval_mode}, "
                 f"gamma_score={self.gamma_score}, min_std={self.min_std}, max_std={self.max_std}")
    
    def load_model_for_eval(self):
        """
        加载 ScoRe-Flow (with_score_gammanet) 模型权重

        支持的 checkpoint 格式：
        1. 完整的 PPOFlowWithScore checkpoint
        2. EMA checkpoint
        """
        data = torch.load(self.base_policy_path, weights_only=False, map_location=self.device)

        log.info(f"Loading ScoRe-Flow model from {self.base_policy_path}")
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
            # 完整的 PPOFlowWithScore checkpoint - 直接加载
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            log.info(f"Loaded complete PPOFlowWithScore checkpoint")
            if missing:
                log.warning(f"Missing keys: {missing[:5]}... (total {len(missing)})")
            if unexpected:
                log.warning(f"Unexpected keys: {unexpected[:5]}... (total {len(unexpected)})")
        else:
            # 单个 FlowMLP checkpoint (预训练模型) - 加载到 actor_old 和 actor_ft
            weights = {k.replace("network.", ""): v for k, v in state_dict.items()}
            self.model.actor_old.load_state_dict(weights, strict=False)
            self.model.actor_ft.policy.load_state_dict(weights, strict=False)
            log.info(f"Loaded pretrained weights to actor_old and actor_ft ({len(weights)} params)")

        log.info(f"Model loaded successfully for evaluation")
    
    def infer(self, cond: dict, num_denoising_steps: int):
        """使用 ScoRe-Flow 采样"""
        timer = Timer()
        
        # 临时设置推理步数
        if hasattr(self.model, 'inference_steps'):
            original_steps = self.model.inference_steps
            self.model.inference_steps = num_denoising_steps
        if hasattr(self.model, 'ft_denoising_steps'):
            original_ft_steps = self.model.ft_denoising_steps
            self.model.ft_denoising_steps = num_denoising_steps
        
        # 使用 get_actions 进行采样
        result = self.model.get_actions(
            cond=cond,
            eval_mode=self.eval_mode,
            save_chains=True,
            ret_logprob=False,
            clip_intermediate_actions=self.clip_intermediate_actions
        )
        
        # 恢复原始设置
        if hasattr(self.model, 'inference_steps'):
            self.model.inference_steps = original_steps
        if hasattr(self.model, 'ft_denoising_steps'):
            self.model.ft_denoising_steps = original_ft_steps
        
        duration = timer()
        
        # 解析返回值并构造 Sample
        trajectories, chains = result
        samples = Sample(trajectories=trajectories, chains=chains)
        
        return samples, duration

