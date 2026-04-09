# MIT License
# Copyright (c) 2026 ScoRe-Flow Authors

"""
Image-based training agent for Frozen-V variants (robomimic square/transport).

继承关系:
    TrainPPOFlowImgAgentFrozenV
        └── TrainPPOFlowAgentFrozenV   (提供 agent_update, 处理 16 个返回值)
            └── TrainPPOFlowAgent      (提供 run, save_model, minibatch_generator 等)
        └── TrainPPOImgFlowAgent       (提供 init_buffer, get_samples_logprobs 等图像相关逻辑)
"""

import logging
import torch
import numpy as np

from agent.finetune.reinflow.train_ppo_flow_img_agent import TrainPPOImgFlowAgent
from agent.finetune.reinflow.train_ppo_flow_agent_frozen_v import TrainPPOFlowAgentFrozenV

log = logging.getLogger(__name__)


class TrainPPOFlowImgAgentFrozenV(TrainPPOFlowAgentFrozenV, TrainPPOImgFlowAgent):
    """
    图像输入版本的 Frozen-V 训练 Agent (robomimic square / transport).

    MRO: TrainPPOFlowImgAgentFrozenV → TrainPPOFlowAgentFrozenV → TrainPPOImgFlowAgent
         → TrainPPOFlowAgent → TrainPPOAgent → TrainAgent

    - init_buffer / get_samples_logprobs: 来自 TrainPPOImgFlowAgent (图像 buffer)
    - agent_update:                       来自 TrainPPOFlowAgentFrozenV (16 返回值解包)
    """

    def __init__(self, cfg):
        # TrainPPOImgFlowAgent.__init__ 会调用 super().__init__(cfg) 链式向上，
        # 最终由 TrainPPOFlowAgent / TrainPPOAgent 完成基础初始化。
        # MRO 保证 TrainPPOFlowAgentFrozenV 的 agent_update 覆盖父类方法。
        TrainPPOImgFlowAgent.__init__(self, cfg)
        log.info("TrainPPOFlowImgAgentFrozenV initialized (frozen velocity field, image input)")
