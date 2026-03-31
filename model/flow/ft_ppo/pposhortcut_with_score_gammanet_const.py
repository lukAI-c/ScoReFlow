# MIT License
# Copyright (c) 2026 ScoRe-Flow Authors
# PPOShortCut with Score-guided drift + Constant Alpha (Ablation)

"""
PPOShortCut + Score 导向 + 固定常数 alpha（消融实验用）

继承自 PPOShortCutWithScore，仅覆盖 get_alpha_t：
    - alpha_constant=0.0  → α(t) ≡ 0，退化为纯 NoisyShortCut（无 score）
    - alpha_constant=1.0  → α(t) ≡ 1，score 全程以常数强度作用（不随 t 衰减）
    - alpha_constant=None → 回退到父类行为（time_mask * gamma_score 或 GammaNet）

与父类的唯一区别：
    get_alpha_t 忽略 time_mask，直接返回 alpha_constant * ones

使用方式（cfg 中）:
    model:
      _target_: model.flow.ft_ppo.pposhortcut_with_score_gammanet_const.PPOShortCutWithScoreConst
      alpha_constant: 0.0   # 或 1.0
      score_scheduler_type: fixed
      ...其余参数与 pposhortcut_with_score_gammanet 完全相同
"""

import torch
from torch import Tensor
import logging

from model.flow.ft_ppo.pposhortcut_with_score_gammanet import PPOShortCutWithScore

log = logging.getLogger(__name__)


class PPOShortCutWithScoreConst(PPOShortCutWithScore):
    """
    消融版本：alpha(t) 为用户指定的定值，不乘 time_mask。

    支持两种典型消融：
        alpha_constant=0.0  → 完全关闭 score 引导，等价于纯 NoisyShortCut
        alpha_constant=1.0  → score 以常数强度作用，验证 time_mask 的必要性

    当 alpha_constant=None 时行为与父类 PPOShortCutWithScore 完全一致。
    """

    def __init__(self,
                 *args,
                 alpha_constant: float = None,   # None=父类行为; 0.0/1.0=常数 alpha
                 **kwargs):
        # alpha_constant 是本类专属参数，不传给父类
        super().__init__(*args, **kwargs)
        self.alpha_constant = alpha_constant

        if alpha_constant is not None:
            logging.info(
                f"PPOShortCutWithScoreConst: alpha_constant={alpha_constant} "
                f"(time_mask DISABLED, score runs at constant α)"
            )
        else:
            logging.info("PPOShortCutWithScoreConst: alpha_constant=None, fallback to parent get_alpha_t")

    # ------------------------------------------------------------------
    # 唯一覆盖：get_alpha_t
    # ------------------------------------------------------------------
    def get_alpha_t(self, t: Tensor) -> Tensor:
        """
        返回常数 alpha(t) = alpha_constant（忽略 time_mask）。

        若 alpha_constant 为 None，退回到父类（带 time_mask）。
        """
        if self.alpha_constant is None:
            return super().get_alpha_t(t)

        B = t.shape[0]
        # 直接返回标量常数，不乘 (1-t)
        alpha_t = torch.full((B, 1, 1), self.alpha_constant, device=self.device)
        return alpha_t  # [B, 1, 1]
