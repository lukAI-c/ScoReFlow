# MIT License

# Copyright (c) 2025 ReinFlow Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Unified Score Function utilities for Flow Matching models.

This module provides a unified interface for computing the score function
derived from the velocity field in flow matching:

    st(x) = (t * bt(x) - x) / (1 - t)

where:
    - bt(x) is the velocity field
    - st(x) is the score function
    - t is the time step in [0, 1)

The stochastic dynamics follow:
    dxt = [bt(xt) + εt·st(xt)] dt + √(2εt) dWt
"""

import math
import logging
import torch
from torch import Tensor

log = logging.getLogger(__name__)


class ScoreFunction:
    """
    Unified score function computation for flow matching models.
    
    This class provides static methods for:
    1. Computing score from velocity field
    2. Epsilon scheduling for stochastic sampling
    3. Stochastic SDE update step
    
    Usage:
        score = ScoreFunction.compute(action, velocity, time)
        epsilon = ScoreFunction.get_epsilon(t, eps_0, schedule='cosine')
        x_next = ScoreFunction.sde_step(x, velocity, score, epsilon, dt)
    """
    
    # Numerical stability constant
    EPS = 1e-5
    
    @staticmethod
    def compute(
        action: Tensor,
        velocity: Tensor,
        time: Tensor,
        eps: float = 1e-5
    ) -> Tensor:
        """
        Compute score function: st(x) = (t * bt(x) - x) / (1 - t)
        
        Based on equation (6) from Lipman et al., 2022 (Flow Matching).
        
        Args:
            action: (B, Ta, Da) current state xt
            velocity: (B, Ta, Da) velocity field bt(x)
            time: (B,) or (B, 1) or scalar, time in [0, 1)
            eps: small constant for numerical stability
        
        Returns:
            score: (B, Ta, Da) score function st(x)
        """
        B = action.shape[0]
        
        # Handle different time tensor shapes
        if isinstance(time, (int, float)):
            t = torch.tensor(time, device=action.device, dtype=action.dtype)
            t = t.view(1, 1, 1).expand(B, 1, 1)
        elif time.dim() == 0:  # scalar tensor
            t = time.view(1, 1, 1).expand(B, 1, 1)
        elif time.dim() == 1:  # (B,)
            t = time.view(B, 1, 1)
        elif time.dim() == 2:  # (B, 1)
            t = time.view(B, 1, 1)
        else:
            t = time
        
        # st(x) = (t * bt(x) - x) / (1 - t)
        score = (t * velocity - action) / (1 - t + eps)
        
        return score
    
    @staticmethod
    def get_epsilon(
        t: float,
        epsilon_0: float,
        schedule: str = 'constant',
        epsilon_min: float = 0.01,
        training_progress: float = 0.0,
        decay_rate: float = 2.0,
        adaptive_epsilon: float = None
    ) -> float:
        """
        Compute epsilon value at time t based on schedule.
        
        Supported schedules:
            - 'constant': εt = ε₀
            - 'linear_decay': εt = ε₀ × (1 - t)
            - 'cosine': εt = ε₀ × 0.5 × (1 + cos(πt))
            - 'sqrt_decay': εt = ε₀ × √(1 - t)
            - 'exponential_decay': εt = ε₀ × exp(-λt)
            - 'warmup_decay': εt = ε₀ × sin(πt)
            - 'quadratic_decay': εt = ε₀ × (1 - t)²
            - 'inverse_sqrt': εt = ε₀ / √(1 + t)
            - 'adaptive': use provided adaptive_epsilon
            - 'training_decay': εt = ε₀ × (1 - 0.5×progress) × (1 - t)
        
        Args:
            t: current time step (0 to 1)
            epsilon_0: base epsilon value
            schedule: schedule type
            epsilon_min: minimum epsilon value
            training_progress: training progress (0 to 1)
            decay_rate: decay rate for exponential schedule
            adaptive_epsilon: learnable epsilon value
        
        Returns:
            epsilon_t: epsilon at current time step
        """
        # 限幅
        if t > 0.95:
            return 0.0
        
        if schedule == 'constant':
            return epsilon_0
        elif schedule == 'linear_decay':
            return max(epsilon_min, epsilon_0 * (1 - t))
        elif schedule == 'cosine':
            return max(epsilon_min, epsilon_0 * 0.5 * (1 + math.cos(math.pi * t)))
        elif schedule == 'sqrt_decay':
            return max(epsilon_min, epsilon_0 * math.sqrt(max(0, 1 - t)))
        elif schedule == 'exponential_decay':
            return max(epsilon_min, epsilon_0 * math.exp(-decay_rate * t))
        elif schedule == 'warmup_decay':
            return max(epsilon_min, epsilon_0 * math.sin(math.pi * t))
        elif schedule == 'quadratic_decay':
            return max(epsilon_min, epsilon_0 * (1 - t) ** 2)
        elif schedule == 'inverse_sqrt':
            return max(epsilon_min, epsilon_0 / math.sqrt(1 + t))
        elif schedule == 'adaptive':
            if adaptive_epsilon is not None:
                return max(epsilon_min, float(adaptive_epsilon))
            return epsilon_0
        elif schedule == 'training_decay':
            decay_factor = 1 - 0.5 * training_progress
            return max(epsilon_min, epsilon_0 * decay_factor * (1 - t))
        else:
            log.warning(f"Unknown epsilon_schedule: {schedule}, using constant")
            return epsilon_0

    @staticmethod
    def sde_step(
        x: Tensor,
        velocity: Tensor,
        score: Tensor,
        epsilon: float,
        dt: float,
        stochastic: bool = True
    ) -> Tensor:
        """
        Perform one SDE step for stochastic sampling.

        SDE dynamics: dxt = [bt(xt) + εt·st(xt)] dt + √(2εt) dWt

        Discrete update:
            x_{k+1} = x_k + [bt + εt·st]·Δt + √(2εt·Δt)·ε

        Args:
            x: (B, Ta, Da) current state
            velocity: (B, Ta, Da) velocity field bt(x)
            score: (B, Ta, Da) score function st(x)
            epsilon: noise coefficient εt
            dt: time step size
            stochastic: whether to add stochastic noise

        Returns:
            x_next: (B, Ta, Da) next state
        """
        # Drift term: [bt + εt·st]·Δt
        drift = (velocity + epsilon * score) * dt

        # Diffusion term: √(2εt·Δt)·ε
        if stochastic and epsilon > 0:
            diffusion = math.sqrt(2 * epsilon * dt) * torch.randn_like(x)
        else:
            diffusion = 0

        return x + drift + diffusion

    @staticmethod
    def ode_step(
        x: Tensor,
        velocity: Tensor,
        dt: float
    ) -> Tensor:
        """
        Perform one ODE step for deterministic sampling.

        ODE dynamics: dxt = bt(xt) dt

        Discrete update:
            x_{k+1} = x_k + bt·Δt

        Args:
            x: (B, Ta, Da) current state
            velocity: (B, Ta, Da) velocity field bt(x)
            dt: time step size

        Returns:
            x_next: (B, Ta, Da) next state
        """
        return x + velocity * dt

    @classmethod
    def get_supported_schedules(cls) -> list:
        """Return list of supported epsilon schedules."""
        return [
            'constant',
            'linear_decay',
            'cosine',
            'sqrt_decay',
            'exponential_decay',
            'warmup_decay',
            'quadratic_decay',
            'inverse_sqrt',
            'adaptive',
            'training_decay'
        ]


class ScoreFunctionMixin:
    """
    Mixin class to add score function capabilities to Flow models.

    Use this mixin by inheriting from it along with your model class:

        class MyFlowModel(nn.Module, ScoreFunctionMixin):
            def __init__(self, ...):
                super().__init__()
                self.epsilon_t = 0.1
                self.epsilon_schedule = 'cosine'
                ...

    The mixin provides:
        - compute_score(action, velocity, time): compute score function
        - get_epsilon_at_time(t, training_progress): get epsilon at time t
        - sde_update(x, velocity, score, epsilon, dt): perform SDE step
    """

    # Default values (can be overridden in child class)
    epsilon_t: float = 0.1
    epsilon_schedule: str = 'cosine'
    epsilon_min: float = 0.01
    epsilon_decay_rate: float = 2.0

    def compute_score(
        self,
        action: Tensor,
        velocity: Tensor,
        time: Tensor
    ) -> Tensor:
        """Compute score function using unified ScoreFunction class."""
        return ScoreFunction.compute(action, velocity, time)

    def get_epsilon_at_time(
        self,
        t: float,
        training_progress: float = 0.0
    ):
        """
        Get epsilon at time t using unified ScoreFunction class.

        Returns:
            float or Tensor: epsilon value at time t

        Note:
            子类可以覆盖此方法以返回 Tensor (用于可学习 epsilon)
            默认实现返回 float (用于固定 epsilon)
        """
        # 检查是否有可学习的 epsilon_logit 参数
        # if hasattr(self, 'epsilon_logit') and hasattr(self, '_get_epsilon_value'):
        #     # 使用子类的可学习 epsilon 实现
        #     # 这种情况下子类应该覆盖此方法
        #     print("Using adaptive epsilon PASS.............................................")
        #     pass

        adaptive_eps = getattr(self, 'adaptive_epsilon', None)
        if adaptive_eps is not None and hasattr(adaptive_eps, 'item'):
            adaptive_eps = adaptive_eps.item()

        return ScoreFunction.get_epsilon(
            t=t,
            epsilon_0=self.epsilon_t,
            schedule=self.epsilon_schedule,
            epsilon_min=getattr(self, 'epsilon_min', 0.01),
            training_progress=training_progress,
            decay_rate=getattr(self, 'epsilon_decay_rate', 2.0),
            adaptive_epsilon=adaptive_eps
        )

    def sde_update(
        self,
        x: Tensor,
        velocity: Tensor,
        score: Tensor,
        epsilon: float,
        dt: float,
        stochastic: bool = True
    ) -> Tensor:
        """Perform SDE step using unified ScoreFunction class."""
        return ScoreFunction.sde_step(x, velocity, score, epsilon, dt, stochastic)

