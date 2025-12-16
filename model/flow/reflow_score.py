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
1-Rectified Flow Policy with Score-based Stochastic Sampling.

Extends the basic ReFlow model to support stochastic sampling using the
score function derived from the velocity field:

    st(x) = (t * bt(x) - x) / (1 - t)

The stochastic dynamics follow:
    dxt = [bt(xt) + εt·st(xt)] dt + √(2εt) dWt
"""

import logging
import torch
import math
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from collections import namedtuple
from model.flow.mlp_flow_score import FlowMLP
from model.flow.score_utils import ScoreFunctionMixin

log = logging.getLogger(__name__)
Sample = namedtuple("Sample", "trajectories chains")


class ReFlowScore(nn.Module, ScoreFunctionMixin):
    """
    ReFlow model with score-based stochastic sampling capability.

    This model learns the velocity field bt(x) for flow matching, and derives
    the score function st(x) = (t*bt(x) - x) / (1-t) for stochastic exploration.

    Uses ScoreFunctionMixin for unified score computation.
    """
    
    def __init__(
        self,
        network: FlowMLP,
        device: torch.device,
        horizon_steps: int,
        action_dim: int,
        act_min: float,
        act_max: float,
        obs_dim: int,
        max_denoising_steps: int,
        seed: int,
        sample_t_type: str = 'uniform',
        epsilon_t: float = 0.1,
        epsilon_schedule: str = 'cosine',
        lamda: float = 10.0, # defalut 1.0
        randn_clip_value: float = 3.0, # defalut 3.0
    ):
        """
        Initialize the ReFlowScore model.

        Args:
            network: FlowMLP network for velocity prediction.
            device: Device to run the model on.
            horizon_steps: Number of steps in the trajectory horizon.
            action_dim: Dimension of the action space.
            act_min: Minimum action value for clipping.
            act_max: Maximum action value for clipping.
            obs_dim: Dimension of the observation space.
            max_denoising_steps: Maximum number of denoising steps.
            seed: Random seed for reproducibility.
            sample_t_type: Type of time sampling ('uniform', 'logitnormal', 'beta').
            epsilon_t: Base exploration noise coefficient.
            epsilon_schedule: Schedule for epsilon ('constant', 'linear_decay', 'cosine').
            lamda: Scaling factor for drift term.
            randn_clip_value: Clipping value for random noise.
        """
        super().__init__()
        if int(max_denoising_steps) <= 0:
            raise ValueError('max_denoising_steps must be a positive integer')
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.network = network.to(device)
        self.device = device
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.data_shape = (self.horizon_steps, self.action_dim)
        self.act_range = (act_min, act_max)
        self.obs_dim = obs_dim
        self.max_denoising_steps = int(max_denoising_steps)
        self.sample_t_type = sample_t_type
        
        # Score-based sampling parameters
        self.epsilon_t = epsilon_t
        self.epsilon_schedule = epsilon_schedule
        self.lamda = lamda
        self.randn_clip_value = randn_clip_value
        
        log.info(f"ReFlowScore initialized with epsilon_t={epsilon_t}, "
                 f"epsilon_schedule={epsilon_schedule}, lamda={lamda}, "
                 f"randn_clip_value={randn_clip_value}")

    def generate_trajectory(self, x1: Tensor, x0: Tensor, t: Tensor) -> Tensor:
        """Generate rectified flow trajectory xt = t * x1 + (1 - t) * x0."""
        t_ = (torch.ones_like(x1, device=self.device) * t.view(x1.shape[0], 1, 1)).to(self.device)
        xt = t_ * x1 + (1 - t_) * x0
        return xt

    def sample_time(self, batch_size: int, time_sample_type: str = 'uniform', **kwargs) -> Tensor:
        """Sample time steps from a specified distribution in [0, 1)."""
        if time_sample_type == 'uniform':
            return torch.rand(batch_size, device=self.device)
        elif time_sample_type == 'logitnormal':
            m = kwargs.get("m", 0)
            s = kwargs.get("s", 1)
            normal_samples = torch.normal(mean=m, std=s, size=(batch_size,), device=self.device)
            return (1 / (1 + torch.exp(-normal_samples))).to(self.device)
        elif time_sample_type == 'beta':
            alpha = kwargs.get("alpha", 1.5)
            beta = kwargs.get("beta", 1.0)
            s = kwargs.get("s", 0.999)
            beta_distribution = torch.distributions.Beta(alpha, beta)
            beta_sample = beta_distribution.sample((batch_size,)).to(self.device)
            return s * (1 - beta_sample)
        else:
            raise ValueError(f'Unknown time_sample_type = {time_sample_type}')

    def generate_target(self, x1: Tensor) -> tuple:
        """Generate training targets for the velocity field."""
        t = self.sample_time(batch_size=x1.shape[0], time_sample_type=self.sample_t_type)
        x0 = torch.randn(x1.shape, dtype=torch.float32, device=self.device)
        xt = self.generate_trajectory(x1, x0, t)
        v = x1 - x0
        return (xt, t), v

    def loss(self, xt: Tensor, t: Tensor, obs: dict, v: Tensor) -> Tensor:
        """Compute the MSE loss between predicted and target velocities."""
        v_hat = self.network(xt, t, obs)
        return F.mse_loss(input=v_hat, target=v)

    # Note: compute_score() and get_epsilon_at_time() are inherited from ScoreFunctionMixin

    def score_loss(self, xt: Tensor, t: Tensor, obs: dict, v: Tensor) -> Tensor:
        """
        Compute score matching loss (optional auxiliary loss).

        The true score for flow matching at time t is:
            st(x) = (t * bt(x) - x) / (1 - t)

        We use the velocity to derive predicted score and match against true score.
        """
        v_hat = self.network(xt, t, obs)
        score_pred = self.compute_score(xt, v_hat, t)
        score_true = self.compute_score(xt, v, t)
        return F.mse_loss(input=score_pred, target=score_true)

    @torch.no_grad()
    def sample(
        self,
        cond: dict,
        inference_steps: int,
        record_intermediate: bool = False,
        clip_intermediate_actions: bool = True,
        z: torch.Tensor = None
    ) -> Sample:
        """Sample trajectories using deterministic ODE (velocity field only)."""
        B = cond['state'].shape[0]
        if record_intermediate:
            x_hat_list = torch.zeros((inference_steps,) + self.data_shape, device=self.device)
        x_hat = z if z is not None else torch.randn((B,) + self.data_shape, device=self.device)
        dt = (1 / inference_steps) * torch.ones_like(x_hat, device=self.device)
        steps = torch.linspace(0, 1-1/inference_steps, inference_steps, device=self.device).repeat(B, 1)

        for i in range(inference_steps):
            t = steps[:, i]
            vt = self.network(x_hat, t, cond)
            x_hat += vt * dt
            if clip_intermediate_actions or i == inference_steps-1:
                x_hat = x_hat.clamp(*self.act_range)
            if record_intermediate:
                x_hat_list[i] = x_hat

        return Sample(trajectories=x_hat, chains=x_hat_list if record_intermediate else None)

    @torch.no_grad()
    def sample_stochastic(
        self,
        cond: dict,
        inference_steps: int,
        epsilon_t: float = None,
        epsilon_schedule: str = None,
        record_intermediate: bool = False,
        clip_intermediate_actions: bool = True,
        z: torch.Tensor = None
    ) -> Sample:
        """
        Sample trajectories using stochastic SDE with score function.

        dxt = [bt(xt) + εt·st(xt)] dt + √(2εt) dWt

        Args:
            cond: dict with 'state' key
            inference_steps: number of denoising steps
            epsilon_t: exploration noise coefficient (uses self.epsilon_t if None)
            epsilon_schedule: schedule type (uses self.epsilon_schedule if None)
            record_intermediate: whether to record trajectory
            clip_intermediate_actions: whether to clip during sampling
            z: initial noise (optional)

        Returns:
            Sample namedtuple with trajectories and optional chains
        """
        B = cond['state'].shape[0]
        eps_base = epsilon_t if epsilon_t is not None else self.epsilon_t
        schedule = epsilon_schedule if epsilon_schedule is not None else self.epsilon_schedule

        if record_intermediate:
            x_hat_list = torch.zeros((inference_steps,) + self.data_shape, device=self.device)

        x_hat = z if z is not None else torch.randn((B,) + self.data_shape, device=self.device)
        dt = 1.0 / inference_steps
        steps = torch.linspace(0, 1 - dt, inference_steps, device=self.device)

        # Temporarily override epsilon settings if provided
        orig_eps = self.epsilon_t
        orig_schedule = self.epsilon_schedule
        if eps_base != orig_eps:
            self.epsilon_t = eps_base
        if schedule != orig_schedule:
            self.epsilon_schedule = schedule

        for i in range(inference_steps):
            t = steps[i]
            t_batch = t.expand(B)

            # Get velocity bt(x)
            vt = self.network(x_hat, t_batch, cond)

            # Get score st(x) using unified ScoreFunctionMixin
            st = self.compute_score(x_hat, vt, t_batch)

            # Compute epsilon at this timestep using unified ScoreFunctionMixin
            eps_t = self.get_epsilon_at_time(t.item()) if eps_base > 0 else 0

            # Stochastic SDE update matching PPOFlow logic
            # Drift: [bt + εt·st]·dt
            drift = vt + eps_t * st
            
            # Diffusion std: √(2εt·Δt)
            diffusion_std = math.sqrt(2 * eps_t * dt)
            
            # Update mean: xt + lamda * drift * dt
            x_mean = x_hat + self.lamda * drift * dt
            
            if clip_intermediate_actions:
                x_mean = x_mean.clamp(*self.act_range)

            # Add noise
            if eps_t > 0:
                noise = torch.randn_like(x_hat)
                noise = noise.clamp(-self.randn_clip_value, self.randn_clip_value)
                x_hat = x_mean + diffusion_std * noise
            else:
                x_hat = x_mean

            # Clip final action
            if i == inference_steps - 1:
                x_hat = x_hat.clamp(*self.act_range)

            if record_intermediate:
                x_hat_list[i] = x_hat

        # Restore original epsilon settings
        self.epsilon_t = orig_eps
        self.epsilon_schedule = orig_schedule

        return Sample(trajectories=x_hat, chains=x_hat_list if record_intermediate else None)

