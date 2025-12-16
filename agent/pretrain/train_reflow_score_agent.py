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
Pre-training ReFlow policy with Score-based stochastic sampling.

This extends the basic ReFlow pretraining to support score-based exploration
during sampling, following the SDE dynamics:

    dxt = [bt(xt) + εt·st(xt)] dt + √(2εt) dWt

where:
    - bt(x) is the velocity field (flow matching)
    - st(x) = (t * bt(x) - x) / (1 - t) is the score function
    - εt is the exploration noise coefficient
"""
import logging
import torch
import numpy as np

log = logging.getLogger(__name__)
from agent.pretrain.train_agent import PreTrainAgent
from model.flow.reflow_score import ReFlowScore


class TrainReFlowScoreAgent(PreTrainAgent):
    """
    Pre-training agent for ReFlow with score-based stochastic sampling.
    
    Key differences from TrainReFlowAgent:
    1. Uses ReFlowScore model instead of ReFlow
    2. Supports stochastic sampling with score function during inference
    3. Can train with score-matching loss in addition to velocity matching
    """
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model: ReFlowScore
        self.ema_model: ReFlowScore
        
        # Verbose settings
        self.verbose_train = False
        self.verbose_loss = True
        self.verbose_test = False
        
        # Score-based parameters from config
        self.epsilon_t = getattr(cfg, 'epsilon_t', 0.1)
        self.epsilon_schedule = getattr(cfg, 'epsilon_schedule', 'cosine')
        
        if self.test_in_mujoco:
            self.test_log_all = True
            self.only_test = False
            
            # Denoising steps for testing
            self.test_denoising_steps = getattr(cfg, 'test_denoising_steps', 4)
            
            self.test_clip_intermediate_actions = True
            self.test_model_type = 'ema'
            
            # Stochastic sampling settings for testing
            self.test_use_stochastic = getattr(cfg, 'test_use_stochastic', True)
            self.test_epsilon_t = getattr(cfg, 'test_epsilon_t', self.epsilon_t)
            self.test_epsilon_schedule = getattr(cfg, 'test_epsilon_schedule', self.epsilon_schedule)
        
        log.info(f"TrainReFlowScoreAgent initialized with epsilon_t={self.epsilon_t}, "
                 f"epsilon_schedule={self.epsilon_schedule}")
    
    def get_loss(self, batch_data):
        """
        Compute training loss for velocity field.
        
        Args:
            batch_data: tuple of (actions, observations)
        
        Returns:
            loss: scalar tensor
        """
        act, cond = batch_data
        (xt, t), v = self.model.generate_target(act)
        
        # Velocity matching loss
        velocity_loss = self.model.loss(xt, t, cond, v)
        
        return velocity_loss
    
    def inference(self, cond: dict):
        """
        Sample actions using stochastic score-based dynamics.
        
        Args:
            cond: dict with 'state' key containing observations
        
        Returns:
            samples: sampled action trajectories
        """
        model = self.ema_model if self.test_model_type == 'ema' else self.model
        
        if self.test_use_stochastic:
            # Use stochastic sampling with score function
            samples = model.sample_stochastic(
                cond,
                inference_steps=self.test_denoising_steps,
                epsilon_t=self.test_epsilon_t,
                epsilon_schedule=self.test_epsilon_schedule,
                record_intermediate=False,
                clip_intermediate_actions=self.test_clip_intermediate_actions
            )
        else:
            # Use deterministic ODE sampling
            samples = model.sample(
                cond,
                inference_steps=self.test_denoising_steps,
                record_intermediate=False,
                clip_intermediate_actions=self.test_clip_intermediate_actions
            )
        
        return samples

