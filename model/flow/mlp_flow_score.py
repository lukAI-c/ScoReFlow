# MIT License

# Copyright (c) 2026 ScoRe-Flow Authors



"""
MLP models for flow matching with learnable stochastic interpolate noise.
"""
import torch
import torch.nn as nn
import logging
import numpy as np
from copy import deepcopy
from typing import Tuple
from torch import Tensor
from model.common.mlp import MLP, ResidualMLP
from model.diffusion.modules import SinusoidalPosEmb
from model.common.modules import SpatialEmb, RandomShiftsAug
from model.common.vit import VitEncoder
from model.flow.score_utils import ScoreFunction, ScoreFunctionMixin
log = logging.getLogger(__name__)
import einops
from typing import List

class FlowMLP(nn.Module, ScoreFunctionMixin):
    def __init__(
        self,
        horizon_steps,
        action_dim,
        cond_dim,
        time_dim=16,
        mlp_dims=[256, 256],
        cond_mlp_dims=None,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.act_dim_total = action_dim * horizon_steps
        self.horizon_steps = horizon_steps
        self.action_dim=action_dim
        self.cond_dim=cond_dim
        self.mlp_dims=mlp_dims
        self.activation_type=activation_type
        self.out_activation_type=out_activation_type
        self.use_layernorm=use_layernorm
        self.residual_style=residual_style

        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )
        
        model = ResidualMLP if residual_style else MLP
        
        # obs encoder
        if cond_mlp_dims:
            self.cond_mlp = MLP(
                [cond_dim] + cond_mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
            )
            self.cond_enc_dim = cond_mlp_dims[-1]
        else:
            self.cond_enc_dim = cond_dim
        input_dim = time_dim + action_dim * horizon_steps + self.cond_enc_dim
        
        # velocity head
        self.mlp_mean = model(
            [input_dim] + mlp_dims + [self.act_dim_total],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )
    
    def forward(
        self,
        action,
        time,
        cond,
        output_embedding=False,
        **kwargs,
    ):
        """
        **Args**:
            action: (B, Ta, Da)
            time: (B,) or int, diffusion step
            cond: dict with key state/rgb; more recent obs at the end
                    state: (B, To, Do)
        **Outpus**:
            velocity. 
            vel: (B, Ta, Da) when output_embedding==False 
            vel,time_emb, cond_emb: when output_embedding==False
        """
        B, Ta, Da = action.shape

        # flatten action chunk
        action = action.view(B, -1)

        # flatten obs history
        state = cond["state"].view(B, -1)

        # obs encoder
        cond_emb = self.cond_mlp(state) if hasattr(self, "cond_mlp") else state
        
        # time encoder
        if isinstance(time, int) or isinstance(time, float):
            time=torch.ones((B,1), device=action.device)* time
        time_emb = self.time_embedding(time.view(B, 1)).view(B, self.time_dim)
        
        # velocity head
        vel_feature = torch.cat([action, time_emb, cond_emb], dim=-1)
        vel = self.mlp_mean(vel_feature)
        
        if output_embedding:
            return vel.view(B, Ta, Da), time_emb, cond_emb
        return vel.view(B, Ta, Da)

    def sample_action(self,cond:dict,inference_steps:int,clip_intermediate_actions:bool,act_range:List[float], z:Tensor=None,save_chains:bool=False):
        """
        simply return action via integration (Euler's method). the initial noise could be specified.
        when `save_chains` is True, also return the denoising trajectory.
        """
        B = cond['state'].shape[0]
        device=cond['state'].device

        x_hat:Tensor=z if z is not None else torch.randn(B, self.horizon_steps, self.action_dim, device=device)
        if save_chains:
            x_chain=torch.zeros((B, inference_steps+1, self.horizon_steps, self.action_dim), device=device)
        dt = (1 / inference_steps) * torch.ones_like(x_hat, device=device)
        steps = torch.linspace(0, 1-1 / inference_steps, inference_steps, device=device).repeat(B, 1)
        for i in range(inference_steps):
            t = steps[:, i]
            vt = self.forward(x_hat, t, cond)
            x_hat += vt * dt
            if clip_intermediate_actions or i == inference_steps-1: # always clip the output action. appended by ReinFlow Authors on 04/25/2025
                x_hat = x_hat.clamp(*act_range)
            if save_chains:
                x_chain[:, i+1] = x_hat
        if save_chains:
            return x_hat, x_chain
        return x_hat

    # def compute_score(self, action: Tensor, vel: Tensor, time: Tensor) -> Tensor:
    #     """
    #     Compute score function based on equation (6) from Lipman et al., 2022:

    #     st(x) = (t * bt(x) - x) / (1 - t)

    #     Args:
    #         action: (B, Ta, Da) current state xt
    #         vel: (B, Ta, Da) velocity field bt(x)
    #         time: (B,) or (B, 1) time in [0, 1)

    #     Returns:
    #         score: (B, Ta, Da) score function st(x)
    #     """
    #     B = action.shape[0]

    #     # Reshape time for broadcasting: (B,) -> (B, 1, 1)
    #     if time.dim() == 1:
    #         time = time.view(B, 1, 1)
    #     elif time.dim() == 2:
    #         time = time.view(B, 1, 1)

    #     # st(x) = (t * bt(x) - x) / (1 - t)
    #     epsilon = 1e-5  # avoid division by zero when t -> 1
    #     score = (time * vel - action) / (1 - time + epsilon)

    #     return score

    def sample_action_stochastic(
        self,
        cond: dict,
        inference_steps: int,
        epsilon_t: float,
        act_range: List[float],
        z: Tensor = None,
        epsilon_schedule: str = 'constant',
        clip_intermediate_actions: bool = True,
        save_chains: bool = False,
    ):
        """
        Stochastic sampling using score-based dynamics:

        dxt = [bt(xt) + εt·st(xt)] dt + √(2εt) dWt

        Discrete update:
        ak+1 = ak + [bt(ak) + εt·st(ak)]·Δt + √(2εt·Δt)·ε

        Args:
            cond: condition dict with 'state' key
            inference_steps: number of denoising steps
            epsilon_t: noise scale coefficient
            act_range: [min, max] for action clipping
            z: initial noise (optional)
            epsilon_schedule: 'constant', 'linear_decay', or 'cosine'
            clip_intermediate_actions: whether to clip during sampling
            save_chains: whether to save trajectory

        Returns:
            x_hat: (B, Ta, Da) sampled actions
            x_chain: (B, steps+1, Ta, Da) trajectory (if save_chains=True)
        """
        B = cond['state'].shape[0]
        device = cond['state'].device

        x_hat: Tensor = z if z is not None else torch.randn(
            B, self.horizon_steps, self.action_dim, device=device
        )

        if save_chains:
            x_chain = torch.zeros(
                (B, inference_steps + 1, self.horizon_steps, self.action_dim),
                device=device
            )
            x_chain[:, 0] = x_hat

        dt = 1.0 / inference_steps
        steps = torch.linspace(0, 1 - dt, inference_steps, device=device)

        for i in range(inference_steps):
            t = steps[i]
            t_batch = t.expand(B)  # (B,)

            # Get velocity bt(x)
            vt = self.forward(x_hat, t_batch, cond)

            # Get score st(x) = (t * bt(x) - x) / (1 - t)
            st = self.compute_score(x_hat, vt, t_batch)

            # Compute epsilon at this timestep
            # if epsilon_schedule == 'constant':
            #     eps_t = epsilon_t
            # elif epsilon_schedule == 'linear_decay':
            #     eps_t = epsilon_t * (1 - t.item())
            # elif epsilon_schedule == 'cosine':
            #     eps_t = epsilon_t * 0.5 * (1 + np.cos(np.pi * t.item()))
            # else:
            #     eps_t = epsilon_t

            # # Stochastic update: ak+1 = ak + [bt + εt·st]·Δt + √(2εt·Δt)·ε
            # drift = (vt + eps_t * st) * dt
            # diffusion = np.sqrt(2 * eps_t * dt) * torch.randn_like(x_hat)
            # x_hat = x_hat + drift + diffusion
            
            eps_t = ScoreFunction.get_epsilon(t.item(), epsilon_t, epsilon_schedule)

            # Stochastic SDE update using unified ScoreFunction
            x_hat = ScoreFunction.sde_step(x_hat, vt, st, eps_t, dt, stochastic=True)

            # Clip if needed
            if clip_intermediate_actions or i == inference_steps - 1:
                x_hat = x_hat.clamp(*act_range)

            if save_chains:
                x_chain[:, i + 1] = x_hat

        if save_chains:
            return x_hat, x_chain
        return x_hat   
class VisionFlowMLP(nn.Module, ScoreFunctionMixin):
    """With ViT backbone"""
    def __init__(
        self,
        backbone: VitEncoder,
        action_dim,
        horizon_steps,
        cond_dim,                       # proprioception only
        img_cond_steps=1,
        time_dim=16,
        mlp_dims=[256, 256],
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
        spatial_emb=0,
        visual_feature_dim=128,         # visual feature dim
        dropout=0,
        num_img=1,                      # currently only supports 1 or 2
        augment=False,
    ):
        super().__init__()
        
        # action chunk
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        self.act_dim_total = action_dim * horizon_steps
        
        # historical proprioception and visual inputs
        self.prop_dim = cond_dim    
        self.img_cond_steps = img_cond_steps
        
        # time
        self.time_dim = time_dim
        
        self.backbone = backbone
        self.mlp_dims = mlp_dims
        self.activation_type = activation_type
        self.out_activation_type = out_activation_type
        self.use_layernorm = use_layernorm
        self.residual_style = residual_style
        self.spatial_emb = spatial_emb
        
        self.dropout = dropout
        self.num_img = num_img
        self.augment = augment
        
        # vision
        self.backbone = backbone
        if augment:
            self.aug = RandomShiftsAug(pad=4)
        if spatial_emb > 0:
            assert spatial_emb > 1, "this is the dimension"
            if num_img == 2:
                self.compress1 = SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=cond_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
                self.compress2 = deepcopy(self.compress1)
            elif num_img == 1:  # TODO: clean up
                self.compress = SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=cond_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
            else:
                raise NotImplementedError(f"num_img={num_img} Currently we only support 1 or 2 image inputs")
            visual_feature_dim = spatial_emb * num_img
        else: # spatial embedding not specified, use default value 128
            self.compress = nn.Sequential(
                nn.Linear(self.backbone.repr_dim, visual_feature_dim),
                nn.LayerNorm(visual_feature_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
            )
        self.cond_enc_dim = visual_feature_dim + self.prop_dim     # rgb and  proprioception      
        
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )
        
        # Flow
        input_dim = (
            time_dim + \
                action_dim * horizon_steps + \
                        self.cond_enc_dim
        )
        
        # output action chunk
        output_dim = action_dim * horizon_steps
        
        # velocity head
        model = ResidualMLP if residual_style else MLP
        self.mlp_mean = model(
            [input_dim] + mlp_dims + [output_dim],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )
    
    def forward(
        self,
        action,
        time,
        cond: dict,
        output_embedding=False,
        **kwargs,
    ):
        """
        inputs:
            action: (B, Ta, Da) action chunk
            time: (B,) or float within [0,1), flow time
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
        outputs:

        TODO long term: more flexible handling of cond
        """
        B, Ta, Da = action.shape
        _, T_rgb, C, H, W = cond["rgb"].shape
        # flatten chunk
        action = action.view(B, -1)

        # flatten history (proprioception, here we use the raw input without encoding)
        state = cond["state"].view(B, -1)

        # Take recent images --- sometimes we want to use fewer img_cond_steps than cond_steps (e.g., 1 image but 3 prio)
        rgb = cond["rgb"][:, -self.img_cond_steps :]
        # concatenate images in cond by channels
        if self.num_img >1:
            rgb = rgb.reshape(B, T_rgb, self.num_img, 3, H, W)
            rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
        elif self.num_img==1:
            rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")
        else:
            raise ValueError(f"self.num_img={self.num_img} <1. ")
        # convert rgb to float32 for augmentation
        rgb = rgb.float()
        
        # visual and proprioceptive embeddings: get vit output - pass in two images separately
        if self.num_img ==2:  # TODO: properly handle multiple images
            rgb1 = rgb[:, 0]
            rgb2 = rgb[:, 1]
            if self.augment:
                rgb1 = self.aug(rgb1)
                rgb2 = self.aug(rgb2)
            feat1 = self.backbone.forward(rgb1)
            feat1 = self.compress1.forward(feat1, state)
            
            feat2 = self.backbone.forward(rgb2)
            feat2 = self.compress2.forward(feat2, state)
            
            feat = torch.cat([feat1, feat2], dim=-1)
        elif self.num_img ==1:  # single image
            if self.augment:
                rgb = self.aug(rgb)
            feat = self.backbone.forward(rgb)
            # compress
            if isinstance(self.compress, SpatialEmb):
                feat = self.compress.forward(feat, state)
            else:
                feat = feat.flatten(1, -1)
                feat = self.compress(feat)
        else:
            raise NotImplementedError(f"num_img={self.num_img} Currently we only support 1 or 2 image inputs")
        cond_encoded = torch.cat([feat, state], dim=-1)   # visual and proprioception inputs. 

        # time embedding
        time = time.view(B, 1)
        time_emb = self.time_embedding(time).view(B, self.time_dim)
        
        # all embeddings: time, visual-proprioceptive
        emb = torch.cat([action, time_emb, cond_encoded], dim=-1)

        # velocity head
        vel = self.mlp_mean(emb)
        if output_embedding:
            return vel.view(B, Ta, Da), time_emb, cond_encoded
        return vel.view(B, Ta, Da)
    
    def sample_action(self, cond: dict, inference_steps: int, clip_intermediate_actions: bool, act_range: List[float], z: Tensor = None, save_chains: bool = False):
        """
        Deterministic sampling via Euler integration.
        When `save_chains` is True, also return the denoising trajectory.

        Args:
            cond: dict with 'state' and 'rgb' keys
            inference_steps: number of denoising steps
            clip_intermediate_actions: whether to clip during sampling
            act_range: [min, max] for action clipping
            z: initial noise (optional)
            save_chains: whether to save trajectory
        """
        B = cond['state'].shape[0]
        device = cond['state'].device

        x_hat: Tensor = z if z is not None else torch.randn(B, self.horizon_steps, self.action_dim, device=device)
        if save_chains:
            x_chain = torch.zeros((B, inference_steps + 1, self.horizon_steps, self.action_dim), device=device)
            x_chain[:, 0] = x_hat

        dt = (1 / inference_steps) * torch.ones_like(x_hat, device=device)
        steps = torch.linspace(0, 1 - 1 / inference_steps, inference_steps, device=device).repeat(B, 1)

        for i in range(inference_steps):
            t = steps[:, i]
            vt = self.forward(x_hat, t, cond)
            x_hat = x_hat + vt * dt
            if clip_intermediate_actions or i == inference_steps - 1:
                x_hat = x_hat.clamp(*act_range)
            if save_chains:
                x_chain[:, i + 1] = x_hat

        if save_chains:
            return x_hat, x_chain
        return x_hat

    # def compute_score(self, action: Tensor, vel: Tensor, time: Tensor) -> Tensor:
    #     """
    #     Compute score function: st(x) = (t * bt(x) - x) / (1 - t)

    #     Args:
    #         action: (B, Ta, Da) current state xt
    #         vel: (B, Ta, Da) velocity field bt(x)
    #         time: (B,) or (B, 1) time in [0, 1)

    #     Returns:
    #         score: (B, Ta, Da) score function st(x)
    #     """
    #     B = action.shape[0]

    #     if time.dim() == 1:
    #         time = time.view(B, 1, 1)
    #     elif time.dim() == 2:
    #         time = time.view(B, 1, 1)

    #     epsilon = 1e-5
    #     score = (time * vel - action) / (1 - time + epsilon)

    #     return score

    def sample_action_stochastic(
        self,
        cond: dict,
        inference_steps: int,
        epsilon_t: float,
        act_range: List[float],
        z: Tensor = None,
        epsilon_schedule: str = 'constant',
        clip_intermediate_actions: bool = True,
        save_chains: bool = False,
    ):
        """
        Stochastic sampling: dxt = [bt(xt) + εt·st(xt)] dt + √(2εt) dWt

        Args:
            cond: dict with 'state' and 'rgb' keys
            inference_steps: number of denoising steps
            epsilon_t: noise scale
            act_range: [min, max] for clipping
            z: initial noise
            epsilon_schedule: 'constant', 'linear_decay', or 'cosine'
            clip_intermediate_actions: whether to clip during sampling
            save_chains: whether to save trajectory

        Returns:
            x_hat: (B, Ta, Da) sampled actions
        """
        B = cond['state'].shape[0]
        device = cond['state'].device

        x_hat: Tensor = z if z is not None else torch.randn(
            B, self.horizon_steps, self.action_dim, device=device
        )

        if save_chains:
            x_chain = torch.zeros(
                (B, inference_steps + 1, self.horizon_steps, self.action_dim),
                device=device
            )
            x_chain[:, 0] = x_hat

        dt = 1.0 / inference_steps
        steps = torch.linspace(0, 1 - dt, inference_steps, device=device) # flow matching 的时间步

        for i in range(inference_steps):
            t = steps[i]
            t_batch = t.expand(B)

            vt = self.forward(x_hat, t_batch, cond)
            st = self.compute_score(x_hat, vt, t_batch)

            # Get epsilon using unified ScoreFunction
            eps_t = ScoreFunction.get_epsilon(t.item(), epsilon_t, epsilon_schedule)

            # Perform SDE step using unified ScoreFunction
            x_hat = ScoreFunction.sde_step(x_hat, vt, st, eps_t, dt, stochastic=True)

            if save_chains:
                x_chain[:, i + 1] = x_hat

        if save_chains:
            return x_hat, x_chain
        return x_hat
    