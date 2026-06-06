# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import math
import random
from pathlib import Path
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
from torch.utils._pytree import tree_map

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.explore_noise_net import ExploreNoiseNet
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.utils.logging import get_logger
from rlinf.utils.nested_dict_process import copy_dict_tensor
from rlinf.utils.pytree import register_pytree_dataclasses


def _to_numpy(x):
    return np.asarray(x.detach().cpu()) if torch.is_tensor(x) else x


@dataclass(frozen=True)
class OpenPi0Config(Pi0Config):
    # config for rl
    config_name: str = "pi0_libero"  # pi0_libero, pi05_libero, pi0_maniskill, pi05_maniskill, pi0_metaworld, pi05_metaworld
    num_images_in_input: int = 2  # number of images in input
    noise_method: str = "flow_sde"  # flow_ode, flow_sde, flow_noise, flow_cps
    # noise config for flow-sde
    noise_level: float = 0.5
    noise_anneal: bool = False
    noise_params: list = field(
        default_factory=lambda: [0.7, 0.3, 400]
    )  # noise_start, noise_end, noise_anneal_steps
    # noise config for flow-noise
    noise_logvar_range: list = field(
        default_factory=lambda: [0.08, 0.16]
    )  # [min_std, max_std]
    # hyper-parameters
    action_chunk: int = 5  # action chunk
    action_env_dim: int = 7  # for environment action dim
    num_steps: int = 10  # denoise steps
    # training config
    train_expert_only: bool = False
    safe_get_logprob: bool = False
    joint_logprob: bool = False  # designed for flow-noise
    double_layer: bool = False  # designed for flow-sde without acceleration
    ignore_last: bool = False  # ignore the last action for noise injection
    # critic
    detach_critic_input: bool = False  # detach critic input with the action expert
    chunk_critic_input: bool = False  # use only the action chunk for critic estimation
    add_value_head: bool = False  # add value head for ppo
    value_after_vlm: bool = False  # value after vlm, pi05 mode
    value_vlm_mode: str = "mean_token"  # last_token, mean_token, first_token

    # TERMINAL_GUIDANCE_PATCH_V1
    guidance_mode: str = "none"  # none, direct_xt, terminal_x1, fmtt_terminal_x1
    guidance_beta: float = 0.1
    guidance_grad_clip_norm: float = 1.0
    guidance_update_rule: str = "euler"  # euler, sigma2_tilt
    guidance_diag_path: str = ""
    # SCORE_FLOW_PATCH_V2
    score_flow_mode: str = "none"  # none, learned_alpha, sigma2_score, value_conditioned
    score_flow_scale: float = 1.0
    score_flow_clip_norm: float = 10.0
    score_flow_alpha_hidden_dim: int = 16
    score_flow_alpha_init_bias: float = -2.0
    score_flow_alpha_max: float = 2.0
    score_flow_use_time_mask: bool = True
    score_flow_condition_source: str = "value_t"
    score_flow_value_scale: float = 10.0
    score_flow_diffusion_mode: str = "none"  # none, modulate_std
    score_flow_diffusion_hidden_dim: int = 16
    score_flow_diffusion_log_scale_max: float = 0.5
    score_flow_diffusion_min_std: float = 1.0e-4
    tr_penalty_mode: str = "none"  # none, scalar_l2, terminal_pullback
    tr_penalty_beta: float = 0.0
    tr_pullback_fd_eps: float = 1.0e-2
    cr_reflow_mode: str = "none"  # none, cr_reflow, cr_reflow_no_anchor
    cr_reflow_kl_epsilon: float = 0.05
    cr_reflow_eta_min: float = 0.01
    cr_reflow_eta_max: float = 10.0
    cr_reflow_weight_clip: float = 10.0
    cr_reflow_anchor_beta: float = 0.1
    cr_reflow_eps: float = 1.0e-6
    # SPEC_FLOW_PATCH_V1
    spec_flow_mode: str = "none"  # none, cheap_draft, logprob, flow, composite
    spec_flow_draft_steps: int = 2
    spec_flow_accept_rate: float = 0.5
    spec_flow_diag_path: str = ""

    # ===== DSRL-specific parameters =====
    use_dsrl: bool = False  # Enable DSRL algorithm
    dsrl_state_dim: int = 8  # Raw state dimension for DSRL encoders
    dsrl_action_noise_dim: int = 32  # Noise dimension output by GaussianPolicy
    dsrl_num_q_heads: int = 10  # Number of Q-networks
    dsrl_agg_q: str = "mean"  # Q aggregation method: 'mean' | 'min'
    dsrl_image_latent_dim: int = 64  # Latent dim for lightweight image encoder
    dsrl_state_latent_dim: int = 64  # Hidden dim for state encoder
    dsrl_hidden_dims: tuple = field(
        default_factory=lambda: (128, 128, 128)
    )  # Hidden dims for Q-head and GaussianPolicy

    # ===== NFT-specific parameters =====
    is_nft: bool = False


class OpenPi0ForRLActionPrediction(PI0Pytorch, BasePolicy):
    """
    Pi0 model for reinforcement learning action prediction.
    """

    config: OpenPi0Config

    @property
    def _no_split_modules(self) -> list[str]:
        if self.config.train_expert_only:
            no_split_modules = [
                "GemmaDecoderLayer",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
            ]
        else:
            no_split_modules = [
                "GemmaMLP",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
            ]
        if self.config.noise_method == "flow_noise":
            no_split_modules.append("ExploreNoiseNet")
        return no_split_modules

    @property
    def _no_split_names(self) -> list[str]:
        return [
            "action_in_proj",
            "action_out_proj",
            "lm_head",
            # --pi0 only--
            "state_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
            # --pi05 only--
            "time_mlp_in",
            "time_mlp_out",
        ]

    def __init__(
        self,
        config: OpenPi0Config,
    ):
        # Override `sample_actions` to prevent parent class polymorphic call
        sample_actions_func = self.sample_actions
        super().__init__(config)
        self.sample_actions = sample_actions_func
        self.logger = get_logger()
        self.global_step = 0
        self._init_score_flow_alpha_net()
        # assert
        assert not (self.config.double_layer and self.config.joint_logprob), (
            "double_layer and joint_logprob can not be set at the same time"
        )

        # rl model init
        if self.config.value_after_vlm:
            proj_width = 2048
        else:
            proj_width = 1024
        # value head
        if self.config.add_value_head:
            if self.config.config_name in ["pi05_maniskill", "pi05_libero"]:
                value_head_hidden_sizes = (1024, 512, 256)
            else:
                value_head_hidden_sizes = (512, 256, 128)
            value_head_activation = "relu"
            self.value_head = ValueHead(
                input_dim=proj_width,
                hidden_sizes=value_head_hidden_sizes,
                output_dim=1,
                activation=value_head_activation,
                bias_last=True,
            )
        self.use_vlm_value = getattr(self.config, "value_after_vlm", False) and getattr(
            self.config, "add_value_head", False
        )
        # noise head for flow-noise
        if self.config.noise_method == "flow_noise":
            self.noise_head = ExploreNoiseNet(
                in_dim=1024,
                out_dim=self.config.action_dim,
                hidden_dims=[128, 64],
                activation_type="tanh",
                noise_logvar_range=self.config.noise_logvar_range,
                noise_scheduler_type="learn",
            )

        # ===== DSRL components initialization =====
        if self.config.use_dsrl:
            from rlinf.models.embodiment.modules.compact_encoders import (
                CompactMultiQHead,
                CompactStateEncoder,
                LightweightImageEncoder64,
            )
            from rlinf.models.embodiment.modules.gaussian_policy import GaussianPolicy

            # Use explicit bfloat16 to match the backbone dtype that will be
            # loaded from the checkpoint later.  At __init__ time the backbone
            # parameters are still float32 (weights are loaded afterwards by
            # safetensors.torch.load_model), so next(self.parameters()).dtype
            # would incorrectly return float32.  Hardcoding bfloat16 here
            # ensures all parameters share a single dtype when FSDP creates
            # its FlatParameter, avoiding the writeback shape-mismatch error.
            _dsrl_dtype = torch.bfloat16

            dsrl_input_dim = (
                self.config.dsrl_state_latent_dim + self.config.dsrl_image_latent_dim
            )  # e.g. 64 + 64 = 128

            self.dsrl_action_noise_net = GaussianPolicy(
                input_dim=dsrl_input_dim,
                output_dim=self.config.dsrl_action_noise_dim,
                hidden_dims=self.config.dsrl_hidden_dims,
                low=None,
                high=None,
                action_horizon=self.config.action_horizon,
            ).to(dtype=_dsrl_dtype)

            self.actor_image_encoder = LightweightImageEncoder64(
                num_images=1,
                latent_dim=self.config.dsrl_image_latent_dim,
                image_size=64,
            ).to(dtype=_dsrl_dtype)
            self.actor_state_encoder = CompactStateEncoder(
                state_dim=self.config.dsrl_state_dim,
                hidden_dim=self.config.dsrl_state_latent_dim,
            ).to(dtype=_dsrl_dtype)
            self.critic_image_encoder = LightweightImageEncoder64(
                num_images=1,
                latent_dim=self.config.dsrl_image_latent_dim,
                image_size=64,
            ).to(dtype=_dsrl_dtype)
            self.critic_state_encoder = CompactStateEncoder(
                state_dim=self.config.dsrl_state_dim,
                hidden_dim=self.config.dsrl_state_latent_dim,
            ).to(dtype=_dsrl_dtype)
            self.q_head = CompactMultiQHead(
                state_dim=self.config.dsrl_state_latent_dim,
                image_dim=self.config.dsrl_image_latent_dim,
                action_dim=self.config.dsrl_action_noise_dim,
                hidden_dims=self.config.dsrl_hidden_dims,
                num_q_heads=self.config.dsrl_num_q_heads,
                output_dim=1,
            ).to(dtype=_dsrl_dtype)

        for name, module in self.named_modules():
            # Set _fsdp_wrap_name to the last part of the path (e.g., "model.action_in_proj" -> "action_in_proj")
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

    def set_global_step(self, global_step):
        self.global_step = global_step

    def setup_wrappers(
        self,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
    ):
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)

    def input_transform(self, obs: dict, transpose=True):
        inputs = tree_map(lambda x: x, obs)
        # process input
        first_process = "prompt" in inputs.keys()
        if first_process:
            inputs.pop("prompt")
        else:
            inputs = {key: inputs[key] for key in inputs.keys() if "/" in key}

        # tensor -> numpy
        inputs = tree_map(_to_numpy, inputs)
        batch_size = next(v.shape[0] for v in inputs.values() if hasattr(v, "shape"))
        # split & transform
        transformed_samples = []
        for i in range(batch_size):
            sample = tree_map(lambda x: x[i], inputs)
            if transpose:
                # convert from [3,256,256] -> [256,256,3]
                sample = tree_map(
                    lambda x: (
                        x.transpose(1, 2, 0) if len(x.shape) == 3 and transpose else x
                    ),
                    sample,
                )
            else:
                sample = tree_map(lambda x: x if len(x.shape) == 3 else x, sample)
            if first_process:
                sample["prompt"] = obs["prompt"][i]
            else:
                sample["prompt"] = "xxxx"
            transformed_sample = self._input_transform(sample)
            transformed_samples.append(transformed_sample)
        # recombine
        inputs = tree_map(
            lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
            *transformed_samples,
        )
        # inputs = tree_map(lambda *x: torch.stack(x, axis=0), inputs)
        if not first_process:
            inputs["tokenized_prompt"] = obs["tokenized_prompt"]
            inputs["tokenized_prompt_mask"] = obs["tokenized_prompt_mask"]
        return inputs

    def output_transform(self, outputs):
        # split & transform
        batch_size = outputs["actions"].shape[0]
        transformed_samples = []
        for i in range(batch_size):
            sample = tree_map(lambda x: np.asarray(x[i].detach().cpu()), outputs)
            sample = self._output_transform(sample)
            transformed_samples.append(sample)
        # recombine
        outputs = tree_map(
            lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
            *transformed_samples,
        )
        outputs["actions"] = outputs["actions"][:, : self.config.action_chunk]
        return outputs

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        elif forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        elif forward_type == ForwardType.NFT:
            return self.nft_forward(**kwargs)
        elif forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)
        elif forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)
        else:
            raise NotImplementedError

    def sft_forward(self, data, **kwargs):
        if hasattr(self, "gradient_checkpointing_disable"):
            self.gradient_checkpointing_disable()
        observation = data["observation"]
        actions = data["actions"]
        return super().forward(observation, actions)

    def prepare_dagger_sft_batch(self, batch):
        """Prepare replay-buffer samples for DAgger SFT updates."""
        device = next(self.parameters()).device
        obs_dict = {}
        obs_prefix_keys = [k for k in batch.keys() if k.startswith("observation/")]
        for key in obs_prefix_keys:
            obs_dict[key] = batch[key]
        if "tokenized_prompt" in batch:
            obs_dict["tokenized_prompt"] = batch["tokenized_prompt"]
        if "tokenized_prompt_mask" in batch:
            obs_dict["tokenized_prompt_mask"] = batch["tokenized_prompt_mask"]

        bsz = batch["action"].shape[0]
        if "model_action" in batch:
            actions = (
                batch["model_action"]
                .reshape(bsz, self.config.action_horizon, self.config.action_dim)
                .clone()
            )
            processed_obs = self.input_transform(obs_dict, transpose=False)
            processed_obs = self.precision_processor(processed_obs)
            observation = _model.Observation.from_dict(processed_obs)
        else:
            obs_dict["actions"] = batch["action"].reshape(
                bsz, self.config.action_chunk, -1
            )
            obs_dict["prompt"] = ["empty" for _ in range(bsz)]
            processed_obs = self.input_transform(obs_dict, transpose=False)
            if "tokenized_prompt" in batch:
                processed_obs["tokenized_prompt"] = batch["tokenized_prompt"]
            if "tokenized_prompt_mask" in batch:
                processed_obs["tokenized_prompt_mask"] = batch["tokenized_prompt_mask"]
            processed_obs = self.precision_processor(processed_obs)
            observation = _model.Observation.from_dict(processed_obs)
            actions = processed_obs["actions"].clone()
            processed_obs.pop("actions")

        register_pytree_dataclasses(observation)
        observation = tree_map(
            lambda x: torch.as_tensor(x, device=device).contiguous().clone(),
            observation,
        )
        return {
            "observation": observation,
            "actions": actions.to(torch.float32).to(device),
        }

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        # get kwargs
        compute_values = kwargs.get("compute_values", False)
        chains = forward_inputs["chains"]
        denoise_inds = forward_inputs["denoise_inds"]
        old_means = kwargs.get("old_means", forward_inputs.get("old_means"))
        old_stds = kwargs.get("old_stds", forward_inputs.get("old_stds"))
        advantages = kwargs.get("advantages", forward_inputs.get("advantages"))
        cr_reflow_mask = kwargs.get(
            "cr_reflow_mask",
            forward_inputs.get("cr_reflow_mask", forward_inputs.get("loss_mask")),
        )
        # input transform
        observation = self.input_transform(forward_inputs, transpose=False)
        observation = _model.Observation.from_dict(observation)
        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        )
        # transfer to device
        device = chains.device
        images = [img.to(device) for img in images]
        img_masks = [img_mask.to(device) for img_mask in img_masks]
        state = state.to(device)
        # get log prob
        (
            log_probs,
            value_t,
            entropy,
            terminal_pullback_loss,
            tr_diag,
            transition_means,
            transition_stds,
        ) = self.get_log_prob_value(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            chains,
            denoise_inds,
            compute_values=compute_values,
            old_means=old_means,
        )
        cr_reflow_loss, cr_reflow_diag = self._compute_cr_reflow_loss(
            chains,
            denoise_inds,
            transition_means,
            transition_stds,
            old_means,
            old_stds,
            advantages,
            value_t,
            cr_reflow_mask,
        )
        log_probs = log_probs[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        entropy = entropy[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        # post process
        log_probs = log_probs.mean(dim=1)
        entropy = entropy.mean(dim=[1, 2, 3], keepdim=False)[
            :, None
        ]  # [:,None] to align with loss-mask shape
        value_t = value_t.mean(dim=-1, keepdim=False)
        return {
            "logprobs": log_probs,
            "values": value_t,
            "entropy": entropy,
            "terminal_pullback_loss": terminal_pullback_loss,
            "tr_diag": tr_diag,
            "cr_reflow_loss": cr_reflow_loss,
            "cr_reflow_diag": cr_reflow_diag,
        }

    def nft_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        """Compute velocity v_theta at explicit (x_t, timesteps) for NFT loss."""
        # obs process
        observation = self.input_transform(forward_inputs, transpose=False)
        observation = _model.Observation.from_dict(observation)
        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        )
        # move device
        device = next(self.parameters()).device
        images = [img.to(device) for img in images]
        img_masks = [m.to(device) for m in img_masks]
        state = state.to(device)
        # nft inputs
        nft_inputs = kwargs["nft_inputs"]
        x_t = nft_inputs["x_t"].to(device)
        t = nft_inputs["timesteps"].to(device)
        # get v_theta
        _, prefix_pad_masks, past_key_values = self._build_prefix_cache(
            images, img_masks, lang_tokens, lang_masks
        )
        compute_values = kwargs.get("compute_values", False)
        v_theta, suffix_out = self.get_velocity(
            state, x_t, t, prefix_pad_masks, past_key_values
        )
        v_theta = v_theta[:, : self.config.action_chunk, :]
        # result
        result: dict[str, Any] = {"v_theta": v_theta, "x_t": x_t, "timesteps": t}
        if compute_values and self.config.add_value_head:
            result["values"] = self._compute_value_from_suffix(suffix_out)[:, None]
        return result

    def obs_processor(self, env_obs):
        # base observation
        processed_obs = {
            "observation/image": env_obs["main_images"],
            "prompt": env_obs["task_descriptions"],
        }
        # state observation
        if "calvin" in self.config.config_name:
            state = env_obs["states"]
            processed_obs["observation/state_ee_pos"] = state[:, :3]
            processed_obs["observation/state_ee_rot"] = state[:, 3:6]
            processed_obs["observation/state_gripper"] = state[:, 6:7]
        else:
            processed_obs["observation/state"] = env_obs["states"]
        # wrist image observation
        if env_obs["wrist_images"] is not None:
            processed_obs["observation/wrist_image"] = env_obs["wrist_images"]
        # extra view image observation
        if env_obs["extra_view_images"] is not None:
            processed_obs["observation/extra_view_image"] = env_obs["extra_view_images"]
        # store used keys
        return processed_obs

    def precision_processor(self, processed_obs):
        device = next(self.parameters()).device
        for key, value in processed_obs.items():
            if isinstance(value, list):
                processed_obs[key] = [
                    item.to(device=device).contiguous()
                    if torch.is_tensor(item)
                    else item
                    for item in value
                ]
            elif torch.is_tensor(value):
                processed_obs[key] = value.to(device=device).contiguous()
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    processed_obs[key][sub_key] = sub_value.to(
                        device=device
                    ).contiguous()
        return processed_obs

    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        compute_values=True,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        to_process_obs = self.obs_processor(env_obs)  # env obs -> policy input obs
        processed_obs = self.input_transform(
            to_process_obs, transpose=False
        )  # policy input obs -> model input obs
        processed_obs = self.precision_processor(
            processed_obs
        )  # obs precision processor
        observation = _model.Observation.from_dict(processed_obs)

        is_dsrl_active = self.config.use_dsrl
        if is_dsrl_active:
            # DSRL mode (both train and eval)

            # Step 1: SAC agent outputs noise
            dsrl_obs = {"images": [env_obs["main_images"]], "states": env_obs["states"]}

            noise_actions, noise_logprob, _ = self.sac_forward(
                dsrl_obs, train=False, mode=mode
            )

            # Step 2: Use noise to sample actual actions from diffusion model
            outputs = self.sample_actions(
                observation,
                noise=noise_actions,
                mode="eval",
                compute_values=compute_values,
            )

            # Step 3: Extract actual actions for environment interaction
            real_actions = self.output_transform(
                {"actions": outputs["actions"], "state": observation.state}
            )["actions"]

            # Return actual actions to environment, but forward_inputs stores noise.
            actions = real_actions
            prev_logprobs = noise_logprob  # SAC noise logprob
            prev_values = outputs.get("prev_values")
            forward_action = noise_actions  # Used for SAC training

        else:
            # Non-DSRL or eval mode
            outputs = self.sample_actions(
                observation, mode=mode, compute_values=compute_values
            )
            actions = self.output_transform(
                {"actions": outputs["actions"], "state": observation.state}
            )["actions"]
            prev_logprobs = outputs["prev_logprobs"]
            prev_values = outputs["prev_values"]
            forward_action = None

        forward_inputs = {
            "chains": outputs["chains"],
            "denoise_inds": outputs["denoise_inds"],
            "tokenized_prompt": processed_obs["tokenized_prompt"],
            "tokenized_prompt_mask": processed_obs["tokenized_prompt_mask"],
            # "action" is the env-executed action, and "model_action" is the original output by the model.
            # For small models, they are consistent. For large models (like pi), "action" is the result after output_transform.
            # For realworld human-in-the-loop training, only "action" can be provided by human.
            "action": actions.reshape(actions.shape[0], -1).contiguous(),
            "model_action": outputs["actions"]
            .reshape(outputs["actions"].shape[0], -1)
            .contiguous(),
        }
        if forward_action is not None:
            forward_inputs["action"] = forward_action

        if self.config.is_nft:
            nft_outputs = {
                key: value for key, value in outputs.items() if key.startswith("nft_")
            }
            forward_inputs.update(nft_outputs)

        # Clone observations to avoid cross-step reference issues.
        cloned_obs = copy_dict_tensor(
            {k: v for k, v in to_process_obs.items() if k != "prompt"}
        )
        forward_inputs.update(cloned_obs)

        # Carry old transition statistics through the rollout->buffer->actor channel
        # only for methods that need teacher-forced transition recomputation.
        if (
            getattr(self.config, "tr_penalty_mode", "none") != "none"
            or self._cr_reflow_enabled()
        ):
            forward_inputs["old_means"] = outputs["old_means"]
            forward_inputs["old_stds"] = outputs["old_stds"]

        result = {
            "prev_logprobs": prev_logprobs,
            "prev_values": prev_values,
            "forward_inputs": forward_inputs,
        }
        return actions, result

    @torch.no_grad()

    def _spec_flow_enabled(self, mode):
        spec_mode = getattr(self.config, "spec_flow_mode", "none")
        return spec_mode != "none"

    def _spec_flow_num_steps_context(self, num_steps):
        class _NumStepsContext:
            def __init__(self, config, value):
                self.config = config
                self.value = int(value)
                self.old_value = int(getattr(config, "num_steps"))

            def __enter__(self):
                try:
                    setattr(self.config, "num_steps", self.value)
                except Exception:
                    object.__setattr__(self.config, "num_steps", self.value)

            def __exit__(self, exc_type, exc, tb):
                try:
                    setattr(self.config, "num_steps", self.old_value)
                except Exception:
                    object.__setattr__(self.config, "num_steps", self.old_value)

        return _NumStepsContext(self.config, num_steps)

    def _spec_flow_flat_norm(self, tensor):
        return tensor.detach().float().reshape(tensor.shape[0], -1).norm(dim=1)

    def _spec_flow_zscore(self, tensor):
        tensor = torch.nan_to_num(tensor.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
        std = tensor.std()
        if std.item() == 0.0 or torch.isnan(std):
            return torch.zeros_like(tensor)
        return (tensor - tensor.mean()) / std

    def _spec_flow_boundary_metrics(self, draft_action):
        previous = getattr(self, "_spec_flow_previous_action", None)
        if previous is None or previous.shape[0] != draft_action.shape[0]:
            boundary_jump = torch.zeros(draft_action.shape[0], device=draft_action.device)
        else:
            boundary_jump = (draft_action[:, 0] - previous[:, -1]).detach().float().norm(dim=-1)
        if draft_action.shape[1] >= 3:
            jerk = (
                draft_action[:, 2:] - 2.0 * draft_action[:, 1:-1] + draft_action[:, :-2]
            ).detach().float().reshape(draft_action.shape[0], -1).norm(dim=1)
        else:
            jerk = torch.zeros(draft_action.shape[0], device=draft_action.device)
        self._spec_flow_previous_action = draft_action.detach()
        return boundary_jump, jerk

    def _spec_flow_score(self, spec_mode, draft_result, full_result):
        draft_action = draft_result["actions"]
        full_action = full_result["actions"]
        draft_logprob = draft_result["prev_logprobs"].detach().float().reshape(draft_action.shape[0], -1).mean(dim=1)
        flow_error = self._spec_flow_flat_norm(draft_action - full_action)
        boundary_jump, jerk = self._spec_flow_boundary_metrics(draft_action)
        if spec_mode == "logprob":
            score = self._spec_flow_zscore(draft_logprob)
        elif spec_mode == "flow":
            score = -self._spec_flow_zscore(flow_error)
        elif spec_mode == "composite":
            score = (
                self._spec_flow_zscore(draft_logprob)
                - self._spec_flow_zscore(flow_error)
                - 0.5 * self._spec_flow_zscore(boundary_jump)
                - 0.25 * self._spec_flow_zscore(jerk)
            )
        else:
            raise ValueError(f"Unsupported spec_flow_mode={spec_mode}")
        metrics = {
            "draft_logprob": draft_logprob,
            "flow_error": flow_error,
            "boundary_jump": boundary_jump,
            "chunk_jerk": jerk,
            "score": score.detach().float(),
        }
        return score, metrics

    def _spec_flow_accept_mask(self, score):
        accept_rate = float(getattr(self.config, "spec_flow_accept_rate", 0.5))
        if accept_rate <= 0.0:
            return torch.zeros_like(score, dtype=torch.bool)
        if accept_rate >= 1.0:
            return torch.ones_like(score, dtype=torch.bool)
        threshold = torch.quantile(score.detach().float(), 1.0 - accept_rate)
        return score >= threshold

    def _spec_flow_blend_tensor(self, draft_value, full_value, accept_mask):
        if not torch.is_tensor(draft_value) or not torch.is_tensor(full_value):
            return full_value
        if draft_value.shape != full_value.shape:
            return full_value
        mask = accept_mask
        while mask.ndim < draft_value.ndim:
            mask = mask.unsqueeze(-1)
        return torch.where(mask, draft_value, full_value)

    def _write_spec_flow_diagnostics(self, rows):
        diag_path = getattr(self.config, "spec_flow_diag_path", "")
        if not diag_path:
            return
        path = Path(str(diag_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "global_step",
            "mode",
            "spec_flow_mode",
            "draft_steps",
            "full_steps",
            "target_accept_rate",
            "actual_accept_rate",
            "estimated_denoise_steps",
            "draft_logprob_mean",
            "flow_error_mean",
            "boundary_jump_mean",
            "chunk_jerk_mean",
            "score_mean",
            "score_std",
        ]
        write_header = not path.exists()
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})

    def sample_actions(
        self,
        observation: _model.Observation,
        noise=None,
        mode="train",
        compute_values=True,
    ) -> torch.Tensor:
        spec_mode = getattr(self.config, "spec_flow_mode", "none")
        if spec_mode == "none":
            return self._sample_actions_full(observation, noise=noise, mode=mode, compute_values=compute_values)

        full_steps = int(getattr(self.config, "num_steps"))
        draft_steps = int(getattr(self.config, "spec_flow_draft_steps", max(1, full_steps // 2)))
        draft_steps = max(1, min(draft_steps, full_steps))

        with self._spec_flow_num_steps_context(draft_steps):
            draft_result = self._sample_actions_full(observation, noise=noise, mode=mode, compute_values=compute_values)

        if spec_mode == "cheap_draft":
            actual_accept_rate = 1.0
            estimated_steps = float(draft_steps)
            selected = draft_result
            metrics = {
                "draft_logprob": draft_result["prev_logprobs"].detach().float().reshape(draft_result["actions"].shape[0], -1).mean(dim=1),
                "flow_error": torch.full((draft_result["actions"].shape[0],), float("nan"), device=draft_result["actions"].device),
                "boundary_jump": torch.full((draft_result["actions"].shape[0],), float("nan"), device=draft_result["actions"].device),
                "chunk_jerk": torch.full((draft_result["actions"].shape[0],), float("nan"), device=draft_result["actions"].device),
                "score": torch.full((draft_result["actions"].shape[0],), float("nan"), device=draft_result["actions"].device),
            }
        else:
            full_result = self._sample_actions_full(observation, noise=noise, mode=mode, compute_values=compute_values)
            score, metrics = self._spec_flow_score(spec_mode, draft_result, full_result)
            accept_mask = self._spec_flow_accept_mask(score)
            actual_accept_rate = float(accept_mask.float().mean().detach().cpu().item())
            estimated_steps = float(draft_steps + full_steps * (1.0 - actual_accept_rate))
            selected = dict(full_result)
            for key in ("actions", "prev_logprobs", "prev_values"):
                selected[key] = self._spec_flow_blend_tensor(draft_result.get(key), full_result.get(key), accept_mask)
            selected["spec_flow_accept_mask"] = accept_mask.detach()
            selected["spec_flow_draft_actions"] = draft_result["actions"].detach()
            selected["spec_flow_full_actions"] = full_result["actions"].detach()

        self._write_spec_flow_diagnostics(
            [
                {
                    "global_step": int(getattr(self, "global_step", 0)),
                    "mode": mode,
                    "spec_flow_mode": spec_mode,
                    "draft_steps": draft_steps,
                    "full_steps": full_steps,
                    "target_accept_rate": float(getattr(self.config, "spec_flow_accept_rate", 0.5)),
                    "actual_accept_rate": actual_accept_rate,
                    "estimated_denoise_steps": estimated_steps,
                    "draft_logprob_mean": float(torch.nanmean(metrics["draft_logprob"]).detach().cpu().item()),
                    "flow_error_mean": float(torch.nanmean(metrics["flow_error"]).detach().cpu().item()),
                    "boundary_jump_mean": float(torch.nanmean(metrics["boundary_jump"]).detach().cpu().item()),
                    "chunk_jerk_mean": float(torch.nanmean(metrics["chunk_jerk"]).detach().cpu().item()),
                    "score_mean": float(torch.nanmean(metrics["score"]).detach().cpu().item()),
                    "score_std": float(torch.nan_to_num(metrics["score"].detach().float(), nan=0.0).std().cpu().item()),
                }
            ]
        )
        return selected

    def _sample_actions_full(
        self,
        observation: _model.Observation,
        noise=None,
        mode="train",
        compute_values=True,
    ) -> torch.Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        device = observation.state.device
        num_steps = self.config.num_steps
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)
        else:
            # DSRL: SAC provides noise, convert dtype to match action_in_proj
            noise = noise.to(self.action_in_proj.weight.dtype)

        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        )

        prefix_output, prefix_pad_masks, past_key_values = self._build_prefix_cache(
            images, img_masks, lang_tokens, lang_masks
        )

        x_t = noise
        # add sde sample and traj collect
        chains = []
        log_probs = []
        values = []
        old_means = []
        old_stds = []
        chains.append(x_t)

        # add value based on the vlm for pi05, expert for pi0
        if self.use_vlm_value:
            values_vlm = self.get_value_from_vlm(prefix_output)
        if self.config.joint_logprob:
            initial_log_prob = self.get_logprob_norm(
                x_t, torch.zeros_like(noise), torch.ones_like(noise)
            )
            log_probs.append(initial_log_prob)

        # In the joint logprob mode, we need to sample the logprob for each denoise step
        # In the non-joint logprob mode, only one denoise step is sampled and ode-sde mix sampling is used
        # denoise index
        collect_nft_state = self.config.is_nft and mode == "train"
        if mode == "train":
            if self.config.joint_logprob or collect_nft_state:
                denoise_inds = torch.arange(num_steps)
            else:
                if self.config.ignore_last:
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 2)] * num_steps
                    )
                else:
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 1)] * num_steps
                    )
        else:
            denoise_inds = torch.tensor([-1] * num_steps)
        denoise_inds = denoise_inds[None].repeat(bsize, 1)

        # collect nft states for nft algorithm
        nft_state = self._init_nft_state(collect_nft_state, x_t, num_steps, device)

        guidance_diag_rows = []
        self._terminal_guidance_active = True
        self._last_terminal_guidance_diag = None
        self._last_score_flow_diag = None

        # denoise step
        for idx in range(num_steps):
            # sample mean var val
            if idx == denoise_inds[0][idx]:
                sample_method = self.config.noise_method
            else:
                sample_method = "flow_ode"
            x_t_prev = x_t
            x_t_mean, x_t_std, value_t, v_t = self.sample_mean_var_val(
                x_t,
                idx,
                state,
                prefix_pad_masks,
                past_key_values,
                sample_method,
                num_steps,
                compute_values,
            )
            old_means.append(x_t_mean.detach())
            old_stds.append(x_t_std.detach())
            score_flow_diag = getattr(self, "_last_score_flow_diag", None)
            if score_flow_diag is not None:
                guidance_diag_rows.append(
                    {
                        "global_step": int(getattr(self, "global_step", 0)),
                        "mode": mode,
                        "denoise_idx": int(idx),
                        **score_flow_diag,
                    }
                )

            guidance_diag = getattr(self, "_last_terminal_guidance_diag", None)
            if guidance_diag is not None:
                guidance_diag_rows.append(
                    {
                        "global_step": int(getattr(self, "global_step", 0)),
                        "mode": mode,
                        "denoise_idx": int(idx),
                        **guidance_diag,
                    }
                )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
            self._update_nft_state(nft_state, idx, x_t_prev, v_t, x_t, sample_method)
            log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)
            # store
            values.append(value_t)
            chains.append(x_t)
            log_probs.append(log_prob)
        self._terminal_guidance_active = False
        x_0 = x_t
        chains = torch.stack(chains, dim=1)
        old_means = torch.stack(old_means, dim=1).detach()
        old_stds = torch.stack(old_stds, dim=1).detach()
        # post process for logprob
        log_probs = torch.stack(log_probs, dim=1)[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        if self.config.joint_logprob:
            log_probs = log_probs.mean(dim=1)
        else:
            log_probs = log_probs[
                torch.arange(log_probs.shape[0]),
                denoise_inds[:, 0],
            ]
        # post process for value
        if self.use_vlm_value:
            values = values_vlm[:, None]
        else:
            values = torch.stack(values, dim=1).mean(dim=-1, keepdim=True)
        result = {
            "actions": x_0,
            "chains": chains,
            "prev_logprobs": log_probs,
            "prev_values": values,
            "denoise_inds": denoise_inds,
            "old_means": old_means,
            "old_stds": old_stds,
        }
        if collect_nft_state:
            result.update(nft_state)
            result["nft_x0"] = x_0.detach()
        if guidance_diag_rows:
            self._write_guidance_diagnostics(guidance_diag_rows)
        return result

    def _get_timesteps(self, denoise_steps, device):
        timesteps = torch.linspace(1, 1 / denoise_steps, denoise_steps, device=device)
        timesteps = torch.cat([timesteps, torch.tensor([0.0], device=device)])
        return timesteps




    def _init_score_flow_alpha_net(self):
        score_flow_mode = getattr(self.config, "score_flow_mode", "none")
        self.score_flow_diffusion_net = None
        if score_flow_mode not in {"learned_alpha", "value_conditioned"}:
            self.score_flow_alpha_net = None
            return
        input_dim = 2 if score_flow_mode == "value_conditioned" else 1
        hidden_dim = int(getattr(self.config, "score_flow_alpha_hidden_dim", 16))
        init_bias = float(getattr(self.config, "score_flow_alpha_init_bias", -2.0))
        self.score_flow_alpha_net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Softplus(),
        )
        torch.nn.init.constant_(self.score_flow_alpha_net[-2].weight, 0.0)
        torch.nn.init.constant_(self.score_flow_alpha_net[-2].bias, init_bias)
        self.score_flow_alpha_net._fsdp_wrap_name = "score_flow_alpha_net"
        diffusion_mode = getattr(self.config, "score_flow_diffusion_mode", "none")
        if score_flow_mode == "value_conditioned":
            if diffusion_mode == "modulate_std":
                diffusion_hidden_dim = int(getattr(self.config, "score_flow_diffusion_hidden_dim", 16))
                self.score_flow_diffusion_net = torch.nn.Sequential(
                    torch.nn.Linear(2, diffusion_hidden_dim),
                    torch.nn.SiLU(),
                    torch.nn.Linear(diffusion_hidden_dim, 1),
                )
                torch.nn.init.constant_(self.score_flow_diffusion_net[-1].weight, 0.0)
                torch.nn.init.constant_(self.score_flow_diffusion_net[-1].bias, 0.0)
                self.score_flow_diffusion_net._fsdp_wrap_name = "score_flow_diffusion_net"
            elif diffusion_mode != "none":
                raise ValueError(f"Unsupported score_flow_diffusion_mode={diffusion_mode}")

        ref_params = [
            param for name, param in self.named_parameters()
            if not name.startswith("score_flow_alpha_net.")
            and not name.startswith("score_flow_diffusion_net.")
            and torch.is_floating_point(param)
        ]
        if ref_params:
            ref_param = next(
                (param for param in ref_params if param.dtype in (torch.bfloat16, torch.float16)),
                ref_params[0],
            )
            self.score_flow_alpha_net.to(device=ref_param.device, dtype=ref_param.dtype)
            if self.score_flow_diffusion_net is not None:
                self.score_flow_diffusion_net.to(device=ref_param.device, dtype=ref_param.dtype)

    def _score_flow_time(self, x_t, t_input):
        if t_input.ndim == x_t.ndim:
            t_scalar = t_input[:, 0, 0].detach()
            t_expanded = t_input.detach()
        else:
            t_scalar = t_input.detach()
            t_expanded = t_scalar[:, None, None].expand_as(x_t)
        return t_scalar, t_expanded

    def _score_flow_clip(self, score_t):
        score_t = torch.nan_to_num(score_t, nan=0.0, posinf=0.0, neginf=0.0)
        raw_norm = torch.linalg.vector_norm(score_t.reshape(score_t.shape[0], -1), dim=1).clamp_min(1.0e-12)
        clip_norm = float(getattr(self.config, "score_flow_clip_norm", 10.0))
        if clip_norm > 0:
            norm_scale = (clip_norm / raw_norm).clamp(max=1.0)
            score_t = score_t * norm_scale.reshape(-1, *([1] * (score_t.ndim - 1)))
        clipped_norm = torch.linalg.vector_norm(score_t.reshape(score_t.shape[0], -1), dim=1)
        return score_t, raw_norm, clipped_norm

    def _score_flow_value_feat(self, value_t, dtype):
        if self.score_flow_alpha_net is None:
            raise RuntimeError("score_flow_alpha_net is only available for learned_alpha/value_conditioned mode")
        alpha_param = next(self.score_flow_alpha_net.parameters())
        if value_t is None:
            return torch.zeros((1, 1), device=alpha_param.device, dtype=alpha_param.dtype)
        value_scale = float(getattr(self.config, "score_flow_value_scale", 10.0))
        value_scale = max(value_scale, 1.0e-12)
        value_feat = torch.tanh(value_t.detach().reshape(-1, 1) / value_scale)
        return value_feat.to(device=alpha_param.device, dtype=alpha_param.dtype)

    def _score_flow_alpha(self, t_scalar, dtype, value_feat=None):
        if self.score_flow_alpha_net is None:
            raise RuntimeError("score_flow_alpha_net is only available for learned_alpha/value_conditioned mode")
        alpha_param = next(self.score_flow_alpha_net.parameters())
        t_in = t_scalar.detach().reshape(-1, 1).to(
            device=alpha_param.device, dtype=alpha_param.dtype
        )
        if getattr(self.config, "score_flow_mode", "none") == "value_conditioned":
            if value_feat is None:
                value_feat = torch.zeros_like(t_in)
            else:
                value_feat = value_feat.to(device=alpha_param.device, dtype=alpha_param.dtype)
                if value_feat.shape[0] == 1 and t_in.shape[0] != 1:
                    value_feat = value_feat.expand(t_in.shape[0], -1)
            net_input = torch.cat([t_in, value_feat], dim=-1)
        else:
            net_input = t_in
        learned_gain = self.score_flow_alpha_net(net_input)
        if bool(getattr(self.config, "score_flow_use_time_mask", True)):
            learned_gain = learned_gain * (1.0 - t_in).clamp_min(0.0)
        scale = float(getattr(self.config, "score_flow_scale", 1.0))
        alpha_t = scale * learned_gain
        alpha_max = float(getattr(self.config, "score_flow_alpha_max", 2.0))
        if alpha_max > 0:
            alpha_t = alpha_t.clamp(max=alpha_max)
        return alpha_t.to(dtype=dtype).reshape(-1, 1, 1)

    def _apply_score_flow_drift(self, x_t, x_t_mean, x_t_std, t_input, delta, v_t, value_t=None):
        self._last_score_flow_diag = None
        score_flow_mode = getattr(self.config, "score_flow_mode", "none")
        if score_flow_mode == "none":
            return x_t_mean, x_t_std
        if score_flow_mode not in {"learned_alpha", "sigma2_score", "value_conditioned"}:
            raise ValueError(f"Unsupported score_flow_mode={score_flow_mode}")

        t_scalar, t_expanded = self._score_flow_time(x_t, t_input)
        score_t = (t_expanded * v_t - x_t) / (1.0 - t_expanded + 1.0e-5)
        score_t, raw_norm, clipped_norm = self._score_flow_clip(score_t)

        if torch.is_tensor(delta):
            delta_t = delta.detach().to(device=x_t.device, dtype=x_t.dtype)
        else:
            delta_t = torch.tensor(delta, device=x_t.device, dtype=x_t.dtype)
        if delta_t.ndim == 1:
            delta_t = delta_t.reshape(-1, 1, 1)

        alpha_mean = 0.0
        std_mult_mean = 1.0
        if score_flow_mode in {"learned_alpha", "value_conditioned"}:
            value_feat = None
            if score_flow_mode == "value_conditioned":
                condition_source = getattr(self.config, "score_flow_condition_source", "value_t")
                if condition_source != "value_t":
                    raise ValueError(f"Unsupported score_flow_condition_source={condition_source}")
                value_feat = self._score_flow_value_feat(value_t, x_t.dtype)
            alpha_t = self._score_flow_alpha(t_scalar, x_t.dtype, value_feat)
            score_increment = delta_t * alpha_t * score_t
            alpha_mean = float(alpha_t.detach().float().mean().cpu().item())
            if (
                score_flow_mode == "value_conditioned"
                and getattr(self.config, "score_flow_diffusion_mode", "none") == "modulate_std"
                and self.score_flow_diffusion_net is not None
            ):
                diffusion_param = next(self.score_flow_diffusion_net.parameters())
                t_scalar_in = t_scalar.detach().reshape(-1, 1).to(
                    device=diffusion_param.device, dtype=diffusion_param.dtype
                )
                value_feat_in = value_feat.to(device=diffusion_param.device, dtype=diffusion_param.dtype)
                if value_feat_in.shape[0] == 1 and t_scalar_in.shape[0] != 1:
                    value_feat_in = value_feat_in.expand(t_scalar_in.shape[0], -1)
                diffusion_input = torch.cat([t_scalar_in, value_feat_in], dim=-1)
                log_resid = self.score_flow_diffusion_net(diffusion_input)
                log_scale_max = float(getattr(self.config, "score_flow_diffusion_log_scale_max", 0.5))
                log_resid = log_scale_max * torch.tanh(log_resid)
                std_mult = torch.exp(log_resid).to(dtype=x_t_std.dtype).reshape(-1, 1, 1)
                min_std = float(getattr(self.config, "score_flow_diffusion_min_std", 1.0e-4))
                x_t_std = (x_t_std * std_mult).clamp_min(min_std)
                std_mult_mean = float(std_mult.detach().float().mean().cpu().item())
        else:
            # x_t_std is the transition std.  Adding (sigma^2 / 2dt) * score * dt
            # to the transition mean is equivalent to 0.5 * sigma^2 * score.
            scale = float(getattr(self.config, "score_flow_scale", 1.0))
            score_increment = scale * 0.5 * x_t_std.detach().pow(2) * score_t
        self._last_score_flow_diag = {
            "score_flow_mode": score_flow_mode,
            "score_flow_scale": float(getattr(self.config, "score_flow_scale", 1.0)),
            "score_flow_alpha": alpha_mean,
            "score_norm": float(raw_norm.detach().float().mean().cpu().item()),
            "score_norm_clipped": float(clipped_norm.detach().float().mean().cpu().item()),
            "score_increment_norm": float(
                score_increment.detach().float().reshape(score_increment.shape[0], -1).norm(dim=1).mean().cpu().item()
            ),
            "transition_std_mean": float(x_t_std.detach().float().mean().cpu().item()),
            "score_flow_std_mult_mean": std_mult_mean,
        }
        return x_t_mean + score_increment, x_t_std


    def _terminal_guidance_enabled(self):
        return (
            bool(getattr(self, "_terminal_guidance_active", False))
            and bool(getattr(self.config, "add_value_head", False))
            and getattr(self.config, "guidance_mode", "none") != "none"
            and float(getattr(self.config, "guidance_beta", 0.0)) != 0.0
        )

    def _compute_value_from_suffix_for_guidance(self, suffix_out):
        if self.config.chunk_critic_input:
            suffix_out_value = torch.mean(
                suffix_out[:, : self.config.action_chunk], dim=1, keepdim=False
            )
        else:
            suffix_out_value = torch.mean(suffix_out, dim=1, keepdim=False)
        return self.value_head(suffix_out_value.to(dtype=torch.float32))[:, 0]

    def _clip_guidance_grad(self, grad):
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        flat = grad.reshape(grad.shape[0], -1)
        raw_norm = torch.linalg.vector_norm(flat, dim=1).clamp_min(1.0e-12)
        clip_norm = float(getattr(self.config, "guidance_grad_clip_norm", 1.0))
        if clip_norm > 0:
            scale = (clip_norm / raw_norm).clamp(max=1.0)
            grad = grad * scale.reshape(-1, *([1] * (grad.ndim - 1)))
        clipped_norm = torch.linalg.vector_norm(
            grad.reshape(grad.shape[0], -1), dim=1
        )
        return grad, raw_norm, clipped_norm

    def _apply_terminal_guidance(
        self,
        x_t,
        x_t_mean,
        x_t_std,
        idx,
        state,
        prefix_pad_masks,
        past_key_values,
        t_input,
        delta,
        sample_method,
    ):
        self._last_terminal_guidance_diag = None
        if not self._terminal_guidance_enabled():
            return x_t_mean

        guidance_mode = getattr(self.config, "guidance_mode", "none")
        beta = float(getattr(self.config, "guidance_beta", 0.0))
        update_rule = getattr(self.config, "guidance_update_rule", "euler")
        if guidance_mode == "fmtt_terminal_x1":
            update_rule = "sigma2_tilt"

        grad = None
        value_xt = torch.zeros(x_t.shape[0], device=x_t.device)
        value_x1 = torch.zeros(x_t.shape[0], device=x_t.device)

        param_states = [(param, param.requires_grad) for param in self.parameters()]
        with torch.enable_grad():
            try:
                for param, _ in param_states:
                    param.requires_grad_(False)

                x_base = x_t.detach().requires_grad_(True)
                if t_input.ndim == x_base.ndim:
                    t_scalar = t_input[:, 0, 0].detach()
                    t_expanded = t_input.detach()
                else:
                    t_scalar = t_input.detach()
                    t_expanded = t_scalar[:, None, None].expand_as(x_base)

                suffix_xt = self.get_suffix_out(
                    state, prefix_pad_masks, past_key_values, x_base, t_scalar
                )
                value_xt = self._compute_value_from_suffix_for_guidance(suffix_xt)

                v_guidance, _ = self.get_velocity(
                    state, x_base, t_scalar, prefix_pad_masks, past_key_values
                )
                x1_hat = x_base + v_guidance * (1 - t_expanded)
                t_terminal = torch.zeros_like(t_scalar)
                suffix_x1 = self.get_suffix_out(
                    state, prefix_pad_masks, past_key_values, x1_hat, t_terminal
                )
                value_x1 = self._compute_value_from_suffix_for_guidance(suffix_x1)

                if guidance_mode == "direct_xt":
                    objective = value_xt
                elif guidance_mode in {"terminal_x1", "fmtt_terminal_x1"}:
                    objective = value_x1
                else:
                    return x_t_mean

                grad = torch.autograd.grad(
                    objective.sum(),
                    x_base,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )[0]
            finally:
                for param, requires_grad in param_states:
                    param.requires_grad_(requires_grad)

        if grad is None:
            return x_t_mean

        grad = grad.detach()
        grad, grad_norm_raw, grad_norm_clipped = self._clip_guidance_grad(grad)
        if update_rule == "sigma2_tilt":
            guidance_scale = x_t_std.detach().pow(2)
        else:
            guidance_scale = delta.detach()
        guided_mean = x_t_mean + beta * guidance_scale * grad

        self._last_terminal_guidance_diag = {
            "guidance_mode": guidance_mode,
            "guidance_update_rule": update_rule,
            "guidance_beta": beta,
            "sample_method": sample_method,
            "denoise_idx_tensor_mean": float(idx.float().mean().detach().cpu().item()),
            "critic_value_xt": float(value_xt.detach().float().mean().cpu().item()),
            "critic_value_x1": float(value_x1.detach().float().mean().cpu().item()),
            "guidance_grad_norm": float(
                grad_norm_raw.detach().float().mean().cpu().item()
            ),
            "guidance_grad_norm_clipped": float(
                grad_norm_clipped.detach().float().mean().cpu().item()
            ),
            "velocity_norm": float(v_guidance.detach().float().norm(dim=-1).mean().cpu().item()),
            "action_norm": float(x_t.detach().float().norm(dim=-1).mean().cpu().item()),
            "guidance_scale": float(guidance_scale.detach().float().mean().cpu().item()),
        }
        return guided_mean

    def _write_guidance_diagnostics(self, rows):
        diag_path = getattr(self.config, "guidance_diag_path", "")
        if not diag_path:
            return
        path = Path(str(diag_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "global_step",
            "mode",
            "denoise_idx",
            "guidance_mode",
            "guidance_update_rule",
            "guidance_beta",
            "sample_method",
            "denoise_idx_tensor_mean",
            "critic_value_xt",
            "critic_value_x1",
            "guidance_grad_norm",
            "guidance_grad_norm_clipped",
            "velocity_norm",
            "action_norm",
            "guidance_scale",
            "score_flow_mode",
            "score_flow_scale",
            "score_flow_alpha",
            "score_norm",
            "score_norm_clipped",
            "score_increment_norm",
            "transition_std_mean",
            "score_flow_std_mult_mean",
        ]
        write_header = not path.exists()
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})


    def sample_mean_var_val(
        self,
        x_t,
        idx,
        state,
        prefix_pad_masks,
        past_key_values,
        sample_method,
        denoise_steps,
        compute_values=True,
    ):
        """
        Sample the mean, variance and value of the action at a given timestep.
        Rollout sample (idx is int) and actor get_log_prob_value (idx is tensor)
        will load this function. `sample_method` is one of flow_ode/flow_sde/
        flow_cps/flow_noise.
        """
        # expand the shape
        bsize = state.shape[0]
        device = state.device
        if isinstance(idx, int):
            idx = torch.tensor(idx).expand(bsize)
        # build parameters
        noise_level = self._get_noise_level(device=device, dtype=x_t.dtype)
        timesteps = self._get_timesteps(denoise_steps, device)
        # input parameters
        t_input = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]
        # velocity prediction
        v_t, suffix_out = self.get_velocity(
            state, x_t, t_input, prefix_pad_masks, past_key_values
        )
        # value prediction
        if (
            self.config.add_value_head
            and compute_values
            and not self.config.value_after_vlm
        ):
            value_t = self._compute_value_from_suffix(suffix_out)
        else:
            value_t = torch.zeros((bsize), device=device)
        # sample mean and variance
        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)

        if sample_method == "flow_ode":
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = torch.zeros_like(t_input)
        elif sample_method == "flow_sde":
            denom_timesteps = torch.where(timesteps == 1, timesteps[1], timesteps)
            sigma_ratio = timesteps / (1 - denom_timesteps)
            sigmas = noise_level * torch.sqrt(sigma_ratio)[:-1]
            sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
            x0_weight = torch.ones_like(t_input) - (t_input - delta)
            x1_weight = t_input - delta - sigma_i**2 * delta / (2 * t_input)
            x_t_std = torch.sqrt(delta) * sigma_i
        elif sample_method == "flow_cps":
            pi = torch.pi
            cos_term = torch.cos(pi * noise_level / 2).to(device)
            sin_term = torch.sin(pi * noise_level / 2).to(device)
            x0_weight = torch.ones_like(t_input) - (t_input - delta)
            x1_weight = (t_input - delta) * cos_term
            x_t_std = (t_input - delta) * sin_term
        elif sample_method == "flow_noise":
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = self.noise_head(suffix_out)
        else:
            raise ValueError(f"Invalid noise method: {sample_method}")
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        x_t_mean, x_t_std = self._apply_score_flow_drift(
            x_t,
            x_t_mean,
            x_t_std,
            t_input,
            delta,
            v_t,
            value_t,
        )
        x_t_mean = self._apply_terminal_guidance(
            x_t,
            x_t_mean,
            x_t_std,
            idx,
            state,
            prefix_pad_masks,
            past_key_values,
            t_input,
            delta,
            sample_method,
        )
        return x_t_mean, x_t_std, value_t, v_t

    def get_suffix_out(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(state, x_t, timestep)
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = (
            "eager"  # noqa: SLF001
        )

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return suffix_out

    def get_velocity(self, state, x_t, timestep, prefix_pad_masks, past_key_values):
        """Compute velocity prediction v_t and raw suffix_out at a given timestep."""
        suffix_out = self.get_suffix_out(
            state, prefix_pad_masks, past_key_values, x_t, timestep
        )
        v_t = self.action_out_proj(suffix_out)
        return v_t, suffix_out

    def _build_prefix_cache(self, images, img_masks, lang_tokens, lang_masks):
        """Embed prefix tokens and compute KV cache for efficient suffix generation."""
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        (prefix_output, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return prefix_output, prefix_pad_masks, past_key_values

    def _compute_value_from_suffix(self, suffix_out):
        """Compute value from suffix output using value head."""
        if self.config.chunk_critic_input:
            suffix_out_value = torch.mean(
                suffix_out[:, : self.config.action_chunk], dim=1, keepdim=False
            )
        else:
            suffix_out_value = torch.mean(suffix_out, dim=1, keepdim=False)
        if self.config.detach_critic_input:
            suffix_out_value = suffix_out_value.detach()
        return self.value_head(suffix_out_value)[:, 0]

    # TODO: to check potential nan here
    def get_logprob_norm(self, sample, mu, sigma):
        # logprob = log p(x|mu,sigma) = -log(sigma) - 0.5 * log(2 * pi) - 0.5 * ((x - mu) / sigma) ** 2
        if self.config.safe_get_logprob:
            log_prob = -torch.pow((sample - mu), 2)
        else:
            mask = sigma == 0
            sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
            constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
                2 * torch.pi * torch.ones_like(sample)
            )
            exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
            log_prob = constant_term + exponent_term
            log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
        return log_prob

    def _tr_action_slice(self, tensor):
        return tensor[..., : self.config.action_chunk, : self.config.action_env_dim]

    def _tr_embed_action_slice(self, tensor, reference):
        full = torch.zeros_like(reference)
        full[:, : self.config.action_chunk, : self.config.action_env_dim] = tensor.to(
            device=reference.device,
            dtype=reference.dtype,
        )
        return full

    def _tr_zero_outputs(self, device, mode=None, beta=None):
        tr_mode = (
            getattr(self.config, "tr_penalty_mode", "none")
            if mode is None
            else mode
        )
        tr_beta = (
            float(getattr(self.config, "tr_penalty_beta", 0.0))
            if beta is None
            else beta
        )
        return torch.zeros((), device=device, dtype=torch.float32), {
            "tr_mode": tr_mode,
            "tr_beta": tr_beta,
            "scalar_l2_disp": 0.0,
            "pullback_disp": 0.0,
        }

    def _tr_sample_method_for_step(self, denoise_inds, step):
        if denoise_inds.ndim == 1:
            active = denoise_inds == step
        else:
            active = denoise_inds[:, min(step, denoise_inds.shape[1] - 1)] == step
        if bool(active.all().detach().cpu().item()):
            return self.config.noise_method
        if bool((~active).all().detach().cpu().item()):
            return "flow_ode"
        raise ValueError(
            "tr_penalty requires a batch-uniform denoise sample method per step"
        )

    def _tr_transition_mean(
        self,
        x_t,
        step,
        state,
        prefix_pad_masks,
        past_key_values,
        sample_method,
        denoise_steps,
        compute_values,
    ):
        idx = torch.full((x_t.shape[0],), step, device=x_t.device, dtype=torch.long)
        x_t_mean, _, _, _ = self.sample_mean_var_val(
            x_t,
            idx,
            state,
            prefix_pad_masks,
            past_key_values,
            sample_method,
            denoise_steps,
            compute_values,
        )
        return x_t_mean

    def _tr_frozen_jvp(
        self,
        x_t,
        vector,
        step,
        state,
        prefix_pad_masks,
        past_key_values,
        sample_method,
        denoise_steps,
        compute_values,
    ):
        eps = max(
            float(getattr(self.config, "tr_pullback_fd_eps", 1.0e-2)),
            1.0e-12,
        )
        x_base = x_t.detach()
        vector_full = self._tr_embed_action_slice(vector.detach(), x_base)
        with torch.no_grad():
            mean_base = self._tr_transition_mean(
                x_base,
                step,
                state,
                prefix_pad_masks,
                past_key_values,
                sample_method,
                denoise_steps,
                compute_values,
            )
            mean_perturbed = self._tr_transition_mean(
                x_base + eps * vector_full,
                step,
                state,
                prefix_pad_masks,
                past_key_values,
                sample_method,
                denoise_steps,
                compute_values,
            )
            jvp = (
                self._tr_action_slice(mean_perturbed)
                - self._tr_action_slice(mean_base)
            ) / eps
        return torch.nan_to_num(jvp.detach(), nan=0.0, posinf=0.0, neginf=0.0)

    def _tr_frozen_vjp(
        self,
        x_t,
        vector,
        step,
        state,
        prefix_pad_masks,
        past_key_values,
        sample_method,
        denoise_steps,
        compute_values,
    ):
        with torch.enable_grad():
            x_base = x_t.detach().requires_grad_(True)
            mean = self._tr_transition_mean(
                x_base,
                step,
                state,
                prefix_pad_masks,
                past_key_values,
                sample_method,
                denoise_steps,
                compute_values,
            )
            mean_slice = self._tr_action_slice(mean)
            grad = torch.autograd.grad(
                mean_slice,
                x_base,
                grad_outputs=vector.detach().to(
                    device=mean_slice.device,
                    dtype=mean_slice.dtype,
                ),
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
        if grad is None:
            grad = torch.zeros_like(x_t)
        grad = self._tr_action_slice(grad)
        return torch.nan_to_num(grad.detach(), nan=0.0, posinf=0.0, neginf=0.0)

    def _compute_terminal_pullback_surrogate(
        self,
        chains,
        delta_means,
        state,
        prefix_pad_masks,
        past_key_values,
        sample_methods,
        denoise_steps,
        compute_values,
        beta,
    ):
        frozen_state = state.detach()
        frozen_prefix_pad_masks = prefix_pad_masks.detach()
        frozen_past_key_values = tree_map(
            lambda x: x.detach() if torch.is_tensor(x) else x,
            past_key_values,
        )

        with torch.no_grad():
            delta_a = torch.zeros_like(delta_means[0].detach())
            for step in range(denoise_steps):
                delta_a = self._tr_frozen_jvp(
                    chains[:, step],
                    delta_a,
                    step,
                    frozen_state,
                    frozen_prefix_pad_masks,
                    frozen_past_key_values,
                    sample_methods[step],
                    denoise_steps,
                    compute_values,
                ) + delta_means[step].detach()
            pullback_disp = torch.linalg.vector_norm(
                delta_a.detach().float().reshape(delta_a.shape[0], -1),
                dim=1,
            )

        param_states = [(param, param.requires_grad) for param in self.parameters()]
        adjoints = [None] * denoise_steps
        try:
            for param, _ in param_states:
                param.requires_grad_(False)
            w = delta_a.detach()
            for step in reversed(range(denoise_steps)):
                # For delta_a_{i+1} = J_i delta_a_i + delta_m_i, the local
                # adjoint for delta_m_i is the current terminal adjoint w.
                adjoints[step] = w.detach()
                if step > 0:
                    w = self._tr_frozen_vjp(
                        chains[:, step],
                        w,
                        step,
                        frozen_state,
                        frozen_prefix_pad_masks,
                        frozen_past_key_values,
                        sample_methods[step],
                        denoise_steps,
                        compute_values,
                    )
        finally:
            for param, requires_grad in param_states:
                param.requires_grad_(requires_grad)

        surrogate_terms = []
        for step, adjoint in enumerate(adjoints):
            surrogate_terms.append(
                (adjoint.float() * delta_means[step].float())
                .reshape(delta_means[step].shape[0], -1)
                .sum(dim=1)
            )
        surrogate_linear = torch.stack(surrogate_terms, dim=1).sum(dim=1).mean()
        penalty_value = 0.5 * beta * pullback_disp.pow(2).mean()
        # Forward value is beta * 0.5 * ||delta_a||^2, while the linear
        # surrogate contributes the desired frozen-sensitivity gradient
        # beta * sum_i <g_i, delta_m_i>.
        penalty = penalty_value + beta * (
            surrogate_linear - surrogate_linear.detach()
        )
        return penalty, float(pullback_disp.mean().detach().cpu().item())

    def _compute_tr_penalty(
        self,
        chains,
        denoise_inds,
        old_means,
        state,
        prefix_pad_masks,
        past_key_values,
        compute_values,
    ):
        mode = getattr(self.config, "tr_penalty_mode", "none")
        beta = float(getattr(self.config, "tr_penalty_beta", 0.0))
        if mode not in {"none", "scalar_l2", "terminal_pullback"}:
            raise ValueError(f"Unsupported tr_penalty_mode={mode}")
        if mode == "none" or beta == 0.0 or old_means is None:
            return self._tr_zero_outputs(state.device, mode, beta)

        chains = chains.detach()
        denoise_inds = denoise_inds.to(device=chains.device)
        old_means = old_means.detach().to(device=chains.device)
        denoise_steps = min(
            int(old_means.shape[1]),
            int(chains.shape[1] - 1),
            int(
                denoise_inds.shape[1]
                if denoise_inds.ndim > 1
                else old_means.shape[1]
            ),
        )
        if denoise_steps <= 0:
            return self._tr_zero_outputs(state.device, mode, beta)

        sample_methods = [
            self._tr_sample_method_for_step(denoise_inds, step)
            for step in range(denoise_steps)
        ]
        delta_means = []
        for step in range(denoise_steps):
            x_t_mean = self._tr_transition_mean(
                chains[:, step],
                step,
                state,
                prefix_pad_masks,
                past_key_values,
                sample_methods[step],
                denoise_steps,
                compute_values,
            )
            old_mean = old_means[:, step].to(
                device=x_t_mean.device,
                dtype=x_t_mean.dtype,
            )
            delta_means.append(
                self._tr_action_slice(x_t_mean) - self._tr_action_slice(old_mean)
            )

        sum_delta = torch.stack(delta_means, dim=1).sum(dim=1)
        sum_delta_flat = sum_delta.float().reshape(sum_delta.shape[0], -1)
        scalar_l2_disp = torch.linalg.vector_norm(
            sum_delta_flat.detach(),
            dim=1,
        )
        pullback_disp = 0.0

        if mode == "scalar_l2":
            penalty = 0.5 * beta * sum_delta_flat.pow(2).sum(dim=1).mean()
        else:
            penalty, pullback_disp = self._compute_terminal_pullback_surrogate(
                chains,
                delta_means,
                state,
                prefix_pad_masks,
                past_key_values,
                sample_methods,
                denoise_steps,
                compute_values,
                beta,
            )

        tr_diag = {
            "tr_mode": mode,
            "tr_beta": beta,
            "scalar_l2_disp": float(scalar_l2_disp.mean().detach().cpu().item()),
            "pullback_disp": pullback_disp,
        }
        return penalty, tr_diag

    def _cr_reflow_enabled(self) -> bool:
        mode = str(getattr(self.config, "cr_reflow_mode", "none"))
        return mode in {"cr_reflow", "cr_reflow_no_anchor"}

    def _cr_zero_outputs(self, device):
        zero = torch.zeros((), device=device)
        return zero, {
            "cr_reflow_mode": str(getattr(self.config, "cr_reflow_mode", "none")),
            "cr_reflow_loss": 0.0,
            "cr_reflow_anchor_loss": 0.0,
            "cr_reflow_eta": 0.0,
            "cr_reflow_weight_mean": 0.0,
            "cr_reflow_weight_max": 0.0,
            "cr_reflow_weight_ess": 0.0,
            "cr_reflow_weight_kl": 0.0,
            "cr_reflow_eta_at_bound": 0.0,
            "cr_reflow_valid_fraction": 0.0,
            "cr_reflow_target_displacement": 0.0,
            "cr_reflow_policy_kl_proxy": 0.0,
            "cr_reflow_policy_displacement": 0.0,
            "cr_reflow_used_advantages": False,
        }

    def _cr_selected_transition_stats(
        self,
        tensor,
        denoise_inds,
        num_terms,
        dtype,
        device,
    ):
        if tensor is None:
            return None
        tensor = tensor.detach().to(device=device, dtype=dtype)
        if tensor.ndim < 2:
            return tensor
        batch_indices = torch.arange(tensor.shape[0], device=device)
        selected = []
        for idx in range(num_terms):
            denoise_ind = denoise_inds[:, idx].to(device=device)
            denoise_ind = denoise_ind.clamp(min=0, max=tensor.shape[1] - 1)
            selected.append(tensor[batch_indices, denoise_ind])
        return torch.stack(selected, dim=1)

    def _cr_prepare_mask(
        self,
        mask,
        batch_size,
        num_terms,
        device,
    ):
        if mask is None:
            full_mask = torch.ones(batch_size, num_terms, device=device, dtype=torch.float32)
            return full_mask, 1.0

        prepared = mask.detach().to(device=device)
        while prepared.ndim > 2:
            prepared = prepared.mean(dim=-1)
        if prepared.ndim == 1:
            prepared = prepared[:, None]
        if prepared.shape[0] != batch_size:
            prepared = prepared.reshape(batch_size, -1)
        if prepared.shape[1] == 1 and num_terms > 1:
            prepared = prepared.expand(-1, num_terms)
        elif prepared.shape[1] != num_terms:
            prepared = prepared.mean(dim=1, keepdim=True).expand(-1, num_terms)

        prepared = prepared[:, :num_terms].float()
        prepared = torch.where(torch.isfinite(prepared), prepared, torch.zeros_like(prepared))
        prepared = (prepared > 0).float()
        valid_fraction = float(prepared.mean().detach().cpu().item()) if prepared.numel() else 0.0
        return prepared, valid_fraction

    def _cr_prepare_advantages(
        self,
        advantages,
        values,
        num_terms,
        device,
        dtype,
        mask,
    ):
        batch_size = values.shape[0]
        if advantages is None:
            return torch.zeros(batch_size, num_terms, device=device, dtype=dtype), False

        adv = advantages.detach().to(device=device, dtype=dtype)
        while adv.ndim > 2:
            adv = adv.mean(dim=-1)
        if adv.ndim == 1:
            adv = adv[:, None]
        if adv.shape[0] != batch_size:
            adv = adv.reshape(batch_size, -1)
        if adv.shape[1] == 1 and num_terms > 1:
            adv = adv.expand(-1, num_terms)
        elif adv.shape[1] != num_terms:
            adv = adv.mean(dim=1, keepdim=True).expand(-1, num_terms)
        adv = adv[:, :num_terms]

        mask_bool = mask > 0
        finite = torch.isfinite(adv) & mask_bool
        if not bool(finite.any()):
            return torch.zeros_like(adv), False
        finite_values = adv[finite]
        adv_mean = finite_values.mean()
        adv_std = finite_values.std(unbiased=False).clamp_min(
            float(getattr(self.config, "cr_reflow_eps", 1.0e-6))
        )
        adv = torch.where(torch.isfinite(adv), adv, adv_mean)
        adv = (adv - adv_mean) / adv_std
        adv = torch.where(mask_bool, adv, torch.zeros_like(adv))
        return adv, True

    def _cr_weights_from_advantages(self, advantages, used_advantages, mask):
        mask = mask.to(device=advantages.device, dtype=torch.float32)
        valid = mask > 0
        num_valid = int(valid.sum().detach().cpu().item())
        if num_valid == 0:
            return torch.zeros_like(advantages, dtype=torch.float32), 0.0, 0.0, 0.0, 0.0

        if not used_advantages:
            return mask, 0.0, 1.0, 0.0, 0.0

        flat_adv = advantages.float()[valid]
        num_weights = int(flat_adv.numel())
        if float(flat_adv.std(unbiased=False).detach().cpu().item()) <= 1.0e-8:
            return mask, 0.0, 1.0, 0.0, 0.0

        epsilon = max(float(getattr(self.config, "cr_reflow_kl_epsilon", 0.05)), 0.0)
        eta_min = max(float(getattr(self.config, "cr_reflow_eta_min", 0.01)), 1.0e-8)
        eta_max = max(float(getattr(self.config, "cr_reflow_eta_max", 10.0)), eta_min)

        def kl_for_eta(eta: float) -> torch.Tensor:
            probs = torch.softmax(flat_adv / eta, dim=0)
            return (probs * (torch.log(probs.clamp_min(1.0e-12)) + math.log(num_weights))).sum()

        use_uniform_fallback = epsilon == 0.0
        if not use_uniform_fallback:
            use_uniform_fallback = (
                float(kl_for_eta(eta_max).detach().cpu().item()) > epsilon
            )

        eta_at_bound = 0.0
        if use_uniform_fallback:
            eta = eta_max
            eta_at_bound = 1.0
        elif float(kl_for_eta(eta_min).detach().cpu().item()) <= epsilon:
            eta = eta_min
            eta_at_bound = 1.0
        else:
            lo = eta_min
            hi = eta_max
            for _ in range(32):
                mid = 0.5 * (lo + hi)
                if float(kl_for_eta(mid).detach().cpu().item()) > epsilon:
                    lo = mid
                else:
                    hi = mid
            eta = hi

        weights = torch.zeros_like(advantages, dtype=flat_adv.dtype)
        if use_uniform_fallback:
            weights[valid] = 1.0
        else:
            probs = torch.softmax(flat_adv / eta, dim=0)
            weights[valid] = probs * float(num_weights)
        clip_value = float(getattr(self.config, "cr_reflow_weight_clip", 10.0))
        if clip_value > 0.0:
            weights = weights.clamp(max=clip_value)
        valid_mean = weights[valid].mean().clamp_min(1.0e-12)
        weights = torch.where(valid, weights / valid_mean, torch.zeros_like(weights))

        flat_weights = weights.float()[valid]
        final_probs = flat_weights / flat_weights.sum().clamp_min(1.0e-12)
        weight_kl = (
            final_probs
            * (torch.log(final_probs.clamp_min(1.0e-12)) + math.log(num_weights))
        ).sum()
        if float(weight_kl.detach().cpu().item()) > epsilon:
            weights = mask
            flat_weights = weights[valid]
            final_probs = flat_weights / flat_weights.sum().clamp_min(1.0e-12)
            weight_kl = (
                final_probs
                * (torch.log(final_probs.clamp_min(1.0e-12)) + math.log(num_weights))
            ).sum()
            eta_at_bound = 1.0
        ess = flat_weights.sum().pow(2) / flat_weights.pow(2).sum().clamp_min(1.0e-12)
        ess_fraction = float((ess / max(num_weights, 1)).detach().cpu().item())
        return (
            weights,
            float(eta),
            ess_fraction,
            float(weight_kl.detach().cpu().item()),
            eta_at_bound,
        )

    def _compute_cr_reflow_loss(
        self,
        chains,
        denoise_inds,
        transition_means,
        transition_stds,
        old_means,
        old_stds,
        advantages,
        values,
        mask,
    ):
        if not self._cr_reflow_enabled():
            return self._cr_zero_outputs(chains.device)
        if transition_means is None or transition_stds is None:
            return self._cr_zero_outputs(chains.device)

        mode = str(getattr(self.config, "cr_reflow_mode", "none"))
        device = transition_means.device
        dtype = transition_means.dtype
        num_terms = int(transition_means.shape[1])
        valid_mask, valid_fraction = self._cr_prepare_mask(
            mask,
            values.shape[0],
            num_terms,
            device,
        )
        denoise_inds = denoise_inds.to(device=device)
        next_inds = (denoise_inds[:, :num_terms] + 1).clamp(max=chains.shape[1] - 1)
        batch_indices = torch.arange(chains.shape[0], device=device)
        targets = []
        for idx in range(num_terms):
            targets.append(chains[batch_indices, next_inds[:, idx]].detach())
        targets = torch.stack(targets, dim=1).to(device=device, dtype=dtype)

        selected_old_means = self._cr_selected_transition_stats(
            old_means,
            denoise_inds,
            num_terms,
            dtype,
            device,
        )
        selected_old_stds = self._cr_selected_transition_stats(
            old_stds,
            denoise_inds,
            num_terms,
            dtype,
            device,
        )
        if selected_old_stds is None:
            selected_old_stds = transition_stds.detach()
        scale = self._tr_action_slice(selected_old_stds).clamp_min(
            float(getattr(self.config, "cr_reflow_eps", 1.0e-6))
        )

        action_means = self._tr_action_slice(transition_means)
        action_targets = self._tr_action_slice(targets)
        reflow_terms = ((action_means - action_targets) / scale).float().pow(2)
        reflow_terms = reflow_terms.reshape(reflow_terms.shape[0], num_terms, -1).mean(dim=2)

        normalized_adv, used_advantages = self._cr_prepare_advantages(
            advantages,
            values,
            num_terms,
            device,
            dtype,
            valid_mask,
        )
        weights, eta, ess_fraction, weight_kl, eta_at_bound = (
            self._cr_weights_from_advantages(
                normalized_adv,
                used_advantages,
                valid_mask,
            )
        )
        weight_sum = weights.float().sum().clamp_min(1.0e-12)
        weighted_reflow = (weights.float() * reflow_terms).sum() / weight_sum
        target_displacement_terms = (
            (action_means.detach() - action_targets.detach()) / scale
        ).float()
        target_displacement_terms = target_displacement_terms.reshape(
            target_displacement_terms.shape[0],
            num_terms,
            -1,
        ).norm(dim=2)
        target_displacement = (
            weights.float() * target_displacement_terms
        ).sum() / weight_sum

        anchor_loss = torch.zeros((), device=device, dtype=weighted_reflow.dtype)
        policy_kl_proxy = torch.zeros((), device=device, dtype=weighted_reflow.dtype)
        policy_displacement = torch.zeros((), device=device, dtype=weighted_reflow.dtype)
        anchor_beta = float(getattr(self.config, "cr_reflow_anchor_beta", 0.1))
        if selected_old_means is not None:
            action_old_means = self._tr_action_slice(selected_old_means)
            policy_delta = ((action_means - action_old_means) / scale).float()
            policy_delta = policy_delta.reshape(policy_delta.shape[0], num_terms, -1)
            policy_mean_shift_terms = policy_delta.pow(2).mean(dim=2)
            policy_kl_proxy = (
                0.5 * (weights.float() * policy_mean_shift_terms).sum() / weight_sum
            )
            policy_displacement_terms = policy_delta.detach().norm(dim=2)
            policy_displacement = (
                weights.float() * policy_displacement_terms
            ).sum() / weight_sum
            if mode == "cr_reflow" and anchor_beta > 0.0:
                anchor_loss = (
                    weights.float() * policy_mean_shift_terms
                ).sum() / weight_sum

        loss = weighted_reflow + anchor_beta * anchor_loss
        diag = {
            "cr_reflow_mode": mode,
            "cr_reflow_loss": float(weighted_reflow.detach().cpu().item()),
            "cr_reflow_anchor_loss": float(anchor_loss.detach().cpu().item()),
            "cr_reflow_eta": eta,
            "cr_reflow_weight_mean": float(weights.detach().float().mean().cpu().item()),
            "cr_reflow_weight_max": float(weights.detach().float().max().cpu().item()),
            "cr_reflow_weight_ess": ess_fraction,
            "cr_reflow_weight_kl": weight_kl,
            "cr_reflow_eta_at_bound": eta_at_bound,
            "cr_reflow_valid_fraction": valid_fraction,
            "cr_reflow_target_displacement": float(
                target_displacement.detach().cpu().item()
            ),
            "cr_reflow_policy_kl_proxy": float(policy_kl_proxy.detach().cpu().item()),
            "cr_reflow_policy_displacement": float(
                policy_displacement.detach().cpu().item()
            ),
            "cr_reflow_used_advantages": used_advantages,
        }
        return loss, diag

    def preprocess_for_train(self, data):
        return data

    def get_log_prob_value(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        chains,
        denoise_inds,
        compute_values=False,
        old_means=None,
    ):
        bsize = state.shape[0]
        batch_indices = torch.arange(bsize, device=chains.device)
        prefix_output, prefix_pad_masks, past_key_values = self._build_prefix_cache(
            images, img_masks, lang_tokens, lang_masks
        )
        chains_log_probs = []
        chains_values = []
        chains_entropy = []
        transition_means = []
        transition_stds = []

        # get log prob
        if self.config.joint_logprob:
            num_steps = self.config.num_steps
            initial_log_prob = self.get_logprob_norm(
                chains[:, 0],
                torch.zeros_like(chains[:, 0]),
                torch.ones_like(chains[:, 0]),
            )
            initial_entropy = self.gaussian_entropy(torch.ones_like(chains[:, 0]))
            chains_log_probs.append(initial_log_prob)
            chains_entropy.append(initial_entropy)
        else:
            num_steps = 1
        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[batch_indices, denoise_ind]
            chains_next = chains[batch_indices, denoise_ind + 1]
            x_t_mean, x_t_std, value_t, _ = self.sample_mean_var_val(
                chains_pre,
                denoise_ind,
                state,
                prefix_pad_masks,
                past_key_values,
                self.config.noise_method,
                self.config.num_steps,
                compute_values,
            )
            log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            entropy = self.gaussian_entropy(x_t_std)
            chains_log_probs.append(log_probs)
            chains_entropy.append(entropy)
            transition_means.append(x_t_mean)
            transition_stds.append(x_t_std)
            if not self.use_vlm_value:
                chains_values.append(value_t)
        if self.use_vlm_value:
            chains_values.append(self.get_value_from_vlm(prefix_output))
        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        chains_values = torch.stack(chains_values, dim=1)

        # entropy is only available for flow-noise method
        if self.config.noise_method == "flow_noise":
            chains_entropy = torch.stack(chains_entropy, dim=1)
        else:
            chains_entropy = torch.zeros_like(chains_log_probs)
        transition_means = torch.stack(transition_means, dim=1)
        transition_stds = torch.stack(transition_stds, dim=1)
        terminal_pullback_loss, tr_diag = self._compute_tr_penalty(
            chains,
            denoise_inds,
            old_means,
            state,
            prefix_pad_masks,
            past_key_values,
            compute_values,
        )
        return (
            chains_log_probs,
            chains_values,
            chains_entropy,
            terminal_pullback_loss,
            tr_diag,
            transition_means,
            transition_stds,
        )

    def get_value_from_vlm(self, prefix_output):
        # prefix_output:
        # pi05: [bs, (256 * 3 + 200) = 968, 2048]
        # pi0: [bs, (256 * 3 + 48) = 816, 1024]
        # token length
        if "pi05_" in self.config.config_name:
            lang_token_len = 200
            all_token_length = 968
        elif "pi0_" in self.config.config_name:
            lang_token_len = 48
            all_token_length = 816

        if self.config.value_vlm_mode == "mean_token":
            prefix_mask = (
                [True] * 256 * self.config.num_images_in_input
                + [False] * 256 * (3 - self.config.num_images_in_input)
                + [True] * lang_token_len
            )
        elif self.config.value_vlm_mode == "last_token":
            prefix_mask = [False] * (all_token_length - 1) + [True] * 1
        elif self.config.value_vlm_mode == "first_token":
            prefix_mask = [True] * 1 + [False] * (all_token_length - 1)
        prefix_out_value = prefix_output[:, prefix_mask, :]
        prefix_out_value = prefix_out_value.mean(dim=1, keepdim=False)
        prefix_out_value = prefix_out_value.to(dtype=torch.float32)
        values_vlm = self.value_head(prefix_out_value)[:, 0]
        return values_vlm

    def gaussian_entropy(self, sigma):
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        entropy = 0.5 * torch.log(2 * math.pi * math.e * (sigma_safe**2))
        return entropy

    def freeze_vlm(self):
        if self.config.train_expert_only:
            # Base freeze: paligemma (SigLIP vision encoder + Gemma)
            self.paligemma_with_expert.paligemma.eval()
            for params in self.paligemma_with_expert.paligemma.parameters():
                params.requires_grad = False

            # ========== DSRL additional freezing ==========
            if self.config.use_dsrl:
                self.logger.info(
                    "[FREEZE_VLM] DSRL mode: freezing gemma_expert parameters"
                )
                self.paligemma_with_expert.gemma_expert.eval()
                for params in self.paligemma_with_expert.gemma_expert.parameters():
                    params.requires_grad = False

                # Freeze projection layers (used in rollout/eval but not optimized).
                # Pi0 has: action_in_proj, action_out_proj, state_proj, action_time_mlp_in/out
                # Pi0.5 has: action_in_proj, action_out_proj, time_mlp_in/out (no state_proj)
                self.logger.info(
                    "[FREEZE_VLM] DSRL mode: freezing projection layers (used in rollout/eval but not optimized)"
                )
                if self.pi05:
                    projection_names = [
                        "action_in_proj",
                        "action_out_proj",
                        "time_mlp_in",
                        "time_mlp_out",
                    ]
                else:
                    projection_names = [
                        "action_in_proj",
                        "action_out_proj",
                        "state_proj",
                        "action_time_mlp",
                    ]
                frozen_count = 0
                for name, param in self.named_parameters():
                    if any(proj_name in name for proj_name in projection_names):
                        param.requires_grad = False
                        frozen_count += 1
                        if frozen_count <= 10:  # Print first 10 for brevity
                            self.logger.info(f"  Froze: {name}")
                if frozen_count > 10:
                    self.logger.info(
                        f"  ... and {frozen_count - 10} more projection layer parameters"
                    )

                # Freeze reinflow_explore_noise_net (only used in reinflow diffuser sampling)
                if hasattr(self, "reinflow_explore_noise_net"):
                    self.logger.info(
                        "[FREEZE_VLM] DSRL mode: freezing reinflow_explore_noise_net (used in non-DSRL rollout but not optimized)"
                    )
                    self.reinflow_explore_noise_net.eval()
                    noise_net_params = 0
                    for params in self.reinflow_explore_noise_net.parameters():
                        params.requires_grad = False
                        noise_net_params += params.numel()
                    self.logger.info(
                        f"  Froze {noise_net_params:,} parameters in reinflow_explore_noise_net"
                    )

    # ===== DSRL-specific methods =====

    def sac_forward(
        self, obs=None, data=None, train=False, return_dist_params=False, **kwargs
    ):
        """SAC forward pass for DSRL.

        Args:
            obs: Observation dict (preferred, matches sac_dsrl).
                 Supports two formats:
                   1. {"images": list of tensors, "states": tensor} - internal format
                   2. {"main_images": tensor, "wrist_images": tensor, "states": tensor} - env format
            data: Dictionary containing observations (legacy, for backward compatibility).
            train: Whether to use data augmentation.
            return_dist_params: Whether to return distribution parameters for logging.

        Returns:
            actions: [B, action_horizon, output_dim] - noise or actual actions
            logprobs: [B] - log probabilities
            dist_params: (mean, std) or None - distribution parameters for logging
        """
        if not self.config.use_dsrl:
            raise ValueError("sac_forward called but use_dsrl=False")

        # Support both call styles: obs (new, from sac_dsrl) or data (legacy)
        if obs is None:
            obs = data.get("obs", data) if data is not None else kwargs.get("obs", {})

        # Handle two obs formats:
        # Format 1 (internal): {"images": [...], "states": ...}
        # Format 2 (env): {"main_images": ..., "wrist_images": ..., "states": ...}
        if "images" not in obs:
            # Convert env format to internal format
            if "main_images" in obs:
                obs = {"images": [obs["main_images"]], "states": obs["states"]}
            else:
                raise ValueError(
                    f"Invalid obs format: {obs.keys()}. Expected 'images' or 'main_images' key."
                )

        # Preprocess images: resize to 64x64, use only agentview camera
        # Returns [B, 1, C, 64, 64] in [-1, 1] range (float32)
        images = self._preprocess_dsrl_images(obs["images"], train=train)
        states = self._preprocess_states(obs["states"])

        # Move to the same device as actor encoders, convert to bfloat16
        device = next(self.actor_image_encoder.parameters()).device
        images = images.to(device=device, dtype=torch.bfloat16)
        states = states.to(device=device, dtype=torch.bfloat16)

        # Extract features (using actor's independent encoder)
        image_features = self.actor_image_encoder(images)  # [B, 64]
        state_features = self.actor_state_encoder(states)  # [B, 64]
        features = torch.cat([state_features, image_features], dim=-1)  # [B, 128]

        # Sample from GaussianPolicy
        mode = kwargs.get("mode", "train")
        deterministic = mode == "eval"

        action_noise, logprobs = self.dsrl_action_noise_net.sample(
            features, deterministic=deterministic
        )

        # Optional: return distribution parameters for logging
        dist_params = None
        if return_dist_params:
            dist = self.dsrl_action_noise_net.forward(features)
            dist_params = (dist.mean, dist.stddev)

        return action_noise, logprobs, dist_params

    def sac_q_forward(
        self,
        obs=None,
        data=None,
        actions=None,
        detach_encoder=False,
        train=False,
        **kwargs,
    ):
        """Q-value forward pass for DSRL.

        Args:
            obs: Observation dict (preferred, matches sac_dsrl).
                 Supports two formats:
                   1. {"images": list of tensors, "states": tensor} - internal format
                   2. {"main_images": tensor, "wrist_images": tensor, "states": tensor} - env format
            data: Dictionary containing observations (legacy, for backward compatibility).
            actions: [B, action_dim] or [B, action_horizon, action_dim]
            detach_encoder: Whether to detach encoder gradients.
            train: Whether to use data augmentation.

        Returns:
            q_values: [B, num_q_heads] - Q-values from all Q-networks.
        """
        if not self.config.use_dsrl:
            raise ValueError("sac_q_forward called but use_dsrl=False")

        # Support both call styles: obs (new, from sac_dsrl) or data (legacy)
        if obs is None:
            obs = data.get("obs", data) if data is not None else kwargs.get("obs", {})
        if actions is None:
            actions = kwargs.get("actions")

        # Handle two obs formats:
        # Format 1 (internal): {"images": [...], "states": ...}
        # Format 2 (env): {"main_images": ..., "wrist_images": ..., "states": ...}
        if "images" not in obs:
            # Convert env format to internal format
            if "main_images" in obs:
                obs = {"images": [obs["main_images"]], "states": obs["states"]}
            else:
                raise ValueError(
                    f"Invalid obs format: {obs.keys()}. Expected 'images' or 'main_images' key."
                )

        # Preprocess images: resize to 64x64, use only agentview camera
        # Returns [B, 1, C, 64, 64] in [-1, 1] range (float32)
        images = self._preprocess_dsrl_images(obs["images"], train=train)
        states = self._preprocess_states(obs["states"])

        # Move to the same device as critic encoders, convert to bfloat16
        device = next(self.critic_image_encoder.parameters()).device
        images = images.to(device=device, dtype=torch.bfloat16)
        states = states.to(device=device, dtype=torch.bfloat16)
        actions = actions.to(device=device, dtype=torch.bfloat16)

        # Extract features (using critic's independent encoder)
        image_features = self.critic_image_encoder(images)
        state_features = self.critic_state_encoder(states)

        # Optionally detach encoder
        if detach_encoder:
            image_features = image_features.detach()
            state_features = state_features.detach()

        # Process actions (DSRL: should be noise, already flattened)
        if actions.dim() == 3:
            actions = actions[:, 0, :]  # [B, action_horizon, dim] -> [B, dim]

        # Compute Q values
        q_values = self.q_head(state_features, image_features, actions)

        return q_values

    # ===== NFT-specific methods =====

    def _init_nft_state(
        self,
        collect_nft_state: bool,
        x_t: torch.Tensor,
        num_steps: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor] | None:
        """Initialize NFT state buffers for rollout sampling."""
        if not collect_nft_state:
            return None
        return {
            "nft_step_index": torch.randint(
                0, num_steps, (x_t.shape[0],), device=device
            ),
            "nft_xcur": torch.zeros_like(x_t),
            "nft_v": torch.zeros_like(x_t),
            "nft_xnext": torch.zeros_like(x_t),
            "nft_noise_level": torch.zeros(
                x_t.shape[0], device=device, dtype=x_t.dtype
            ),
        }

    def _update_nft_state(
        self,
        nft_state: dict[str, torch.Tensor] | None,
        idx: int,
        x_t_prev: torch.Tensor,
        v_t: torch.Tensor,
        x_t: torch.Tensor,
        sample_method: str,
    ) -> None:
        """Update NFT state buffers for the selected denoising step."""
        if nft_state is None:
            return
        mask = nft_state["nft_step_index"] == idx
        if not mask.any():
            return
        mask_bc = mask[:, None, None]
        nft_state["nft_xcur"] = torch.where(
            mask_bc, x_t_prev.detach(), nft_state["nft_xcur"]
        )
        nft_state["nft_v"] = torch.where(mask_bc, v_t.detach(), nft_state["nft_v"])
        nft_state["nft_xnext"] = torch.where(
            mask_bc, x_t.detach(), nft_state["nft_xnext"]
        )
        noise_level = self._get_noise_level(
            device=x_t.device, dtype=x_t.dtype, sample_method=sample_method
        )
        nft_state["nft_noise_level"] = torch.where(
            mask,
            torch.full_like(nft_state["nft_noise_level"], float(noise_level.item())),
            nft_state["nft_noise_level"],
        )

    def _get_noise_level(
        self, device: torch.device, dtype: torch.dtype, sample_method: str | None = None
    ) -> torch.Tensor:
        method = sample_method or self.config.noise_method
        if method == "flow_ode":
            return torch.zeros((), device=device, dtype=dtype)
        if self.config.noise_anneal:
            noise_start, noise_end, anneal_steps = self.config.noise_params
            noise_level = (
                noise_start
                + (noise_end - noise_start)
                * min(self.global_step, anneal_steps)
                / anneal_steps
            )
        else:
            noise_level = self.config.noise_level
        return torch.tensor(noise_level, device=device, dtype=dtype)

    def _preprocess_dsrl_images(self, images, train=False):
        """Preprocess images for DSRL: resize to 64x64, use only agentview camera.

        Args:
            images: List of tensors.
                Can be [B, H, W, C] (NHWC) from environment or
                [B, C, H, W] (NCHW) from processed data.
                For Libero: images[0] is agentview, images[1] is wrist.
            train: Whether to use data augmentation (placeholder for now).

        Returns:
            Tensor of shape [B, 1, C, 64, 64] - only agentview, resized, in [-1, 1].
        """

        # Extract only agentview camera (first image in the list)
        if isinstance(images, list):
            agentview_img = images[0]
        else:
            # Assume it's already a tensor
            agentview_img = images

        # Detect and convert NHWC -> NCHW (environment outputs NHWC)
        if agentview_img.shape[-1] == 3:
            # NHWC format: [B, H, W, C] -> [B, C, H, W]
            agentview_img = agentview_img.permute(0, 3, 1, 2)

        B, C, H, W = agentview_img.shape
        target_size = 64

        # ===== UNIFIED VALUE RANGE HANDLING =====
        # Convert to float32 and normalize to [0, 1] for PyTorch resize
        if agentview_img.dtype == torch.uint8:
            # [0, 255] -> [0, 1]
            agentview_img = agentview_img.float() / 255.0
        else:
            # Check if in [-1, 1] range
            if agentview_img.min() < 0:
                # [-1, 1] -> [0, 1]
                agentview_img = (agentview_img + 1.0) / 2.0
            # else: already in [0, 1] range, assume correctly normalized
        # ===========================================

        # Clamp to ensure valid range
        agentview_img = agentview_img.clamp(0.0, 1.0)

        # ===== GPU-ACCELERATED RESIZE (aligned with PIL behavior) =====
        # PyTorch bilinear with align_corners=False approximates PIL's behavior
        resized_img = F.interpolate(
            agentview_img,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )
        # =============================================================

        # Convert back to [-1, 1] range (to match PIL-based pipeline)
        resized_img = resized_img * 2.0 - 1.0  # [0, 1] -> [-1, 1]

        # Add num_images dimension: [B, C, 64, 64] -> [B, 1, C, 64, 64]
        resized_img = resized_img.unsqueeze(1)

        return resized_img

    def _preprocess_states(self, states):
        """
        Preprocess states: flatten to 2D and convert to bfloat16.

        Args:
            states: [B, ...] any shape

        Returns:
            states: [B, state_dim] flattened states as bfloat16
        """
        if states.dim() > 2:
            states = states.reshape(states.shape[0], -1)
        # Convert to bfloat16 to match encoder's dtype
        if states.dtype != torch.bfloat16:
            states = states.to(torch.bfloat16)
        return states
