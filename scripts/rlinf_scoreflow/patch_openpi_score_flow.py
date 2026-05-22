#!/usr/bin/env python3
"""Patch RLinf OpenPI with the original Score-Flow score drift.

This patch is intentionally narrow.  It adds only the original Score-Flow
mechanism to RLinf/OpenPI:

    dx_t = [v_theta(x_t, t, s) + alpha_psi(t) * score_t] dt + noise
    score_t = (t * v_theta(x_t, t, s) - x_t) / (1 - t)

It does not add terminal action guidance, FMTT, direct x_t critic guidance,
Spec-Flow, pIRL diagnostics, or sigma^2 score-coefficient ablations.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PATCH_MARKER = "# SCOREFLOW_ORIGINAL_RLINF_PATCH_V1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rlinf-root",
        type=Path,
        required=True,
        help="Path to the RLinf repository root.",
    )
    return parser.parse_args()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"Could not find patch anchor: {label}")
    return text.replace(old, new, 1)


def patch_openpi_action_model(path: Path) -> bool:
    text = path.read_text()
    if PATCH_MARKER in text:
        return False

    backup = path.with_suffix(path.suffix + ".scoreflow_original_backup")
    if not backup.exists():
        shutil.copy2(path, backup)

    score_flow_fields = (
        f"    {PATCH_MARKER}\n"
        '    score_flow_mode: str = "none"  # none, learned_alpha\n'
        "    score_flow_scale: float = 1.0\n"
        "    score_flow_clip_norm: float = 10.0\n"
        "    score_flow_alpha_hidden_dim: int = 16\n"
        "    score_flow_alpha_init_bias: float = -2.0\n"
        "    score_flow_alpha_max: float = 2.0\n"
        "    score_flow_use_time_mask: bool = True\n"
    )
    text = replace_once(
        text,
        '    value_vlm_mode: str = "mean_token"  # last_token, mean_token, first_token\n',
        '    value_vlm_mode: str = "mean_token"  # last_token, mean_token, first_token\n'
        "\n"
        + score_flow_fields,
        "OpenPI config fields",
    )

    text = replace_once(
        text,
        "        self.global_step = 0\n",
        "        self.global_step = 0\n"
        "        self._init_score_flow_alpha_net()\n",
        "OpenPI Score-Flow AlphaNet init",
    )

    score_flow_methods = r'''
    def _init_score_flow_alpha_net(self):
        hidden_dim = int(getattr(self.config, "score_flow_alpha_hidden_dim", 16))
        init_bias = float(getattr(self.config, "score_flow_alpha_init_bias", -2.0))
        self.score_flow_alpha_net = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Softplus(),
        )
        torch.nn.init.constant_(self.score_flow_alpha_net[-2].weight, 0.0)
        torch.nn.init.constant_(self.score_flow_alpha_net[-2].bias, init_bias)

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
        flat = score_t.reshape(score_t.shape[0], -1)
        raw_norm = torch.linalg.vector_norm(flat, dim=1).clamp_min(1.0e-12)
        clip_norm = float(getattr(self.config, "score_flow_clip_norm", 10.0))
        if clip_norm > 0:
            scale = (clip_norm / raw_norm).clamp(max=1.0)
            score_t = score_t * scale.reshape(-1, *([1] * (score_t.ndim - 1)))
        return score_t

    def _score_flow_alpha(self, t_scalar, dtype):
        t_in = t_scalar.detach().reshape(-1, 1).to(
            device=t_scalar.device, dtype=torch.float32
        )
        alpha_t = self.score_flow_alpha_net(t_in)
        if bool(getattr(self.config, "score_flow_use_time_mask", True)):
            alpha_t = alpha_t * (1.0 - t_in).clamp_min(0.0)
        alpha_t = alpha_t * float(getattr(self.config, "score_flow_scale", 1.0))
        alpha_max = float(getattr(self.config, "score_flow_alpha_max", 2.0))
        if alpha_max > 0:
            alpha_t = alpha_t.clamp(max=alpha_max)
        return alpha_t.to(dtype=dtype).reshape(-1, 1, 1)

    def _apply_score_flow_drift(self, x_t, x_t_mean, t_input, delta, v_t):
        score_flow_mode = getattr(self.config, "score_flow_mode", "none")
        if score_flow_mode == "none":
            return x_t_mean
        if score_flow_mode != "learned_alpha":
            raise ValueError(f"Unsupported score_flow_mode={score_flow_mode}")

        t_scalar, t_expanded = self._score_flow_time(x_t, t_input)
        score_t = (t_expanded * v_t - x_t) / (1.0 - t_expanded + 1.0e-5)
        score_t = self._score_flow_clip(score_t)

        if torch.is_tensor(delta):
            delta_t = delta.detach().to(device=x_t.device, dtype=x_t.dtype)
        else:
            delta_t = torch.tensor(delta, device=x_t.device, dtype=x_t.dtype)
        if delta_t.ndim == 1:
            delta_t = delta_t.reshape(-1, 1, 1)

        alpha_t = self._score_flow_alpha(t_scalar, x_t.dtype)
        return x_t_mean + delta_t * alpha_t * score_t

'''
    text = replace_once(
        text,
        "    def sample_mean_var_val(\n",
        score_flow_methods + "    def sample_mean_var_val(\n",
        "Score-Flow methods",
    )

    text = replace_once(
        text,
        "        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight\n"
        "        return x_t_mean, x_t_std, value_t, v_t\n",
        "        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight\n"
        "        x_t_mean = self._apply_score_flow_drift(\n"
        "            x_t, x_t_mean, t_input, delta, v_t\n"
        "        )\n"
        "        return x_t_mean, x_t_std, value_t, v_t\n",
        "Score-Flow drift application",
    )

    path.write_text(text)
    return True


def main() -> None:
    args = parse_args()
    model_path = (
        args.rlinf_root
        / "rlinf"
        / "models"
        / "embodiment"
        / "openpi"
        / "openpi_action_model.py"
    )
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    changed = patch_openpi_action_model(model_path)
    print(f"patched={changed} path={model_path}")


if __name__ == "__main__":
    main()
