#!/usr/bin/env python3
"""Patch RLinf to preserve raw per-state LIBERO evaluation outcomes."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REMOTE_LIBERO_ENV = Path("rlinf/envs/libero/libero_env.py")
REMOTE_EVAL_RUNNER = Path("rlinf/runners/embodied_eval_runner.py")
REMOTE_ROLLOUT_WORKER = Path("rlinf/workers/rollout/hf/huggingface_worker.py")
BACKUP_SUFFIX = ".pre_pirl_official_eval_backup"
LIBERO_MARKER = "PIRL_OFFICIAL_EVAL_LIBERO_PATCH_V1"
RUNNER_MARKER = "PIRL_OFFICIAL_EVAL_RUNNER_PATCH_V1"
ROLLOUT_MARKER = "PIRL_OFFICIAL_EVAL_ROLLOUT_PATCH_V1"

LIBERO_ANCHOR = '''        episode_info["episode_len"] = self.elapsed_steps.copy()

        # Use success episode_len for reward if already succeeded, else current elapsed
'''

LIBERO_REPLACEMENT = f'''        episode_info["episode_len"] = self.elapsed_steps.copy()

        # {LIBERO_MARKER}
        episode_info["reset_state_id"] = self.reset_state_ids.copy()
        episode_info["task_id"] = self.task_ids.copy()
        episode_info["trial_id"] = self.trial_ids.copy()

        # Use success episode_len for reward if already succeeded, else current elapsed
'''

RUNNER_IMPORT_ANCHOR = '''import typing

from rlinf.scheduler import Channel
'''

RUNNER_HELPERS = f'''import json
import typing
from pathlib import Path

import numpy as np
import torch

from rlinf.scheduler import Channel


# {RUNNER_MARKER}
def _json_scalar(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    elif isinstance(value, np.generic):
        value = value.item()
    return value


def _write_official_episode_artifact(path, env_results):
    required = ("reset_state_id", "task_id", "trial_id", "success_once")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for shard_index, result in enumerate(env_results):
            if result is None:
                continue
            missing = [key for key in required if key not in result]
            if missing:
                raise RuntimeError(
                    f"Official evaluation shard {{shard_index}} is missing {{missing}}"
                )
            lengths = {{key: len(result[key]) for key in required}}
            if len(set(lengths.values())) != 1:
                raise RuntimeError(
                    f"Official evaluation shard {{shard_index}} has mismatched lengths: "
                    f"{{lengths}}"
                )
            for local_index in range(next(iter(lengths.values()))):
                row = {{
                    "shard_index": shard_index,
                    "local_index": local_index,
                    "reset_state_id": int(
                        _json_scalar(result["reset_state_id"][local_index])
                    ),
                    "task_id": int(_json_scalar(result["task_id"][local_index])),
                    "trial_id": int(_json_scalar(result["trial_id"][local_index])),
                    "success": bool(_json_scalar(result["success_once"][local_index])),
                }}
                handle.write(json.dumps(row, sort_keys=True) + "\\n")
'''

RUNNER_EVALUATE_ANCHOR = '''        env_results = env_handle.wait()
        rollout_handle.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
'''

RUNNER_EVALUATE_REPLACEMENT = '''        env_results = env_handle.wait()
        rollout_handle.wait()
        official_artifact = self.cfg.runner.get("official_episode_artifact", None)
        if official_artifact:
            _write_official_episode_artifact(official_artifact, env_results)
        eval_metrics_list = [results for results in env_results if results is not None]
'''

ROLLOUT_IMPORT_ANCHOR = '''import copy
import gc
from typing import Any, Literal
'''

ROLLOUT_IMPORT_REPLACEMENT = f'''import copy
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Literal


# {ROLLOUT_MARKER}
def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
'''

ROLLOUT_CHECKPOINT_ANCHOR = '''        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            self.hf_model.load_state_dict(model_dict)
'''

ROLLOUT_CHECKPOINT_REPLACEMENT = '''        if self.cfg.runner.get("ckpt_path", None):
            checkpoint_path = Path(self.cfg.runner.ckpt_path).resolve()
            model_dict = torch.load(checkpoint_path, map_location="cpu")
            self.hf_model.load_state_dict(model_dict)
            receipt_path = self.cfg.runner.get("official_checkpoint_receipt", None)
            if receipt_path and self._rank == 0:
                receipt = {
                    "checkpoint_loaded": True,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": _sha256_file(checkpoint_path),
                    "state_dict_keys": len(model_dict),
                    "rollout_rank": self._rank,
                }
                receipt_output = Path(receipt_path)
                receipt_output.parent.mkdir(parents=True, exist_ok=True)
                receipt_output.write_text(
                    json.dumps(receipt, sort_keys=True) + "\\n", encoding="utf-8"
                )
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def replace_once(text: str, old: str, new: str, label: str) -> tuple[str, bool]:
    count = text.count(old)
    if count == 1:
        return text.replace(old, new, 1), True
    if new in text:
        return text, False
    raise RuntimeError(f"Expected exactly one {label} anchor, found {count}")


def patch_libero_text(text: str) -> tuple[str, bool]:
    return replace_once(text, LIBERO_ANCHOR, LIBERO_REPLACEMENT, "LIBERO metrics")


def patch_runner_text(text: str) -> tuple[str, bool]:
    text, import_changed = replace_once(
        text,
        RUNNER_IMPORT_ANCHOR,
        RUNNER_HELPERS,
        "evaluation runner imports",
    )
    text, evaluate_changed = replace_once(
        text,
        RUNNER_EVALUATE_ANCHOR,
        RUNNER_EVALUATE_REPLACEMENT,
        "evaluation runner raw artifact",
    )
    return text, import_changed or evaluate_changed


def patch_rollout_text(text: str) -> tuple[str, bool]:
    text, import_changed = replace_once(
        text,
        ROLLOUT_IMPORT_ANCHOR,
        ROLLOUT_IMPORT_REPLACEMENT,
        "rollout worker imports",
    )
    text, checkpoint_changed = replace_once(
        text,
        ROLLOUT_CHECKPOINT_ANCHOR,
        ROLLOUT_CHECKPOINT_REPLACEMENT,
        "rollout worker checkpoint load",
    )
    return text, import_changed or checkpoint_changed


def patch_file(path: Path, patcher, check: bool) -> bool:
    if not path.exists():
        raise FileNotFoundError(path)
    patched, changed = patcher(path.read_text(encoding="utf-8"))
    if not changed or check:
        return changed
    backup = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(patched, encoding="utf-8")
    return True


def main() -> None:
    args = parse_args()
    libero_path = args.rlinf_root / REMOTE_LIBERO_ENV
    runner_path = args.rlinf_root / REMOTE_EVAL_RUNNER
    rollout_path = args.rlinf_root / REMOTE_ROLLOUT_WORKER
    libero_changed = patch_file(libero_path, patch_libero_text, args.check)
    runner_changed = patch_file(runner_path, patch_runner_text, args.check)
    rollout_changed = patch_file(rollout_path, patch_rollout_text, args.check)
    print(
        f"libero_patched={libero_changed and not args.check} "
        f"libero_would_patch={libero_changed if args.check else False} "
        f"runner_patched={runner_changed and not args.check} "
        f"runner_would_patch={runner_changed if args.check else False} "
        f"rollout_patched={rollout_changed and not args.check} "
        f"rollout_would_patch={rollout_changed if args.check else False} "
        f"check={args.check}"
    )


if __name__ == "__main__":
    main()
