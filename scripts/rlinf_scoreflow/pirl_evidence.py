#!/usr/bin/env python3
"""Hash and parse evidence referenced by pi_RL protocol artifacts."""

from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    if not path.is_dir():
        raise ValueError(f"model path must be an existing directory: {path}")
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"model path contains no files: {path}")
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(candidate)))
    return digest.hexdigest()


def read_command(path: Path) -> list[str]:
    if not path.is_file():
        raise ValueError(f"command file does not exist: {path}")
    return shlex.split(path.read_text(encoding="utf-8"))


def parse_command(path: Path) -> tuple[dict[str, str], tuple[str, ...], str | None]:
    overrides: dict[str, str] = {}
    duplicates: list[str] = []
    config_name: str | None = None
    tokens = read_command(path)
    for index, token in enumerate(tokens):
        if token == "--config-name" and index + 1 < len(tokens):
            config_name = tokens[index + 1]
        if "=" not in token or token.startswith("--"):
            continue
        key, value = token.split("=", 1)
        key = key.lstrip("+")
        if key in overrides:
            duplicates.append(key)
        overrides[key] = value
    return overrides, tuple(duplicates), config_name


def expected_method_overrides(method: str) -> dict[str, str]:
    shared = {
        "actor.model.openpi.noise_method": "flow_noise",
        "actor.model.openpi.score_flow_mode": "none",
        "actor.model.openpi.tr_penalty_mode": "none",
        "actor.model.openpi.cr_reflow_mode": "none",
    }
    if method == "flow_noise_baseline":
        return shared
    if method == "scoreflow_original":
        return {**shared, "actor.model.openpi.score_flow_mode": "learned_alpha"}
    if method == "tr_scalar_l2":
        return {**shared, "actor.model.openpi.tr_penalty_mode": "scalar_l2"}
    if method in ("tr_pullback", "tr_pullback_matched"):
        return {**shared, "actor.model.openpi.tr_penalty_mode": "terminal_pullback"}
    if method == "cr_reflow_no_anchor":
        return {
            **shared,
            "actor.model.openpi.cr_reflow_mode": "cr_reflow_no_anchor",
            "actor.model.openpi.cr_reflow_anchor_beta": "0.0",
        }
    if method == "cr_reflow":
        return {**shared, "actor.model.openpi.cr_reflow_mode": "cr_reflow"}
    raise ValueError(f"unsupported method: {method}")


def reproducibility_errors(
    artifact: dict[str, Any],
    training_path: Path,
) -> list[str]:
    errors: list[str] = []
    path = Path(str(artifact.get("reproducibility_bundle", "")))
    if not path.is_file():
        return errors
    try:
        bundle = json.loads(path.read_text(encoding="utf-8"))
        if bundle.get("status") != "done" or bundle.get("exit_code") != 0:
            errors.append("reproducibility bundle must record done with exit_code 0")
        expected_command = str(Path(str(artifact.get("command_file", ""))).resolve())
        if bundle.get("command_file") != expected_command:
            errors.append("reproducibility bundle command_file must match evaluation")
        if bundle.get("command_sha256") != artifact.get("command_sha256"):
            errors.append("reproducibility bundle command_sha256 must match evaluation")
        if bundle.get("protocol_artifact") != str(training_path.resolve()):
            errors.append("reproducibility bundle protocol_artifact must match training")
        if bundle.get("protocol_artifact_sha256") != artifact.get(
            "training_artifact_sha256"
        ):
            errors.append(
                "reproducibility bundle protocol_artifact_sha256 must match training"
            )
        config_snapshot = bundle.get("config_snapshot")
        if not isinstance(config_snapshot, dict):
            errors.append("reproducibility bundle must contain config_snapshot")
        else:
            base_config = Path(str(config_snapshot.get("base_config", "")))
            if not base_config.is_file():
                errors.append("reproducibility base_config must reference an existing file")
            elif config_snapshot.get("base_config_sha256") != sha256_file(base_config):
                errors.append("reproducibility base_config_sha256 does not match")
    except (json.JSONDecodeError, OSError, TypeError):
        errors.append("reproducibility bundle must contain valid JSON")
    return errors


def raw_episode_errors(
    artifact: dict[str, Any],
    official: dict[str, Any],
    task_results: Any,
) -> list[str]:
    path = Path(str(artifact.get("raw_episode_artifact", "")))
    if not path.is_file():
        return []
    errors: list[str] = []
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if any(not isinstance(row.get("success"), bool) for row in rows):
            errors.append("raw episode success values must be booleans")
        pairs = {(row.get("task_id"), row.get("trial_id")) for row in rows}
        reset_ids = {row.get("reset_state_id") for row in rows}
        expected_pairs = {
            (task_id, trial_id)
            for task_id in range(official["task_count"])
            for trial_id in range(official["states_per_task"])
        }
        if len(rows) != official["total_states"] or pairs != expected_pairs:
            errors.append("raw episode evidence must contain the exact official task-state set")
        if len(reset_ids) != official["total_states"]:
            errors.append("raw episode reset_state_id values must be unique")
        if artifact.get("successes") != sum(row.get("success") is True for row in rows):
            errors.append("reported successes must match raw episode evidence")
        task_successes = {
            task_id: sum(
                row.get("success") is True
                for row in rows
                if row.get("task_id") == task_id
            )
            for task_id in range(official["task_count"])
        }
        if isinstance(task_results, list) and any(
            task_successes.get(task.get("task_id")) != task.get("successes")
            for task in task_results
        ):
            errors.append("task_results successes must match raw episode evidence")
    except (json.JSONDecodeError, OSError, TypeError, AttributeError):
        errors.append("raw episode evidence must contain valid JSON objects")
    return errors
