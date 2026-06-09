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
