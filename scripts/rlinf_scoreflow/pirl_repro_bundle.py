#!/usr/bin/env python3
"""Write a reproducibility bundle for an aligned pi_RL run."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from .pirl_evidence import parse_command, read_command, sha256_file
except ImportError:
    from pirl_evidence import parse_command, read_command, sha256_file


def git_state(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"root": str(root.resolve()), "revision": None, "dirty": None}
    if not (root / ".git").exists():
        return result
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if revision.returncode == 0:
        result["revision"] = revision.stdout.strip()
    if status.returncode == 0:
        result["dirty"] = bool(status.stdout.strip())
    return result


def config_snapshot(command_file: Path) -> dict[str, str]:
    _, _, config_name = parse_command(command_file)
    tokens = read_command(command_file)
    try:
        config_path = Path(tokens[tokens.index("--config-path") + 1])
    except (ValueError, IndexError):
        raise ValueError("command must contain --config-path") from None
    if config_name is None:
        raise ValueError("command must contain --config-name")
    base_config = config_path / f"{config_name}.yaml"
    if not base_config.is_file():
        raise ValueError(f"base config does not exist: {base_config}")
    return {
        "config_name": config_name,
        "base_config": str(base_config.resolve()),
        "base_config_sha256": sha256_file(base_config),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--command-file", type=Path, required=True)
    parser.add_argument("--protocol-artifact", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--exit-code", type=int)
    args = parser.parse_args()

    bundle = {
        "status": args.status,
        "exit_code": args.exit_code,
        "command_file": str(args.command_file.resolve()),
        "command_sha256": sha256_file(args.command_file),
        "protocol_artifact": str(args.protocol_artifact.resolve()),
        "protocol_artifact_sha256": sha256_file(args.protocol_artifact),
        "config_snapshot": config_snapshot(args.command_file),
        "local_code": git_state(args.local_root),
        "rlinf_code": git_state(args.rlinf_root),
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
            "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
        },
    }
    args.output.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
