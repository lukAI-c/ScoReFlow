#!/usr/bin/env python3
"""Build a strict pi_RL official evaluation artifact from raw episode JSONL."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .pirl_protocol import load_protocol, sha256_file, validate_evaluation_artifact
except ImportError:
    from pirl_protocol import load_protocol, sha256_file, validate_evaluation_artifact


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return [center - margin, center + margin]


def read_episodes(path: Path) -> list[dict[str, Any]]:
    episodes = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = {"reset_state_id", "task_id", "trial_id", "success"}
        missing = required - row.keys()
        if missing:
            raise ValueError(f"line {line_number} is missing fields: {sorted(missing)}")
        if not isinstance(row["success"], bool):
            raise ValueError(f"line {line_number} success must be a boolean")
        episodes.append(row)
    return episodes


def read_checkpoint_receipt(
    path: Path,
    expected_provenance: str,
) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("checkpoint_loaded") is not True:
        raise ValueError("checkpoint load receipt must confirm checkpoint_loaded=true")
    checkpoint_path = receipt.get("checkpoint_path")
    if not isinstance(checkpoint_path, str) or not checkpoint_path.endswith(
        "/actor/model_state_dict/full_weights.pt"
    ):
        raise ValueError("checkpoint load receipt has an invalid checkpoint_path")
    expected_sha256 = expected_provenance.removeprefix("sha256:")
    if receipt.get("checkpoint_sha256") != expected_sha256:
        raise ValueError("checkpoint load receipt SHA-256 does not match provenance")
    if not Path(checkpoint_path).is_file():
        raise ValueError("checkpoint load receipt path does not exist")
    if sha256_file(Path(checkpoint_path)) != expected_sha256:
        raise ValueError("checkpoint file SHA-256 does not match provenance")
    if not isinstance(receipt.get("state_dict_keys"), int) or receipt["state_dict_keys"] <= 0:
        raise ValueError("checkpoint load receipt must contain a positive state_dict_keys")
    return receipt


def build_artifact(args: argparse.Namespace) -> dict[str, Any]:
    protocol = load_protocol(args.protocol)
    official = protocol["official_evaluation"]
    checkpoint_receipt = read_checkpoint_receipt(
        args.checkpoint_receipt,
        args.checkpoint_provenance,
    )
    episodes = read_episodes(args.raw_episodes)
    reset_ids = [int(row["reset_state_id"]) for row in episodes]
    task_trial_ids = [
        (int(row["task_id"]), int(row["trial_id"])) for row in episodes
    ]
    if len(episodes) != official["total_states"]:
        raise ValueError(
            f"expected {official['total_states']} episodes, got {len(episodes)}"
        )
    if len(set(reset_ids)) != official["total_states"]:
        raise ValueError("reset_state_id values must be unique across all 500 episodes")
    if len(set(task_trial_ids)) != official["total_states"]:
        raise ValueError("(task_id, trial_id) pairs must be unique across all episodes")

    task_counts = Counter(task_id for task_id, _ in task_trial_ids)
    expected_tasks = set(range(official["task_count"]))
    if set(task_counts) != expected_tasks:
        raise ValueError(f"task IDs must be exactly {sorted(expected_tasks)}")

    task_results = []
    for task_id in sorted(task_counts):
        rows = [row for row in episodes if int(row["task_id"]) == task_id]
        trial_ids = {int(row["trial_id"]) for row in rows}
        expected_trials = set(range(official["states_per_task"]))
        if trial_ids != expected_trials:
            raise ValueError(
                f"task {task_id} trial IDs must be exactly 0.."
                f"{official['states_per_task'] - 1}"
            )
        successes = sum(bool(row["success"]) for row in rows)
        task_results.append(
            {
                "task_id": task_id,
                "evaluated_states": len(rows),
                "successes": successes,
                "success_rate": successes / len(rows),
                "wilson_95_interval": wilson_interval(successes, len(rows)),
            }
        )

    successes = sum(bool(row["success"]) for row in episodes)
    artifact = {
        "protocol_id": protocol["protocol_id"],
        "suite": args.suite,
        "method": args.method,
        "status": "done",
        "total_states": len(episodes),
        "successes": successes,
        "success_rate": successes / len(episodes),
        "success_percent": successes * 100 / len(episodes),
        "wilson_95_interval": wilson_interval(successes, len(episodes)),
        "checkpoint_provenance": args.checkpoint_provenance,
        "checkpoint_path": checkpoint_receipt["checkpoint_path"],
        "checkpoint_load_receipt": str(args.checkpoint_receipt),
        "checkpoint_load_receipt_sha256": sha256_file(args.checkpoint_receipt),
        "checkpoint_state_dict_keys": checkpoint_receipt["state_dict_keys"],
        "command_file": str(args.command_file),
        "command_sha256": sha256_file(args.command_file),
        "raw_episode_artifact": str(args.raw_episodes),
        "raw_episode_sha256": sha256_file(args.raw_episodes),
        "training_artifact": str(args.training_artifact.resolve()),
        "training_artifact_sha256": sha256_file(args.training_artifact),
        "terminal_status_file": str(args.terminal_status.resolve()),
        "terminal_status_sha256": sha256_file(args.terminal_status),
        "reproducibility_bundle": str(args.reproducibility_bundle.resolve()),
        "reproducibility_bundle_sha256": sha256_file(args.reproducibility_bundle),
        "task_results": task_results,
    }
    result = validate_evaluation_artifact(artifact, protocol)
    artifact["official_comparison_eligible"] = result.compliant
    artifact["evaluation_protocol_errors"] = list(result.errors)
    if not result.compliant:
        raise ValueError("; ".join(result.errors))
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=None)
    parser.add_argument("--raw-episodes", type=Path, required=True)
    parser.add_argument("--command-file", type=Path, required=True)
    parser.add_argument("--checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--training-artifact", type=Path, required=True)
    parser.add_argument("--terminal-status", type=Path, required=True)
    parser.add_argument("--reproducibility-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--checkpoint-provenance", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.protocol is None:
        args.protocol = Path(__file__).parent / "protocols" / "pirl_pi05_libero.json"
    artifact = build_artifact(args)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact, sort_keys=True))


if __name__ == "__main__":
    main()
