#!/usr/bin/env python3
"""Create and validate pi_RL-aligned LIBERO protocol artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROTOCOL_PATH = Path(__file__).parent / "protocols" / "pirl_pi05_libero.json"
SHA256_PROVENANCE_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ValidationResult:
    compliant: bool
    errors: tuple[str, ...]


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_training_fields(protocol: dict[str, Any], suite: str) -> dict[str, Any]:
    suite_protocol = protocol["suites"][suite]
    return {
        "model_family": protocol["model_family"],
        "train_epochs": protocol["train_epochs"],
        "global_batch_size": protocol["global_batch_size"],
        "parallel_environments": protocol["parallel_environments"],
        "rollout_epochs": protocol["rollout_epochs"],
        "actor_lr": protocol["actor_lr"],
        "critic_lr": protocol["critic_lr"],
        "reward_discount": protocol["reward_discount"],
        "gae_lambda": protocol["gae_lambda"],
        "clip_ratio": protocol["clip_ratio"],
        "flow_noise_min_logvar": protocol["flow_noise_min_logvar"],
        "flow_noise_max_logvar": protocol["flow_noise_max_logvar"],
        "flow_noise_entropy_bonus": protocol["flow_noise_entropy_bonus"],
        "interaction_steps": suite_protocol["interaction_steps"],
        "update_epochs": suite_protocol["update_epochs"],
        "action_prediction_horizon": suite_protocol["action_prediction_horizon"],
        "action_replan_horizon": suite_protocol["action_replan_horizon"],
        "denoise_steps": suite_protocol["denoise_steps"],
        "scheduler": suite_protocol["scheduler"],
    }


def is_immutable_provenance(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_PROVENANCE_PATTERN.fullmatch(value))


def validate_training_artifact(
    artifact: dict[str, Any],
    protocol: dict[str, Any],
) -> ValidationResult:
    errors: list[str] = []
    suite = artifact.get("suite")
    if suite not in protocol["suites"]:
        return ValidationResult(False, (f"unsupported suite: {suite}",))
    if artifact.get("protocol_id") != protocol["protocol_id"]:
        errors.append(f"protocol_id must be {protocol['protocol_id']}")
    for field, expected in expected_training_fields(protocol, suite).items():
        actual = artifact.get("training", {}).get(field)
        if actual != expected:
            errors.append(f"training.{field}: expected {expected!r}, got {actual!r}")
    if not is_immutable_provenance(artifact.get("model_provenance")):
        errors.append("model_provenance must be sha256:<64 lowercase hex characters>")
    if not isinstance(artifact.get("command_sha256"), str) or not SHA256_PATTERN.fullmatch(
        artifact["command_sha256"]
    ):
        errors.append("command_sha256 must contain 64 lowercase hex characters")
    return ValidationResult(not errors, tuple(errors))


def validate_evaluation_artifact(
    artifact: dict[str, Any],
    protocol: dict[str, Any],
) -> ValidationResult:
    errors: list[str] = []
    official = protocol["official_evaluation"]
    suite = artifact.get("suite")
    if suite not in protocol["suites"]:
        errors.append(f"unsupported suite: {suite}")
    if artifact.get("protocol_id") != protocol["protocol_id"]:
        errors.append(f"protocol_id must be {protocol['protocol_id']}")
    if artifact.get("status") != "done":
        errors.append("status must be done")
    if artifact.get("total_states") != official["total_states"]:
        errors.append(f"total_states must be {official['total_states']}")
    task_results = artifact.get("task_results")
    if not isinstance(task_results, list) or len(task_results) != official["task_count"]:
        errors.append(f"task_results must contain {official['task_count']} tasks")
    else:
        task_ids: list[Any] = []
        success_total = 0
        for index, task in enumerate(task_results):
            task_ids.append(task.get("task_id"))
            if task.get("evaluated_states") != official["states_per_task"]:
                errors.append(
                    f"task_results[{index}].evaluated_states must be "
                    f"{official['states_per_task']}"
                )
            successes = task.get("successes")
            if not isinstance(successes, int) or not 0 <= successes <= official["states_per_task"]:
                errors.append(f"task_results[{index}].successes is invalid")
            else:
                success_total += successes
        expected_task_ids = set(range(official["task_count"]))
        if set(task_ids) != expected_task_ids:
            errors.append(
                f"task_results task IDs must be exactly {sorted(expected_task_ids)}"
            )
        if artifact.get("successes") != success_total:
            errors.append(f"successes must equal task result sum {success_total}")
        expected_rate = success_total / official["total_states"]
        if artifact.get("success_rate") != expected_rate:
            errors.append(f"success_rate must equal {expected_rate}")
        if artifact.get("success_percent") != expected_rate * 100:
            errors.append(f"success_percent must equal {expected_rate * 100}")
    if not is_immutable_provenance(artifact.get("checkpoint_provenance")):
        errors.append("checkpoint_provenance must be sha256:<64 lowercase hex characters>")
    checkpoint_path = artifact.get("checkpoint_path")
    if not isinstance(checkpoint_path, str) or not checkpoint_path.endswith(
        "/actor/model_state_dict/full_weights.pt"
    ):
        errors.append("checkpoint_path must identify RLinf actor full_weights.pt")
    if not artifact.get("checkpoint_load_receipt"):
        errors.append("checkpoint_load_receipt is required")
    if (
        not isinstance(artifact.get("checkpoint_state_dict_keys"), int)
        or artifact["checkpoint_state_dict_keys"] <= 0
    ):
        errors.append("checkpoint_state_dict_keys must be a positive integer")
    if not artifact.get("raw_episode_artifact"):
        errors.append("raw_episode_artifact is required")
    for field in (
        "checkpoint_load_receipt_sha256",
        "command_sha256",
        "raw_episode_sha256",
    ):
        value = artifact.get(field)
        if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
            errors.append(f"{field} must contain 64 lowercase hex characters")
    if artifact.get("official_comparison_eligible") is False:
        errors.append("official_comparison_eligible must not be false")
    return ValidationResult(not errors, tuple(errors))


def write_training_artifact(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    command = args.command_file.read_bytes()
    artifact = {
        "protocol_id": protocol["protocol_id"],
        "suite": args.suite,
        "method": args.method,
        "model_provenance": args.model_provenance,
        "command_file": str(args.command_file),
        "command_sha256": hashlib.sha256(command).hexdigest(),
        "training": {
            "model_family": args.model_family,
            "train_epochs": args.train_epochs,
            "global_batch_size": args.global_batch_size,
            "parallel_environments": args.parallel_environments,
            "rollout_epochs": args.rollout_epochs,
            "actor_lr": args.actor_lr,
            "critic_lr": args.critic_lr,
            "reward_discount": args.reward_discount,
            "gae_lambda": args.gae_lambda,
            "clip_ratio": args.clip_ratio,
            "flow_noise_min_logvar": args.flow_noise_min_logvar,
            "flow_noise_max_logvar": args.flow_noise_max_logvar,
            "flow_noise_entropy_bonus": args.flow_noise_entropy_bonus,
            "interaction_steps": args.interaction_steps,
            "update_epochs": args.update_epochs,
            "action_prediction_horizon": args.action_prediction_horizon,
            "action_replan_horizon": args.action_replan_horizon,
            "denoise_steps": args.denoise_steps,
            "scheduler": args.scheduler,
        },
    }
    result = validate_training_artifact(artifact, protocol)
    artifact["training_protocol_compliant"] = result.compliant
    artifact["training_protocol_errors"] = list(result.errors)
    artifact["official_comparison_eligible"] = False
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")


def validate_artifact(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    if args.kind == "training":
        result = validate_training_artifact(artifact, protocol)
    else:
        result = validate_evaluation_artifact(artifact, protocol)
    print(json.dumps({"compliant": result.compliant, "errors": result.errors}))
    if not result.compliant:
        raise SystemExit(1)


def add_common_protocol_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    emit = subparsers.add_parser("emit-training")
    add_common_protocol_arg(emit)
    emit.add_argument("--output", type=Path, required=True)
    emit.add_argument("--command-file", type=Path, required=True)
    emit.add_argument("--suite", required=True)
    emit.add_argument("--method", required=True)
    emit.add_argument("--model-provenance", required=True)
    emit.add_argument("--model-family", required=True)
    emit.add_argument("--train-epochs", type=int, required=True)
    emit.add_argument("--global-batch-size", type=int, required=True)
    emit.add_argument("--parallel-environments", type=int, required=True)
    emit.add_argument("--rollout-epochs", type=int, required=True)
    emit.add_argument("--actor-lr", type=float, required=True)
    emit.add_argument("--critic-lr", type=float, required=True)
    emit.add_argument("--reward-discount", type=float, required=True)
    emit.add_argument("--gae-lambda", type=float, required=True)
    emit.add_argument("--clip-ratio", type=float, required=True)
    emit.add_argument("--flow-noise-min-logvar", type=float, required=True)
    emit.add_argument("--flow-noise-max-logvar", type=float, required=True)
    emit.add_argument("--flow-noise-entropy-bonus", type=float, required=True)
    emit.add_argument("--interaction-steps", type=int, required=True)
    emit.add_argument("--update-epochs", type=int, required=True)
    emit.add_argument("--action-prediction-horizon", type=int, required=True)
    emit.add_argument("--action-replan-horizon", type=int, required=True)
    emit.add_argument("--denoise-steps", type=int, required=True)
    emit.add_argument("--scheduler", action=argparse.BooleanOptionalAction, required=True)
    emit.set_defaults(func=write_training_artifact)

    validate = subparsers.add_parser("validate")
    add_common_protocol_arg(validate)
    validate.add_argument("--kind", choices=("training", "evaluation"), required=True)
    validate.add_argument("--artifact", type=Path, required=True)
    validate.set_defaults(func=validate_artifact)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
