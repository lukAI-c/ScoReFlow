#!/usr/bin/env python3
"""Create and validate pi_RL-aligned LIBERO protocol artifacts."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .pirl_evidence import (
        expected_method_overrides,
        parse_command,
        raw_episode_errors,
        reproducibility_errors,
        sha256_file,
        sha256_tree,
    )
except ImportError:
    from pirl_evidence import (
        expected_method_overrides,
        parse_command,
        raw_episode_errors,
        reproducibility_errors,
        sha256_file,
        sha256_tree,
    )


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


def expected_training_overrides(protocol: dict[str, Any], suite: str) -> dict[str, str]:
    fields = expected_training_fields(protocol, suite)
    scheduler = "cosine" if fields["scheduler"] else "constant"
    return {
        "runner.max_epochs": str(fields["train_epochs"]),
        "algorithm.rollout_epoch": str(fields["rollout_epochs"]),
        "algorithm.eval_rollout_epoch": "1",
        "env.train.total_num_envs": str(fields["parallel_environments"]),
        "env.eval.total_num_envs": str(protocol["official_evaluation"]["total_states"]),
        "actor.global_batch_size": str(fields["global_batch_size"]),
        "algorithm.update_epoch": str(fields["update_epochs"]),
        "algorithm.gamma": str(fields["reward_discount"]),
        "algorithm.gae_lambda": str(fields["gae_lambda"]),
        "algorithm.clip_ratio_high": str(fields["clip_ratio"]),
        "algorithm.clip_ratio_low": str(fields["clip_ratio"]),
        "env.train.max_episode_steps": str(fields["interaction_steps"]),
        "env.train.max_steps_per_rollout_epoch": str(fields["interaction_steps"]),
        "env.eval.max_episode_steps": str(fields["interaction_steps"]),
        "env.eval.max_steps_per_rollout_epoch": str(fields["interaction_steps"]),
        "actor.optim.lr": str(fields["actor_lr"]),
        "actor.optim.value_lr": str(fields["critic_lr"]),
        "actor.model.openpi.config_name": protocol["approved_model"]["config_name"],
        "actor.model.num_action_chunks": str(fields["action_replan_horizon"]),
        "actor.model.num_steps": str(fields["denoise_steps"]),
        "actor.model.openpi.noise_logvar_range": (
            f"[{fields['flow_noise_min_logvar']},{fields['flow_noise_max_logvar']}]"
        ),
        "algorithm.entropy_bonus": str(fields["flow_noise_entropy_bonus"]),
        "actor.optim.lr_scheduler": scheduler,
        "actor.model.openpi.noise_method": "flow_noise",
    }


def expected_evaluation_overrides(protocol: dict[str, Any], suite: str) -> dict[str, str]:
    suite_protocol = protocol["suites"][suite]
    official = protocol["official_evaluation"]
    interaction_steps = str(suite_protocol["interaction_steps"])
    return {
        "algorithm.eval_rollout_epoch": "1",
        "env.eval.total_num_envs": str(official["total_states"]),
        "env.eval.auto_reset": "true",
        "env.eval.ignore_terminations": "true",
        "env.eval.group_size": "1",
        "env.eval.use_fixed_reset_state_ids": "true",
        "env.eval.use_ordered_reset_state_ids": "true",
        "env.eval.max_episode_steps": interaction_steps,
        "env.eval.max_steps_per_rollout_epoch": interaction_steps,
        "env.eval.video_cfg.save_video": "false",
        "actor.model.num_action_chunks": str(suite_protocol["action_replan_horizon"]),
        "actor.model.num_steps": str(suite_protocol["denoise_steps"]),
        "actor.model.openpi.noise_method": "flow_noise",
    }


def command_errors(
    command_file: Path,
    required: dict[str, str],
    expected_config: str,
) -> list[str]:
    try:
        overrides, duplicates, config_name = parse_command(command_file)
    except (OSError, ValueError) as exc:
        return [str(exc)]
    errors = [f"duplicate command override: {key}" for key in duplicates]
    if config_name != expected_config:
        errors.append(f"config name must be {expected_config}, got {config_name}")
    for key, expected in required.items():
        actual = overrides.get(key)
        if not override_values_equal(actual, expected):
            errors.append(f"command {key}: expected {expected!r}, got {actual!r}")
    return errors


def override_values_equal(actual: str | None, expected: str) -> bool:
    if actual == expected:
        return True
    if actual is None:
        return False
    try:
        return json.loads(actual.lower()) == json.loads(expected.lower())
    except json.JSONDecodeError:
        return False


def is_immutable_provenance(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_PROVENANCE_PATTERN.fullmatch(value))


def verify_file_reference(
    artifact: dict[str, Any],
    path_field: str,
    hash_field: str,
    errors: list[str],
) -> None:
    path_value = artifact.get(path_field)
    expected_hash = artifact.get(hash_field)
    if not isinstance(path_value, str) or not Path(path_value).is_file():
        errors.append(f"{path_field} must reference an existing file")
        return
    actual_hash = sha256_file(Path(path_value))
    if expected_hash != actual_hash:
        errors.append(f"{hash_field} does not match {path_field}")


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
    if artifact.get("approved_model_id") != protocol["approved_model"]["id"]:
        errors.append("approved_model_id does not match protocol")
    model_path = Path(str(artifact.get("model_path", "")))
    try:
        actual_model_digest = sha256_tree(model_path)
    except ValueError as exc:
        errors.append(str(exc))
    else:
        if artifact.get("model_provenance") != f"sha256:{actual_model_digest}":
            errors.append("model_provenance does not match model_path")
    command_path = Path(str(artifact.get("command_file", "")))
    verify_file_reference(artifact, "command_file", "command_sha256", errors)
    try:
        required_overrides = {
            **expected_training_overrides(protocol, suite),
            **expected_method_overrides(str(artifact.get("method", ""))),
        }
    except ValueError as exc:
        errors.append(str(exc))
        required_overrides = expected_training_overrides(protocol, suite)
    errors.extend(
        command_errors(
            command_path,
            required_overrides,
            protocol["suites"][suite]["config_name"],
        )
    )
    for field, expected in expected_training_fields(protocol, suite).items():
        actual = artifact.get("training", {}).get(field)
        if actual != expected:
            errors.append(f"training.{field}: expected {expected!r}, got {actual!r}")
    return ValidationResult(not errors, tuple(errors))


def validate_evaluation_artifact(
    artifact: dict[str, Any],
    protocol: dict[str, Any],
) -> ValidationResult:
    errors: list[str] = []
    official = protocol["official_evaluation"]
    suite = artifact.get("suite")
    if suite not in protocol["suites"]:
        return ValidationResult(False, (f"unsupported suite: {suite}",))
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
        task_ids = [task.get("task_id") for task in task_results]
        success_total = sum(task.get("successes", -1) for task in task_results)
        if set(task_ids) != set(range(official["task_count"])):
            errors.append("task_results task IDs must be exactly 0..9")
        if any(task.get("evaluated_states") != official["states_per_task"] for task in task_results):
            errors.append(f"every task must contain {official['states_per_task']} states")
        if artifact.get("successes") != success_total:
            errors.append(f"successes must equal task result sum {success_total}")
        expected_rate = success_total / official["total_states"]
        if artifact.get("success_rate") != expected_rate:
            errors.append(f"success_rate must equal {expected_rate}")
        if artifact.get("success_percent") != expected_rate * 100:
            errors.append(f"success_percent must equal {expected_rate * 100}")
    for path_field, hash_field in (
        ("checkpoint_load_receipt", "checkpoint_load_receipt_sha256"),
        ("command_file", "command_sha256"),
        ("raw_episode_artifact", "raw_episode_sha256"),
        ("training_artifact", "training_artifact_sha256"),
        ("terminal_status_file", "terminal_status_sha256"),
        ("reproducibility_bundle", "reproducibility_bundle_sha256"),
    ):
        verify_file_reference(artifact, path_field, hash_field, errors)
    checkpoint_path = Path(str(artifact.get("checkpoint_path", "")))
    if not checkpoint_path.is_file():
        errors.append("checkpoint_path must reference existing full_weights.pt")
    elif artifact.get("checkpoint_provenance") != f"sha256:{sha256_file(checkpoint_path)}":
        errors.append("checkpoint_provenance does not match checkpoint_path")
    receipt_path = Path(str(artifact.get("checkpoint_load_receipt", "")))
    if receipt_path.is_file() and checkpoint_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("checkpoint_loaded") is not True:
                errors.append("checkpoint receipt must confirm checkpoint_loaded")
            if receipt.get("checkpoint_path") != str(checkpoint_path):
                errors.append("checkpoint receipt path must match checkpoint_path")
            if receipt.get("checkpoint_sha256") != sha256_file(checkpoint_path):
                errors.append("checkpoint receipt SHA-256 must match checkpoint_path")
            state_dict_keys = receipt.get("state_dict_keys")
            if not isinstance(state_dict_keys, int) or state_dict_keys <= 0:
                errors.append("checkpoint receipt state_dict_keys must be positive")
            if artifact.get("checkpoint_state_dict_keys") != state_dict_keys:
                errors.append("checkpoint_state_dict_keys must match checkpoint receipt")
        except (json.JSONDecodeError, OSError, TypeError):
            errors.append("checkpoint receipt must contain valid JSON")
    terminal_path = Path(str(artifact.get("terminal_status_file", "")))
    if terminal_path.is_file():
        try:
            terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
            if terminal.get("status") != "done" or terminal.get("exit_code") != 0:
                errors.append("terminal status must record done with exit_code 0")
        except (json.JSONDecodeError, OSError, TypeError):
            errors.append("terminal status must contain valid JSON")
    errors.extend(raw_episode_errors(artifact, official, task_results))
    training_path = Path(str(artifact.get("training_artifact", "")))
    if training_path.is_file():
        try:
            training = json.loads(training_path.read_text(encoding="utf-8"))
            training_result = validate_training_artifact(training, protocol)
            if not training_result.compliant:
                errors.append("training_artifact is not compliant")
            try:
                overrides, _, _ = parse_command(Path(str(artifact.get("command_file", ""))))
            except (OSError, ValueError):
                overrides = {}
            for key in ("actor.model.model_path", "rollout.model.model_path"):
                if overrides.get(key) != training.get("model_path"):
                    errors.append(f"evaluation {key} must match training model_path")
            if overrides.get("runner.ckpt_path") != str(checkpoint_path):
                errors.append("evaluation runner.ckpt_path must match checkpoint_path")
            if artifact.get("method") != training.get("method"):
                errors.append("evaluation method must match training method")
        except (json.JSONDecodeError, OSError, TypeError):
            errors.append("training_artifact must contain valid JSON")
    errors.extend(reproducibility_errors(artifact, training_path))
    errors.extend(
        command_errors(
            Path(str(artifact.get("command_file", ""))),
            expected_evaluation_overrides(protocol, suite),
            protocol["suites"][suite]["config_name"],
        )
    )
    return ValidationResult(not errors, tuple(errors))


def write_training_artifact(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    model_digest = sha256_tree(args.model_path)
    artifact = {
        "protocol_id": protocol["protocol_id"],
        "suite": args.suite,
        "method": args.method,
        "approved_model_id": protocol["approved_model"]["id"],
        "model_path": str(args.model_path.resolve()),
        "model_provenance": f"sha256:{model_digest}",
        "command_file": str(args.command_file.resolve()),
        "command_sha256": sha256_file(args.command_file),
        "training": expected_training_fields(protocol, args.suite),
        "official_comparison_eligible": False,
    }
    result = validate_training_artifact(artifact, protocol)
    artifact["training_protocol_compliant"] = result.compliant
    artifact["training_protocol_errors"] = list(result.errors)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    if not result.compliant:
        raise SystemExit("; ".join(result.errors))


def validate_artifact(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    result = (
        validate_training_artifact(artifact, protocol)
        if args.kind == "training"
        else validate_evaluation_artifact(artifact, protocol)
    )
    print(json.dumps({"compliant": result.compliant, "errors": result.errors}))
    if not result.compliant:
        raise SystemExit(1)


def digest_path(args: argparse.Namespace) -> None:
    print(f"sha256:{sha256_tree(args.path)}")


def digest_file(args: argparse.Namespace) -> None:
    print(f"sha256:{sha256_file(args.path)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    emit = subparsers.add_parser("emit-training")
    emit.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    emit.add_argument("--output", type=Path, required=True)
    emit.add_argument("--command-file", type=Path, required=True)
    emit.add_argument("--model-path", type=Path, required=True)
    emit.add_argument("--suite", required=True)
    emit.add_argument("--method", required=True)
    emit.set_defaults(func=write_training_artifact)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    validate.add_argument("--kind", choices=("training", "evaluation"), required=True)
    validate.add_argument("--artifact", type=Path, required=True)
    validate.set_defaults(func=validate_artifact)
    digest = subparsers.add_parser("digest")
    digest.add_argument("--path", type=Path, required=True)
    digest.set_defaults(func=digest_path)
    file_digest = subparsers.add_parser("digest-file")
    file_digest.add_argument("--path", type=Path, required=True)
    file_digest.set_defaults(func=digest_file)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
