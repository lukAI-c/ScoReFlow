from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.rlinf_scoreflow.patch_rlinf_pirl_evaluator import (
    LIBERO_ANCHOR,
    RUNNER_EVALUATE_ANCHOR,
    RUNNER_IMPORT_ANCHOR,
    ROLLOUT_CHECKPOINT_ANCHOR,
    ROLLOUT_IMPORT_ANCHOR,
    patch_libero_text,
    patch_runner_text,
    patch_rollout_text,
)
from scripts.rlinf_scoreflow.pirl_official_eval import build_artifact, read_checkpoint_receipt
from scripts.rlinf_scoreflow.pirl_protocol import (
    expected_evaluation_overrides,
    expected_training_fields,
    expected_training_overrides,
    load_protocol,
    sha256_file,
    sha256_tree,
    validate_evaluation_artifact,
)


class PiRLOfficialEvalTest(unittest.TestCase):
    def make_valid_inputs(
        self,
        root: Path,
        *,
        duplicate_states: bool = False,
        eval_extra_tokens: tuple[str, ...] = (),
    ) -> argparse.Namespace:
        protocol_path = Path("scripts/rlinf_scoreflow/protocols/pirl_pi05_libero.json")
        protocol = load_protocol(protocol_path)
        suite = "libero_spatial"
        model = root / "model"
        model.mkdir()
        (model / "weights.bin").write_bytes(b"pi05-sft")
        training_command = root / "training-command.txt"
        training_command.write_text(
            " ".join(
                [
                    "python",
                    "train.py",
                    "--config-name",
                    protocol["suites"][suite]["config_name"],
                    *(
                        f"{key}={value}"
                        for key, value in expected_training_overrides(protocol, suite).items()
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        training_artifact = root / "training-artifact.json"
        training_artifact.write_text(
            json.dumps(
                {
                    "protocol_id": protocol["protocol_id"],
                    "suite": suite,
                    "method": "flow_noise_baseline",
                    "approved_model_id": protocol["approved_model"]["id"],
                    "model_path": str(model),
                    "model_provenance": f"sha256:{sha256_tree(model)}",
                    "command_file": str(training_command),
                    "command_sha256": sha256_file(training_command),
                    "training": expected_training_fields(protocol, suite),
                }
            ),
            encoding="utf-8",
        )
        checkpoint = root / "checkpoint/actor/model_state_dict/full_weights.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"rl-weights")
        checkpoint_sha = sha256_file(checkpoint)
        receipt = root / "checkpoint_load_receipt.json"
        receipt.write_text(
            json.dumps(
                {
                    "checkpoint_loaded": True,
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha,
                    "state_dict_keys": 123,
                    "rollout_rank": 0,
                }
            ),
            encoding="utf-8",
        )
        raw = root / "episodes.jsonl"
        rows = [
            {
                "reset_state_id": 0 if duplicate_states else task_id * 50 + trial_id,
                "task_id": task_id,
                "trial_id": trial_id,
                "success": not (task_id == 9 and trial_id == 49),
            }
            for task_id in range(10)
            for trial_id in range(50)
        ]
        raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        command = root / "evaluation-command.txt"
        command.write_text(
            " ".join(
                [
                    "python",
                    "eval.py",
                    "--config-name",
                    protocol["suites"][suite]["config_name"],
                    *(f"{key}={value}" for key, value in expected_evaluation_overrides(protocol, suite).items()),
                    f"actor.model.model_path={model}",
                    f"rollout.model.model_path={model}",
                    f"runner.ckpt_path={checkpoint}",
                    *eval_extra_tokens,
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        terminal = root / "terminal_status.json"
        terminal.write_text('{"status":"done","exit_code":0}\n', encoding="utf-8")
        base_config = root / "libero_spatial_ppo_openpi_pi05.yaml"
        base_config.write_text("{}\n", encoding="utf-8")
        reproducibility = root / "reproducibility.json"
        reproducibility.write_text(
            json.dumps(
                {
                    "status": "done",
                    "exit_code": 0,
                    "command_file": str(command.resolve()),
                    "command_sha256": sha256_file(command),
                    "protocol_artifact": str(training_artifact.resolve()),
                    "protocol_artifact_sha256": sha256_file(training_artifact),
                    "config_snapshot": {
                        "config_name": protocol["suites"][suite]["config_name"],
                        "base_config": str(base_config.resolve()),
                        "base_config_sha256": sha256_file(base_config),
                    },
                }
            ),
            encoding="utf-8",
        )
        return argparse.Namespace(
            protocol=protocol_path,
            raw_episodes=raw,
            command_file=command,
            checkpoint_receipt=receipt,
            training_artifact=training_artifact,
            terminal_status=terminal,
            reproducibility_bundle=reproducibility,
            output=root / "artifact.json",
            suite=suite,
            method="flow_noise_baseline",
            checkpoint_provenance=f"sha256:{checkpoint_sha}",
        )

    def test_build_artifact_requires_and_summarizes_exact_500_states(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = build_artifact(self.make_valid_inputs(Path(temp_dir)))

            self.assertEqual(artifact["successes"], 499)
            self.assertEqual(artifact["success_percent"], 99.8)
            self.assertTrue(artifact["official_comparison_eligible"])

    def test_duplicate_state_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self.make_valid_inputs(Path(temp_dir), duplicate_states=True)

            with self.assertRaisesRegex(ValueError, "reset_state_id values must be unique"):
                build_artifact(args)

    def test_wrong_evaluation_command_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self.make_valid_inputs(
                Path(temp_dir),
                eval_extra_tokens=("env.eval.total_num_envs=8",),
            )

            with self.assertRaisesRegex(ValueError, "duplicate command override"):
                build_artifact(args)

    def test_remote_patch_is_idempotent(self) -> None:
        libero, changed = patch_libero_text(LIBERO_ANCHOR)
        self.assertTrue(changed)
        self.assertFalse(patch_libero_text(libero)[1])
        runner, changed = patch_runner_text(RUNNER_IMPORT_ANCHOR + RUNNER_EVALUATE_ANCHOR)
        self.assertTrue(changed)
        self.assertFalse(patch_runner_text(runner)[1])
        rollout, changed = patch_rollout_text(ROLLOUT_IMPORT_ANCHOR + ROLLOUT_CHECKPOINT_ANCHOR)
        self.assertTrue(changed)
        self.assertFalse(patch_rollout_text(rollout)[1])

    def test_checkpoint_receipt_must_match_actual_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self.make_valid_inputs(Path(temp_dir))
            receipt = json.loads(args.checkpoint_receipt.read_text(encoding="utf-8"))
            receipt["checkpoint_sha256"] = "b" * 64
            args.checkpoint_receipt.write_text(json.dumps(receipt), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "does not match provenance"):
                read_checkpoint_receipt(args.checkpoint_receipt, args.checkpoint_provenance)

    def test_malformed_terminal_evidence_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self.make_valid_inputs(Path(temp_dir))
            artifact = build_artifact(args)
            args.terminal_status.write_text("{invalid", encoding="utf-8")
            artifact["terminal_status_sha256"] = sha256_file(args.terminal_status)

            result = validate_evaluation_artifact(
                artifact,
                load_protocol(args.protocol),
            )

            self.assertFalse(result.compliant)
            self.assertIn("terminal status must contain valid JSON", result.errors)

    def test_prepare_uses_live_rlinf_denoise_config_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = self.make_valid_inputs(root)
            checkpoint_dir = Path(json.loads(args.checkpoint_receipt.read_text())["checkpoint_path"]).parents[2]
            config_dir = root / "rlinf/examples/embodiment/config"
            config_dir.mkdir(parents=True)
            (config_dir / "libero_spatial_ppo_openpi_pi05.yaml").write_text(
                "{}\n",
                encoding="utf-8",
            )
            env = {
                **os.environ,
                "RLINF_ROOT": str(root / "rlinf"),
                "MODEL_DIR": str(root / "model"),
                "RL_CHECKPOINT_DIR": str(checkpoint_dir),
                "CHECKPOINT_PROVENANCE": args.checkpoint_provenance,
                "TRAINING_ARTIFACT": str(args.training_artifact),
                "EXP_ROOT": str(root / "logs"),
                "PYTHON_BIN": sys.executable,
                "PATCH_RLINF": "0",
                "PREPARE_ONLY": "1",
            }
            subprocess.run(
                ["bash", "scripts/rlinf_scoreflow/run_pirl_official_evaluation.sh"],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )
            command = (
                root
                / "logs"
                / "libero_spatial_flow_noise_baseline_official_eval"
                / "command.txt"
            ).read_text(encoding="utf-8")

            self.assertIn("actor.model.num_steps=3", command)
            self.assertNotIn("actor.model.openpi.num_steps", command)
            self.assertTrue(
                (
                    root
                    / "logs"
                    / "libero_spatial_flow_noise_baseline_official_eval"
                    / "reproducibility.json"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
