from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
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
from scripts.rlinf_scoreflow.pirl_official_eval import (
    build_artifact,
    read_checkpoint_receipt,
)


class PiRLOfficialEvalTest(unittest.TestCase):
    def test_build_artifact_requires_and_summarizes_exact_500_states(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "episodes.jsonl"
            command = root / "command.txt"
            output = root / "artifact.json"
            receipt = root / "checkpoint_load_receipt.json"
            rows = [
                {
                    "reset_state_id": task_id * 50 + trial_id,
                    "task_id": task_id,
                    "trial_id": trial_id,
                    "success": not (task_id == 9 and trial_id == 49),
                }
                for task_id in range(10)
                for trial_id in range(50)
            ]
            raw.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            command.write_text("evaluate\n", encoding="utf-8")
            receipt.write_text(
                json.dumps(
                    {
                        "checkpoint_loaded": True,
                        "checkpoint_path": "/checkpoints/global_step_500/actor/model_state_dict/full_weights.pt",
                        "checkpoint_sha256": "a" * 64,
                        "state_dict_keys": 123,
                        "rollout_rank": 0,
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                protocol=Path("scripts/rlinf_scoreflow/protocols/pirl_pi05_libero.json"),
                raw_episodes=raw,
                command_file=command,
                checkpoint_receipt=receipt,
                output=output,
                suite="libero_spatial",
                method="flow_noise_baseline",
                checkpoint_provenance=f"sha256:{'a' * 64}",
            )

            artifact = build_artifact(args)

            self.assertEqual(artifact["successes"], 499)
            self.assertEqual(artifact["success_percent"], 99.8)
            self.assertTrue(artifact["official_comparison_eligible"])
            self.assertEqual(artifact["checkpoint_state_dict_keys"], 123)
            self.assertEqual(artifact["task_results"][9]["successes"], 49)

    def test_duplicate_state_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "episodes.jsonl"
            command = root / "command.txt"
            receipt = root / "checkpoint_load_receipt.json"
            rows = [
                {
                    "reset_state_id": 0,
                    "task_id": task_id,
                    "trial_id": trial_id,
                    "success": True,
                }
                for task_id in range(10)
                for trial_id in range(50)
            ]
            raw.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            command.write_text("evaluate\n", encoding="utf-8")
            receipt.write_text(
                json.dumps(
                    {
                        "checkpoint_loaded": True,
                        "checkpoint_path": "/checkpoints/global_step_500/actor/model_state_dict/full_weights.pt",
                        "checkpoint_sha256": "a" * 64,
                        "state_dict_keys": 123,
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                protocol=Path("scripts/rlinf_scoreflow/protocols/pirl_pi05_libero.json"),
                raw_episodes=raw,
                command_file=command,
                checkpoint_receipt=receipt,
                output=root / "artifact.json",
                suite="libero_spatial",
                method="flow_noise_baseline",
                checkpoint_provenance=f"sha256:{'a' * 64}",
            )

            with self.assertRaisesRegex(ValueError, "reset_state_id values must be unique"):
                build_artifact(args)

    def test_remote_patch_is_idempotent(self) -> None:
        libero, changed = patch_libero_text(LIBERO_ANCHOR)
        self.assertTrue(changed)
        self.assertFalse(patch_libero_text(libero)[1])

        runner_source = RUNNER_IMPORT_ANCHOR + RUNNER_EVALUATE_ANCHOR
        runner, changed = patch_runner_text(runner_source)
        self.assertTrue(changed)
        self.assertFalse(patch_runner_text(runner)[1])

        rollout_source = ROLLOUT_IMPORT_ANCHOR + ROLLOUT_CHECKPOINT_ANCHOR
        rollout, changed = patch_rollout_text(rollout_source)
        self.assertTrue(changed)
        self.assertFalse(patch_rollout_text(rollout)[1])

    def test_checkpoint_receipt_must_match_expected_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            receipt = Path(temp_dir) / "receipt.json"
            receipt.write_text(
                json.dumps(
                    {
                        "checkpoint_loaded": True,
                        "checkpoint_path": "/checkpoint/actor/model_state_dict/full_weights.pt",
                        "checkpoint_sha256": "b" * 64,
                        "state_dict_keys": 1,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "does not match provenance"):
                read_checkpoint_receipt(receipt, f"sha256:{'a' * 64}")

    def test_prepare_uses_live_rlinf_denoise_config_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_dir = root / "checkpoint"
            checkpoint_weights = checkpoint_dir / "actor/model_state_dict/full_weights.pt"
            checkpoint_weights.parent.mkdir(parents=True)
            checkpoint_weights.write_bytes(b"weights")
            env = {
                **os.environ,
                "RLINF_ROOT": str(root / "rlinf"),
                "MODEL_DIR": str(root / "model"),
                "RL_CHECKPOINT_DIR": str(checkpoint_dir),
                "CHECKPOINT_PROVENANCE": f"sha256:{'a' * 64}",
                "EXP_ROOT": str(root / "logs"),
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
            self.assertIn(f"runner.ckpt_path={checkpoint_weights}", command)
            self.assertIn(f"rollout.model.model_path={root / 'model'}", command)

    def test_prepare_rejects_missing_rl_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env = {
                **os.environ,
                "RLINF_ROOT": str(root / "rlinf"),
                "MODEL_DIR": str(root / "model"),
                "RL_CHECKPOINT_DIR": str(root / "missing-checkpoint"),
                "CHECKPOINT_PROVENANCE": f"sha256:{'a' * 64}",
                "PATCH_RLINF": "0",
                "PREPARE_ONLY": "1",
            }

            result = subprocess.run(
                ["bash", "scripts/rlinf_scoreflow/run_pirl_official_evaluation.sh"],
                check=False,
                env=env,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("RL checkpoint weights not found", result.stderr)


if __name__ == "__main__":
    unittest.main()
