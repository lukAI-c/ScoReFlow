from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

from scripts.rlinf_scoreflow.pirl_evidence import expected_method_overrides
from scripts.rlinf_scoreflow.pirl_protocol import (
    expected_training_fields,
    expected_training_overrides,
    load_protocol,
    sha256_file,
    sha256_tree,
    validate_evaluation_artifact,
    validate_training_artifact,
    write_training_artifact,
)


class PiRLProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_protocol()

    def make_training_artifact(
        self,
        root: Path,
        *,
        extra_tokens: tuple[str, ...] = (),
    ) -> dict[str, object]:
        model = root / "model"
        model.mkdir()
        (model / "weights.bin").write_bytes(b"pi05-sft")
        suite = "libero_spatial"
        overrides = expected_training_overrides(self.protocol, suite)
        overrides.update(expected_method_overrides("flow_noise_baseline"))
        command = root / "command.txt"
        tokens = [
            "python",
            "train.py",
            "--config-name",
            self.protocol["suites"][suite]["config_name"],
            *(f"{key}={value}" for key, value in overrides.items()),
            *extra_tokens,
        ]
        command.write_text(" ".join(tokens) + "\n", encoding="utf-8")
        return {
            "protocol_id": self.protocol["protocol_id"],
            "suite": suite,
            "method": "flow_noise_baseline",
            "approved_model_id": self.protocol["approved_model"]["id"],
            "model_path": str(model),
            "model_provenance": f"sha256:{sha256_tree(model)}",
            "command_file": str(command),
            "command_sha256": sha256_file(command),
            "training": expected_training_fields(self.protocol, suite),
        }

    def test_published_spatial_training_contract_is_compliant(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = self.make_training_artifact(Path(temp_dir))

            result = validate_training_artifact(artifact, self.protocol)

            self.assertTrue(result.compliant, result.errors)

    def test_duplicate_locked_override_is_not_compliant(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = self.make_training_artifact(
                Path(temp_dir),
                extra_tokens=("runner.max_epochs=1",),
            )
            artifact["command_sha256"] = sha256_file(Path(str(artifact["command_file"])))

            result = validate_training_artifact(artifact, self.protocol)

            self.assertFalse(result.compliant)
            self.assertTrue(any("duplicate command override" in error for error in result.errors))
            self.assertTrue(any("runner.max_epochs" in error for error in result.errors))

    def test_every_locked_training_override_rejects_trailing_replacement(self) -> None:
        for key in expected_training_overrides(self.protocol, "libero_spatial"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp_dir:
                artifact = self.make_training_artifact(
                    Path(temp_dir),
                    extra_tokens=(f"{key}=invalid",),
                )
                artifact["command_sha256"] = sha256_file(Path(str(artifact["command_file"])))

                result = validate_training_artifact(artifact, self.protocol)

                self.assertFalse(result.compliant)
                self.assertTrue(any(key in error for error in result.errors))

    def test_tampered_command_hash_is_not_compliant(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = self.make_training_artifact(Path(temp_dir))
            Path(str(artifact["command_file"])).write_text("tampered\n", encoding="utf-8")

            result = validate_training_artifact(artifact, self.protocol)

            self.assertFalse(result.compliant)
            self.assertIn("command_sha256 does not match command_file", result.errors)

    def test_old_reduced_budget_field_is_not_compliant(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = self.make_training_artifact(Path(temp_dir))
            artifact["training"] = dict(artifact["training"])
            artifact["training"]["train_epochs"] = 15

            result = validate_training_artifact(artifact, self.protocol)

            self.assertFalse(result.compliant)
            self.assertTrue(any("training.train_epochs" in error for error in result.errors))

    def test_direct_screening_emitter_rejects_full_budget(self) -> None:
        with self.assertRaisesRegex(SystemExit, r"must be in \[1, 499\]"):
            write_training_artifact(
                argparse.Namespace(
                    protocol=Path(__file__).parents[1]
                    / "scripts/rlinf_scoreflow/protocols/pirl_pi05_libero.json",
                    screening_max_epochs=500,
                )
            )

    def test_screening_emitter_requires_command_budget_to_match_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact = self.make_training_artifact(root)
            output = root / "screening.json"

            with self.assertRaises(SystemExit):
                write_training_artifact(
                    argparse.Namespace(
                        protocol=Path(__file__).parents[1]
                        / "scripts/rlinf_scoreflow/protocols/pirl_pi05_libero.json",
                        output=output,
                        command_file=Path(str(artifact["command_file"])),
                        model_path=Path(str(artifact["model_path"])),
                        suite="libero_spatial",
                        method="flow_noise_baseline",
                        screening_max_epochs=20,
                    )
                )

    def test_method_label_must_match_final_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = self.make_training_artifact(Path(temp_dir))
            artifact["method"] = "cr_reflow"

            result = validate_training_artifact(artifact, self.protocol)

            self.assertFalse(result.compliant)
            self.assertTrue(
                any("cr_reflow_mode" in error for error in result.errors),
                result.errors,
            )

    def test_training_scalar_cannot_pass_official_evaluation(self) -> None:
        result = validate_evaluation_artifact(
            {
                "protocol_id": self.protocol["protocol_id"],
                "suite": "libero_spatial",
                "status": "done",
                "total_states": 8,
            },
            self.protocol,
        )

        self.assertFalse(result.compliant)
        self.assertIn("total_states must be 500", result.errors)

    def test_nonexistent_evaluation_evidence_is_rejected(self) -> None:
        result = validate_evaluation_artifact(
            {
                "protocol_id": self.protocol["protocol_id"],
                "suite": "libero_spatial",
                "status": "done",
                "total_states": 500,
                "successes": 500,
                "success_rate": 1.0,
                "success_percent": 100.0,
                "task_results": [
                    {"task_id": task_id, "evaluated_states": 50, "successes": 50}
                    for task_id in range(10)
                ],
                "checkpoint_path": "/missing/full_weights.pt",
                "command_file": "/missing/command.txt",
                "raw_episode_artifact": "/missing/episodes.jsonl",
                "checkpoint_load_receipt": "/missing/receipt.json",
                "training_artifact": "/missing/training.json",
                "terminal_status_file": "/missing/terminal.json",
            },
            self.protocol,
        )

        self.assertFalse(result.compliant)
        self.assertTrue(any("existing file" in error for error in result.errors))


if __name__ == "__main__":
    unittest.main()
