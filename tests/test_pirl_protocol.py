from __future__ import annotations

import unittest

from scripts.rlinf_scoreflow.pirl_protocol import (
    expected_training_fields,
    load_protocol,
    validate_evaluation_artifact,
    validate_training_artifact,
)


class PiRLProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_protocol()
        self.provenance = f"sha256:{'a' * 64}"
        self.command_sha256 = "b" * 64

    def test_published_spatial_training_contract_is_compliant(self) -> None:
        artifact = {
            "protocol_id": self.protocol["protocol_id"],
            "suite": "libero_spatial",
            "model_provenance": self.provenance,
            "command_sha256": self.command_sha256,
            "training": expected_training_fields(self.protocol, "libero_spatial"),
        }

        result = validate_training_artifact(artifact, self.protocol)

        self.assertTrue(result.compliant)
        self.assertEqual(result.errors, ())

    def test_old_reduced_budget_run_is_not_compliant(self) -> None:
        training = expected_training_fields(self.protocol, "libero_spatial")
        training.update(
            {
                "train_epochs": 15,
                "global_batch_size": 8,
                "parallel_environments": 4,
                "rollout_epochs": 1,
                "denoise_steps": 10,
            }
        )
        artifact = {
            "protocol_id": self.protocol["protocol_id"],
            "suite": "libero_spatial",
            "model_provenance": self.provenance,
            "command_sha256": self.command_sha256,
            "training": training,
        }

        result = validate_training_artifact(artifact, self.protocol)

        self.assertFalse(result.compliant)
        self.assertEqual(len(result.errors), 5)

    def test_training_scalar_cannot_pass_official_evaluation(self) -> None:
        result = validate_evaluation_artifact(
            {
                "status": "done",
                "total_states": 8,
                "checkpoint_provenance": self.provenance,
            },
            self.protocol,
        )

        self.assertFalse(result.compliant)
        self.assertIn("total_states must be 500", result.errors)

    def test_complete_500_state_evaluation_is_compliant(self) -> None:
        result = validate_evaluation_artifact(
            {
                "protocol_id": self.protocol["protocol_id"],
                "suite": "libero_spatial",
                "status": "done",
                "total_states": 500,
                "successes": 500,
                "checkpoint_provenance": self.provenance,
                "raw_episode_artifact": "episodes.jsonl",
                "task_results": [
                    {"task_id": task_id, "evaluated_states": 50, "successes": 50}
                    for task_id in range(10)
                ],
            },
            self.protocol,
        )

        self.assertTrue(result.compliant)


if __name__ == "__main__":
    unittest.main()
