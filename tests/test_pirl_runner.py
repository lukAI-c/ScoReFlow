from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "rlinf_scoreflow" / "run_score_flow_benchmark.sh"


class PiRLRunnerTest(unittest.TestCase):
    def run_prepare(self, temp_root: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
        model_dir = temp_root / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "weights.bin").write_bytes(b"pi05-sft")
        config_dir = temp_root / "rlinf/examples/embodiment/config"
        config_dir.mkdir(parents=True, exist_ok=True)
        for config_name in (
            "libero_spatial_ppo_openpi_pi05",
            "libero_10_ppo_openpi_pi05",
        ):
            (config_dir / f"{config_name}.yaml").write_text("{}\n", encoding="utf-8")
        env = {
            **os.environ,
            "RLINF_ROOT": str(temp_root / "rlinf"),
            "EXP_ROOT": str(temp_root / "experiments"),
            "PYTHON_BIN": sys.executable,
            "PATCH_RLINF": "0",
            "PREPARE_ONLY": "1",
            "SEEDS": "42",
            "METHODS": "flow_noise_baseline",
            "MODEL_DIR": str(model_dir),
            **overrides,
        }
        return subprocess.run(
            ["bash", str(RUNNER)],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_official_spatial_prepare_emits_valid_exact_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            result = self.run_prepare(
                temp_root,
                PIRL_OFFICIAL_PROTOCOL="1",
                SUITES="libero_spatial",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            run_dir = (
                temp_root
                / "experiments"
                / "libero_spatial"
                / "libero_spatial_flow_noise_baseline_seed42"
            )
            command = (run_dir / "command.txt").read_text(encoding="utf-8")
            artifact = json.loads(
                (run_dir / "training_protocol.json").read_text(encoding="utf-8")
            )

            self.assertIn("runner.max_epochs=500", command)
            self.assertIn("runner.save_interval=40", command)
            self.assertIn("actor.global_batch_size=2048", command)
            self.assertIn("env.train.total_num_envs=64", command)
            self.assertIn("actor.model.openpi.config_name=pi05_libero", command)
            self.assertNotIn("actor.model.action_horizon", command)
            self.assertIn("actor.model.num_steps=3", command)
            self.assertNotIn("actor.model.openpi.num_steps", command)
            self.assertIn("++actor.model.openpi.noise_logvar_range=\\[0.04\\,0.10\\]", command)
            self.assertIn("++actor.optim.lr_scheduler=constant", command)
            self.assertTrue(artifact["training_protocol_compliant"])

    def test_official_long_prepare_uses_existing_cosine_scheduler_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            result = self.run_prepare(
                temp_root,
                PIRL_OFFICIAL_PROTOCOL="1",
                SUITES="libero_10",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            command = (
                temp_root
                / "experiments"
                / "libero_10"
                / "libero_10_flow_noise_baseline_seed42"
                / "command.txt"
            ).read_text(encoding="utf-8")
            self.assertIn("actor.optim.total_training_steps=500", command)
            self.assertIn("actor.optim.lr_scheduler=cosine", command)
            self.assertNotIn("++actor.optim.lr_scheduler=cosine", command)

    def test_cr_reflow_method_parameters_are_explicitly_tunable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            result = self.run_prepare(
                temp_root,
                PIRL_OFFICIAL_PROTOCOL="1",
                SUITES="libero_spatial",
                METHODS="cr_reflow",
                CR_REFLOW_KL_EPSILON="0.02",
                CR_REFLOW_ETA_MIN="0.005",
                CR_REFLOW_ETA_MAX="20.0",
                CR_REFLOW_WEIGHT_CLIP="5.0",
                CR_REFLOW_ANCHOR_BETA="0.03",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            command = (
                temp_root
                / "experiments"
                / "libero_spatial"
                / "libero_spatial_cr_reflow_seed42"
                / "command.txt"
            ).read_text(encoding="utf-8")
            self.assertIn("cr_reflow_kl_epsilon=0.02", command)
            self.assertIn("cr_reflow_eta_min=0.005", command)
            self.assertIn("cr_reflow_eta_max=20.0", command)
            self.assertIn("cr_reflow_weight_clip=5.0", command)
            self.assertIn("cr_reflow_anchor_beta=0.03", command)

    def test_development_non_libero_prepare_remains_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.run_prepare(Path(temp_dir), SUITES="maniskill")

            self.assertEqual(result.returncode, 0, result.stderr)

    def test_official_trailing_override_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.run_prepare(
                Path(temp_dir),
                PIRL_OFFICIAL_PROTOCOL="1",
                SUITES="libero_spatial",
                EXTRA_OVERRIDES="runner.max_epochs=1",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("EXTRA_OVERRIDES is forbidden", result.stderr)

    def test_official_non_libero_suite_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.run_prepare(
                Path(temp_dir),
                PIRL_OFFICIAL_PROTOCOL="1",
                SUITES="maniskill",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("only supports LIBERO suites", result.stderr)


if __name__ == "__main__":
    unittest.main()
