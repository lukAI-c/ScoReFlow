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
        env = {
            **os.environ,
            "RLINF_ROOT": str(temp_root / "rlinf"),
            "EXP_ROOT": str(temp_root / "experiments"),
            "PYTHON_BIN": sys.executable,
            "PATCH_RLINF": "0",
            "PREPARE_ONLY": "1",
            "SEEDS": "42",
            "METHODS": "flow_noise_baseline",
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
                MODEL_PROVENANCE=f"sha256:{'a' * 64}",
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
            self.assertIn("actor.global_batch_size=2048", command)
            self.assertIn("env.train.total_num_envs=64", command)
            self.assertIn("actor.model.openpi.num_steps=3", command)
            self.assertIn("actor.model.openpi.noise_logvar_range=\\[0.04\\,0.10\\]", command)
            self.assertTrue(artifact["training_protocol_compliant"])

    def test_development_non_libero_prepare_remains_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.run_prepare(Path(temp_dir), SUITES="maniskill")

            self.assertEqual(result.returncode, 0, result.stderr)

    def test_official_non_libero_suite_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.run_prepare(
                Path(temp_dir),
                PIRL_OFFICIAL_PROTOCOL="1",
                SUITES="maniskill",
                MODEL_PROVENANCE=f"sha256:{'a' * 64}",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("only supports LIBERO suites", result.stderr)


if __name__ == "__main__":
    unittest.main()
