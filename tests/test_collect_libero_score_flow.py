from __future__ import annotations

import unittest

import pandas as pd

from scripts.rlinf_scoreflow.collect_libero_score_flow import summarize


def scalar_row(run_name: str, seed: str, value: float) -> dict[str, object]:
    return {
        "suite": "libero_spatial",
        "method": "cr_reflow",
        "seed": seed,
        "run_name": run_name,
        "tag": "eval/success_at_end",
        "step": 0,
        "value": value,
        "event_path": f"/tmp/{run_name}/events.out.tfevents.test",
    }


def manifest_row(run_name: str, seed: str, status: str | None) -> dict[str, object]:
    return {
        "suite": "libero_spatial",
        "method": "cr_reflow",
        "seed": seed,
        "run_name": run_name,
        "status": status,
    }


class CollectorStatusTest(unittest.TestCase):
    def test_only_explicit_done_rows_enter_aggregates(self) -> None:
        manifest = pd.DataFrame(
            [
                manifest_row("libero_spatial_cr_reflow_seed42", "42", "done"),
                manifest_row("libero_spatial_cr_reflow_seed43", "43", "prepared"),
                manifest_row("libero_spatial_cr_reflow_seed44", "44", "running"),
                manifest_row("libero_spatial_cr_reflow_seed45", "45", "failed"),
                manifest_row("libero_spatial_cr_reflow_seed46", "46", None),
            ]
        )
        scalars = pd.DataFrame(
            [
                scalar_row("libero_spatial_cr_reflow_seed42", "42", 1.0),
                scalar_row("libero_spatial_cr_reflow_seed43", "43", 0.8),
                scalar_row("libero_spatial_cr_reflow_seed44", "44", 0.6),
                scalar_row("libero_spatial_cr_reflow_seed45", "45", 0.4),
                scalar_row("libero_spatial_cr_reflow_seed46", "46", 0.2),
                scalar_row("libero_spatial_cr_reflow_seed47", "47", 0.0),
            ]
        )

        summary = summarize(manifest, scalars, 2, 3, 0.25)
        statuses = summary.set_index("run_name")["status"].to_dict()

        self.assertEqual(statuses["libero_spatial_cr_reflow_seed42"], "done")
        self.assertEqual(statuses["libero_spatial_cr_reflow_seed43"], "prepared")
        self.assertEqual(statuses["libero_spatial_cr_reflow_seed44"], "running")
        self.assertEqual(statuses["libero_spatial_cr_reflow_seed45"], "failed")
        self.assertEqual(statuses["libero_spatial_cr_reflow_seed46"], "unknown")
        self.assertEqual(statuses["libero_spatial_cr_reflow_seed47"], "unknown")
        self.assertTrue((summary["num_seeds"] == 1).all())
        self.assertFalse(summary["main_table"].any())
        self.assertTrue((summary["final_success_mean"] == 1.0).all())

    def test_scalar_only_rows_are_unknown_without_manifest(self) -> None:
        scalars = pd.DataFrame(
            [scalar_row("libero_spatial_cr_reflow_seed47", "47", 0.5)]
        )

        summary = summarize(pd.DataFrame(), scalars, 1, 3, 0.25)

        self.assertEqual(summary.loc[0, "status"], "unknown")
        self.assertNotIn("num_seeds", summary)

    def test_manifest_only_missing_status_is_unknown(self) -> None:
        manifest = pd.DataFrame(
            [manifest_row("libero_spatial_cr_reflow_seed48", "48", None)]
        )

        summary = summarize(manifest, pd.DataFrame(), 1, 3, 0.25)

        self.assertEqual(summary.loc[0, "status"], "unknown")
        self.assertNotIn("num_seeds", summary)


if __name__ == "__main__":
    unittest.main()
