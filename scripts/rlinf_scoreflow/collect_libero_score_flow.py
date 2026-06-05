#!/usr/bin/env python3
"""Collect RLinf Score-Flow benchmark results into CSV summaries."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any


KNOWN_SUITES = (
    "metaworld_mt50",
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "calvin_d_d",
    "maniskill",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument("--collapse-window", type=int, default=3)
    parser.add_argument("--collapse-threshold", type=float, default=0.25)
    return parser.parse_args()


def load_manifest(exp_root: Path) -> pd.DataFrame:
    import pandas as pd

    path = exp_root / "run_manifest.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    for col in ["suite", "method", "seed", "run_name"]:
        if col in frame:
            frame[col] = frame[col].astype(str)
    return frame


def split_run_name(run_name: str) -> tuple[str, str, str]:
    if "_seed" not in run_name:
        return "unknown", run_name, "unknown"
    prefix, seed = run_name.rsplit("_seed", 1)
    for suite in KNOWN_SUITES:
        marker = f"{suite}_"
        if prefix.startswith(marker):
            return suite, prefix[len(marker) :], seed
    return "unknown", prefix, seed


def load_scalars(exp_root: Path) -> pd.DataFrame:
    import pandas as pd

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for event_path in sorted(exp_root.glob("**/events.out.tfevents.*")):
        run_name = next(
            (parent.name for parent in [event_path.parent, *event_path.parents] if "_seed" in parent.name),
            event_path.parent.name,
        )
        suite, method, seed = split_run_name(run_name)
        try:
            acc = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
            acc.Reload()
        except Exception:
            continue
        for tag in acc.Tags().get("scalars", []):
            for scalar in acc.Scalars(tag):
                rows.append(
                    {
                        "suite": suite,
                        "method": method,
                        "seed": seed,
                        "run_name": run_name,
                        "tag": tag,
                        "step": scalar.step,
                        "value": scalar.value,
                        "event_path": str(event_path),
                    }
                )
    return pd.DataFrame(rows)


def select_tag(scalars: pd.DataFrame, keywords: tuple[str, ...]) -> str | None:
    if scalars.empty:
        return None
    tags = sorted(str(tag) for tag in scalars["tag"].dropna().unique())
    for tag in tags:
        lower = tag.lower()
        if "eval" in lower and all(keyword in lower for keyword in keywords):
            return tag
    for tag in tags:
        lower = tag.lower()
        if all(keyword in lower for keyword in keywords):
            return tag
    return None


def selected_metric_tags(scalars: pd.DataFrame) -> dict[str, str]:
    if scalars.empty:
        return {}
    candidates = {
        "final_success": ("success",),
        "final_return": ("return",),
        "final_reward": ("reward",),
        "final_avg_subtasks": ("subtask",),
        "final_len1": ("len", "1"),
        "final_len2": ("len", "2"),
        "final_len3": ("len", "3"),
        "final_len4": ("len", "4"),
        "final_len5": ("len", "5"),
        "final_approx_kl": ("approx", "kl"),
        "final_scalar_l2_disp": ("scalar", "l2", "disp"),
        "final_pullback_disp": ("pullback", "disp"),
        "final_terminal_pullback_loss": ("terminal", "pullback", "loss"),
        "final_cr_reflow_loss": ("cr", "reflow", "loss"),
        "final_cr_reflow_anchor_loss": ("cr", "reflow", "anchor"),
        "final_cr_reflow_eta": ("cr", "reflow", "eta"),
        "final_cr_reflow_weight_ess": ("cr", "reflow", "ess"),
        "final_cr_reflow_weight_max": ("cr", "reflow", "weight", "max"),
        "final_cr_reflow_valid_fraction": ("cr", "reflow", "valid", "fraction"),
        "final_cr_reflow_chain_kl_proxy": ("cr", "reflow", "chain", "kl"),
        "final_cr_reflow_chain_displacement": ("cr", "reflow", "chain", "displacement"),
    }
    tags: dict[str, str] = {}
    for name, keywords in candidates.items():
        tag = select_tag(scalars, keywords)
        if tag is not None:
            tags[name] = tag
    return tags


def collapse_flag(values: pd.DataFrame, window: int, threshold: float) -> bool | float:
    if values.empty:
        return math.nan
    tail = values.sort_values("step")["value"].tail(max(window, 1))
    return bool((tail <= threshold).all())


def summarize(
    manifest: pd.DataFrame,
    scalars: pd.DataFrame,
    min_seeds: int,
    collapse_window: int,
    collapse_threshold: float,
) -> pd.DataFrame:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    metric_tags = selected_metric_tags(scalars)
    if scalars.empty:
        return manifest.copy()

    for run_name, group in scalars.groupby("run_name"):
        suite, method, seed = split_run_name(str(run_name))
        row: dict[str, Any] = {"suite": suite, "method": method, "seed": seed, "run_name": run_name}
        for name, tag in metric_tags.items():
            values = group[group["tag"] == tag].sort_values("step")
            row[name] = values["value"].iloc[-1] if not values.empty else math.nan
            row[f"{name}_tag"] = tag
            if name == "final_success":
                row["terminal_collapse"] = collapse_flag(
                    values,
                    collapse_window,
                    collapse_threshold,
                )
        rows.append(row)

    summary = pd.DataFrame(rows)
    if not manifest.empty:
        summary = manifest.merge(summary, on=["suite", "method", "seed", "run_name"], how="outer")

    agg_rows: list[dict[str, Any]] = []
    complete = summary[summary.get("status", "done").fillna("done") == "done"]
    for (suite, method), group in complete.groupby(["suite", "method"]):
        seed_count = group["seed"].astype(str).nunique()
        row = {"suite": suite, "method": method, "num_seeds": seed_count, "main_table": seed_count >= min_seeds}
        for metric in metric_tags:
            if metric in group:
                row[f"{metric}_mean"] = group[metric].mean()
                row[f"{metric}_std"] = group[metric].std()
        if "terminal_collapse" in group:
            row["terminal_collapse_rate"] = group["terminal_collapse"].mean()
        agg_rows.append(row)
    aggregate = pd.DataFrame(agg_rows)
    if aggregate.empty:
        return summary
    return summary.merge(aggregate, on=["suite", "method"], how="left")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.exp_root)
    scalars = load_scalars(args.exp_root)
    summary = summarize(
        manifest,
        scalars,
        args.min_seeds,
        args.collapse_window,
        args.collapse_threshold,
    )

    manifest.to_csv(args.output_dir / "scoreflow_benchmark_manifest.csv", index=False)
    scalars.to_csv(args.output_dir / "scoreflow_benchmark_raw_scalars.csv", index=False)
    summary.to_csv(args.output_dir / "scoreflow_benchmark_summary.csv", index=False)


if __name__ == "__main__":
    main()
