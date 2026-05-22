#!/usr/bin/env python3
"""Collect RLinf LIBERO Score-Flow results into CSV summaries."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-seeds", type=int, default=3)
    return parser.parse_args()


def load_manifest(exp_root: Path) -> pd.DataFrame:
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
    parts = prefix.split("_")
    if len(parts) >= 3 and parts[0] == "libero":
        return "_".join(parts[:2]), "_".join(parts[2:]), seed
    return "unknown", prefix, seed


def load_scalars(exp_root: Path) -> pd.DataFrame:
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


def summarize(manifest: pd.DataFrame, scalars: pd.DataFrame, min_seeds: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    success_tag = select_tag(scalars, ("success",))
    reward_tag = select_tag(scalars, ("reward",)) or select_tag(scalars, ("return",))
    if scalars.empty:
        return manifest.copy()

    for run_name, group in scalars.groupby("run_name"):
        suite, method, seed = split_run_name(str(run_name))
        row: dict[str, Any] = {"suite": suite, "method": method, "seed": seed, "run_name": run_name}
        for name, tag in [("final_success", success_tag), ("final_reward", reward_tag)]:
            if tag is None:
                row[name] = math.nan
                continue
            values = group[group["tag"] == tag].sort_values("step")
            row[name] = values["value"].iloc[-1] if not values.empty else math.nan
            row[f"{name}_tag"] = tag
        rows.append(row)

    summary = pd.DataFrame(rows)
    if not manifest.empty:
        summary = manifest.merge(summary, on=["suite", "method", "seed", "run_name"], how="outer")

    agg_rows: list[dict[str, Any]] = []
    complete = summary[summary.get("status", "done").fillna("done") == "done"]
    for (suite, method), group in complete.groupby(["suite", "method"]):
        seed_count = group["seed"].astype(str).nunique()
        row = {"suite": suite, "method": method, "num_seeds": seed_count, "main_table": seed_count >= min_seeds}
        for metric in ["final_success", "final_reward"]:
            if metric in group:
                row[f"{metric}_mean"] = group[metric].mean()
                row[f"{metric}_std"] = group[metric].std()
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
    summary = summarize(manifest, scalars, args.min_seeds)

    manifest.to_csv(args.output_dir / "scoreflow_libero_manifest.csv", index=False)
    scalars.to_csv(args.output_dir / "scoreflow_libero_raw_scalars.csv", index=False)
    summary.to_csv(args.output_dir / "scoreflow_libero_summary.csv", index=False)


if __name__ == "__main__":
    main()
