#!/usr/bin/env python3
"""Collect RLinf Score-Flow benchmark results into CSV summaries."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    from .pirl_protocol import load_protocol, validate_evaluation_artifact
except ImportError:
    from pirl_protocol import load_protocol, validate_evaluation_artifact


KNOWN_SUITES = (
    "metaworld_mt50",
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "calvin_d_d",
    "maniskill",
)

MetricTagSelection = tuple[str, str]


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


def load_official_evaluations(exp_root: Path) -> pd.DataFrame:
    import pandas as pd

    protocol = load_protocol()
    rows: list[dict[str, Any]] = []
    for artifact_path in sorted(exp_root.glob("**/evaluation_artifact.json")):
        try:
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            result = validate_evaluation_artifact(artifact, protocol)
            rows.append(
                {
                    "suite": artifact.get("suite"),
                    "method": artifact.get("method"),
                    "successes": artifact.get("successes"),
                    "total_states": artifact.get("total_states"),
                    "success_percent": artifact.get("success_percent"),
                    "artifact_path": str(artifact_path),
                    "official_comparison_eligible": result.compliant,
                    "validation_errors": "; ".join(result.errors),
                }
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            rows.append(
                {
                    "artifact_path": str(artifact_path),
                    "official_comparison_eligible": False,
                    "validation_errors": str(exc),
                }
            )
    return pd.DataFrame(rows)


def select_tag(
    scalars: pd.DataFrame,
    keywords: tuple[str, ...],
    preferred: tuple[str, ...] = (),
) -> tuple[str | None, str | None]:
    if scalars.empty:
        return None, None
    tags = sorted(str(tag) for tag in scalars["tag"].dropna().unique())
    tag_set = set(tags)
    for tag in preferred:
        if tag in tag_set:
            return tag, "preferred"
        train_tag = f"train/{tag}"
        if train_tag in tag_set:
            return train_tag, "preferred"
    for tag in tags:
        lower = tag.lower()
        if "eval" in lower and all(keyword in lower for keyword in keywords):
            return tag, "fallback_eval_keyword"
    for tag in tags:
        lower = tag.lower()
        if all(keyword in lower for keyword in keywords):
            return tag, "fallback_keyword"
    return None, None


def selected_metric_tags(scalars: pd.DataFrame) -> dict[str, MetricTagSelection]:
    if scalars.empty:
        return {}
    candidates = {
        "final_success": (("success",), ()),
        "final_return": (("return",), ()),
        "final_reward": (("reward",), ()),
        "final_avg_subtasks": (("subtask",), ()),
        "final_len1": (("len", "1"), ()),
        "final_len2": (("len", "2"), ()),
        "final_len3": (("len", "3"), ()),
        "final_len4": (("len", "4"), ()),
        "final_len5": (("len", "5"), ()),
        "final_approx_kl": (("approx", "kl"), ("actor/approx_kl",)),
        "final_scalar_l2_disp": (("scalar", "l2", "disp"), ("actor/scalar_l2_disp",)),
        "final_pullback_disp": (("pullback", "disp"), ("actor/pullback_disp",)),
        "final_terminal_pullback_loss": (
            ("terminal", "pullback", "loss"),
            ("actor/terminal_pullback_loss",),
        ),
        "final_cr_reflow_loss": (("cr", "reflow", "loss"), ("actor/cr_reflow_loss",)),
        "final_cr_reflow_actor_loss": (
            ("cr", "reflow", "actor", "loss"),
            ("actor/cr_reflow_actor_loss",),
        ),
        "final_cr_reflow_anchor_loss": (
            ("cr", "reflow", "anchor"),
            ("actor/cr_reflow_anchor_loss",),
        ),
        "final_cr_reflow_eta": (("cr", "reflow", "eta"), ("actor/cr_reflow_eta",)),
        "final_cr_reflow_weight_ess": (
            ("cr", "reflow", "ess"),
            ("actor/cr_reflow_weight_ess",),
        ),
        "final_cr_reflow_weight_kl": (
            ("cr", "reflow", "weight", "kl"),
            ("actor/cr_reflow_weight_kl",),
        ),
        "final_cr_reflow_eta_at_bound": (
            ("cr", "reflow", "eta", "bound"),
            ("actor/cr_reflow_eta_at_bound",),
        ),
        "final_cr_reflow_uniform_fallback": (
            ("cr", "reflow", "uniform", "fallback"),
            ("actor/cr_reflow_uniform_fallback",),
        ),
        "final_cr_reflow_weight_max": (
            ("cr", "reflow", "weight", "max"),
            ("actor/cr_reflow_weight_max",),
        ),
        "final_cr_reflow_valid_fraction": (
            ("cr", "reflow", "valid", "fraction"),
            ("actor/cr_reflow_valid_fraction",),
        ),
        "final_cr_reflow_target_displacement": (
            ("cr", "reflow", "target", "displacement"),
            ("actor/cr_reflow_target_displacement",),
        ),
        "final_cr_reflow_policy_kl_proxy": (
            ("cr", "reflow", "policy", "kl"),
            ("actor/cr_reflow_policy_kl_proxy",),
        ),
        "final_cr_reflow_policy_displacement": (
            ("cr", "reflow", "policy", "displacement"),
            ("actor/cr_reflow_policy_displacement",),
        ),
        "final_cr_reflow_mode_code": (
            ("cr", "reflow", "mode", "code"),
            ("actor/cr_reflow_mode_code",),
        ),
    }
    tags: dict[str, MetricTagSelection] = {}
    for name, (keywords, preferred) in candidates.items():
        tag, source = select_tag(scalars, keywords, preferred)
        if tag is not None:
            tags[name] = (tag, source or "unknown")
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
        summary = manifest.copy()
        if "status" in summary:
            summary["status"] = summary["status"].fillna("unknown").astype(str)
        elif not summary.empty:
            summary["status"] = "unknown"
        if not summary.empty:
            summary["has_scalar_evidence"] = False
            summary["evidence_type"] = "manifest_only"
            summary["official_comparison_eligible"] = False
        return summary

    for run_name, group in scalars.groupby("run_name"):
        suite, method, seed = split_run_name(str(run_name))
        row: dict[str, Any] = {
            "suite": suite,
            "method": method,
            "seed": seed,
            "run_name": run_name,
            "has_scalar_evidence": True,
            "evidence_type": "training_tensorboard_scalar",
            "official_comparison_eligible": False,
        }
        for name, (tag, tag_source) in metric_tags.items():
            values = group[group["tag"] == tag].sort_values("step")
            value = values["value"].iloc[-1] if not values.empty else math.nan
            row[name] = value
            row[f"{name}_has_evidence"] = bool(pd.notna(value))
            row[f"{name}_tag"] = tag
            row[f"{name}_tag_source"] = tag_source
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
    if "status" in summary:
        summary["status"] = summary["status"].fillna("unknown").astype(str)
    else:
        summary["status"] = "unknown"
    summary["has_scalar_evidence"] = summary["has_scalar_evidence"].eq(True)
    summary["official_comparison_eligible"] = False
    for metric in metric_tags:
        evidence_column = f"{metric}_has_evidence"
        summary[evidence_column] = summary[evidence_column].eq(True)

    agg_rows: list[dict[str, Any]] = []
    complete = summary[
        (summary["status"] == "done") & summary["has_scalar_evidence"]
    ]
    for (suite, method), group in complete.groupby(["suite", "method"]):
        success_evidence = (
            group[group["final_success_has_evidence"]]
            if "final_success_has_evidence" in group
            else group.iloc[0:0]
        )
        success_seed_count = success_evidence["seed"].astype(str).nunique()
        row = {
            "suite": suite,
            "method": method,
            "num_seeds": success_seed_count,
            "main_table": success_seed_count >= min_seeds,
        }
        for metric in metric_tags:
            metric_group = group[group[f"{metric}_has_evidence"]]
            row[f"{metric}_num_seeds"] = metric_group["seed"].astype(str).nunique()
            row[f"{metric}_mean"] = metric_group[metric].mean()
            row[f"{metric}_std"] = metric_group[metric].std()
        if "terminal_collapse" in success_evidence:
            row["terminal_collapse_rate"] = success_evidence["terminal_collapse"].mean()
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
    official_evaluations = load_official_evaluations(args.exp_root)
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
    official_evaluations.to_csv(
        args.output_dir / "pirl_official_evaluations.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
