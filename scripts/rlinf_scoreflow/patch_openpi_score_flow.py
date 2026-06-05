#!/usr/bin/env python3
"""Install the local OpenPI Score-Flow/TR/CR-Reflow model into RLinf."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import shutil
from pathlib import Path


LIVE_MODEL = Path(__file__).resolve().parents[2] / "openpi_action_model.LIVE.py"
REMOTE_MODEL = Path("rlinf/models/embodiment/openpi/openpi_action_model.py")
BACKUP_SUFFIX = ".pre_scoreflow_cr_reflow_backup"
UNEXPECTED_SUFFIX = ".unexpected_scoreflow_target.diff"
MAX_EVIDENCE_DIFF_LINES = 2000
EXPECTED_MARKERS = (
    "TERMINAL_GUIDANCE_PATCH_V1",
    "SCORE_FLOW_PATCH_V2",
    "tr_penalty_mode",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rlinf-root",
        type=Path,
        required=True,
        help="Path to the RLinf repository root.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=LIVE_MODEL,
        help="Source OpenPI model file to install.",
    )
    return parser.parse_args()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def has_expected_markers(text: str) -> bool:
    return all(marker in text for marker in EXPECTED_MARKERS)


def write_unexpected_evidence(source: Path, target: Path, source_text: str, target_text: str) -> Path:
    evidence_path = target.with_suffix(target.suffix + UNEXPECTED_SUFFIX)
    diff_lines = list(
        difflib.unified_diff(
            target_text.splitlines(keepends=True),
            source_text.splitlines(keepends=True),
            fromfile=str(target),
            tofile=str(source),
        )
    )
    truncated = len(diff_lines) > MAX_EVIDENCE_DIFF_LINES
    if truncated:
        diff_lines = diff_lines[:MAX_EVIDENCE_DIFF_LINES]
        diff_lines.append("\n[diff truncated]\n")

    evidence = [
        "# Unexpected OpenPI target state\n",
        f"source={source}\n",
        f"target={target}\n",
        f"source_sha256={sha256_text(source_text)}\n",
        f"target_sha256={sha256_text(target_text)}\n",
        f"expected_markers={','.join(EXPECTED_MARKERS)}\n",
        f"target_has_expected_markers={has_expected_markers(target_text)}\n",
        f"diff_truncated={truncated}\n",
        "\n## Diff\n",
        *diff_lines,
    ]
    evidence_path.write_text("".join(evidence))
    return evidence_path


def compatibility_reason(source: Path, target: Path, source_text: str, target_text: str) -> str:
    if source_text == target_text:
        return "identical"

    backup = target.with_suffix(target.suffix + BACKUP_SUFFIX)
    if backup.exists() and sha256_text(target_text) == sha256_text(backup.read_text()):
        return "matches_pre_cr_backup"

    if has_expected_markers(target_text):
        return "expected_scoreflow_tr_markers"

    evidence_path = write_unexpected_evidence(source, target, source_text, target_text)
    raise RuntimeError(
        "Refusing to overwrite unexpected OpenPI target. "
        f"Wrote hash/diff evidence to {evidence_path}"
    )


def install_model(source: Path, target: Path) -> tuple[bool, str]:
    if not source.exists():
        raise FileNotFoundError(source)
    if not target.exists():
        raise FileNotFoundError(target)

    source_text = source.read_text()
    target_text = target.read_text()
    reason = compatibility_reason(source, target, source_text, target_text)
    if reason == "identical":
        return False, reason

    backup = target.with_suffix(target.suffix + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(target, backup)
    target.write_text(source_text)
    return True, reason


def main() -> None:
    args = parse_args()
    model_path = args.rlinf_root / REMOTE_MODEL
    changed, reason = install_model(args.source, model_path)
    print(f"patched={changed} compatibility={reason} source={args.source} path={model_path}")


if __name__ == "__main__":
    main()
