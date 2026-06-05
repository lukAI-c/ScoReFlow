#!/usr/bin/env python3
"""Install the local OpenPI Score-Flow/TR/CR-Reflow model into RLinf."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


LIVE_MODEL = Path(__file__).resolve().parents[2] / "openpi_action_model.LIVE.py"
REMOTE_MODEL = Path("rlinf/models/embodiment/openpi/openpi_action_model.py")
BACKUP_SUFFIX = ".pre_scoreflow_cr_reflow_backup"


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


def install_model(source: Path, target: Path) -> bool:
    if not source.exists():
        raise FileNotFoundError(source)
    if not target.exists():
        raise FileNotFoundError(target)

    source_text = source.read_text()
    target_text = target.read_text()
    if source_text == target_text:
        return False

    backup = target.with_suffix(target.suffix + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(target, backup)
    target.write_text(source_text)
    return True


def main() -> None:
    args = parse_args()
    model_path = args.rlinf_root / REMOTE_MODEL
    changed = install_model(args.source, model_path)
    print(f"patched={changed} source={args.source} path={model_path}")


if __name__ == "__main__":
    main()
