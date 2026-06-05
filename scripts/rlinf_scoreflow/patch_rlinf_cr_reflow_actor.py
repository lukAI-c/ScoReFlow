#!/usr/bin/env python3
"""Patch RLinf FSDP actor worker to consume CR-Reflow policy outputs."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REMOTE_ACTOR = Path("rlinf/workers/actor/fsdp_actor_worker.py")
BACKUP_SUFFIX = ".pre_cr_reflow_actor_backup"
MARKER = "CR_REFLOW_ACTOR_PATCH_V1"


HELPER_BLOCK = f'''

# {MARKER}
def _cr_reflow_actor_mode(cfg: DictConfig) -> str:
    mode = OmegaConf.select(cfg, "actor.model.openpi.cr_reflow_mode", default="none")
    return str(mode or "none")


def _cr_reflow_actor_active(cfg: DictConfig) -> bool:
    return _cr_reflow_actor_mode(cfg) in {{"cr_reflow", "cr_reflow_no_anchor"}}


def _cr_reflow_scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean().cpu().item())
    if isinstance(value, (int, float, bool)):
        return float(value)
    return 0.0


def _cr_reflow_forward_inputs(forward_inputs, advantages, loss_mask):
    if forward_inputs is None:
        return None
    enriched = dict(forward_inputs)
    if isinstance(advantages, torch.Tensor):
        enriched["advantages"] = advantages.detach()
    if isinstance(loss_mask, torch.Tensor):
        enriched["cr_reflow_mask"] = loss_mask.detach()
    return enriched


def _cr_reflow_record_metrics(metrics_data, output_dict) -> None:
    diag = output_dict.get("cr_reflow_diag", {{}}) or {{}}
    for key, value in diag.items():
        if isinstance(value, str):
            metrics_data[f"actor/{{key}}"] = value
        elif isinstance(value, (int, float, bool, torch.Tensor)):
            metrics_data[f"actor/{{key}}"] = _cr_reflow_scalar(value)
'''


FORWARD_INPUTS_OLD = '''                    forward_inputs = batch.get("forward_inputs", None)

                    kwargs = {}
'''

FORWARD_INPUTS_NEW = '''                    forward_inputs = batch.get("forward_inputs", None)
                    cr_reflow_active = _cr_reflow_actor_active(self.cfg)
                    if cr_reflow_active:
                        forward_inputs = _cr_reflow_forward_inputs(
                            forward_inputs,
                            advantages,
                            loss_mask,
                        )

                    kwargs = {}
'''


POLICY_LOSS_OLD = '''                    loss, metrics_data = policy_loss(**kwargs)

                    entropy_loss = torch.tensor(
'''

POLICY_LOSS_NEW = '''                    loss, metrics_data = policy_loss(**kwargs)

                    if cr_reflow_active and not kwargs["critic_warmup"]:
                        cr_reflow_loss = output_dict.get("cr_reflow_loss", None)
                        if cr_reflow_loss is None:
                            raise RuntimeError(
                                "CR-Reflow actor mode is active but the policy did not "
                                "return cr_reflow_loss"
                            )
                        metrics_data["actor/ppo_policy_loss"] = _cr_reflow_scalar(loss)
                        metrics_data["actor/cr_reflow_actor_loss"] = _cr_reflow_scalar(
                            cr_reflow_loss
                        )
                        loss = cr_reflow_loss

                    entropy_loss = torch.tensor(
'''


TR_HOOK_OLD = '''                    if tr_loss is not None and not kwargs["critic_warmup"]:
                        loss = loss + tr_loss
                        _trd = output_dict.get("tr_diag", {}) or {}
                        for _k, _v in _trd.items():
                            if isinstance(_v, (int, float)):
                                metrics_data[f"actor/{_k}"] = float(_v)

                    if self.enable_sft_co_train:
'''

TR_HOOK_NEW = '''                    if tr_loss is not None and not kwargs["critic_warmup"]:
                        loss = loss + tr_loss
                        _trd = output_dict.get("tr_diag", {}) or {}
                        for _k, _v in _trd.items():
                            if isinstance(_v, (int, float)):
                                metrics_data[f"actor/{_k}"] = float(_v)

                    if cr_reflow_active:
                        _cr_reflow_record_metrics(metrics_data, output_dict)

                    if self.enable_sft_co_train:
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rlinf-root",
        type=Path,
        required=True,
        help="Path to the RLinf repository root.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate that the actor worker can be patched without writing files.",
    )
    return parser.parse_args()


def replace_once(text: str, old: str, new: str, label: str) -> tuple[str, bool]:
    count = text.count(old)
    if count == 1:
        return text.replace(old, new, 1), True
    if new in text:
        return text, False
    raise RuntimeError(f"Expected exactly one {label} anchor, found {count}")


def patch_text(text: str) -> tuple[str, bool]:
    changed = False
    if MARKER not in text:
        import_anchor = "from rlinf.workers.rollout.utils import RankMapper\n"
        text, did_replace = replace_once(
            text,
            import_anchor,
            import_anchor + HELPER_BLOCK,
            "helper insertion",
        )
        changed = changed or did_replace
    text, did_replace = replace_once(
        text,
        FORWARD_INPUTS_OLD,
        FORWARD_INPUTS_NEW,
        "forward_inputs enrichment",
    )
    changed = changed or did_replace
    text, did_replace = replace_once(
        text,
        POLICY_LOSS_OLD,
        POLICY_LOSS_NEW,
        "CR-Reflow actor loss selection",
    )
    changed = changed or did_replace
    text, did_replace = replace_once(
        text,
        TR_HOOK_OLD,
        TR_HOOK_NEW,
        "CR-Reflow diagnostic logging",
    )
    changed = changed or did_replace
    return text, changed


def patch_actor(path: Path, check: bool) -> bool:
    if not path.exists():
        raise FileNotFoundError(path)

    source_text = path.read_text()
    patched_text, changed = patch_text(source_text)
    if not changed or check:
        return changed

    backup = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(patched_text)
    return True


def main() -> None:
    args = parse_args()
    actor_path = args.rlinf_root / REMOTE_ACTOR
    changed = patch_actor(actor_path, args.check)
    print(
        f"patched={changed and not args.check} "
        f"would_patch={changed if args.check else False} "
        f"check={args.check} path={actor_path}"
    )


if __name__ == "__main__":
    main()
