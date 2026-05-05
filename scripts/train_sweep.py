#!/usr/bin/env python3
import sys
import argparse
import traceback
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml  # noqa: E402
from src.vit.training.train import train  # noqa: E402


def _load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _inject_tags(config: dict, extra_tags: list[str]) -> dict:
    existing = config.get("tags", [])
    merged = existing + [t for t in extra_tags if t not in existing]
    config["tags"] = merged
    return config


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def run_sweep(config_paths: list[str], extra_tags: list[str]) -> None:
    total = len(config_paths)
    succeeded = []
    failed = []

    print(f"\n{'=' * 70}")
    print(f"  SWEEP: {total} config(s) to run")
    if extra_tags:
        print(f"  Extra tags: {extra_tags}")
    print(f"{'=' * 70}\n")

    sweep_start = time.time()

    for idx, config_path in enumerate(config_paths, start=1):
        print(f"\n{'=' * 70}")
        print(f"  [{idx}/{total}] {config_path}")
        print(f"{'=' * 70}")

        run_start = time.time()
        try:
            config = _load_config(config_path)
            config = _inject_tags(config, extra_tags)

            experiment = config.get("experiment_name", Path(config_path).stem)
            print(f"  Experiment : {experiment}")
            print(f"  Tags       : {config.get('tags', [])}")
            print(
                f"  Checkpoint : {config.get('output', {}).get('checkpoint_dir', 'N/A')}"
            )
            print()

            train(config)

            duration = time.time() - run_start
            print(f"\n  Completed in {_fmt_duration(duration)}")
            succeeded.append(config_path)

        except Exception:
            duration = time.time() - run_start
            print(f"\n  FAILED after {_fmt_duration(duration)}")
            print(f"  {'=' * 60}")
            traceback.print_exc()
            print(f"  {'=' * 60}")
            failed.append(config_path)

    total_duration = time.time() - sweep_start

    print(f"\n{'=' * 70}")
    print(f"  SWEEP COMPLETE  ({_fmt_duration(total_duration)} total)")
    print(f"{'=' * 70}")
    print(f"  Succeeded ({len(succeeded)}/{total}):")
    for p in succeeded:
        print(f"    OK  {p}")
    if failed:
        print(f"  Failed ({len(failed)}/{total}):")
        for p in failed:
            print(f"    !!  {p}")
    print(f"{'=' * 70}\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ViT training sequentially over multiple config files."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        metavar="CONFIG",
        help="One or more YAML config file paths to run in order.",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=[],
        metavar="TAG",
        help="Extra W&B tags to inject into every run (e.g. sweep-01 loss-ablation).",
    )
    args = parser.parse_args()

    run_sweep(args.configs, args.tags)
