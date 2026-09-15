#!/usr/bin/env python3
"""Public workflow entry point. Run with --help for usage."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aha_publish import paths
from aha_publish.commands import entry, parser, require, run, selected_tasks

def main():
    from aha_publish.commands import environment
    from aha_publish.calibration.settings import runtime_environment
    p = parser("Step 4: run calibrated behavior trees and save logs for scoring.", tasks=True)
    p.add_argument("--failures", default="none", help="none for clean runs; all or comma-separated injected failure types.")
    p.add_argument("--waypoints", default="", help="Comma-separated injection waypoint indices.")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--checks", choices=["full", "detectors"], default="full", help="full includes condition VLM and slip/collision confirmation.")
    p.add_argument("--model", help="Override the VLM model.")
    p.add_argument("--redo", action="store_true", help="Rerun cases already present in the output directory.")
    args = p.parse_args()
    if args.workers < 1:
        p.error("--workers must be positive")
    tasks = selected_tasks(args)
    for task in tasks:
        require(paths.BT_DIR / f"{task}.bt_conditions.json", args.dry_run)
        require(paths.TTM_CONTEXT_DIR / f"{task}.llm_context.json", args.dry_run)
    env = runtime_environment(environment(), tasks, dry_run=args.dry_run)
    options = ["--out", paths.RUNS_DIR, "--workers", args.workers]
    for task in tasks:
        options += ["--task", task]
    # The batch backend includes a clean baseline; 'none' selects only it.
    if args.failures != "all":
        options += ["--failures", args.failures]
    if args.waypoints:
        options += ["--waypoints", args.waypoints]
    if args.checks == "full":
        options += ["--vlm-prepost", "--vlm-confirm", "slip,collision"]
    if args.model:
        options += ["--vlm-model", args.model]
    if args.redo:
        options += ["--redo"]
    run("running.run_all_tasks_all_failures", options, args.dry_run, env=env)


if __name__ == "__main__":
    entry(main)
