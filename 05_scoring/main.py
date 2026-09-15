#!/usr/bin/env python3
"""Public workflow entry point. Run with --help for usage."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aha_publish import paths
from aha_publish.commands import entry, parser, require, run, selected_tasks

def main():
    p = parser("Step 5: build checkpoint samples and score detection performance.")
    p.add_argument("--task", action="append", help="Restrict sample extraction to task names.")
    p.add_argument("--bt-root", type=Path, default=paths.RUNS_DIR)
    p.add_argument("--baseline-root", type=Path, help="Optional previously assessed GPT baseline grids (B2).")
    p.add_argument("--samples", type=Path, help="Rescore an existing sample CSV, skipping extraction.")
    p.add_argument("--out-dir", type=Path, default=paths.SCORES_DIR)
    p.add_argument("--methods", nargs="+", choices=["B2", "A1", "A2", "A3", "A4"], default=["A1", "A2", "A3", "A4"])
    p.add_argument("--unit", choices=["checkpoint", "episode"], default="checkpoint")
    p.add_argument("--window", choices=["to-injection", "full"], default="to-injection")
    p.add_argument("--uncertain-as", choices=["success", "failure"], default="success")
    p.add_argument("--common-episodes", action="store_true")
    p.add_argument("--matrix-image", action="store_true")
    args = p.parse_args()
    samples = args.samples or args.out_dir / "detection_samples.csv"
    if args.samples:
        require(samples, args.dry_run)
    else:
        require(args.bt_root, args.dry_run)
        options = ["--bt-root", args.bt_root, "--out", samples]
        if args.task:
            options += ["--tasks", *args.task]
        if args.baseline_root:
            require(args.baseline_root, args.dry_run)
            options += ["--grid-root", args.baseline_root]
        run("scoring.make_detection_samples", options, args.dry_run)
    options = ["--samples", samples, "--out-dir", args.out_dir, "--methods", *args.methods,
               "--unit", args.unit, "--window", args.window, "--uncertain-as", args.uncertain_as,
               "--no-list-errors"]
    if args.common_episodes:
        options += ["--common-episodes"]
    if args.matrix_image:
        options += ["--matrix-image", "--matrix-image-methods", *args.methods]
    run("scoring.score_detection_samples", options, args.dry_run)


if __name__ == "__main__":
    entry(main)
