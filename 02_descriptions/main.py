#!/usr/bin/env python3
"""Public workflow entry point. Run with --help for usage."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aha_publish import paths
from aha_publish.commands import entry, parser, require, run, selected_tasks

def main():
    p = parser("Step 2: capture waypoint evidence and generate task descriptions.", tasks=True)
    p.add_argument("--provider", choices=["openai", "claude"], default="openai")
    p.add_argument("--model", help="Override the description model.")
    p.add_argument("--reuse-evidence", action="store_true", help="Reuse grids and gripper sequences previously generated in this publication output folder.")
    args = p.parse_args()
    for task in selected_tasks(args):
        context = paths.TTM_CONTEXT_DIR / f"{task}.llm_context.json"
        grid = paths.DESCRIPTION_DIR / f"{task}_ALL_WAYPOINTS_COMBINED.png"
        gripper = paths.GRIPPER_DIR / f"{task}.json"
        require(context, args.dry_run)
        if not args.reuse_evidence:
            run("descriptions.waypoints_screenshot", ["--single", task, "--gripper"], args.dry_run)
            run("descriptions.evidence", ["--task", task], args.dry_run)
            run("descriptions.extract_gripper_sequence", [task, "--headless"], args.dry_run)
        require(grid, args.dry_run)
        require(gripper, args.dry_run)
        options = ["--task", task, "--provider", args.provider, "--grid-image", grid,
                   "--ttm-context", context, "--gripper-sequence", gripper,
                   "--gripper-sequence-input"]
        if args.model:
            options += ["--openai-model" if args.provider == "openai" else "--claude-model", args.model]
        run("descriptions.cli", options, args.dry_run)
        require(paths.DESCRIPTION_DIR / f"{task}_ALL_WAYPOINTS_COMBINED.{args.provider}.multimodal_analysis.json", args.dry_run)


if __name__ == "__main__":
    entry(main)
