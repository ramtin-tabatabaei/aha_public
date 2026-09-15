#!/usr/bin/env python3
"""Public workflow entry point. Run with --help for usage."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aha_publish import paths
from aha_publish.commands import entry, parser, require, run, selected_tasks

def main():
    p = parser("Step 3: generate behavior-tree conditions from task descriptions.", tasks=True)
    p.add_argument("--provider", choices=["openai", "claude"], default="openai", help="Provider used to generate the BT.")
    p.add_argument("--description-provider", choices=["openai", "claude"], default="openai")
    p.add_argument("--model", help="Override the BT generation and review model.")
    p.add_argument("--no-reviewer", action="store_true")
    args = p.parse_args()
    for task in selected_tasks(args):
        description = paths.DESCRIPTION_DIR / f"{task}_ALL_WAYPOINTS_COMBINED.{args.description_provider}.multimodal_analysis.json"
        require(description, args.dry_run)
        options = ["--task-context", description, "--provider", args.provider, "--generate-only", "--regenerate"]
        if args.model:
            options += ["--model", args.model]
        if args.no_reviewer:
            options += ["--no-reviewer"]
        run("behavior_trees.cli", options, args.dry_run)
        require(paths.BT_DIR / f"{task}.bt_conditions.json", args.dry_run)


if __name__ == "__main__":
    entry(main)
