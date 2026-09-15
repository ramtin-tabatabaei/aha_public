#!/usr/bin/env python3
"""Public workflow entry point. Run with --help for usage."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aha_publish import paths
from aha_publish.commands import entry, parser, require, run, selected_tasks

def main():
    p = parser("Step 1: inspect existing RLBench TTM models and write context JSON.", tasks=True)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()
    for task in selected_tasks(args):
        target = paths.TTM_CONTEXT_DIR / f"{task}.llm_context.json"
        if args.skip_existing and target.exists():
            continue
        require(paths.RLBENCH_ROOT / "rlbench/task_ttms" / f"{task}.ttm", args.dry_run)
        if not args.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
        run("ttm.inspect_ttm", [task, "--save", target], args.dry_run)
        require(target, args.dry_run)


if __name__ == "__main__":
    entry(main)
