"""CLI argument handling and entry points for BT maker."""

from aha_publish import paths

import re
from copy import copy

from . import llm_generation
from .gui import *

def available_task_contexts(directory: Path | None = None) -> list[Path]:
    """List selectable task-context JSON files, newest-friendly and sorted by name.

    Looks in the directory of DEFAULT_TASK_CONTEXT_PATH (overridable via the
    BT_TASK_CONTEXT_PATH env var). Skips the *.usage.json token-accounting files
    that sit next to each analysis JSON.
    """
    directory = directory or DEFAULT_TASK_CONTEXT_PATH.parent
    if not directory.is_dir():
        return []
    return [
        path
        for path in sorted(directory.glob("*.json"))
        if not path.name.endswith(".usage.json")
    ]

def discover_tasks(directory: Path | None = None) -> list[tuple[str, Path | None]]:
    """Return every task as (task_name, analysis_path_or_None).

    A task is runnable only when it has an analysis JSON; tasks with only a grid
    image map to None. Sorted by task name so the menu order is stable.
    """
    directory = directory or DEFAULT_TASK_CONTEXT_PATH.parent
    tasks: dict[str, Path | None] = {
        task_name_from_context_path(path): path
        for path in available_task_contexts(directory)
    }
    if directory.is_dir():
        for image in sorted(directory.glob("*.png")):
            tasks.setdefault(task_name_from_context_path(image), None)
    return sorted(tasks.items())

def prompt_for_task_context(directory: Path | None = None) -> list[Path] | None:
    """Show a numbered menu and return analysis paths for numbers or ranges.

    Image-only tasks are listed but not selectable. Returns None when there is
    nothing to choose from or stdin is not a terminal, so callers can fall back
    to the default. Exits cleanly if the user quits.
    """
    tasks = discover_tasks(directory)
    if not tasks or not sys.stdin.isatty():
        return None

    runnable = {
        index: path
        for index, (_, path) in enumerate(tasks, start=1)
        if path is not None
    }
    print(f"\nAvailable tasks ({len(runnable)} of {len(tasks)} ready to run):")
    for index, (name, path) in enumerate(tasks, start=1):
        marker = "" if path is not None else "  — no analysis yet (run task_description_generator)"
        print(f"  {index:>3}. {name}{marker}")

    while True:
        try:
            choice = input(
                f"\nSelect task(s) [1-{len(tasks)}] "
                "(e.g. 31-40 or 1,3-5; q to quit): "
            ).strip()
        except EOFError:
            return None
        if choice.lower() in ("q", "quit", "exit"):
            print("No task selected. Exiting.")
            sys.exit(0)
        selected: dict[int, None] = {}
        try:
            for part in choice.split(","):
                bounds = part.strip().split("-")
                if len(bounds) not in (1, 2) or not all(b.strip().isdecimal() for b in bounds):
                    raise ValueError
                start, end = int(bounds[0]), int(bounds[-1])
                if not 1 <= start <= end <= len(tasks):
                    raise ValueError
                selected.update((index, None) for index in range(start, end + 1))
        except ValueError:
            print(f"Please enter numbers or ascending ranges between 1 and {len(tasks)} (e.g. 1,3-5).")
            continue
        missing = [tasks[index - 1][0] for index in selected if index not in runnable]
        if missing:
            print(f"No analysis JSON yet for: {', '.join(missing)}. Pick tasks without the marker.")
            continue
        return [runnable[index] for index in selected]

def normalize_direct_waypoint_args(argv: list[str]) -> tuple[list[str], Path | None]:
    """Accept path-like args such as --waypoints_description/task.analysis.json."""
    normalized = []
    direct_waypoint = None

    for arg in argv:
        looks_like_path_option = (
            arg.startswith("--")
            and arg.endswith(".json")
            and ("/" in arg or "\\" in arg)
        )
        if looks_like_path_option:
            if direct_waypoint is not None:
                raise ValueError("Only one direct waypoint JSON path can be provided.")
            direct_waypoint = Path(arg[2:])
        else:
            normalized.append(arg)

    return normalized, direct_waypoint

def parse_args() -> argparse.Namespace:
    try:
        argv, direct_waypoint = normalize_direct_waypoint_args(sys.argv[1:])
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    parser = argparse.ArgumentParser(
        description=(
            "Generate task verification preconditions and postconditions from "
            "a task context JSON or a Claude/OpenAI analysis JSON file."
        )
    )
    parser.add_argument(
        "--provider",
        choices=("claude", "openai"),
        default=DEFAULT_PROVIDER,
        help="Model provider to call. Defaults to the value configured in this file.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model to use for both Agent 1 generation and Agent 2 review. "
            "Overrides the provider-specific model env vars."
        ),
    )
    parser.add_argument(
        "--generator-model",
        default=None,
        help="Model to use for Agent 1 generation only.",
    )
    parser.add_argument(
        "--reviewer-model",
        default=None,
        help="Model to use for Agent 2 review only.",
    )
    parser.add_argument(
        "--task-context",
        type=Path,
        default=None,
        help=(
            "Path to task_context.json or an analysis JSON file such as "
            "output/example.openai.analysis.json. If omitted (and no waypoint "
            "JSON is given), select numbers or ranges interactively. Multiple tasks "
            "require --generate or --generate-only and are saved without opening the GUI."
        ),
    )
    parser.add_argument(
        "waypoint_json",
        nargs="?",
        type=Path,
        help=(
            "Optional waypoint analysis JSON. You can also pass it in the "
            "legacy form --waypoints_description/example.openai.analysis.json."
        ),
    )
    parser.add_argument(
        "--review-output",
        type=Path,
        default=None,
        help=(
            "Path where the curated GUI review JSON will be saved. "
            "Defaults to <task-name>.bt_conditions.json in prepared_BTs."
        ),
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="Local GUI host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Local GUI port.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start the GUI server without opening a browser tab.",
    )
    parser.add_argument(
        "--no-auto-generate",
        action="store_true",
        help="Open the GUI without generating the first draft automatically.",
    )
    parser.add_argument(
        "--generate",
        "--regenerate",
        dest="generate",
        action="store_true",
        help=(
            "Generate a fresh BT from the LLM on startup and overwrite the saved "
            "prepared BT for the task with it. By default (no flag) the studio "
            "loads the predefined saved BT and leaves it untouched until you press "
            "Save. (--regenerate is a legacy alias.)"
        ),
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Run the old terminal-only condition generation flow.",
    )
    reviewer_group = parser.add_mutually_exclusive_group()
    reviewer_group.add_argument(
        "--reviewer",
        dest="reviewer",
        action="store_true",
        default=None,
        help="Run the Agent 2 structural reviewer over the draft (default: BT_USE_REVIEWER env, ON).",
    )
    reviewer_group.add_argument(
        "--no-reviewer",
        dest="reviewer",
        action="store_false",
        help="Skip the Agent 2 structural reviewer.",
    )
    args = parser.parse_args(argv)
    if direct_waypoint is not None:
        if args.waypoint_json is not None:
            parser.error("Provide only one waypoint JSON path.")
        args.waypoint_json = direct_waypoint
    return args

def normalize_model_name(model: str | None) -> str | None:
    if model is None:
        return None
    model = model.strip()
    if re.match(r"^gpt\d", model):
        model = "gpt-" + model[3:]
    return model or None

def apply_model_overrides(args: argparse.Namespace) -> None:
    shared_model = normalize_model_name(args.model)
    generator_model = normalize_model_name(args.generator_model) or shared_model
    reviewer_model = normalize_model_name(args.reviewer_model) or shared_model
    if not generator_model and not reviewer_model:
        return

    if args.provider == "claude":
        if generator_model:
            llm_generation.GENERATOR_CLAUDE_MODEL = generator_model
        if reviewer_model:
            llm_generation.CLAUDE_MODEL = reviewer_model
    else:
        if generator_model:
            llm_generation.GENERATOR_OPENAI_MODEL = generator_model
        if reviewer_model:
            llm_generation.OPENAI_MODEL = reviewer_model

def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.provider = args.provider.strip().lower()
    apply_model_overrides(args)

    # A positional waypoint JSON takes precedence over --task-context. When
    # neither is supplied, ask the user to pick a task (honoring an explicit
    # BT_TASK_CONTEXT_PATH env override, and falling back to the default path
    # for non-interactive runs).
    selected = args.waypoint_json or args.task_context
    if selected is None:
        if os.environ.get("BT_TASK_CONTEXT_PATH"):
            selected = DEFAULT_TASK_CONTEXT_PATH
        else:
            selected = prompt_for_task_context() or DEFAULT_TASK_CONTEXT_PATH
    args.task_contexts = selected if isinstance(selected, list) else [selected]
    if len(args.task_contexts) > 1:
        if not (args.generate or args.generate_only):
            print("ERROR: Multiple tasks require --generate or --generate-only.", file=sys.stderr)
            sys.exit(2)
        if args.review_output is not None:
            print("ERROR: --review-output cannot be used with multiple tasks; each task needs its own output.", file=sys.stderr)
            sys.exit(2)
    args.task_context = args.task_contexts[0]

    args.review_output_explicit = args.review_output is not None
    args.task_context = resolve_project_path(args.task_context)
    task_context = load_task_context_object(args.task_context) if args.task_context.exists() else {}
    args.task_name = task_name_from_context(args.task_context, task_context)

    if args.review_output is None:
        args.review_output = default_review_output_path(
            args.task_context,
            args.task_name,
        )
    else:
        args.review_output = resolve_project_path(args.review_output)
    return args

def run_generate_only(args: argparse.Namespace) -> None:
    provider = args.provider.strip().lower()
    ensure_input_files(args)
    task_context = load_task_context(args.task_context)
    failure_definitions = load_failure_definitions()
    try:
        client = build_client(provider)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    provider_label = "Claude" if provider == "claude" else "OpenAI"
    print(f"Calling {provider_label} API...")
    print(f"Task context: {args.task_context}")
    print(f"Condition catalogue: {CONDITION_RULES_PATH}")

    result = generate_conditions(
        failure_definitions,
        task_context,
        client,
        provider,
        task_context_path=args.task_context,
        reviewer=getattr(args, "reviewer", None),
    )
    print_conditions(result)
    output_path = save_review_file(args, {"generated": result})
    print(f"\nSaved BT conditions -> {output_path}")

def main():
    args = finalize_args(parse_args())

    if len(args.task_contexts) > 1:
        print(f"Generating {len(args.task_contexts)} tasks sequentially without opening the GUI.")
        failed = []
        for index, path in enumerate(args.task_contexts, start=1):
            print(f"\n[{index}/{len(args.task_contexts)}] {path.name}")
            try:
                task_args = copy(args)
                task_args.waypoint_json = path
                task_args.review_output = None
                task_args = finalize_args(task_args)
                run_generate_only(task_args)
            except (Exception, SystemExit) as exc:
                failed.append(path.name)
                print(f"FAILED [{path.name}]: {exc}", file=sys.stderr)
        print(f"\nProcessed {len(args.task_contexts) - len(failed)} task(s); {len(failed)} failed: {failed}")
        if failed:
            sys.exit(1)
        return

    if args.generate_only:
        run_generate_only(args)
        return

    run_gui(args)

if __name__ == "__main__":
    main()
