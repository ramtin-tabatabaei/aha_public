"""Task execution, output writing, usage reporting, and task discovery."""

from aha_publish import paths

from .providers import *

def analyze_task(task_paths: TaskInputs, args: argparse.Namespace, client) -> tuple[dict[str, Any], Usage]:
    waypoints = discover_waypoints(task_paths, args)
    text_payload = build_text_payload(task_paths, args, waypoints)

    if args.write_prompt:
        prompt_path = Path(args.write_prompt).expanduser()
        if args.all:
            prompt_path = prompt_path.with_name(
                f"{prompt_path.stem}.{task_paths.task_name}{prompt_path.suffix}"
            )
        prompt_path.write_text(text_payload)

    if args.dry_run:
        return {
            "task": task_paths.task_name,
            "dry_run": True,
            "expected_waypoint_ids": waypoints,
            "prompt_chars": len(text_payload),
        }, Usage()

    if args.provider == "openai":
        return analyze_with_openai(task_paths, args, text_payload, client)
    if args.provider == "claude":
        return analyze_with_claude(task_paths, args, text_payload, client)
    raise ValueError(f"Unknown provider '{args.provider}'. Choose 'openai' or 'claude'.")


def print_analysis_summary(task_paths: TaskInputs, analysis: dict[str, Any]) -> None:
    print(f"\nTask: {task_paths.task_name}")
    print(f"Output: {task_paths.output_json_path}")
    description = analysis.get("overall_description")
    if description:
        print(f"Overall: {description}")
    waypoints = analysis.get("waypoints", [])
    print(f"Waypoint entries: {len(waypoints)}")


def model_name_for(args: argparse.Namespace) -> str:
    if args.provider == "openai":
        return args.openai_model
    if args.provider == "claude":
        return args.claude_model
    return args.provider


def format_cost(cost: float | None) -> str:
    if cost is None:
        return "n/a (set --input-price-per-1m and --output-price-per-1m)"
    return f"${cost:.4f}"


def usage_record(args: argparse.Namespace, usage: Usage, label: str) -> dict[str, Any]:
    cost = usage.cost(args.input_price_per_1m, args.output_price_per_1m)
    return {
        "label": label,
        "provider": args.provider,
        "model": model_name_for(args),
        "calls": usage.calls,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "input_price_per_1m": args.input_price_per_1m,
        "output_price_per_1m": args.output_price_per_1m,
        "cost_usd": cost,
    }


def print_usage_summary(args: argparse.Namespace, usage: Usage, label: str) -> None:
    cost = usage.cost(args.input_price_per_1m, args.output_price_per_1m)
    print(
        f"{label}: {usage.input_tokens} input + {usage.output_tokens} output "
        f"= {usage.total_tokens} tokens over {usage.calls} call(s) "
        f"on {model_name_for(args)}; cost {format_cost(cost)}"
    )


def add_held_object_to_description(
    desc_path: Path,
    task_name: str,
    gripper_sequence_path: Path | None = None,
) -> str:
    """Stamp per-waypoint gripper_state into an EXISTING description JSON from the
    simulator gripper sequence
    (outputs/gripper_sequences/<task>.json).

    held_object stays the description generator's own answer; stamping only
    clears it where the gripper is commanded open.

    No API call. Returns:
      'patched'   - description updated with simulator ground truth,
      'truncated' - the sequence covers fewer waypoints than the description, so
                    it was NOT written (re-capture the sequence in the sim first),
      'skipped'   - no sequence or description file available.
    """
    desc_path = Path(desc_path)
    try:
        from .gripper_ground_truth import patch_description_obj, SEQ_DIR
    except Exception as exc:  # noqa: BLE001
        print(f"  [held-object] unavailable: {exc}")
        return "skipped"
    seq_path = gripper_sequence_path or SEQ_DIR / f"{task_name}.json"
    if not seq_path.exists():
        print(f"  [held-object] {task_name}: no gripper sequence; skipped")
        return "skipped"
    if not desc_path.exists():
        print(f"  [held-object] {task_name}: no description at {desc_path.name}; skipped")
        return "skipped"
    seq = json.loads(seq_path.read_text())
    desc = json.loads(desc_path.read_text())
    max_desc = max((w.get("waypoint") for w in desc.get("waypoints", [])
                    if isinstance(w, dict) and isinstance(w.get("waypoint"), int)), default=-1)
    max_seq = max((w.get("index", -1) for w in seq.get("waypoints", [])), default=-1)
    if max_seq < max_desc:
        # Truncated capture (e.g. BT-runner dump that stopped on task success):
        # writing it would leave trailing waypoints wrong. Skip rather than fabricate.
        print(f"  [held-object] {task_name}: sequence covers wp0..{max_seq} but "
              f"description has wp0..{max_desc}; SKIPPED — re-capture in sim first.")
        return "truncated"
    if not any("held_object" in w for w in desc.get("waypoints", []) if isinstance(w, dict)):
        # A description generated before held_object joined the prompt schema has
        # no answer to keep, so every waypoint would be stamped empty. Say so
        # instead of silently leaving the field blank.
        print(f"  [held-object] {task_name}: description carries no held_object field; "
              f"re-generate it — nothing to keep.")
    patched, _notes = patch_description_obj(desc, seq)
    desc_path.write_text(json.dumps(patched, indent=2))
    held_count = sum(1 for w in patched.get("waypoints", []) if w.get("held_object"))
    print(f"  [held-object] {task_name}: {len(_notes)} waypoints, {held_count} holding "
          f"→ {desc_path.name}")
    return "patched"


def run_one(task_name: str, args: argparse.Namespace, client, grid_image: Path | None = None) -> tuple[Path, Usage]:
    task_paths = build_task_paths(args, task_name, grid_image)
    print(f"\nAnalyzing {task_paths.task_name}")
    print_task_category(task_paths.task_name)
    print(f"Grid image: {task_paths.grid_image_path}")

    # Refresh BEFORE building the prompt, so generation and stamping read the
    # same newly captured evidence. Stage 2 normally captures it in its worker.
    if getattr(args, "extract_gripper_sequence", False):
        from aha_publish.descriptions.extract_gripper_sequence import extract as _extract_gripper, MIN_TIP_TRAVEL_M
        _grip_data = _extract_gripper(task_paths.task_name)
        if _grip_data.get('tip_travel_m', 0) < MIN_TIP_TRAVEL_M:
            raise ValueError('Gripper capture failed: the arm did not execute a trajectory.')
        _grip_out = task_paths.gripper_sequence_path
        _grip_out.parent.mkdir(parents=True, exist_ok=True)
        _grip_out.write_text(json.dumps(_grip_data, indent=2))
        print(f"  [gripper-extract] {_grip_data['n_waypoints']} waypoints → {_grip_out}")

    analysis, usage = analyze_task(task_paths, args, client)
    task_paths.output_json_path.parent.mkdir(parents=True, exist_ok=True)
    task_paths.output_json_path.write_text(json.dumps(analysis, indent=2))

    # Stamp the simulator's commanded gripper_state into the generated
    # description from the captured gripper sequence. Provider-correct: patches
    # THIS run's output file, not whichever provider sorts first.
    if getattr(args, "add_held_object", False):
        add_held_object_to_description(
            task_paths.output_json_path,
            task_paths.task_name,
            task_paths.gripper_sequence_path,
        )

    print_analysis_summary(task_paths, analysis)

    if not args.dry_run:
        usage_path = task_paths.output_json_path.with_suffix(".usage.json")
        usage_path.write_text(
            json.dumps(usage_record(args, usage, task_paths.task_name), indent=2)
        )
        print_usage_summary(args, usage, f"Cost [{task_paths.task_name}]")

    return task_paths.output_json_path, usage


def discover_generator_tasks(photo_dir: Path, provider: str) -> list[tuple[str, bool, bool]]:
    """Return [(task_name, has_grid_image, already_generated)] per task, sorted.

    Only tasks that actually have a grid image in photo_dir are listed — a task
    without one has nothing to analyze, so it is not offered. already_generated
    is True when the analysis JSON this run would write already exists
    (re-running overwrites it).
    """
    tasks: dict[str, bool] = {}
    if photo_dir.exists():
        for image in find_all_grid_images(photo_dir):
            task_name = extract_task_from_grid_image(image)
            analysis = image.with_suffix(f".{provider}.multimodal_analysis.json")
            tasks[task_name] = analysis.exists()
    return sorted((name, True, generated) for name, generated in tasks.items())


def prompt_for_task(photo_dir: Path, provider: str) -> list[str] | None:
    """Show a numbered menu and return tasks selected by numbers or inclusive ranges.

    Returns None when there is nothing to choose from or stdin is not a
    terminal, so callers can fall back to the default. Exits cleanly if quit.
    """
    tasks = discover_generator_tasks(photo_dir, provider)
    if not tasks or not sys.stdin.isatty():
        return None

    generated = sum(1 for _, _, done in tasks if done)
    print(f"\nAvailable tasks ({len(tasks)} with a grid image, {generated} already generated):")
    selectable: dict[int, str] = {}
    for index, (name, _has_image, done) in enumerate(tasks, start=1):
        marker = "  — already generated (will overwrite)" if done else ""
        selectable[index] = name
        print(f"  {index:>3}. {name}{marker}")

    while True:
        try:
            choice = input(
                f"\nSelect task(s) to generate [1-{len(tasks)}] "
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
        return [selectable[index] for index in selected]
