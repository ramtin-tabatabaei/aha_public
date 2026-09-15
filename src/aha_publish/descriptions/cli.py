"""CLI argument parsing and main entry point for task description generation."""

from aha_publish import paths

from .runner import *

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="General multimodal robot task inspector for combined waypoint grid images."
    )
    parser.add_argument(
        "--task",
        default=os.getenv("AHA_TASK_NAME"),
        help=(
            "Task name to analyze. If omitted (and not --all), you are prompted "
            "to pick tasks interactively using numbers or ranges. Defaults to $AHA_TASK_NAME when set."
        ),
    )
    parser.add_argument("--all", action="store_true", help="Analyze every *_ALL_WAYPOINTS_COMBINED image in --photo-dir.")
    parser.add_argument("--photo-dir", default=str(DEFAULT_PHOTO_DIR))
    parser.add_argument("--simulator", default=os.getenv("AHA_SIMULATOR", "auto"), help="Optional simulator label. Defaults to auto-inference from supplied paths.")
    parser.add_argument("--provider", default=os.getenv("AHA_LLM_PROVIDER", PROVIDER), choices=["openai", "claude"])
    parser.add_argument("--openai-model", default=os.getenv("OPENAI_MODEL", OPENAI_MODEL))
    parser.add_argument("--claude-model", default=os.getenv("CLAUDE_MODEL", CLAUDE_MODEL))
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument(
        "--input-price-per-1m",
        type=float,
        default=env_float("AHA_INPUT_PRICE_PER_1M"),
        help="Input price per 1M tokens, used to report cost. Defaults to $AHA_INPUT_PRICE_PER_1M if set.",
    )
    parser.add_argument(
        "--output-price-per-1m",
        type=float,
        default=env_float("AHA_OUTPUT_PRICE_PER_1M"),
        help="Output price per 1M tokens, used to report cost. Defaults to $AHA_OUTPUT_PRICE_PER_1M if set.",
    )

    parser.add_argument("--grid-image", help="Combined waypoint grid image. Defaults to --photo-dir/{task}_ALL_WAYPOINTS_COMBINED.png.")
    parser.add_argument("--image-path", action="append", default=[], help="Extra image input as LABEL=PATH, repeatable.")
    parser.add_argument("--gripper-sequence", help="Gripper sequence JSON path. Defaults to --gripper-sequence-dir/{task}.json.")
    parser.add_argument("--gripper-sequence-dir", default=str(DEFAULT_GRIPPER_SEQUENCE_DIR))
    parser.add_argument("--ttm-context", help="Scene/object relationship context path.")
    parser.add_argument("--scene-context-path", action="append", default=[], help="Extra scene/context path as LABEL=PATH, repeatable.")
    parser.add_argument("--task-config", help="Task config path.")
    parser.add_argument("--task-context-path", action="append", default=[], help="Extra task config/context path as LABEL=PATH, repeatable.")
    parser.add_argument("--output-json", help="Output JSON path. Cannot be used with --all.")

    parser.add_argument("--waypoints", help="Comma-separated waypoint ids matching grid rows.")
    parser.add_argument("--camera-count", type=int, help="Number of camera columns in the combined image, if known.")
    parser.add_argument("--max-chars-text-file", type=int, default=MAX_CHARS_PER_TEXT_FILE)

    parser.set_defaults(
        include_grid_image=True,
        include_ttm_context=True,
        # ON by default: the model decides held_object from this table, the same
        # file drives the post-generation gripper_state stamping (add_held_object),
        # and the prompt is written around it. With it off the model never learns
        # where the gripper releases, so its prose ("the held remote") contradicts
        # the stamped fields.
        include_gripper_sequence=True,
        include_task_config=False,
        include_extra_fields=False,
        add_held_object=True,
        extract_gripper_sequence=False,
        held_object_only=False,
    )
    parser.add_argument("--no-grid-image", action="store_false", dest="include_grid_image")
    parser.add_argument("--no-ttm-context", action="store_false", dest="include_ttm_context")
    parser.add_argument("--gripper-sequence-input", action="store_true", dest="include_gripper_sequence",
                        help="Feed the per-waypoint simulator gripper action + held object to the model at "
                             "generation time (measured data; lets it tell grasp-and-carry from a button press).")
    parser.add_argument("--no-gripper-sequence-input", action="store_false", dest="include_gripper_sequence")
    parser.add_argument("--no-task-config", action="store_false", dest="include_task_config")

    # Explicit opt-ins for text inputs that are disabled by default.
    parser.add_argument("--task-config-text", action="store_true", dest="include_task_config")

    # Optional OUTPUT fields, off by default (not consumed by downstream bt_maker/detectors).
    parser.add_argument("--extra-fields", action="store_true", dest="include_extra_fields",
                        help="Emit optional output fields unused by downstream code: key_scene_objects.relationships "
                             "and per-waypoint approx_motion_from_previous_waypoint, waypoint_relationship_used, scene_context_used.")

    # Gripper ground-truth: stamp per-waypoint gripper_state from the simulator
    # gripper sequence (outputs/gripper_sequences/<task>.json),
    # and clear held_object wherever that sequence has the gripper open. The
    # model owns held_object itself — the grasp latch never sees a control that
    # is pinched and turned (a knob/crank is not a registered graspable object).
    parser.add_argument("--add-held-object", action="store_true", dest="add_held_object",
                        help="After generating, stamp the simulator's commanded gripper_state into the "
                             "description from the captured gripper sequence. ON by default.")
    parser.add_argument("--no-held-object", action="store_false", dest="add_held_object",
                        help="Disable the stamping pass entirely; keep the VLM's own gripper_state and "
                             "held_object untouched.")
    parser.add_argument("--extract-gripper-sequence", action="store_true", dest="extract_gripper_sequence",
                        help="Re-capture the gripper sequence in CoppeliaSim before stamping (needs the sim). "
                             "Off by default; the existing sequence file is reused otherwise.")
    parser.add_argument("--held-object-only", action="store_true", dest="held_object_only",
                        help="Do NOT call the API. Only stamp gripper_state into existing description file(s) "
                             "for --task/--all from the captured gripper sequence.")

    parser.add_argument(
        "--naming",
        choices=("original", "descriptive"),
        default=DEFAULT_NAMING_MODE if DEFAULT_NAMING_MODE in ("original", "descriptive") else "original",
        help=(
            "Object-naming mode. 'original' (default) copies exact simulator/TTM names "
            "verbatim; 'descriptive' generates clear, image-based, task-relevant names. "
            "Default can also be set via $AHA_NAMING_MODE."
        ),
    )

    parser.add_argument("--write-prompt", help="Write the assembled text prompt to this path.")
    parser.add_argument("--dry-run", action="store_true", help="Build inputs and write a dry-run JSON without calling an API.")

    args = parser.parse_args()
    if args.all and args.output_json:
        parser.error("--output-json cannot be used with --all because each task writes its own result.")
    if args.all and args.grid_image:
        parser.error("--grid-image cannot be used with --all because images are discovered from --photo-dir.")
    return args


def main() -> None:
    args = parse_args()
    provider = args.provider.strip().lower()
    args.provider = provider

    # No-API mode: only stamp held_object into existing description file(s).
    if args.held_object_only:
        photo_dir = Path(args.photo_dir).expanduser()
        if args.all:
            tasks = [extract_task_from_grid_image(p) for p in find_all_grid_images(photo_dir)]
        elif args.task:
            tasks = [args.task]
        else:
            print("ERROR: --held-object-only needs --task <name> or --all.", file=sys.stderr)
            sys.exit(2)
        counts = {"patched": 0, "truncated": 0, "skipped": 0}
        trunc = []
        for task_name in tasks:
            desc_path = photo_dir / f"{task_name}_ALL_WAYPOINTS_COMBINED.{provider}.multimodal_analysis.json"
            gripper_sequence_path = (
                as_path(args.gripper_sequence)
                or Path(args.gripper_sequence_dir).expanduser() / f"{task_name}.json"
            )
            result = add_held_object_to_description(desc_path, task_name, gripper_sequence_path)
            counts[result] = counts.get(result, 0) + 1
            if result == "truncated":
                trunc.append(task_name)
        print(f"\nheld-object: patched {counts['patched']}, "
              f"skipped-truncated {counts['truncated']}, skipped {counts['skipped']}.")
        if trunc:
            print(f"  Truncated gripper sequences (re-capture in sim, then re-run): {trunc}")
        return

    # When no task was given, ask which tasks to generate.
    # ($AHA_TASK_NAME already populates args.task above, so it is honored here.)
    selected_tasks = [args.task] if args.task else []
    if not args.all and not args.task:
        selected_tasks = prompt_for_task(Path(args.photo_dir).expanduser(), provider)
        if not selected_tasks:
            print(
                "ERROR: No task selected. Pass --task <name>, use --all, or set $AHA_TASK_NAME.",
                file=sys.stderr,
            )
            sys.exit(2)
        args.task = selected_tasks[0]

    if len(selected_tasks) > 1 and (args.output_json or args.grid_image):
        print("ERROR: --output-json and --grid-image cannot be used with multiple selected tasks.", file=sys.stderr)
        sys.exit(2)

    client = None if args.dry_run else build_client(provider)

    try:
        if args.all or len(selected_tasks) > 1:
            if args.all:
                image_paths = find_all_grid_images(Path(args.photo_dir).expanduser())
                if not image_paths:
                    raise FileNotFoundError(f"No combined waypoint images found in {args.photo_dir}")
                batch = [(extract_task_from_grid_image(path), path) for path in image_paths]
            else:
                batch = [(name, None) for name in selected_tasks]
            outputs = []
            failed = []
            total_usage = Usage()
            for task_name, image_path in batch:
                try:
                    output_path, usage = run_one(task_name, args, client, image_path)
                    outputs.append(output_path)
                    total_usage += usage
                except Exception as exc:  # noqa: BLE001 — one bad task must not abort the batch
                    failed.append(task_name)
                    print(f"FAILED [{task_name}]: {exc}", file=sys.stderr)
            print(f"\nProcessed {len(outputs)} task(s); {len(failed)} failed: {failed}")
            if not args.dry_run:
                print_usage_summary(args, total_usage, "Cost [TOTAL]")
            return

        run_one(args.task, args, client)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
