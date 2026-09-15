"""Depth, scene, text-context, and prompt payload assembly."""

from aha_publish import paths

from .paths import *
from .prompts import optional_field_replacements

def load_json_file(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(errors="replace"))
    except json.JSONDecodeError:
        return {}


def parse_waypoints(value: str | None) -> list[int] | None:
    if not value:
        return None
    waypoints = []
    for item in value.split(","):
        stripped = item.strip()
        if stripped:
            waypoints.append(int(stripped))
    return waypoints


def gripper_sequence_waypoints(task_paths: TaskInputs) -> list[int]:
    sequence = load_json_file(task_paths.gripper_sequence_path)
    waypoints = []
    for item in sequence.get("waypoints", []):
        index = item.get("index")
        if index is None:
            continue
        try:
            waypoint = int(index)
        except (TypeError, ValueError):
            continue
        if waypoint >= 0:
            waypoints.append(waypoint)
    return sorted(set(waypoints))


def gripper_sequence_block(task_paths: TaskInputs, args: argparse.Namespace) -> str:
    """Per-waypoint gripper action + held object, from the simulator gripper sequence.

    Measured data only (no interpretation): it lets the model tell grasp-and-carry from a
    button press, and know which reorientations turn a held object. Gated by
    --gripper-sequence / --no-gripper-sequence.
    """
    if not getattr(args, "include_gripper_sequence", False):
        return "Gripper sequence input disabled."
    sequence = load_json_file(task_paths.gripper_sequence_path)
    items = sequence.get("waypoints") if sequence else None
    if not items:
        raise ValueError(
            f'Missing or empty gripper sequence: {task_paths.gripper_sequence_path}. '
            'Run 02_descriptions/main.py to capture this task\'s evidence first.'
        )
    lines = [
        f"path: {task_paths.gripper_sequence_path}",
        f"initial_state: {sequence.get('initial_state', 'unknown')}; "
        f"graspable_objects: {sequence.get('graspable_objects', [])}",
        "Per-waypoint gripper action, how far the fingers ended up apart "
        "(gripper_open_amount: 1.0 fully open, 0.0 shut against each other), and "
        "what the simulator latched afterward. held '(none)' means the latch "
        "recorded nothing, NOT that the gripper is empty: only objects the task "
        "registered as graspable can appear there, so a gripped knob, handle, lid "
        "or door never does.",
        "",
        "| waypoint | gripper_action | state_after | gripper_open_amount | held_object |",
        "|---|---|---|---|---|",
    ]
    for item in sorted(items, key=lambda w: w.get("index", 0)):
        held = item.get("held_after") or []
        held_txt = ", ".join(held) if held else "(none)"
        open_amount = item.get("gripper_open_amount")
        open_txt = f"{open_amount:.3f}" if isinstance(open_amount, (int, float)) else "?"
        lines.append(
            f"| wp{item.get('index')} | {item.get('action', '?')} | "
            f"{item.get('state_after', '?')} | {open_txt} | {held_txt} |"
        )
    return "\n".join(lines)


def discover_waypoints(task_paths: TaskInputs, args: argparse.Namespace) -> list[int]:
    explicit = parse_waypoints(args.waypoints)
    if explicit is not None:
        return explicit

    gripper_waypoints = gripper_sequence_waypoints(task_paths)
    if gripper_waypoints:
        return gripper_waypoints

    return [0, 1, 2, 3]


def enabled_text_contexts(task_paths: TaskInputs, args: argparse.Namespace) -> list[tuple[str, Path]]:
    contexts = []
    if args.include_ttm_context:
        contexts.extend(
            (entry.label, entry.path)
            for entry in task_paths.scene_context_paths
        )
    if args.include_task_config:
        contexts.extend(
            (entry.label, entry.path)
            for entry in task_paths.task_context_paths
        )
    return contexts


def text_context_block(task_paths: TaskInputs, args: argparse.Namespace) -> str:
    blocks = []
    for label, path in enabled_text_contexts(task_paths, args):
        blocks.append(
            "\n".join([
                f"## {label}",
                f"path: {path}",
                read_context_file(path, args.max_chars_text_file),
            ])
        )
    if not blocks:
        return "No optional text context enabled."
    return "\n\n".join(blocks)


def group_paths_as_json(entries: list[PathEntry]) -> list[dict[str, str]]:
    return [
        {
            "label": entry.label,
            "path": str(entry.path),
        }
        for entry in entries
    ]


def grouped_input_inventory(task_paths: TaskInputs) -> str:
    return json.dumps({
        "images": group_paths_as_json(task_paths.image_paths),
        "gripper_sequence": str(task_paths.gripper_sequence_path) if task_paths.gripper_sequence_path else None,
        "scene_context": group_paths_as_json(task_paths.scene_context_paths),
        "task_context": group_paths_as_json(task_paths.task_context_paths),
        "outputs": group_paths_as_json(task_paths.output_paths),
    }, indent=2)


def camera_count_text(args: argparse.Namespace) -> str:
    if args.camera_count:
        return f"{args.camera_count} camera column(s)"
    return "not fixed; infer it from the grid image and visible camera labels"


def build_text_payload(
    task_paths: TaskInputs,
    args: argparse.Namespace,
    waypoints: list[int],
) -> str:
    naming = resolve_naming_mode(args)
    user_prompt = (
        USER_PROMPT_TEMPLATE
        .replace("__TASK_NAME__", task_paths.task_name)
        .replace("__WAYPOINT_COUNT__", str(len(waypoints)))
        .replace("__WAYPOINT_IDS_JSON__", json.dumps(waypoints))
        .replace("__CAMERA_COUNT_TEXT__", camera_count_text(args))
        .replace("__NAME_FIELD_HINT__", NAME_FIELD_HINT[naming])
        .replace("__NAMING_RULE__", NAMING_RULE[naming])
    )
    for placeholder, snippet in optional_field_replacements(args).items():
        user_prompt = user_prompt.replace(placeholder, snippet)

    return "\n\n".join([
        user_prompt,
        "# Task and simulator",
        json.dumps({
            "task": task_paths.task_name,
            "simulator": task_paths.simulator,
            "grid_image_path": str(task_paths.grid_image_path),
            "expected_waypoint_ids": waypoints,
            "camera_count": args.camera_count,
        }, indent=2),
        "# Enabled input flags",
        json.dumps({
            "include_grid_image": args.include_grid_image,
            "include_ttm_context": args.include_ttm_context,
            "include_gripper_sequence": getattr(args, "include_gripper_sequence", False),
            "include_task_config": args.include_task_config,
        }, indent=2),
        "# Grouped input paths",
        grouped_input_inventory(task_paths),
        "# Gripper / grasp sequence input",
        gripper_sequence_block(task_paths, args),
        "# Scene/task context input",
        text_context_block(task_paths, args),
    ])
