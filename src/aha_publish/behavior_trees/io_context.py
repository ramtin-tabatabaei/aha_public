"""Task-context and path I/O helpers for BT maker."""

from aha_publish import paths

from .prompts import *

def parse_json_response(raw: str) -> dict:
    """Strip accidental markdown fences and parse JSON."""
    clean = raw.strip()
    if clean.startswith("```"):
        clean = "\n".join(
            line for line in clean.splitlines()
            if not line.strip().startswith("```")
        ).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        preview = clean[:900]
        if len(clean) > len(preview):
            preview += "..."
        raise ValueError(
            f"Model returned invalid JSON: {e}. Response preview: {preview}"
        ) from e

@lru_cache(maxsize=128)
def _read_json_file_cached(path_text: str, mtime_ns: int) -> dict:
    # mtime_ns is part of the cache key, so edits invalidate automatically.
    del mtime_ns
    return json.loads(Path(path_text).read_text(encoding="utf-8"))

def load_json_file(path: Path) -> dict:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return _read_json_file_cached(str(resolved), stat.st_mtime_ns)

_TOUCHING_KEYWORDS = frozenset({
    "touch", "touching", "touched",
    "contact", "contacting",
    "grasp", "grasping", "grasped",
    "grip", "gripping", "gripped",
    "hold", "holding", "held",
    "enclose", "enclosing", "enclosed",
    "between finger", "finger", "align",
})
_OCCLUDED_KEYWORDS = frozenset({
    "occluded", "hidden", "inside", "closed", "behind",
})


def _proximity_score(desc: str) -> int:
    d = desc.lower()
    if any(kw in d for kw in _OCCLUDED_KEYWORDS):
        return -1
    if any(kw in d for kw in _TOUCHING_KEYWORDS):
        return 2
    if "near" in d or "close" in d or "adjacent" in d or "immediate" in d:
        return 1
    return 0


def waypoint_object_names(wp: dict) -> list[str]:
    """Per-waypoint object list, replacing the removed ``relevant_objects`` field.

    Built from the objects actually present at the waypoint: the held object
    plus every object the waypoint records a distance to
    (``approx_distance_to_objects`` keys). Held object first, then distance keys,
    deduped. Falls back to nothing when neither is present (callers also draw on
    ``key_scene_objects`` roles).
    """
    if not isinstance(wp, dict):
        return []
    names: list[str] = []
    held = wp.get("held_object")
    if held:
        names.append(str(held))
    for name in (wp.get("approx_distance_to_objects") or {}):
        text = str(name)
        if text and text not in names:
            names.append(text)
    return names


def _infer_held_from_proximity(wp: dict, graspable_names: set | None = None) -> str | None:
    """Return the relevant object most likely being held based on proximity data.

    Uses approx_distance_to_objects scores — only returns an object if it
    reaches touching level (score >= 2); otherwise returns None.

    When ``graspable_names`` is provided (a set derived from the task's declared
    graspable objects), only those objects are eligible: a touching object that is
    not graspable is a contact/press target (e.g. a button on a press task), not a
    held object, so it must not be recovered as ``held_object``.
    """
    relevant = waypoint_object_names(wp)
    distances = wp.get("approx_distance_to_objects") or {}
    best_obj, best_score = None, -1
    for obj in relevant:
        if graspable_names is not None and obj not in graspable_names:
            continue
        score = _proximity_score(distances.get(obj, ""))
        if score > best_score:
            best_score, best_obj = score, obj
    return best_obj if best_score >= 2 else None


# gripper_state_source values written by the description pipeline's stamping
# pass. Under both, gripper_state is the simulator's commanded open/closed and
# held_object was decided against the measured gripper table — by the grasp-latch
# cascade ("simulator_ground_truth") or by the description generator itself,
# which is given that table in its prompt ("vlm_held_object"). The generator owns
# it by default because the latch cannot see a control that is pinched and turned
# rather than grasped (a clock crank is never a registered graspable object).
# Anything else means the description was never stamped, so its gripper fields
# are pure image guesses.
STAMPED_GRIPPER_SOURCES = ("simulator_ground_truth", "vlm_held_object")


def has_stamped_gripper_fields(task_obj) -> bool:
    """True when the description's gripper_state / held_object are stamped."""
    return (
        isinstance(task_obj, dict)
        and task_obj.get("gripper_state_source") in STAMPED_GRIPPER_SOURCES
    )


def _sanitize_held_object(task_obj: dict) -> dict:
    """Correct held_object/gripper_state entries that contradict the waypoint's
    known objects (held object + approx_distance_to_objects keys).

    When held_object names an object not among the waypoint's known objects, try
    to recover the correct one from approx_distance_to_objects proximity scores.
    If a relevant object is at
    touching level it becomes the corrected held_object and gripper_state is
    updated to match. If no touching object is found, held_object is cleared
    and gripper_state is set to 'closed'.
    """
    waypoints = task_obj.get("waypoints")
    if not isinstance(waypoints, list):
        return task_obj
    # Declared graspable objects (authoritative when the field is present, even if
    # empty). Only these can actually be held; a touched non-graspable object is a
    # contact/press target. None means "no graspability info" → don't over-filter.
    _pd = task_obj.get("placement_distribution")
    _grasp_list = _pd.get("graspable_objects") if isinstance(_pd, dict) else None
    graspable_names = (
        {str(o.get("name")) for o in _grasp_list if isinstance(o, dict) and o.get("name")}
        if isinstance(_grasp_list, list) else None
    )
    ground_truth = has_stamped_gripper_fields(task_obj)
    new_waypoints = []
    for wp in waypoints:
        if not isinstance(wp, dict):
            new_waypoints.append(wp)
            continue
        held = wp.get("held_object")
        relevant = waypoint_object_names(wp)
        gs = str(wp.get("gripper_state", ""))

        if held is not None and relevant and held not in relevant:
            # Case 1: held_object names a wrong object (not a known waypoint object).
            # Recover the correct one from proximity, or fall back to just closed.
            wp = dict(wp)
            correct = _infer_held_from_proximity(wp, graspable_names)
            wp["held_object"] = correct
            if "holding" in gs.lower():
                if correct:
                    wp["gripper_state"] = re.sub(
                        r"holding\s+object\s*:\s*\w+",
                        f"holding object: {correct}",
                        gs,
                        flags=re.IGNORECASE,
                    )
                else:
                    wp["gripper_state"] = "closed"

        elif held is None and relevant is not None and any(
            kw in gs.lower() for kw in ("closed", "holding", "closing")
        ):
            # Case 2: held_object is null but gripper_state says closed.
            wp = dict(wp)
            if ground_truth:
                # held_object comes from the simulator and is reliable: null means
                # nothing is grasped, so a closed gripper here is a press/contact
                # (e.g. a button on a press task). Trust it — do not invent a held
                # object from proximity and never flip the closed state to open.
                wp["held_object"] = None
                wp["gripper_state"] = "closed"
            else:
                # VLM-sourced context: held_object/gripper_state are guesses. Check
                # whether a graspable object is actually being touched:
                # - if yes → recover held_object and update gripper_state to match.
                # - if no  → the VLM mislabeled a staging/approach pose as closed;
                #            correct to open so the BT gets the right pre-condition.
                correct = _infer_held_from_proximity(wp, graspable_names)
                if correct:
                    wp["held_object"] = correct
                    wp["gripper_state"] = f"holding object: {correct}"
                else:
                    wp["gripper_state"] = "open"
        new_waypoints.append(wp)
    return {**task_obj, "waypoints": new_waypoints}


def load_task_context(path: Path) -> str:
    """Load task context verbatim, preserving explicit gripper/holding states."""
    return json.dumps(load_json_file(path), indent=2)

def load_task_context_object(path: Path) -> dict:
    return load_json_file(path)

def task_name_from_context_path(path: Path) -> str:
    name = path.name
    if name.endswith(".json"):
        name = name[:-5]
    for marker in (
        "_ALL_WAYPOINTS_COMBINED",
        ".openai.multimodal_analysis",
        ".openai.analysis",
        ".multimodal_analysis",
        ".analysis",
    ):
        if marker in name:
            name = name.split(marker)[0]
    return name or path.stem

def task_name_from_context(path: Path, task_context: dict | None = None) -> str:
    task_context = task_context or {}
    for key in ("task", "task_name", "name"):
        value = str(task_context.get(key) or "").strip()
        if value:
            return value
    return task_name_from_context_path(path)

def legacy_review_output_path(task_context_path: Path, task_name: str | None = None) -> Path:
    task_name = task_name or task_name_from_context_path(task_context_path)
    return PREPARED_BTS_DIR / f"{task_name}.bt_conditions.json"

def default_review_output_path(
    task_context_path: Path,
    task_name: str | None = None,
) -> Path:
    task_name = task_name or task_name_from_context_path(task_context_path)
    return PREPARED_BTS_DIR / f"{task_name}.bt_conditions.json"

def review_output_candidates(args: argparse.Namespace) -> list[Path]:
    candidates = [args.review_output]
    if not getattr(args, "review_output_explicit", False):
        legacy_path = legacy_review_output_path(args.task_context, getattr(args, "task_name", None))
        if legacy_path != args.review_output:
            candidates.append(legacy_path)
    return candidates

def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        PROJECT_ROOT / path,
        AHA_SCRIPTS_DIR / path,
        SCRIPT_DIR / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return (PROJECT_ROOT / path).resolve()

def project_relative_str(path: Path | str) -> str:
    """Serialize a path portably: repo-relative when inside the repo, else absolute.

    Keeps saved *.bt_conditions.json files free of machine-specific absolute
    paths so they resolve on any checkout via resolve_project_path().
    """
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(p)

def ensure_input_files(args: argparse.Namespace) -> None:
    if not args.task_context.exists():
        raise FileNotFoundError(f"Task context file not found: {args.task_context}")

def load_design_html() -> str:
    return (SCRIPT_DIR / "design.html").read_text()
