"""Apply simulator gripper ground truth to generated task descriptions.

The sequence input lives at:
  outputs/gripper_sequences/<task>.json

It is produced by ``extract_gripper_sequence.py`` and records per-waypoint
open/close actions plus actual grasped objects. This module patches generated
description JSON so downstream BT generation uses the simulator-confirmed
``gripper_state`` instead of an image guess.

``held_object`` is NOT re-derived here: the description generator decides it,
having been given this same sequence table in its prompt, and this module only
clears it where the gripper is commanded open. The simulator grasp latch cannot
own that field because it never sees a control that is pinched and turned rather
than grasped — a clock crank is not a registered graspable object, so
``held_after`` stays empty while the robot holds and turns it.
"""

from aha_publish import paths

from .config import *

SEQ_DIR = DEFAULT_GRIPPER_SEQUENCE_DIR
DESC_DIR = paths.DESCRIPTION_DIR


def derive_waypoint_states(seq: dict, held_by_index: dict) -> list[dict]:
    """Return [{index, gripper_state, held_object}] from a captured sequence.

    ``held_by_index`` maps a waypoint index to the object held while the gripper
    is closed there (see :func:`_vlm_held_by_index`). A held object is reported
    only where the sequence has the gripper commanded closed, so the commanded
    action always wins over a claimed hold.
    """
    out = []
    cmd_state = seq.get("initial_state", "Open")
    for wp in seq.get("waypoints", []):
        idx = wp.get("index")
        action = wp.get("action", "none")
        if action == "open":
            cmd_state = "Open"
        elif action == "close":
            cmd_state = "Closed"

        held_obj = held_by_index.get(idx) if cmd_state == "Closed" else None

        if held_obj:
            out.append({
                "index": idx,
                "gripper_state": "holding",
                "held_object": held_obj,
            })
        else:
            out.append({
                "index": idx,
                "gripper_state": "open" if cmd_state == "Open" else "closed",
                "held_object": None,
            })
    return out


def _vlm_held_by_index(desc: dict) -> dict:
    """Waypoint index -> the held object the DESCRIPTION ITSELF names.

    The generation prompt hands the model the measured gripper table (open/close
    action, state_after, gripper_open_amount, held_after) and has it name the
    held object per waypoint, so its answer already covers the cases the
    simulator grasp latch is blind to — a crank or knob pinched and turned is
    never a registered graspable object, so ``held_after`` stays empty there
    (change_clock). The model is told to answer with key_scene_objects names;
    :func:`descriptive_name_map` catches the case where it gives the simulator
    handle instead.
    """
    out: dict = {}
    for waypoint in desc.get("waypoints", []) or []:
        idx = waypoint.get("waypoint")
        held = waypoint.get("held_object")
        if idx is None or not isinstance(held, str):
            continue
        held = held.strip()
        if held and held.lower() not in ("null", "none", "uncertain", "not applicable"):
            out[idx] = held
    return out


def descriptive_name_map(desc: dict) -> dict:
    """Map raw simulator object names to the description's object names.

    Applied to the held object so a model that answers with an ``original_name``
    handle still lands on the ``key_scene_objects`` name downstream reads.
    """
    out = {}
    for item in desc.get("key_scene_objects", []) or []:
        name = item.get("name")
        original = item.get("original_name")
        if name and original and original != name:
            out[original] = name
    return out


def patch_description_obj(desc: dict, seq: dict) -> tuple[dict, list[str]]:
    """Stamp the commanded gripper state into ``desc``.

    ``gripper_state`` is the simulator's, because open/closed is the commanded
    action and no image can beat it. ``held_object`` stays the description
    generator's own answer, cleared only where the gripper is commanded open.
    """
    states = {
        state["index"]: state
        for state in derive_waypoint_states(seq, _vlm_held_by_index(desc))
    }
    name_map = descriptive_name_map(desc)
    notes = []
    last = None
    for waypoint in desc.get("waypoints", []):
        idx = waypoint.get("waypoint")
        if idx is None:
            continue

        state = states.get(idx)
        if state is None:
            if last is not None:
                waypoint["gripper_state"] = last["gripper_state"]
                waypoint["held_object"] = last["held_object"]
                notes.append(
                    f"wp{idx}: (no seq entry) carried -> {waypoint['gripper_state']!r}"
                )
            continue

        old = waypoint.get("gripper_state")
        if state["gripper_state"] == "holding":
            held = name_map.get(state["held_object"], state["held_object"])
            waypoint["gripper_state"] = f"holding object: {held}"
            waypoint["held_object"] = held
        elif state["gripper_state"] == "open":
            waypoint["gripper_state"] = "open"
            waypoint["held_object"] = None
        else:
            # Closed with no held object is a real closed/no-hold state, common
            # for button presses: the fingers shut against each other with
            # nothing between them.
            waypoint["gripper_state"] = "closed"
            waypoint["held_object"] = None

        last = {
            "gripper_state": waypoint["gripper_state"],
            "held_object": waypoint["held_object"],
        }
        notes.append(f"wp{idx}: {old!r} -> {waypoint['gripper_state']!r}")

    # Omit the legacy marker, including when re-stamping an older description.
    desc.pop("gripper_state_source", None)
    return desc, notes


def find_description_file(task: str, desc_dir: Path = DESC_DIR) -> Path | None:
    """The canonical description JSON for a task.

    The generator writes ``<grid>.<provider>.multimodal_analysis.json``; extra
    variants (``...openai.IMPROVEDTTM.multimodal_analysis.json``) are experiments
    that must not be patched in place of the real output, so the file with the
    fewest name segments wins.
    """
    candidates = sorted(desc_dir.glob(f"{task}_ALL_WAYPOINTS_COMBINED*.json"))
    candidates = [path for path in candidates if "usage" not in path.name]
    candidates.sort(key=lambda path: (path.name.count("."), path.name))
    return candidates[0] if candidates else None


def patch_task(
    task: str,
    seq_dir: Path = SEQ_DIR,
    desc_dir: Path = DESC_DIR,
) -> bool:
    seq_path = seq_dir / f"{task}.json"
    desc_path = find_description_file(task, desc_dir)
    if not seq_path.exists():
        print(f"{task}: no gripper sequence ({seq_path.name}); skipped")
        return False
    if not desc_path:
        print(f"{task}: no description file; skipped")
        return False

    seq = json.loads(seq_path.read_text())
    desc = json.loads(desc_path.read_text())
    desc, notes = patch_description_obj(desc, seq)
    desc_path.write_text(json.dumps(desc, indent=2))
    print(f"{task}: patched {len(notes)} waypoints -> {desc_path.name}")
    for note in notes:
        print(f"    {note}")
    return True


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    for task in argv:
        patch_task(task)
