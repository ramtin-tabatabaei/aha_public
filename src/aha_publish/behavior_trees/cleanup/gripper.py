"""Gripper-state condition helpers and enforcement passes."""

from aha_publish import paths

from ..semantics import *


def gripper_condition_from_text(text: str) -> str | None:
    match = re.search(
        r"\bgripper_condition\s*=\s*(Open|Closed)\b",
        str(text),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1).capitalize()


def section_gripper_condition(conditions: list[dict]) -> str | None:
    for condition in conditions:
        state = gripper_condition_from_text(condition.get("condition", ""))
        if state:
            return state
    return None


def gripper_condition_block(state: str, reason: str) -> dict:
    return normalize_condition(
        {
            "condition": f"gripper_condition = {state}",
            "failure_links": [
                {
                    "failure": "ExecutionSequenceMismatch",
                    "reason": reason,
                }
            ],
        }
    )


def remove_gripper_condition_blocks(conditions: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for condition in conditions:
        parts = [
            part
            for part in condition_parts(condition.get("condition", ""))
            if not gripper_condition_from_text(part)
        ]
        if not parts:
            continue
        kept.append({**condition, "condition": " and ".join(parts)})
    return kept


def object_in_gripper_block(obj: str, reason: str) -> dict:
    return normalize_condition(
        {
            "condition": f"object_in_gripper({obj}) == True",
            "failure_links": [
                {
                    "failure": "ExecutionSequenceMismatch",
                    "reason": reason,
                }
            ],
        }
    )


def stage_creates_held_object(stage: dict) -> bool:
    text = " ".join(
        str(stage.get(key, ""))
        for key in ("name", "summary", "visual_summary", "robot_action")
    ).lower()
    return any(
        token in text
        for token in (
            "close gripper",
            "close_gripper",
            "closing",
            "grasp",
            "grasped",
            "pick up",
            "pickup",
            "picking",
            "lift",
            "lifting",
            "lifted",
        )
    )


def section_has_object_in_gripper(conditions: list[dict]) -> bool:
    return any(has_object_in_gripper_text(condition.get("condition", "")) for condition in conditions)


def section_has_object_for_press(conditions: list[dict]) -> bool:
    return any(
        re.search(r"\bobject_for_press\(\w+\)", c.get("condition", ""))
        for c in conditions
    )



def strip_object_in_gripper_for_objects(condition: dict, objects: set[str]) -> dict | None:
    if not objects:
        return condition
    kept = []
    for part in condition_parts(condition.get("condition", "")):
        match = re.search(r"\bobject_in_gripper\((\w+)\)", part)
        if match and match.group(1) in objects:
            continue
        kept.append(part)
    if not kept:
        return None
    return {**condition, "condition": " and ".join(kept)}


def section_has_gripper_state(conditions: list[dict]) -> bool:
    return any(has_gripper_state_text(condition.get("condition", "")) for condition in conditions)


def default_gripper_state_for_stage(stage: dict, section: str) -> str:
    text = " ".join(
        str(stage.get(key, ""))
        for key in ("name", "summary", "visual_summary", "robot_action")
    ).lower()
    if any(token in text for token in ("open gripper", "open_gripper", "release", "drop", "dunk", "leave", "leaving")):
        return "gripper_condition = Open"
    if any(token in text for token in ("approach", "near", "positioned", "aligned", "ready")):
        return "gripper_condition = Open"
    if any(token in text for token in ("close gripper", "close_gripper", "closing", "grasp", "pick up", "pickup", "picking")):
        return "gripper_condition = Open" if section == "preconditions" else "gripper_condition = Closed"
    return "gripper_condition = Closed"


def gripper_state_condition(stage: dict, section: str) -> dict:
    state = default_gripper_state_for_stage(stage, section)
    return normalize_condition(
        {
            "condition": state,
            "failure_links": [
                {
                    "failure": "ExecutionSequenceMismatch",
                    "reason": "Checks the gripper open/closed state required for this stage.",
                }
            ],
        }
    )


def enforce_release_breaks_holding(stages: list[dict]) -> list[dict]:
    """A released object cannot be held at the start of the next waypoint."""
    for index in range(len(stages) - 1):
        released_objects = released_objects_from_conditions(
            stages[index].get("postconditions", [])
        )
        if not released_objects:
            continue

        next_stage = stages[index + 1]
        next_preconditions = []
        removed_objects: set[str] = set()
        for condition in next_stage.get("preconditions", []):
            condition_objects = objects_in_gripper_from_conditions([condition])
            condition = strip_object_in_gripper_for_objects(condition, released_objects)
            if condition is None:
                removed_objects.update(condition_objects & released_objects)
                continue
            if condition_objects & released_objects:
                removed_objects.update(condition_objects & released_objects)
            next_preconditions.append(condition)

        if not removed_objects:
            continue

        next_stage["preconditions"] = merge_condition_blocks(next_preconditions)
        if stage_creates_held_object(next_stage):
            existing_post = {
                condition.get("condition")
                for condition in next_stage.get("postconditions", [])
            }
            for obj in sorted(removed_objects):
                condition_text_value = f"object_in_gripper({obj}) == True"
                if condition_text_value in existing_post:
                    continue
                next_stage.setdefault("postconditions", []).append(
                    object_in_gripper_block(
                        obj,
                        "Confirms the object is held after this grasp or lift action.",
                    )
                )
            next_stage["postconditions"] = merge_condition_blocks(
                next_stage.get("postconditions", [])
            )

    return stages


def enforce_held_object_name_continuity(stages: list[dict]) -> list[dict]:
    """Keep object_in_gripper(...) naming the SAME object across one hold sequence.

    The grasp waypoint commits to a single object (the selected object).  Later
    continuing-hold waypoints sometimes name a *different* object in
    object_in_gripper(...) because the LLM described the stage by the thing being
    moved rather than the thing actually grasped (e.g. open_window: the gripper
    holds the handle, but the "push window outward" stage was written as
    object_in_gripper(window)).  Physically only the grasped object is in the
    gripper, so every object_in_gripper in the same uninterrupted hold sequence is
    rewritten to the first/grasped object.  A release (gripper opens or
    gripper_released) ends the sequence and clears the held object, so a genuine
    re-grasp of a new object after a release is left untouched.
    """
    held: str | None = None
    for stage in stages:
        for section in ("preconditions", "postconditions"):
            new_section = []
            changed = False
            for condition in stage.get(section, []):
                objects = objects_in_gripper_from_conditions([condition])
                if objects:
                    if held is None:
                        held = sorted(objects)[0]
                    if any(obj != held for obj in objects):
                        condition = {
                            **condition,
                            "condition": re.sub(
                                r"\bobject_in_gripper\(\w+\)",
                                f"object_in_gripper({held})",
                                condition.get("condition", ""),
                            ),
                        }
                        changed = True
                new_section.append(condition)
            if changed:
                stage[section] = merge_condition_blocks(new_section)

        post_conditions = stage.get("postconditions", [])
        if (released_objects_from_conditions(post_conditions)
                or section_gripper_condition(post_conditions) == "Open"):
            held = None

    return stages


def enforce_gripper_condition_continuity(stages: list[dict]) -> list[dict]:
    """Propagate gripper state across waypoints, preferring object_in_gripper over gripper_condition = Closed.

    Two things this function does:

    1. Carry-forward preconditions: sets the next stage's gripper precondition to
       match the last known state.  When the gripper is Closed and the held object
       is known, writes object_in_gripper(<obj>) instead of gripper_condition = Closed.

    2. Postcondition replacement: once the first close event is recorded, any later
       stage whose postcondition still carries gripper_condition = Closed (but we
       know the held object) has that block replaced with object_in_gripper(<obj>).
       The first close postcondition is left as-is — it marks the actual close action.

    The held object is found by scanning forward from the close event so that cases
    where object_in_gripper first appears in a later stage (e.g. open_jar lifts the
    lid two stages after grasping) are handled correctly.
    """
    def _first_held_object_from(start: int) -> str | None:
        for i in range(start, len(stages)):
            # Stop when the previous stage opened the gripper — don't bleed across
            # an open/close boundary into a different object's hold sequence.
            if i > start:
                prev_post = stages[i - 1].get("postconditions", [])
                if section_gripper_condition(prev_post) == "Open":
                    break
            all_conds = stages[i].get("preconditions", []) + stages[i].get("postconditions", [])
            held = objects_in_gripper_from_conditions(all_conds)
            if held:
                return sorted(held)[0]
        return None

    _placement_re = re.compile(r"\b(?:on|inside|next_to)\(", re.IGNORECASE)

    # Pre-pass: fix carry waypoints whose POST has a spurious gripper_condition = Open.
    # A waypoint is a carry (not a release) if its POST has no gripper_released and no
    # placement predicate AND the NEXT waypoint's PRE still expects object_in_gripper —
    # meaning the object is still being held across the boundary.
    for i, stage in enumerate(stages):
        pre = stage.get("preconditions", [])
        held = objects_in_gripper_from_conditions(pre)
        if not held:
            continue
        post = stage.get("postconditions", [])
        if released_objects_from_conditions(post):
            continue
        if any(_placement_re.search(c.get("condition", "")) for c in post):
            continue
        if section_gripper_condition(post) != "Open":
            continue
        # Only fix if the very next waypoint still holds the same object.
        if i + 1 >= len(stages):
            continue
        next_held = objects_in_gripper_from_conditions(stages[i + 1].get("preconditions", []))
        if not (held & next_held):
            continue
        stage["postconditions"] = remove_gripper_condition_blocks(post)
        for obj in sorted(held):
            stage["postconditions"].append(
                object_in_gripper_block(obj, "Object still held — gripper_condition = Open removed.")
            )
        stage["postconditions"] = merge_condition_blocks(stage["postconditions"])

    last_known_state: str | None = None
    last_held_object: str | None = None
    first_close_seen: bool = False

    for index, stage in enumerate(stages):
        post_conditions = stage.get("postconditions", [])
        post_state = section_gripper_condition(post_conditions)

        if post_state:
            last_known_state = post_state
            if post_state == "Open":
                # When the gripper opens while still holding an object, add gripper_released.
                if last_held_object and not released_objects_from_conditions(stage.get("postconditions", [])):
                    stage["postconditions"].append(normalize_condition({
                        "condition": f"gripper_released({last_held_object}) == True",
                        "failure_links": [{"failure": "ExecutionSequenceMismatch",
                                           "reason": "Object released as the gripper opens."}],
                    }))
                    stage["postconditions"] = merge_condition_blocks(stage["postconditions"])
                last_held_object = None
                first_close_seen = False
            elif post_state == "Closed" and last_held_object is None:
                last_held_object = _first_held_object_from(index)

        held_in_post = objects_in_gripper_from_conditions(post_conditions)
        if held_in_post:
            last_held_object = sorted(held_in_post)[0]
            last_known_state = "Closed"
            first_close_seen = True

        # Press stage: gripper closed on a button/contact but no object held.
        # Treat as Closed state so the next stage gets gripper_condition = Closed,
        # but do NOT set last_held_object (nothing to propagate as object_in_gripper).
        if section_has_object_for_press(post_conditions):
            last_known_state = "Closed"
            first_close_seen = True

        # Replace gripper_condition = Closed in this stage's postconditions with
        # object_in_gripper once we're past the first close event and know the object.
        # Skip press stages — object_for_press must not be replaced.
        if (first_close_seen
                and last_known_state == "Closed"
                and last_held_object
                and section_gripper_condition(stage.get("postconditions", [])) == "Closed"
                and not section_has_object_for_press(stage.get("postconditions", []))):
            stage["postconditions"] = remove_gripper_condition_blocks(stage.get("postconditions", []))
            stage["postconditions"].append(
                object_in_gripper_block(
                    last_held_object,
                    "Carries forward the held object from the previous waypoint.",
                )
            )
            stage["postconditions"] = merge_condition_blocks(stage["postconditions"])

        if post_state == "Closed" and not first_close_seen:
            first_close_seen = True

        if index + 1 >= len(stages) or last_known_state is None:
            continue

        next_stage = stages[index + 1]
        next_preconditions = next_stage.get("preconditions", [])

        if section_has_object_in_gripper(next_preconditions) or section_has_object_for_press(next_preconditions):
            continue

        if last_known_state == "Closed" and last_held_object:
            next_stage["preconditions"] = remove_gripper_condition_blocks(next_preconditions)
            next_stage["preconditions"].append(
                object_in_gripper_block(
                    last_held_object,
                    "Carries forward the held object from the previous waypoint.",
                )
            )
            next_stage["preconditions"] = merge_condition_blocks(next_stage["preconditions"])
        else:
            pre_state = section_gripper_condition(next_preconditions)
            if pre_state == last_known_state:
                continue
            next_stage["preconditions"] = remove_gripper_condition_blocks(next_preconditions)
            next_stage["preconditions"].append(
                gripper_condition_block(
                    last_known_state,
                    "Carries forward the gripper state from the previous waypoint.",
                )
            )
            next_stage["preconditions"] = merge_condition_blocks(next_stage["preconditions"])

    return stages
