"""Selected-object and grasp-result precondition enforcement."""

from aha_publish import paths

from ..semantics import *
from .gripper import (
    section_gripper_condition,
    remove_gripper_condition_blocks,
    gripper_condition_block,
    objects_in_gripper_from_conditions,
    section_has_object_in_gripper,
)


def enforce_grasp_result_preconditions(stages: list[dict]) -> list[dict]:
    """Add alignment/orientation before the first waypoint that creates a grasp."""
    already_grasped: set[str] = set()
    for stage in stages:
        grasped_objects = object_in_gripper_postcondition_objects(
            stage.get("postconditions", [])
        )
        if not grasped_objects:
            continue

        preconditions = stage.get("preconditions", [])
        if section_has_object_in_gripper(preconditions):
            already_grasped.update(grasped_objects)
            continue

        new_grasp_objects = {
            obj for obj in grasped_objects
            if obj not in already_grasped and not is_invalid_condition_argument(obj)
        }
        already_grasped.update(grasped_objects)
        if not new_grasp_objects:
            continue

        # A waypoint that creates object_in_gripper should start open, not closed.
        if section_gripper_condition(preconditions) == "Closed":
            preconditions = remove_gripper_condition_blocks(preconditions)
            preconditions.append(
                gripper_condition_block(
                    "Open",
                    "Gripper must be open before creating a new grasp.",
                )
            )

        stage["preconditions"] = merge_condition_blocks(preconditions)

    return stages


def _strip_selected_object(cond: dict) -> dict | None:
    """Strip ONLY selected_object parts; leave alignment and orientation intact."""
    kept = []
    for part in condition_parts(cond.get("condition", "")):
        if re.search(r"\bselected_object\(\w+\)", part):
            continue
        kept.append(part)
    if not kept:
        return None
    return {**cond, "condition": " and ".join(kept)}


def _strip_all_pre_grasp(cond: dict) -> dict | None:
    """Strip selected_object, end_effector_aligned_with, and gripper_oriented_for."""
    kept = []
    for part in condition_parts(cond.get("condition", "")):
        if re.search(r"\bselected_object\(\w+\)", part):
            continue
        if re.search(r"\bend_effector_aligned_with\(\w+\)", part):
            continue
        if re.search(r"\bgripper_oriented_for\(\w+\)", part):
            continue
        kept.append(part)
    if not kept:
        return None
    return {**cond, "condition": " and ".join(kept)}


def enforce_pre_grasp_conditions(stages: list[dict]) -> list[dict]:
    """Ensure selected_object is present exactly once per stage and matches
    the alignment/orientation already in that stage's preconditions.

    Strip rules:
    - ALL stages, postconditions: strip selected_object + alignment + orientation.
    - Stage 0 preconditions: strip all three.
    - All other preconditions: strip selected_object only (keep alignment/orientation).

    Inject rule (stage index > 0):
    - If the stage's preconditions contain end_effector_aligned_with(X) or
      gripper_oriented_for(X), inject selected_object(X) to match.
    - Otherwise, if this is the first stage that creates object_in_gripper(X),
      inject selected_object(X) based on the postcondition.
    - One selected_object per stage; alignment/orientation are the authority
      for which object is being selected.
    """
    # Pass 1: identify first-grasp stage per object (used as fallback when
    # there is no alignment/orientation to read the object from).
    already_grasped: set[str] = set()
    first_grasp_stage: dict[str, int] = {}

    for index, stage in enumerate(stages):
        post_objects = object_in_gripper_postcondition_objects(stage.get("postconditions", []))
        pre_held = objects_in_gripper_from_conditions(stage.get("preconditions", []))
        for obj in post_objects:
            if obj not in pre_held and obj not in already_grasped and not is_invalid_condition_argument(obj):
                first_grasp_stage[obj] = index
        already_grasped.update(post_objects)

    # Pass 2: strip
    for i, stage in enumerate(stages):
        for section in ("preconditions", "postconditions"):
            stripper = _strip_all_pre_grasp if (section == "postconditions" or i == 0) else _strip_selected_object
            kept = []
            for cond in stage.get(section, []):
                result = stripper(cond)
                if result is not None:
                    kept.append(result)
            stage[section] = merge_condition_blocks(kept)

    # Pass 3: inject selected_object at the right stage
    for i, stage in enumerate(stages):
        if i == 0:
            continue
        preconditions = stage.get("preconditions", [])

        # Collect objects named by alignment/orientation still present in preconditions.
        ao_objects: set[str] = set()
        for cond in preconditions:
            for part in condition_parts(cond.get("condition", "")):
                m = re.search(r"\bend_effector_aligned_with\((\w+)\)", part)
                if m and not is_invalid_condition_argument(m.group(1)):
                    ao_objects.add(m.group(1))
                m = re.search(r"\bgripper_oriented_for\((\w+)\)", part)
                if m and not is_invalid_condition_argument(m.group(1)):
                    ao_objects.add(m.group(1))

        if ao_objects:
            # Alignment/orientation is the authority — selected_object must match.
            if len(ao_objects) == 1:
                grasp_obj = next(iter(ao_objects))
            else:
                # Tie-break: prefer the object_in_gripper postcondition match.
                post_objects = object_in_gripper_postcondition_objects(stage.get("postconditions", []))
                grasp_obj = next((o for o in sorted(ao_objects) if o in post_objects), sorted(ao_objects)[0])
        else:
            # No alignment/orientation — use first-grasp postcondition as fallback.
            objs = [obj for obj, idx in first_grasp_stage.items() if idx == i]
            if not objs:
                continue
            post_objects = object_in_gripper_postcondition_objects(stage.get("postconditions", []))
            grasp_obj = next((o for o in sorted(objs) if o in post_objects), sorted(objs)[0])

        preconditions.append(normalize_condition({
            "condition": f"selected_object({grasp_obj}) == True",
            "failure_links": [{"failure": "WrongObjectSelection", "reason": "Confirms the robot has committed to the task-specified object before grasping."}],
        }))
        stage["preconditions"] = merge_condition_blocks(preconditions)

        # Fix object_in_gripper in postconditions to match grasp_obj.
        new_post = []
        for cond in stage.get("postconditions", []):
            fixed = re.sub(
                r"\bobject_in_gripper\(\w+\)",
                f"object_in_gripper({grasp_obj})",
                cond.get("condition", ""),
            )
            new_post.append({**cond, "condition": fixed})
        stage["postconditions"] = new_post

    return stages
