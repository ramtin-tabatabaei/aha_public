"""Cleanup pipeline — ordered pass list and stage-level filtering helpers."""

from aha_publish import paths

from ..semantics import *
from .predicates import apply_allowed_predicates_filter, strip_color_qualifiers_from_object_found
from .object_scope import (
    filter_object_not_found_scope,
    hoist_object_not_found_conditions,
    ensure_object_found_for_all_objects,
    fix_placement_condition_timing,
)
from .alignment import (
    condition_has_end_effector_alignment,
    condition_has_orientation,
    strip_end_effector_alignment,
    strip_end_effector_alignment_for_objects,
    strip_orientation,
    strip_orientation_for_objects,
    stage_allows_wrong_object_selection,
    stage_allows_end_effector_alignment,
)
from .selected_object import (
    enforce_grasp_result_preconditions,
    enforce_pre_grasp_conditions,
)
from .gripper import (
    section_has_object_in_gripper,
    section_has_gripper_state,
    section_gripper_condition,
    objects_in_gripper_from_conditions,
    gripper_state_condition,
    enforce_release_breaks_holding,
    enforce_gripper_condition_continuity,
    enforce_held_object_name_continuity,
)


def filter_stage_condition_blocks(
    stage: dict,
    conditions: list[dict],
    section: str = "preconditions",
) -> list[dict]:
    kept = []
    allows_wrong_object = stage_allows_wrong_object_selection(stage)
    allows_alignment = stage_allows_end_effector_alignment(stage)
    has_held_object = section_has_object_in_gripper(conditions)
    held_objects = objects_in_gripper_from_conditions(conditions)
    section_is_closed = section == "preconditions" and section_gripper_condition(conditions) == "Closed"

    for condition in conditions:
        failure = (condition.get("failure_links") or [{}])[0].get("failure")
        if failure == "WrongObjectSelection" and not allows_wrong_object:
            continue
        if section_is_closed and condition_has_end_effector_alignment(condition):
            condition = strip_end_effector_alignment(condition)
            if condition is None:
                continue
        if section_is_closed and condition_has_orientation(condition):
            condition = strip_orientation(condition)
            if condition is None:
                continue
        if held_objects and condition_has_end_effector_alignment(condition):
            condition = strip_end_effector_alignment_for_objects(condition, held_objects)
            if condition is None:
                continue
        if section != "preconditions" and condition_has_end_effector_alignment(condition):
            condition = strip_end_effector_alignment(condition)
            if condition is None:
                continue
        if section != "preconditions" and condition_has_orientation(condition):
            condition = strip_orientation(condition)
            if condition is None:
                continue
        if condition_has_end_effector_alignment(condition) and not allows_alignment:
            condition = strip_end_effector_alignment(condition)
            if condition is None:
                continue
        if condition_has_orientation(condition) and not allows_alignment:
            condition = strip_orientation(condition)
            if condition is None:
                continue
        if has_held_object and has_gripper_state_text(condition.get("condition", "")):
            condition = {
                **condition,
                "condition": " and ".join(
                    part
                    for part in condition_parts(condition.get("condition", ""))
                    if not has_gripper_state_text(part)
                ),
            }
            if not condition["condition"]:
                continue
        if held_objects and condition_has_orientation(condition):
            condition = strip_orientation_for_objects(condition, held_objects)
            if condition is None:
                continue
        kept.append(condition)

    if not section_has_object_in_gripper(kept) and not section_has_gripper_state(kept):
        kept.append(gripper_state_condition(stage, section))
        kept = merge_condition_blocks(kept)

    return kept


def apply_condition_cleanup_pipeline(stages: list[dict]) -> list[dict]:
    """Apply all deterministic repair/cleanup passes in one ordered pipeline."""
    stages = hoist_object_not_found_conditions(stages)
    stages = apply_allowed_predicates_filter(stages)
    stages = filter_object_not_found_scope(stages)
    stages = strip_color_qualifiers_from_object_found(stages)
    stages = ensure_object_found_for_all_objects(stages)
    stages = fix_placement_condition_timing(stages)
    stages = enforce_release_breaks_holding(stages)
    stages = enforce_gripper_condition_continuity(stages)
    stages = enforce_grasp_result_preconditions(stages)
    stages = enforce_pre_grasp_conditions(stages)
    stages = ensure_object_found_for_all_objects(stages)
    return enforce_held_object_name_continuity(stages)
