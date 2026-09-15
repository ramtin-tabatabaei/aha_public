"""Alignment and orientation condition management passes."""

from aha_publish import paths

from ..semantics import *


def condition_has_end_effector_alignment(condition: dict) -> bool:
    return "end_effector_aligned_with(" in str(condition.get("condition", ""))


def condition_has_orientation(condition: dict) -> bool:
    return "gripper_oriented_for(" in str(condition.get("condition", ""))


def strip_end_effector_alignment(condition: dict) -> dict | None:
    parts = [
        part.strip()
        for part in str(condition.get("condition", "")).split(" and ")
        if part.strip()
    ]
    kept = [
        part
        for part in parts
        if "end_effector_aligned_with(" not in part
        and not re.search(r"\b" + "dist" "ance" + r"\(\s*end_effector\s*,", part)
    ]
    if not kept:
        return None
    return {**condition, "condition": " and ".join(kept)}


def strip_end_effector_alignment_for_objects(condition: dict, objects: set[str]) -> dict | None:
    if not objects:
        return condition
    kept = []
    for part in condition_parts(condition.get("condition", "")):
        match = re.search(r"\bend_effector_aligned_with\((\w+)\)", part)
        if match and match.group(1) in objects:
            continue
        kept.append(part)
    if not kept:
        return None
    return {**condition, "condition": " and ".join(kept)}


def strip_orientation(condition: dict) -> dict | None:
    parts = [
        part.strip()
        for part in str(condition.get("condition", "")).split(" and ")
        if part.strip()
    ]
    kept = [part for part in parts if "gripper_oriented_for(" not in part]
    if not kept:
        return None
    return {**condition, "condition": " and ".join(kept)}


def strip_orientation_for_objects(condition: dict, objects: set[str]) -> dict | None:
    if not objects:
        return condition
    parts = [
        part.strip()
        for part in str(condition.get("condition", "")).split(" and ")
        if part.strip()
    ]
    kept = []
    for part in parts:
        match = re.search(r"\bgripper_oriented_for\((\w+)\)", part)
        if match and match.group(1) in objects:
            continue
        kept.append(part)
    if not kept:
        return None
    return {**condition, "condition": " and ".join(kept)}


def stage_allows_wrong_object_selection(stage: dict) -> bool:
    text = " ".join(
        str(stage.get(key, ""))
        for key in ("name", "summary", "visual_summary", "robot_action")
    ).lower()
    return any(
        token in text
        for token in (
            "pick up",
            "pickup",
            "picking",
            "grasp",
            "at the grasp",
            "grasp pose",
            "ready to grasp",
            "before grasp",
            "pre-grasp",
            "closing",
            "close gripper",
            "close_gripper",
            "engage",
            "push",
            "press",
            "manipulate",
            "manipulating",
            "manipulation",
        )
    )


def stage_allows_end_effector_alignment(stage: dict) -> bool:
    text = " ".join(
        str(stage.get(key, ""))
        for key in ("name", "summary", "visual_summary", "robot_action")
    ).lower()
    middle_motion = any(
        token in text
        for token in (
            "carry",
            "carries",
            "carrying",
            "transport",
            "transfer",
            "moving",
            "move toward",
            "moves toward",
            "toward",
            "lift",
            "lifting",
            "lifted",
        )
    )
    commitment_stage = any(
        token in text
        for token in (
            "pick up",
            "pickup",
            "picking",
            "grasp",
            "at the grasp",
            "grasp pose",
            "ready to grasp",
            "before grasp",
            "pre-grasp",
            "closing",
            "close gripper",
            "close_gripper",
            "engage",
            "push",
            "press",
            "manipulate",
            "manipulating",
            "manipulation",
        )
    )
    if middle_motion and not commitment_stage:
        return False
    if commitment_stage:
        return True

    return False
