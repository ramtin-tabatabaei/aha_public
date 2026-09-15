"""Object-scope filtering and ObjectNotFound condition management."""

from aha_publish import paths

from ..semantics import *


def identify_task_objects_from_conditions(
    stages: list[dict],
) -> tuple[set[str], set[str]]:
    """Scan non-ObjectNotFound conditions to identify the primary object and target.

    Returns (primary_objects, target_objects).
    """
    primary_objects: set[str] = set()
    target_objects: set[str] = set()
    for stage in stages:
        for section in ("preconditions", "postconditions"):
            for cond in stage.get(section, []):
                failure = (cond.get("failure_links") or [{}])[0].get("failure")
                if failure == "ObjectNotFound":
                    continue
                text = cond.get("condition", "")
                for m in re.finditer(r"\bselected_object\((\w+)", text):
                    if not is_invalid_condition_argument(m.group(1)):
                        primary_objects.add(m.group(1))
                for m in re.finditer(r"\bobject_in_gripper\((\w+)\)", text):
                    if not is_invalid_condition_argument(m.group(1)):
                        primary_objects.add(m.group(1))
                for m in re.finditer(r"\bgripper_released\((\w+)\)", text):
                    if not is_invalid_condition_argument(m.group(1)):
                        primary_objects.add(m.group(1))
                for m in re.finditer(r"\bobject_for_press\((\w+)\)", text):
                    if not is_invalid_condition_argument(m.group(1)):
                        primary_objects.add(m.group(1))
                for m in re.finditer(r"\bclosed\((\w+)\)", text):
                    if not is_invalid_condition_argument(m.group(1)):
                        primary_objects.add(m.group(1))
                for m in re.finditer(r"\bend_effector_aligned_with\((\w+)\)", text):
                    if not is_invalid_condition_argument(m.group(1)):
                        primary_objects.add(m.group(1))
                for m in re.finditer(r"\b(?:on|inside|next_to)\((\w+),\s*(\w+)\)", text):
                    if not is_invalid_condition_argument(m.group(1)):
                        primary_objects.add(m.group(1))
                    if not is_invalid_condition_argument(m.group(2)):
                        target_objects.add(m.group(2))
                for m in re.finditer(r"\bobject_oriented_for_object\((\w+),\s*(\w+)\)", text):
                    if not is_invalid_condition_argument(m.group(1)):
                        primary_objects.add(m.group(1))
                    if not is_invalid_condition_argument(m.group(2)):
                        target_objects.add(m.group(2))
    return primary_objects, target_objects


def filter_object_not_found_scope(stages: list[dict]) -> list[dict]:
    """Keep ObjectNotFound conditions only for the primary object and placement target.

    Any ObjectNotFound condition for an object not identified as the manipulated
    object or the placement target is removed.
    """
    primary_objects, target_objects = identify_task_objects_from_conditions(stages)
    allowed_objects = primary_objects | target_objects
    if not allowed_objects:
        return stages

    for stage in stages:
        for section in ("preconditions", "postconditions"):
            kept = []
            for cond in stage.get(section, []):
                failure = (cond.get("failure_links") or [{}])[0].get("failure")
                if failure != "ObjectNotFound":
                    kept.append(cond)
                    continue
                objects_in_cond = re.findall(
                    r"\bObject_found\(([^)]+)\)",
                    cond.get("condition", ""),
                    re.IGNORECASE,
                )
                found_objects = {
                    w.strip()
                    for group in objects_in_cond
                    for w in group.split(",")
                    if not is_invalid_condition_argument(w)
                }
                if found_objects & allowed_objects:
                    kept.append(cond)
            stage[section] = kept
    return stages


def hoist_object_not_found_conditions(stages: list[dict]) -> list[dict]:
    if not stages:
        return stages

    first_stage = stages[0]
    collected = []
    seen = {
        condition.get("condition")
        for condition in first_stage.get("preconditions", [])
        if (condition.get("failure_links") or [{}])[0].get("failure") == "ObjectNotFound"
    }

    for stage_index, stage in enumerate(stages):
        for section in ("preconditions", "postconditions"):
            kept = []
            for condition in stage.get(section, []):
                failure = (condition.get("failure_links") or [{}])[0].get("failure")
                if failure != "ObjectNotFound":
                    kept.append(condition)
                    continue
                if stage_index == 0 and section == "preconditions":
                    kept.append(condition)
                    continue
                key = condition.get("condition")
                if key not in seen:
                    collected.append(condition)
                    seen.add(key)
            stage[section] = kept

    first_stage.setdefault("preconditions", []).extend(collected)
    first_stage["preconditions"] = merge_condition_blocks(first_stage["preconditions"])
    return stages


def ensure_object_found_for_all_objects(stages: list[dict]) -> list[dict]:
    """Consolidate all Object_found checks into a single Object_found(obj1, obj2, ...) == True in stage-0 preconditions.

    Only include objects that appear in primary manipulation conditions (selected_object,
    object_in_gripper, gripper_released, on/inside/next_to).  Approach-only objects such as
    simulation boundaries or mesh wrappers that only appear in end_effector_aligned_with are
    intentionally excluded from this list.
    """
    if not stages:
        return stages

    def _collect(text: str, referenced: set[str]) -> None:
        _junk = frozenset({"true", "false", "and", "or", ""})
        for pattern in (
            r"\bselected_object\((\w+)",
            r"\bobject_in_gripper\((\w+)\)",
            r"\bgripper_released\((\w+)\)",
            r"\bobject_for_press\((\w+)\)",
            r"\bclosed\((\w+)\)",
        ):
            for m in re.finditer(pattern, text, re.IGNORECASE):
                word = m.group(1).strip()
                if word.lower() not in _junk and not is_invalid_condition_argument(word):
                    referenced.add(word)
        for m in re.finditer(r"\b(?:on|inside|next_to)\((\w+),\s*(\w+)\)", text, re.IGNORECASE):
            for grp in (m.group(1), m.group(2)):
                word = grp.strip()
                if word.lower() not in _junk and not is_invalid_condition_argument(word):
                    referenced.add(word)
        for m in re.finditer(r"\bobject_oriented_for_object\((\w+),\s*(\w+)\)", text, re.IGNORECASE):
            for grp in (m.group(1), m.group(2)):
                word = grp.strip()
                if word.lower() not in _junk and not is_invalid_condition_argument(word):
                    referenced.add(word)

    referenced: set[str] = set()
    for stage in stages:
        for section in ("preconditions", "postconditions"):
            for cond in stage.get(section, []):
                if (cond.get("failure_links") or [{}])[0].get("failure") == "ObjectNotFound":
                    continue
                _collect(cond.get("condition", ""), referenced)

    # Fallback: if no manipulation-specific predicates found, use alignment conditions
    if not referenced:
        _junk = frozenset({"true", "false", "and", "or", ""})
        for stage in stages:
            for section in ("preconditions", "postconditions"):
                for cond in stage.get(section, []):
                    if (cond.get("failure_links") or [{}])[0].get("failure") == "ObjectNotFound":
                        continue
                    text = cond.get("condition", "")
                    for pattern in (
                        r"\bend_effector_aligned_with\((\w+)\)",
                        r"\bgripper_oriented_for\((\w+)\)",
                    ):
                        for m in re.finditer(pattern, text, re.IGNORECASE):
                            word = m.group(1).strip()
                            if word.lower() not in _junk and not is_invalid_condition_argument(word):
                                referenced.add(word)

    if not referenced:
        return stages

    first = stages[0]
    first["preconditions"] = [
        c for c in first.get("preconditions", [])
        if not re.search(r"\bObject_found\(", c.get("condition", ""), re.IGNORECASE)
    ]

    combined = ", ".join(sorted(referenced))
    new_cond = normalize_condition({
        "condition": f"Object_found({combined}) == True",
        "failure_links": [{
            "failure": "ObjectNotFound",
            "reason": "Confirms all task objects are perceived before execution.",
        }],
    })
    first["preconditions"] = [new_cond] + first.get("preconditions", [])
    return stages


def fix_placement_condition_timing(stages: list[dict]) -> list[dict]:
    """Move placement-result conditions (inside, on, next_to) from preconditions to
    postconditions whenever the same stage also holds the object in the gripper.

    The contradiction: object_in_gripper(X) == True in preconditions means X is still
    held, so inside/on/next_to(X, target) cannot be true yet — those predicates describe
    the physical result of releasing X and can only be checked at the END of the stage.
    """
    for stage in stages:
        preconditions = stage.get("preconditions", [])

        held_objects: set[str] = set()
        for cond in preconditions:
            for m in re.finditer(r"\bobject_in_gripper\((\w+)\)", cond.get("condition", "")):
                held_objects.add(m.group(1))

        if not held_objects:
            continue

        kept: list[dict] = []
        to_promote: list[dict] = []
        for cond in preconditions:
            text = cond.get("condition", "")
            m = re.search(r"\b(?:inside|on|next_to)\((\w+),\s*\w+\)", text)
            if m and m.group(1) in held_objects:
                to_promote.append(cond)
            else:
                kept.append(cond)

        if not to_promote:
            continue

        stage["preconditions"] = kept
        existing = {c.get("condition") for c in stage.get("postconditions", [])}
        additions = [c for c in to_promote if c.get("condition") not in existing]
        stage.setdefault("postconditions", []).extend(additions)
        if additions:
            stage["postconditions"] = merge_condition_blocks(stage["postconditions"])

    return stages
