"""Task-context semantic parsing and orientation inference."""

from aha_publish import paths

from .conditions import *

def parse_task_context_for_semantics(task_context: str | dict | None) -> tuple[dict[int, dict], dict[str, str]]:
    if task_context is None:
        return {}, {}
    if isinstance(task_context, str):
        try:
            parsed = json.loads(task_context)
        except json.JSONDecodeError:
            return {}, {}
    else:
        parsed = task_context
    if not isinstance(parsed, dict):
        return {}, {}

    source_stages = parsed.get("stages")
    if source_stages is None:
        source_stages = parsed.get("waypoints", [])

    stage_lookup: dict[int, dict] = {}
    for index, stage in enumerate(source_stages or []):
        if not isinstance(stage, dict):
            continue
        stage_number = stage.get("stage", stage.get("stage_number", stage.get("waypoint", index)))
        for key in {index, stage_number}:
            if isinstance(key, int):
                stage_lookup[key] = stage

    scene_roles = {}
    for item in parsed.get("key_scene_objects", []) or []:
        if not isinstance(item, dict):
            continue
        name = clean_object_name(str(item.get("name") or "").strip())
        if not name:
            continue
        scene_roles[name] = " ".join(
            str(item.get(key) or "")
            for key in ("role", "relationships")
        ).lower()
    return stage_lookup, scene_roles

def stage_with_context(stage: dict, stage_contexts: dict[int, dict] | None = None) -> dict:
    stage_contexts = stage_contexts or {}
    stage_id = stage.get("stage")
    context = stage_contexts.get(stage_id) if isinstance(stage_id, int) else None
    if context is None:
        context = stage_contexts.get(int(stage_id)) if str(stage_id).isdigit() else None
    if not context:
        return stage
    merged = {**context, **stage}
    if not merged.get("summary"):
        merged["summary"] = " ".join(
            str(context.get(key, ""))
            for key in ("summary", "visual_summary", "robot_action")
            if context.get(key)
        )
    return merged

def objects_in_gripper_from_conditions(conditions: list[dict]) -> set[str]:
    held: set[str] = set()
    for condition in conditions:
        for match in re.finditer(r"\bobject_in_gripper\((\w+)\)", condition.get("condition", "")):
            if not is_invalid_condition_argument(match.group(1)):
                held.add(match.group(1))
    return held


def released_objects_from_conditions(conditions: list[dict]) -> set[str]:
    released: set[str] = set()
    for condition in conditions:
        for match in re.finditer(r"\bgripper_released\((\w+)\)", condition.get("condition", "")):
            if not is_invalid_condition_argument(match.group(1)):
                released.add(match.group(1))
    return released


def object_in_gripper_postcondition_objects(conditions: list[dict]) -> set[str]:
    objects: set[str] = set()
    for condition in conditions:
        for match in re.finditer(
            r"\bobject_in_gripper\((\w+)\)\s*==\s*True\b",
            condition.get("condition", ""),
        ):
            obj = match.group(1)
            if not is_invalid_condition_argument(obj):
                objects.add(obj)
    return objects


