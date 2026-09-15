"""Condition text parsing and predicate normalization helpers."""

from aha_publish import paths

from .failure_definitions import *

def approach_alignment_condition(obj: str) -> str:
    return f"end_effector_aligned_with({obj}) == true"

def condition_text(condition) -> str:
    if isinstance(condition, dict):
        return (
            condition.get("condition")
            or condition.get("text")
            or condition.get("description")
            or ""
        )
    return str(condition)

def normalize_failure_link(link) -> dict:
    if isinstance(link, dict):
        failure = link.get("failure") or link.get("name") or link.get("failure_mode") or ""
        reason = link.get("reason") or link.get("rationale") or link.get("explanation") or ""
        return {"failure": canonical_failure_name(failure), "reason": str(reason)}
    return {"failure": canonical_failure_name(str(link)), "reason": ""}

def condition_failure_links(condition) -> list[dict]:
    if not isinstance(condition, dict):
        return []
    links = (
        condition.get("failure_links")
        or condition.get("related_failures")
        or condition.get("failures")
        or []
    )
    if isinstance(links, str):
        links = [{"failure": links, "reason": ""}]
    normalized = [normalize_failure_link(link) for link in links]
    return normalized[:1]

MEASURED_SPATIAL_FUNCTION_RE = re.compile(
    r"\b(?:" + "dist" "ance" + r"|vertical_delta|horizontal_delta)\s*\("
)

# Allowed condition predicate patterns — one per predicate_template in the
# condition catalogue.  Any condition part not matching one of these is dropped.
ALLOWED_CONDITION_PATTERNS: list[re.Pattern] = [
    # ObjectNotFound
    re.compile(r"^Object_found\(\w+(?:,\s*\w+)*\)\s*==\s*True$", re.IGNORECASE),
    # WrongObjectSelection
    re.compile(r"^selected_object\(\w+(?:,\s*\w+)*\)\s*==\s*True$", re.IGNORECASE),
    # WrongPosition
    re.compile(r"^end_effector_aligned_with\(\w+\)\s*==\s*true$", re.IGNORECASE),
    re.compile(r"^on\(\w+,\s*\w+\)\s*==\s*true$", re.IGNORECASE),
    re.compile(r"^inside\(\w+,\s*\w+\)\s*==\s*true$", re.IGNORECASE),
    re.compile(r"^next_to\(\w+,\s*\w+\)\s*==\s*true$", re.IGNORECASE),
    # WrongOrientation
    re.compile(r"^gripper_oriented_for\(\w+\)\s*==\s*True$", re.IGNORECASE),
    # object_oriented_for_object intentionally removed — there is no detector to
    # check object-to-object orientation, so it is dropped from every BT.
    # ExecutionSequenceMismatch
    re.compile(r"^gripper_condition\s*=\s*(?:Open|Closed)$", re.IGNORECASE),
    re.compile(r"^object_in_gripper\(\w+\)\s*==\s*True$", re.IGNORECASE),
    re.compile(r"^gripper_released\(\w+\)\s*==\s*True$", re.IGNORECASE),
    re.compile(r"^object_for_press\(\w+\)\s*==\s*True$", re.IGNORECASE),
    re.compile(r"^closed\(\w+\)\s*==\s*True$", re.IGNORECASE),
    re.compile(r"^open\(\w+\)\s*==\s*True$", re.IGNORECASE),
    re.compile(r"^end_effector_in_contact_with\(\w+\)\s*==\s*True$", re.IGNORECASE),
]

INVALID_CONDITION_ARGUMENTS = frozenset(
    {
        "",
        "selected",
        "selection",
        "object",
        "obj",
        "item",
        "thing",
        "target",
        "reference",
        "region",
        "placement_region",
        "spawn_boundary",
        "spawnboundary",
        "boundary",
        "waypoint",
    }
)

def is_invalid_condition_argument(name: str) -> bool:
    normalized = str(name or "").strip().lower()
    if normalized in INVALID_CONDITION_ARGUMENTS:
        return True
    if re.fullmatch(r"(?:wp|waypoint)\d*", normalized):
        return True
    if normalized.endswith("_boundary") or normalized.endswith("boundary"):
        return True
    return False

def valid_condition_arguments(args: list[str]) -> list[str]:
    return [arg.strip() for arg in args if not is_invalid_condition_argument(arg)]

# Trailing tokens that are simulator scene-graph cruft, not part of the object's
# real-world name (e.g. RLBench/CoppeliaSim shapes like basket_ball_hoop_respondable
# or crackers_box_visible). Stripped generically — no per-object mapping. Instance
# numbers (block0, block_1) are deliberately NOT stripped so distinct objects stay
# distinguishable.
SIMULATOR_NAME_SUFFIXES = frozenset(
    {
        "respondable",
        "visible",
        "visual",
        "invisible",
        "collision",
        "convex",
        "dynamic",
        "static",
        "mesh",
        "shape",
        "proxy",
        "link",
        "part",
        "sub",
        "ungrouped",
        "dummy",
        "sensor",
        "joint",
        "frame",
    }
)

@lru_cache(maxsize=16384)
def clean_object_name(name: str) -> str:
    """Strip trailing simulator-internal suffixes from an object identifier.

    e.g. basket_ball_hoop_respondable -> basket_ball_hoop,
         crackers_box_visible_respondable -> crackers_box.
    Reverts to the original if cleaning would empty the name or reduce it to a
    generic/invalid argument, so it can never break a condition.
    """
    raw = str(name).strip()
    if not raw:
        return raw
    tokens = raw.split("_")
    while len(tokens) > 1 and tokens[-1].lower() in SIMULATOR_NAME_SUFFIXES:
        tokens.pop()
    cleaned = "_".join(tokens)
    if not cleaned or is_invalid_condition_argument(cleaned):
        return raw
    return cleaned

@lru_cache(maxsize=16384)
def clean_condition_object_names(text: str) -> str:
    """Rewrite every object argument inside predicate(...) calls via clean_object_name.

    Predicate names and non-parenthesized predicates (e.g. gripper_condition = Open)
    are left untouched; only the comma-separated arguments are cleaned.
    """
    def _replace(match: re.Match) -> str:
        predicate, args = match.group(1), match.group(2)
        if not args.strip():
            return match.group(0)
        cleaned = ", ".join(clean_object_name(arg) for arg in args.split(","))
        return f"{predicate}({cleaned})"

    return re.sub(r"\b(\w+)\(([^)]*)\)", _replace, str(text))

def condition_part_with_valid_arguments(part: str) -> str | None:
    text = part.strip()
    object_found = re.fullmatch(
        r"Object_found\(([^)]+)\)\s*==\s*True",
        text,
        flags=re.IGNORECASE,
    )
    if object_found:
        args = valid_condition_arguments(object_found.group(1).split(","))
        if not args:
            return None
        return f"Object_found({', '.join(args)}) == True"

    selected = re.fullmatch(
        r"selected_object\(([^)]+)\)\s*==\s*True",
        text,
        flags=re.IGNORECASE,
    )
    if selected:
        args = [arg.strip() for arg in selected.group(1).split(",")]
        if not args or is_invalid_condition_argument(args[0]):
            return None
        return text

    single_arg = re.fullmatch(
        r"(end_effector_aligned_with|gripper_oriented_for|object_in_gripper|gripper_released|object_for_press|end_effector_in_contact_with|closed|open)\((\w+)\)\s*==\s*(true|True)",
        text,
        flags=re.IGNORECASE,
    )
    if single_arg and is_invalid_condition_argument(single_arg.group(2)):
        return None

    relation = re.fullmatch(
        r"(on|inside|next_to)\((\w+),\s*(\w+)\)\s*==\s*(true|True)",
        text,
        flags=re.IGNORECASE,
    )
    if relation and (
        is_invalid_condition_argument(relation.group(2))
        or is_invalid_condition_argument(relation.group(3))
    ):
        return None

    gripper_condition = re.fullmatch(
        r"gripper_condition\s*=\s*(Open|Closed)",
        text,
        flags=re.IGNORECASE,
    )
    if gripper_condition:
        return f"gripper_condition = {gripper_condition.group(1).capitalize()}"

    return text

def is_allowed_condition_part(part: str) -> bool:
    return any(p.match(part.strip()) for p in ALLOWED_CONDITION_PATTERNS)

def filter_condition_to_allowed_predicates(cond: dict) -> dict | None:
    """Remove parts of a condition whose predicate is not in the allowed list.

    Returns None when no parts survive (the whole block should be dropped).
    """
    parts = condition_parts(cond.get("condition", ""))
    kept = []
    for part in parts:
        if not is_allowed_condition_part(part):
            continue
        validated = condition_part_with_valid_arguments(part)
        if validated:
            kept.append(validated)
    if not kept:
        return None
    return {**cond, "condition": " and ".join(kept)}

@lru_cache(maxsize=16384)
def normalize_spatial_condition_text(text: str) -> str:
    """Normalize spatial checks to qualitative predicates only."""
    clean = " ".join(str(text).strip().split())
    if not clean:
        return clean
    clean = re.sub(
        r"\bgripper_is_open\s*==\s*true\b",
        "gripper_condition = Open",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\bgripper_is_closed\s*==\s*true\b",
        "gripper_condition = Closed",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\bgripper_condition\s*(?:==|=)\s*['\"]?open['\"]?\b",
        "gripper_condition = Open",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\bgripper_condition\s*(?:==|=)\s*['\"]?closed['\"]?\b",
        "gripper_condition = Closed",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\b(?:door_closed|lid_closed|drawer_closed|object_closed)\((\w+)\)\s*==\s*(?:true|True)\b",
        r"closed(\1) == True",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\b((?:\w*_)?(?:door|lid|drawer)(?:_\w*)?)_closed\s*==\s*(?:true|True)\b",
        r"closed(\1) == True",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\b(?:door_open|lid_open|drawer_open|object_open)\((\w+)\)\s*==\s*(?:true|True)\b",
        r"open(\1) == True",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\b((?:\w*_)?(?:door|lid|drawer)(?:_\w*)?)_open(?:ed)?\s*==\s*(?:true|True)\b",
        r"open(\1) == True",
        clean,
        flags=re.IGNORECASE,
    )
    if (
        "postconditions_satisfied" in clean
        or clean.strip().lower() == "robot_state == ready"
        or re.search(r"\bstage_\d+_completed\s*==\s*true\b", clean)
    ):
        return ""

    clean = re.sub(
        r"\b(\w+)_(on|inside|next_to)\((\w+)\)\s*==\s*true\b",
        r"\2(\1, \3) == true",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\b(?:on|inside|next_to)\(((?:\w*_)?(?:door|lid|drawer)(?:_\w*)?),\s*(?:microwave|fridge|refrigerator|oven|grill|cabinet|drawer|box|jar|laptop)\)\s*==\s*true\b",
        r"closed(\1) == True",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\b(\w+)_point_cloud_cluster_valid\s*==\s*true\b",
        r"Object_found(\1) == True",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\b(\w+)_detected\s*==\s*true\b",
        r"Object_found(\1) == True",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\b(\w+)_visible_in_camera\s*==\s*true\b",
        r"Object_found(\1) == True",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\bobject_pose_confidence\((\w+)\)\s*>\s*\d+(?:\.\d+)?\b",
        r"Object_found(\1) == True",
        clean,
    )
    clean = re.sub(
        r"\b(\w+)_pose_confidence\s*>\s*\d+(?:\.\d+)?\b",
        r"Object_found(\1) == True",
        clean,
    )
    clean = re.sub(
        r"\bselected_object_id\s*==\s*locked_target_(\w+)_id\b",
        r"selected_object(\1) == True",
        clean,
    )
    clean = re.sub(
        r"\bcontact\(gripper,\s*(\w+)\)\s*==\s*true\b",
        "gripper_condition = Closed",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\bgripper_correct_grasp\(\w+\)\s*==\s*True\b",
        "gripper_condition = Closed",
        clean,
    )
    clean = re.sub(
        r"\b(\w+)_in_gripper\s*==\s*true\b",
        r"object_in_gripper(\1) == True",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\bgripper_contains\((\w+)\)\s*==\s*true\b",
        r"object_in_gripper(\1) == True",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\b(\w+)_attached_to_end_effector\s*==\s*true\b",
        r"object_in_gripper(\1) == True",
        clean,
        flags=re.IGNORECASE,
    )
    return remove_measured_spatial_parts(clean)

@lru_cache(maxsize=16384)
def remove_measured_spatial_parts(text: str) -> str:
    parts = [
        part.strip()
        for part in str(text).split(" and ")
        if part.strip()
    ]
    kept = [
        part
        for part in parts
        if not MEASURED_SPATIAL_FUNCTION_RE.search(part)
        and not re.search(r"\b\d+(?:\.\d+)?\s*cm\b", part, re.IGNORECASE)
        and not re.search(r"(?:<|>|<=|>=)\s*\d", part)
        and not re.search(r"\b(?:radius|diameter|width|height)\b", part)
    ]
    return " and ".join(kept)

def is_waypoint_name(name: str) -> bool:
    return bool(re.fullmatch(r"(?:waypoint|wp)\d+", str(name), re.IGNORECASE))

def first_end_effector_object(text: str) -> str | None:
    match = re.search(r"end_effector_aligned_with\((\w+)\)", text)
    if match and not is_waypoint_name(match.group(1)):
        return match.group(1)
    return None

def normalize_wrong_position_text(text: str) -> str:
    if re.search(r"\b(?:on|inside|next_to)\(\w+,\s*\w+\)\s*==\s*true\b", text):
        return text

    obj = first_end_effector_object(text)
    if obj:
        return approach_alignment_condition(obj)

    return text

def normalize_condition(condition) -> dict:
    text = normalize_spatial_condition_text(condition_text(condition))
    links = condition_failure_links(condition) or infer_failure_links(text)
    failure = canonical_failure_name((links[0] if links else {}).get("failure", ""))
    if failure == "WrongPosition":
        text = normalize_wrong_position_text(text)
    text = clean_condition_object_names(text)
    return {
        "condition": text,
        # Always empty: reviewers add/remove separate condition blocks rather
        # than swapping alternatives (see BT_CONDITION_GUIDELINES["alternatives"]).
        "alternatives": [],
        "failure_links": links,
    }

def merge_condition_blocks(conditions: list[dict]) -> list[dict]:
    """Deduplicate conditions by canonical text. Each check stays as its own block."""
    seen: set[str] = set()
    unique: list[dict] = []
    for item in conditions:
        normalized = normalize_condition(item)
        text = normalized["condition"]
        if not text:
            continue
        key = " ".join(text.lower().split())
        if key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return unique

def expand_and_conditions(conditions: list[dict]) -> list[dict]:
    """Split condition blocks that contain ' and ' into individual blocks.

    Each atomic predicate becomes its own block with independently inferred failure_links.
    This undoes any ' and ' joining that may have arrived from model output.
    """
    expanded: list[dict] = []
    for cond in conditions:
        parts = [p.strip() for p in cond.get("condition", "").split(" and ") if p.strip()]
        if len(parts) <= 1:
            expanded.append(cond)
            continue
        for part in parts:
            expanded.append({
                "condition": part,
                "alternatives": [],
                "failure_links": infer_failure_links(part),
            })
    return expanded

@lru_cache(maxsize=16384)
def condition_parts(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(text).split(" and ") if part.strip())

def has_object_in_gripper_text(text: str) -> bool:
    return bool(re.search(r"\bobject_in_gripper\(\w+\)\s*==\s*True\b", str(text)))

def has_gripper_state_text(text: str) -> bool:
    return bool(
        re.search(
            r"\bgripper_condition\s*=\s*(?:Open|Closed)\b",
            str(text),
            flags=re.IGNORECASE,
        )
    )
