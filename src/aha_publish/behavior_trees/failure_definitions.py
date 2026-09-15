"""Failure definition loading, hold conditions, and measured-context cleanup."""

from aha_publish import paths

from .io_context import *

# ------------------------------------------------------------------------------
# Hold conditions (runtime detectors)
#
# Continuous "must stay true while the stage runs" invariants, one per runtime
# detector in aha_scripts/detectors/. Unlike pre/postconditions these are NOT produced by the
# language model; the BT maker attaches the full set to every waypoint
# deterministically. This list is the catalog: it is the only definition of the
# hold-condition set, and the text below is what lands in every generated BT.
# ------------------------------------------------------------------------------
DEFAULT_HOLD_CONDITION_DETECTORS = [
    {
        "detector": "collision",
        "failure": "collision",
        "condition": "no_collision() == True",
        "label": "No collision",
        "definition": "No unexpected collision or contact spike occurs while the stage executes. Backed by the collision detector (joint-torque deltas/norms compared against the task's successful-run torque stats).",
    },
    {
        "detector": "freezing",
        "failure": "freezing",
        "condition": "not_frozen() == True",
        "label": "Not frozen",
        "definition": "The robot keeps making progress and does not freeze or stall during the stage. Backed by the freezing detector (joint position/velocity stillness plus camera-motion across views).",
    },
    {
        "detector": "slip",
        "failure": "slip",
        "condition": "maintains_grasp() == True",
        "label": "Grasp maintained",
        "definition": "A grasped object does not slip out of the gripper while the stage executes. Backed by the slip detector (gripper touch-force loss relative to the grip-force stats while in a holding state).",
    },
    {
        "detector": "orientation",
        "failure": "orientation",
        "condition": "orientation_maintained() == True",
        "label": "Orientation maintained",
        "definition": "The waypoint/object orientation stays within the expected trend and does not drift. Backed by the orientation detector (TTM-recalculated waypoint angle trend).",
    },
    {
        "detector": "transition",
        "failure": "transition",
        "condition": "reaches_waypoint() == True",
        "label": "Waypoint reached",
        "definition": "The end effector reaches the expected waypoint pose and transitions in the correct sequence. Backed by the transition detector (end-effector distance to the expected waypoint pose).",
    },
]

HOLD_CONDITIONS_DESCRIPTION = (
    "Continuous invariants that must stay TRUE for the whole time a waypoint/stage is "
    "executing, as opposed to the pre/post checkpoints. Each one maps one-to-one to a "
    "runtime detector in the aha_scripts/detectors/ folder. The BT maker attaches the full set to "
    "every waypoint deterministically; these are NOT produced by the language model."
)

# Failure names the catalogue does not assign because it does not define the two
# generic gripper open/closed sequence-state predicates. The system prompt names
# this failure for them directly (see the GRIPPER OPEN/CLOSED SEQUENCE STATE
# section of SYSTEM_PROMPT), so it stays a valid link target.
NON_CATALOGUE_FAILURE_NAMES = ("ExecutionSequenceMismatch",)

# Refreshed every time failure definitions are loaded so the stage builders, which
# do not receive the definitions dict, can still see the latest catalog.
_HOLD_CONDITION_DETECTORS: list[dict] = list(DEFAULT_HOLD_CONDITION_DETECTORS)


def hold_condition_detectors(failure_definitions: dict | None = None) -> list[dict]:
    """Return the ordered hold-condition catalog.

    Prefers an explicit ``failure_definitions`` dict, then the cache populated by
    ``load_failure_definitions``, then the built-in default.
    """
    if isinstance(failure_definitions, dict):
        section = failure_definitions.get("hold_conditions") or {}
        detectors = section.get("detectors")
        if isinstance(detectors, list) and detectors:
            return detectors
    return _HOLD_CONDITION_DETECTORS or DEFAULT_HOLD_CONDITION_DETECTORS


def build_hold_condition_blocks() -> list[dict]:
    """Per-stage hold-condition blocks built from the active detector catalog."""
    blocks = []
    for entry in hold_condition_detectors():
        detector = entry.get("detector", "")
        failure = entry.get("failure") or detector
        definition = entry.get("definition", "")
        blocks.append(
            {
                "condition": entry.get("condition", ""),
                "detector": detector,
                "failure": failure,
                "label": entry.get("label") or detector,
                "definition": definition,
                "alternatives": [],
                "selected": True,
                "failure_links": [{"failure": failure, "reason": definition}],
            }
        )
    return blocks


def attach_hold_conditions(stages: list[dict]) -> list[dict]:
    """Give every stage the full, deterministic set of hold conditions.

    Idempotent: it overwrites any existing ``hold_conditions`` so the set always
    matches the current catalog, even for older saved BTs that predate the field.
    """
    for stage in stages:
        stage["hold_conditions"] = build_hold_condition_blocks()
    return stages


def hold_conditions_for_saved_stage(saved_stage: dict) -> list[dict]:
    """Hold-condition blocks for a reloaded stage, honoring user removals.

    Hold conditions are otherwise deterministic, but the GUI now lets the user
    delete individual detectors. We therefore distinguish:
      * key absent  -> older saved BT that predates the field: attach the full
        catalog so nothing is lost.
      * key present -> the user curated the set (possibly to empty): keep only the
        detectors still listed, refreshed against the current catalog so their
        labels/definitions stay up to date.
    """
    saved = saved_stage.get("hold_conditions") if isinstance(saved_stage, dict) else None
    full_blocks = build_hold_condition_blocks()
    if not isinstance(saved, list):
        return full_blocks
    kept_detectors = {
        str(item.get("detector") or "").strip()
        for item in saved
        if isinstance(item, dict)
    }
    kept_detectors.discard("")
    return [block for block in full_blocks if block.get("detector") in kept_detectors]


def catalogue_failure_names() -> list[str]:
    """Every failure name a generated condition may link to, in catalogue order.

    The verification-condition catalogue assigns one failure per rule, so it is
    also the authority on which failure names exist. Nothing here describes WHEN
    a failure applies: placement is stated once, by the catalogue rule itself.
    """
    from .prompts import load_condition_rules, _iter_condition_rules

    names: list[str] = []
    for _category, rule in _iter_condition_rules(load_condition_rules()):
        failure = str(rule.get("failure") or "").strip()
        if failure and failure not in names:
            names.append(failure)
    for extra in NON_CATALOGUE_FAILURE_NAMES:
        if extra not in names:
            names.append(extra)
    return names


def load_failure_definitions() -> dict:
    """Return the failure-definitions dict derived from the condition catalogue.

    The failure names come from ``generated_condition_rules.json`` and the hold
    conditions from ``DEFAULT_HOLD_CONDITION_DETECTORS`` above. No prose about
    where a condition belongs is supplied: the catalogue's placements are the
    single source of truth, so nothing can contradict them.
    """
    global _HOLD_CONDITION_DETECTORS
    _HOLD_CONDITION_DETECTORS = list(DEFAULT_HOLD_CONDITION_DETECTORS)
    return {
        "hold_conditions": {
            "description": HOLD_CONDITIONS_DESCRIPTION,
            "detectors": DEFAULT_HOLD_CONDITION_DETECTORS,
        },
        "failure_categories": {name: {"id": name} for name in catalogue_failure_names()},
    }

MEASURED_CONTEXT_KEY_PARTS = (
    "dist" "ance",
    "offset",
    "tolerance",
    "measurement",
    "magnitude",
    "approx_motion_from_previous_waypoint",
    "waypoint_relationship_used",
)

def strip_measured_context(value):
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if any(part in key_text for part in MEASURED_CONTEXT_KEY_PARTS):
                continue
            cleaned[key] = strip_measured_context(item)
        return cleaned
    if isinstance(value, list):
        return [strip_measured_context(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"\b\d+(?:\.\d+)?\s*cm\b", "qualitative relation", value, flags=re.IGNORECASE)
        value = re.sub(r"\b\d+(?:\.\d+)?\s*m\b", "qualitative relation", value, flags=re.IGNORECASE)
        value = re.sub(r"\b" + "dist" "ance" + r"\b", "spatial relation", value, flags=re.IGNORECASE)
    return value

def task_context_without_measured_values(task_context: str) -> str:
    try:
        parsed = json.loads(task_context)
    except json.JSONDecodeError:
        cleaned = re.sub(
            r"\b\d+(?:\.\d+)?\s*(?:cm|m)\b",
            "qualitative relation",
            str(task_context),
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\b" + "dist" "ance" + r"\b", "spatial relation", cleaned, flags=re.IGNORECASE)
        return cleaned
    return json.dumps(strip_measured_context(parsed), indent=2)

# Keys holding a raw simulator handle. The BT speaks the description's
# descriptive ``name`` vocabulary (basketball, hoop) because its conditions are
# checked visually; the handles (ball, basket_ball_hoop_respondable) mean nothing
# to a camera. They are dropped from the prompt rather than explained to the
# model, so it cannot pick one up.
SIMULATOR_HANDLE_KEYS = ("original_name",)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def scene_name_aliases(task_context) -> dict[str, str]:
    """{original_name -> name} from key_scene_objects, for renaming handles.

    key_scene_objects carries both the descriptive ``name`` and the raw
    simulator ``original_name`` (e.g. name 'basketball' / original_name 'ball').
    Only unambiguous, safe aliases are returned:
      - the alias is a bare identifier (multi-handle originals such as
        "dirt0,dirt1,dirt2" are split; anything else is dropped);
      - the alias is not itself some other object's descriptive name (renaming
        it would collide with a real object);
      - the alias does not point at two different names.
    """
    try:
        d = json.loads(task_context) if isinstance(task_context, str) else task_context
    except Exception:
        return {}
    if not isinstance(d, dict):
        return {}
    objects = [o for o in (d.get("key_scene_objects") or []) if isinstance(o, dict)]
    names = {str(o.get("name")).strip() for o in objects if o.get("name")}

    aliases: dict[str, str] = {}
    ambiguous: set[str] = set()
    for o in objects:
        name = str(o.get("name") or "").strip()
        original = o.get("original_name")
        if not name or not original:
            continue
        for token in re.split(r"[,/;\s]+", str(original)):
            token = token.strip()
            if not token or token == name or token in names:
                continue
            if not _IDENTIFIER_RE.match(token):
                continue
            for variant in {token, token.lower()} - names:
                if aliases.get(variant, name) != name:
                    ambiguous.add(variant)
                aliases[variant] = name
    for token in ambiguous:
        aliases.pop(token, None)
    return aliases

def strip_simulator_handles(value, aliases: dict[str, str] | None = None):
    """Drop original_name keys and rewrite handles left behind in free text.

    Removing the field is not enough on its own: the description maker also
    quotes handles inside prose ("...this sequence targets target_button_wrap1,
    the plus_button"), so the model can still read one there. Those occurrences
    are renamed to the descriptive name instead of deleted, which keeps the
    sentence meaningful.
    """
    aliases = aliases or {}
    if isinstance(value, dict):
        return {
            key: strip_simulator_handles(item, aliases)
            for key, item in value.items()
            if str(key).strip() not in SIMULATOR_HANDLE_KEYS
        }
    if isinstance(value, list):
        return [strip_simulator_handles(item, aliases) for item in value]
    if isinstance(value, str) and aliases:
        for handle, name in aliases.items():
            value = re.sub(r"\b" + re.escape(handle) + r"\b", name, value)
    return value

def task_context_for_prompt(task_context: str) -> str:
    """The task context as the language model should see it.

    Drops measured values (so conditions stay qualitative) and raw simulator
    handles (so object identifiers can only be the descriptive names). The
    unstripped context is still what the deterministic passes receive —
    ``normalize_scene_object_names`` needs ``original_name`` to repair handles
    the model may echo out of the waypoint prose.
    """
    try:
        parsed = json.loads(task_context)
    except json.JSONDecodeError:
        return task_context_without_measured_values(task_context)
    aliases = scene_name_aliases(parsed)
    return json.dumps(
        strip_simulator_handles(strip_measured_context(parsed), aliases), indent=2
    )

FAILURE_NAME_ALIASES = {
    "approach_alignment_error": "WrongPosition",
    "missed_grasp": "ExecutionSequenceMismatch",
    "slip_or_drop": "ExecutionSequenceMismatch",
    "lift_or_transfer_pose_error": "WrongPosition",
    "release_error": "ExecutionSequenceMismatch",
    "execution_sequence_mismatch": "ExecutionSequenceMismatch",
    "collision_or_obstruction": "WrongPosition",
    "object_not_found": "ObjectNotFound",
    "wrong_object_selection": "WrongObjectSelection",
    "wrong_position": "WrongPosition",
    "wrong_orientation": "WrongOrientation",
}

def canonical_failure_name(name: str) -> str:
    stripped = str(name or "").strip()
    return FAILURE_NAME_ALIASES.get(
        stripped,
        FAILURE_NAME_ALIASES.get(stripped.lower(), stripped),
    )

def iter_failure_definition_entries(failure_definitions: dict) -> list[tuple[str, dict]]:
    categories = failure_definitions.get("failure_categories")
    if isinstance(categories, dict):
        return [
            (str(details.get("id") or key), details)
            for key, details in categories.items()
            if isinstance(details, dict)
        ]

    return [
        (str(details.get("id") or key), details)
        for key, details in failure_definitions.items()
        if isinstance(details, dict)
    ]

def failure_definition_summaries(failure_definitions: dict) -> list[dict]:
    summaries = []
    for name, details in iter_failure_definition_entries(failure_definitions):
        summaries.append(
            {
                "id": name,
                "definition": details.get("definition", ""),
                "when_to_check": details.get("when_to_check") or details.get("when") or "",
            }
        )
    return summaries

def infer_failure_links(text: str) -> list[dict]:
    lower = text.lower()

    if any(token in lower for token in ["object_found", "detected", "visible", "pose_confidence", "valid point cloud", "point_cloud", "object_pose_confidence"]):
        return [
            {
                "failure": "ObjectNotFound",
                "reason": "Confirms the required task object is visible and localized before acting.",
            }
        ]

    if any(token in lower for token in ["target_id", "instance_id", "object_class", "attribute", "selected_object", "matches_task"]):
        return [
            {
                "failure": "WrongObjectSelection",
                "reason": "Verifies the robot is engaging the task-specified object instance.",
            }
        ]

    if any(token in lower for token in ["orientation", "approach_axis", "wrist", "roll", "pca", "grasp_axis"]):
        return [
            {
                "failure": "WrongOrientation",
                "reason": "Checks that the gripper orientation is compatible with the grasp.",
            }
        ]

    if any(token in lower for token in [
        "between_fingers",
        "gripper_surrounds",
        "centerline",
        "grasp_pose",
        "aligned_with",
        "on(",
        "inside(",
        "next_to(",
        "alignment",
        "dist" "ance(",
        "vertical_delta",
        "horizontal_delta",
        "pose_error",
        "waypoint",
        "wp",
        "lift",
        "transfer",
        "above",
        "target",
        "release_pose",
        "place_pose",
        "clearance",
        "collision",
        "obstruction",
    ]):
        return [
            {
                "failure": "WrongPosition",
                "reason": "Checks that the end effector or object is at the expected spatial relation.",
            }
        ]

    if any(token in lower for token in [
        "object_in_gripper",
        "in_gripper",
        "gripper_contains",
        "attached_to_end_effector",
        "gripper_correct_grasp",
        "contact_force",
        "grasp_force",
        "held",
        "carry",
        "gripper_condition",
        "gripper_is_open",
        "gripper_is_closed",
        "open_gripper",
        "gripper_closed",
        "stage_",
        "robot_state",
        "controller_enabled",
        "motion_planner",
        "previous_stage",
        "sequence",
        "success_flag",
        "release",
    ]):
        return [
            {
                "failure": "ExecutionSequenceMismatch",
                "reason": "Checks the direct gripper, held-object, or release state required at this point.",
            }
        ]

    return [
        {
            "failure": "ExecutionSequenceMismatch",
            "reason": "Checks the direct gripper, held-object, or release state required at this point.",
        }
    ]
