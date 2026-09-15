"""JSON-driven system prompts and response schemas for BT condition generation.

All verification-condition knowledge is loaded from ``generated_condition_rules.json``.
This module contains only:

* the output structure;
* generic waypoint-primitive semantics;
* generic predicate/object constraints;
* generic catalogue-application logic; and
* generic structural-review logic.

Predicate templates, failure links, applicable primitives, pre/postcondition
placement, occurrence, continuity, required roles, and condition-specific
review rules must be defined in the external JSON catalogue.
"""

from __future__ import annotations

from aha_publish import paths

import json
import os
from pathlib import Path
from typing import Any, Iterable

from .config import *


# ---------------------------------------------------------------------------
# Paths and primitive vocabulary
# ---------------------------------------------------------------------------

try:
    _DEFAULT_SCRIPT_DIR = Path(SCRIPT_DIR)
except NameError:
    _DEFAULT_SCRIPT_DIR = (paths.SOURCE_DIR / 'behavior_trees')

CONDITION_RULES_PATH = Path(
    os.environ.get(
        "CONDITION_RULES_PATH",
        str(_DEFAULT_SCRIPT_DIR / "generated_condition_rules.json"),
    )
)

WAYPOINT_PRIMITIVES = ("move", "grasp", "transport", "place", "release", "push")
_VALID_CONDITION_POSITIONS = {"precondition", "postcondition"}
# observation_moment is fully determined by condition_position, so it is derived
# here rather than stored in the catalogue.
_MOMENT_FOR_POSITION = {
    "precondition": "start_of_waypoint",
    "postcondition": "end_of_waypoint",
}

# Generic gripper open/closed sequence-state predicates. The external catalogue
# intentionally does not define them (they carry authoritative gripper state
# across waypoints rather than detecting a failure-linked condition), but the
# prompt still needs them, so they are appended to the predicate whitelist and
# explained in STEP 2. Their failure link is ExecutionSequenceMismatch.
_EXTRA_SEQUENCE_PREDICATES = (
    "gripper_condition = Open",
    "gripper_condition = Closed",
)


# ---------------------------------------------------------------------------
# External catalogue loading and validation
# ---------------------------------------------------------------------------

def load_condition_rules(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the external verification-condition catalogue.

    The catalogue is the sole source of truth for all predicates and their
    failure/timing assignments. A missing or malformed catalogue is therefore a
    configuration error rather than something that should be silently repaired.
    """

    rules_path = Path(path) if path is not None else CONDITION_RULES_PATH
    if not rules_path.exists():
        raise FileNotFoundError(
            f"Verification-condition catalogue not found: {rules_path}. "
            "Set CONDITION_RULES_PATH or place generated_condition_rules.json "
            "beside this module."
        )

    try:
        with rules_path.open("r", encoding="utf-8") as stream:
            rules = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {rules_path} at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc

    validate_condition_rules(rules)
    return rules


def _iter_condition_rules(
    rules: dict[str, Any],
) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    for category in rules.get("categories", []):
        for rule in category.get("rules", []):
            yield category, rule


def _normalise_placements(rule: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a rule's placements in one canonical list.

    Preferred JSON format:

    ``placements: [{primitive, condition_position, occurrence}]``

    A legacy single ``timing`` object is also accepted and converted to one
    placement so existing catalogues can be migrated incrementally.

    ``observation_moment`` is not stored in the catalogue: it is fully
    determined by ``condition_position`` (precondition -> start_of_waypoint,
    postcondition -> end_of_waypoint) and derived here.
    """

    placements = rule.get("placements")
    if isinstance(placements, list) and placements:
        result: list[dict[str, Any]] = []
        for item in placements:
            placement = dict(item)
            placement["observation_moment"] = _MOMENT_FOR_POSITION.get(
                placement.get("condition_position")
            )
            result.append(placement)
        return result

    timing = rule.get("timing")
    if isinstance(timing, dict) and timing:
        occurrence = timing.get("occurrence", rule.get("occurrence"))

        def _as_list(value) -> list:
            if value is None:
                return []
            return list(value) if isinstance(value, (list, tuple)) else [value]

        def _placement(primitive, position) -> dict[str, Any]:
            return {
                "primitive": primitive,
                "condition_position": position,
                "observation_moment": _MOMENT_FOR_POSITION.get(position),
                "occurrence": occurrence,
            }

        # Per-primitive overrides of the default check_type.
        per_primitive = timing.get("per_primitive") or []
        overrides = {
            entry.get("primitive"): entry
            for entry in per_primitive
            if isinstance(entry, dict) and entry.get("primitive") is not None
        }

        default_check = timing.get("check_type", timing.get("condition_position"))

        primitives = timing.get("applicable_primitives", []) or []
        # A task-global rule may intentionally have no primitive.
        if not primitives:
            return [_placement(None, position) for position in _as_list(default_check)]

        placements: list[dict[str, Any]] = []
        for primitive in primitives:
            override = overrides.get(primitive)
            check = override.get("check_type", default_check) if override else default_check
            for position in _as_list(check):
                placements.append(_placement(primitive, position))
        return placements

    raise ValueError(
        f"Rule {rule.get('source_condition_id', '<unknown>')} must define "
        "a non-empty 'placements' list or a legacy 'timing' object."
    )


def _rule_instructions(rule: dict[str, Any]) -> list[str]:
    instructions: list[str] = []

    raw = rule.get("generation_instructions")
    if isinstance(raw, list):
        instructions.extend(str(item) for item in raw if str(item).strip())

    single = rule.get("generation_rule")
    if single:
        instructions.append(str(single))

    return instructions


def _rule_prohibitions(rule: dict[str, Any]) -> list[str]:
    prohibitions: list[str] = []
    for key in ("prohibitions", "constraints"):
        raw = rule.get(key)
        if isinstance(raw, list):
            prohibitions.extend(str(item) for item in raw if str(item).strip())
    return prohibitions


def validate_condition_rules(rules: dict[str, Any]) -> None:
    """Validate catalogue invariants before either agent receives the prompt."""

    if not isinstance(rules, dict):
        raise ValueError("The condition catalogue root must be a JSON object.")

    categories = rules.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError("The condition catalogue must contain non-empty categories.")

    seen_ids: set[str] = set()
    seen_predicate_failure_pairs: set[tuple[str, str]] = set()
    total_rules = 0

    for category_index, category in enumerate(categories):
        if not isinstance(category, dict):
            raise ValueError(f"Category {category_index} must be a JSON object.")

        category_name = category.get("category") or category.get("name")
        if not isinstance(category_name, str) or not category_name.strip():
            raise ValueError(f"Category {category_index} has no valid name.")

        category_rules = category.get("rules")
        if not isinstance(category_rules, list) or not category_rules:
            raise ValueError(f"Category '{category_name}' has no rules.")

        for rule_index, rule in enumerate(category_rules):
            total_rules += 1
            if not isinstance(rule, dict):
                raise ValueError(
                    f"Rule {rule_index} in category '{category_name}' must be an object."
                )

            source_id = rule.get("source_condition_id")
            if not isinstance(source_id, str) or not source_id.strip():
                raise ValueError(
                    f"Rule {rule_index} in category '{category_name}' has no "
                    "source_condition_id."
                )
            if source_id in seen_ids:
                raise ValueError(
                    f"Duplicate source_condition_id '{source_id}'. The catalogue "
                    "must obey one input condition ID -> one output rule."
                )
            seen_ids.add(source_id)

            predicate = rule.get("predicate_template")
            if not isinstance(predicate, str) or not predicate.strip():
                raise ValueError(f"Rule '{source_id}' has no predicate_template.")

            failure = rule.get("failure")
            if not isinstance(failure, str) or not failure.strip():
                raise ValueError(f"Rule '{source_id}' has no failure name.")

            pair = (predicate.strip(), failure.strip())
            if pair in seen_predicate_failure_pairs:
                raise ValueError(
                    f"Duplicate predicate/failure rule {pair!r}. Combine its valid "
                    "uses into one rule's placements array instead of duplicating it."
                )
            seen_predicate_failure_pairs.add(pair)

            placements = _normalise_placements(rule)
            for placement_index, placement in enumerate(placements):
                position = placement.get("condition_position")
                if position not in _VALID_CONDITION_POSITIONS:
                    raise ValueError(
                        f"Rule '{source_id}' placement {placement_index} has invalid "
                        f"condition_position {position!r}."
                    )
                # observation_moment is derived from condition_position, so a
                # valid position guarantees a valid, consistent moment.

                primitive = placement.get("primitive")
                if primitive is not None and primitive not in WAYPOINT_PRIMITIVES:
                    raise ValueError(
                        f"Rule '{source_id}' placement {placement_index} uses "
                        f"unknown primitive {primitive!r}."
                    )

    if total_rules == 0:
        raise ValueError("The condition catalogue contains no rules.")


# ---------------------------------------------------------------------------
# Catalogue renderers shared by Agent 1 and Agent 2
# ---------------------------------------------------------------------------

def render_predicate_whitelist(rules: dict[str, Any]) -> str:
    """Render only predicate templates explicitly present in the JSON."""

    templates: list[str] = []
    for _category, rule in _iter_condition_rules(rules):
        template = rule["predicate_template"].strip()
        if template not in templates:
            templates.append(template)
    # Generic gripper sequence-state predicates the catalogue does not define.
    for extra in _EXTRA_SEQUENCE_PREDICATES:
        if extra not in templates:
            templates.append(extra)
    return "\n".join(f"* {template}" for template in templates)


def render_condition_catalogue(rules: dict[str, Any]) -> str:
    """Render all condition-specific knowledge from the external JSON."""

    lines: list[str] = []

    taxonomy_name = rules.get("taxonomy_name")
    if taxonomy_name:
        lines.append(f"Catalogue: {taxonomy_name}")

    design_summary = rules.get("design_summary")
    if design_summary:
        lines.append(f"Scope: {design_summary}")

    if lines:
        lines.append("")

    for category, rule in _iter_condition_rules(rules):
        category_name = category.get("category") or category.get("name") or ""
        category_explanation = category.get("category_explanation") or category.get(
            "explanation"
        )

        lines.append(f"### {rule['source_condition_id']} — {category_name}")
        if category_explanation:
            lines.append(f"Category meaning: {category_explanation}")
        if rule.get("rule_title"):
            lines.append(f"Rule title: {rule['rule_title']}")
        lines.append(f"Predicate template: {rule['predicate_template']}")
        lines.append(f"Failure link: {rule['failure']}")

        role = rule.get("condition_role") or rule.get("role")
        if role:
            lines.append(f"Condition role: {role}")

        required = rule.get("required")
        if required is not None:
            lines.append(f"Required when placement matches: {bool(required)}")

        meaning = rule.get("predicate_meaning") or rule.get("why_checked")
        if meaning:
            lines.append(f"Predicate meaning: {meaning}")

        observation = rule.get("expected_success_observation")
        if observation:
            lines.append(f"Expected successful observation: {observation}")

        reason = rule.get("reason")
        if reason:
            lines.append(f"Failure-link reason: {reason}")

        lines.append("Valid placements:")
        for placement in _normalise_placements(rule):
            primitive = placement.get("primitive") or "task_global"
            occurrence = placement.get("occurrence") or rule.get("occurrence")
            line = (
                f"  - primitive={primitive}; "
                f"position={placement['condition_position']}; "
                f"observation={placement['observation_moment']}"
            )
            if occurrence:
                line += f"; occurrence={occurrence}"
            lines.append(line)

        instructions = _rule_instructions(rule)
        if instructions:
            lines.append("Generation instructions:")
            lines.extend(f"  - {instruction}" for instruction in instructions)

        prohibitions = _rule_prohibitions(rule)
        if prohibitions:
            lines.append("Prohibitions:")
            lines.extend(f"  - {prohibition}" for prohibition in prohibitions)

        continuity = rule.get("continuity")
        if continuity:
            if isinstance(continuity, list):
                lines.append("Continuity requirements:")
                lines.extend(f"  - {item}" for item in continuity)
            else:
                lines.append(f"Continuity requirement: {continuity}")

        lines.append("")

    return "\n".join(lines).rstrip()


def predicate_for_placement(
    primitive: str | None,
    position: str,
    rules: dict[str, Any] | None = None,
) -> tuple[str | None, str | None]:
    """Return ``(predicate_template, failure)`` for the catalogue rule that has a
    placement matching ``primitive`` and ``position`` (condition_position), else
    ``(None, None)``.

    This lets deterministic post-processing reference catalogue predicates by their
    role (e.g. "the release-postcondition predicate") instead of hard-coding a
    predicate name, so renaming a predicate in the JSON does not silently break it.
    """
    if rules is None:
        rules = load_condition_rules()
    for _category, rule in _iter_condition_rules(rules):
        for placement in _normalise_placements(rule):
            if (
                placement.get("primitive") == primitive
                and placement.get("condition_position") == position
            ):
                return rule.get("predicate_template"), rule.get("failure")
    return None, None


def gripper_state_predicates() -> tuple[str, ...]:
    """The generic gripper open/closed sequence-state predicates. These live here
    (not in the external catalogue) but are still a single source of truth other
    modules read instead of hard-coding the literal strings."""
    return _EXTRA_SEQUENCE_PREDICATES


# ---------------------------------------------------------------------------
# Agent 1 prompt template — no predicate or failure is hard-coded here
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = r"""You are a robotics engineer specialising in task verification.

You receive:

1. an ordered robot-waypoint sequence;
2. authoritative per-waypoint gripper and held-object state;
3. valid physical-object identifiers;
4. available failure modes; and
5. an externally generated verification-condition catalogue.

For every waypoint, classify the waypoint primitive and instantiate only the verification conditions authorised by the catalogue.

Respond ONLY with a JSON object in this exact structure:

{
  "stages": [
    {
      "stage": 0,
      "name": "short description of what happens at this waypoint",
      "primitive": "move",
      "preconditions": [
        {
          "condition": "one instantiated catalogue predicate",
          "alternatives": [],
          "failure_links": [
            {
              "failure": "exact catalogue failure name",
              "reason": "short tooltip reason grounded in the catalogue"
            }
          ]
        }
      ],
      "postconditions": []
    }
  ]
}

=== STRUCTURAL RULES ===

* Every condition block contains exactly one predicate.
* Never join predicates with "and" or "or".
* Put separate checks in separate condition blocks.
* alternatives must always be [].
* Every condition has exactly one failure_links entry.
* Use the exact failure name assigned by the matching catalogue rule.
* Every waypoint has exactly one primitive selected from: move, grasp, transport, place, release, or push.
* Do not duplicate an instantiated predicate within the same condition list.
* Do not create any predicate, failure mapping, placement rule, or occurrence
  rule that is absent from the external catalogue, except the two generic
  gripper_condition open/closed sequence-state predicates defined in STEP 2.

=== ALLOWED PREDICATE TEMPLATES ===

Use only these templates, all loaded from the external JSON catalogue:

[[PREDICATE_WHITELIST]]

Instantiate placeholders using exact visible physical-object identifiers from
the task context. Do not emit placeholder tokens in the final stages.

Conditions may relate only entities and states permitted by their catalogue
rules. Do not use camera flags, numerical thresholds, distances, angles,
dimensions, waypoint identifiers, simulator-only identifiers, or hidden planner
states unless the catalogue explicitly contains such a predicate.

Never invent, shorten, or generalise an object identifier. When multiple objects
share a type, use the complete attribute-qualified identifier supplied in
key_scene_objects. Object identifiers must remain lowercase underscore-separated
tokens.

Every object identifier must be the "name" field of a key_scene_objects entry.
Those names are the only object vocabulary you have; the conditions are checked
visually, so an identifier that names nothing a camera can see is invalid. When
the waypoint prose calls an object something else, map it back to the
key_scene_objects "name" before emitting it.

=== STEP 1: CLASSIFY EACH WAYPOINT ===

Use the waypoint description together with authoritative gripper and held-object
state. Structured state overrides ambiguous free text.

move
* Empty-arm free-space motion.
* Includes approaching, reaching, retracting, hovering, and reorienting without
  establishing a new task-relevant contact or grasp.
* A waypoint aimed at a later press or push is still move until contact occurs.

grasp
* Closes the gripper on an object and establishes that the object is held.

transport
* Moves, lifts, lowers, or reorients an object while maintaining an existing
  grasp.

place
* Positions a held object at its intended target while the grasp is maintained.
* It occurs before a distinct release action.

release
* Opens the gripper and relinquishes the held object.
* It may establish a final object-to-object relation when the catalogue permits
  such a condition.

push
* Manipulates an object through non-grasping contact.
* Includes pressing, pushing, sliding, and contact-based opening or closing.
* It applies only when the waypoint establishes or maintains contact; an
  approach-only waypoint is move.
* If robot_action explicitly says the robot pushes or is pushing an object,
  classify that waypoint as push. Do not
  classify it as grasp, transport, or place, or require the pushed object to
  be held. Moving into position to push later is still move.
* Pressing is ALWAYS push. When the waypoint description says the robot presses,
  pushes down on, depresses, taps, or pokes an object — a button, switch, key,
  lever, pedal, or anything else — the primitive is push, never grasp, transport
  or place. A press moves the object without ever holding it.
* A closed gripper does NOT make a press into a grasp. Fingertips closed into a
  tool for pressing are still non-grasping contact. Wording such as "closes its
  fingertips in preparation", "with closed fingertips", or "presses with the
  closed gripper" describes a push, not a grasp.
* The object being pressed moves because the gripper displaces it. That is not
  transport: transport requires the object to be held. Lowering a closed gripper
  onto a button and driving it down is one push waypoint.

=== HELD-OBJECT GATE (applies before every classification) ===

held_object is authoritative and decides which primitives are even available.

* When held_object is null at a waypoint, the robot holds nothing there. That
  waypoint can ONLY be move or push. grasp, transport, place and release are
  forbidden, whatever the gripper state or the free text suggests.
* When held_object is null at EVERY waypoint of the task, the task involves no
  grasping at all. Every waypoint is move or push, and no waypoint may carry a
  held-object condition such as object_in_gripper(...).
* grasp is permitted only where held_object goes from null to a named object.
  transport and place are permitted only where held_object names an object at
  both the start and the end of the waypoint. release is permitted only where
  held_object goes from a named object back to null.
* Never infer holding from gripper_state alone. gripper_state = "closed" with
  held_object = null means the gripper is closed but empty.

Classify from what physically happens in that waypoint, not from the overall
task intent.

=== STEP 2: APPLY THE EXTERNAL CONDITION CATALOGUE ===

[[CONDITION_CATALOGUE]]

For each catalogue rule:

* instantiate it only when one of its valid placements matches the classified
  waypoint, condition position, observation moment, and occurrence rule;
* bind every placeholder to an exact task-context entity or state;
* attach the rule's exact failure name;
* use the catalogue's reason when supplied, otherwise write a concise tooltip
  grounded in its predicate meaning and expected successful observation;
* obey every generation instruction, prohibition, role requirement, and
  continuity requirement;
* treat one catalogue rule with multiple placements as one condition type with
  several authorised uses, not as several new condition definitions; and
* omit the condition when its placement or task-specific applicability is not
  satisfied.

=== GRIPPER OPEN/CLOSED SEQUENCE STATE ===

The catalogue does not define the gripper open/closed sequence-state predicates.
In ADDITION to the catalogue predicates, you may use exactly these two generic
Grasp predicates to record the authoritative gripper state, and no
others outside the catalogue:

* gripper_condition = Open
* gripper_condition = Closed

General rules:

* Attach the failure name ExecutionSequenceMismatch to them.
* Read the required open/closed value from the authoritative gripper block in the
  task context; never assume a fixed value.
* Do NOT add a gripper_condition predicate when a there is one of the condtions defined in the Category: Grasp. Those take precedence.
* Use exactly one Grasp predicate per condition list: either one
  catalogue Grasp predicate or one gripper_condition predicate,
  never both for the same state.

Placement by primitive:

* move: use one gripper_condition predicate as the Grasp precondition
  (Open or Closed per the authoritative state), and preserve that same open or
  closed state in the postcondition. A move never holds an object.
* grasp: use gripper_condition = Open as the precondition — the gripper is open
  before it closes. The held-object postcondition comes from the catalogue.
* release: the catalogue's release predicate (the Grasp predicate whose placement
  is primitive=release, position=postcondition) belongs ONLY here, and only as a
  postcondition. Never put it on a move, grasp, transport, place or push waypoint,
  and never in a precondition — a waypoint that merely ends with an open gripper,
  such as a retreat after the release already happened, does NOT get it. The
  release is stated once, on the waypoint that actually relinquishes the object.
* push: for the waypoint that first establishes contact, use exactly one of
  gripper_condition = Open or gripper_condition = Closed as the precondition,
  selected from the authoritative gripper block. Do not assume a push requires a
  closed gripper. When a push waypoint ends after contact has terminated,
  preserve the authoritative gripper state with the matching gripper_condition
  predicate.

Continuity:

* Where a gripper_condition predicate is used, take its Open/Closed value straight
  from the authoritative per-waypoint gripper state in the task description:
    - the POSTcondition of waypoint N uses waypoint N's own gripper state;
    - the PRECONDITION of waypoint N uses waypoint N-1's gripper state, because a
      waypoint begins in whatever state the previous waypoint left behind.
  Never use waypoint N's own state for its precondition. Read both values from the
  description; do not infer them from the waypoint's wording or from what the
  action seems to require. Example: if the description reports waypoint 5 open and
  waypoint 6 closed, then waypoint 6's precondition is gripper_condition = Open and
  its postcondition is gripper_condition = Closed.
* Waypoint 0 always begins with the gripper open: its precondition gripper state
  is always gripper_condition = Open (the arm starts empty with an open gripper).
  Never emit gripper_condition = Closed as the precondition of waypoint 0. This
  rule is absolute and OVERRIDES the authoritative per-waypoint gripper state: even
  if the authoritative gripper block reports waypoint 0 as closed, the waypoint 0
  precondition must still be gripper_condition = Open.
* If waypoint i ends with gripper_condition = Open, waypoint i+1 begins with
  gripper_condition = Open unless it changes the state through an explicit action.
* If waypoint i ends with gripper_condition = Closed, waypoint i+1 begins with
  gripper_condition = Closed unless a more specific catalogue Grasp
  predicate is authoritative.
* After a release (gripper_released), a subsequent empty-arm move begins with
  gripper_condition = Open.

=== STEP 3: GENERIC TIMING AND CONTINUITY ===

* A precondition is observable at the start of its waypoint.
* A postcondition is observable at the end of its waypoint.
* A state required to begin an action belongs in preconditions.
* A state established by an action belongs in postconditions.
* Preserve authoritative state continuity between consecutive waypoints.
* Apply any predicate-specific continuity requirement only when it is defined by
  that catalogue rule.
* Do not represent sequence continuity using a new predicate unless such a
  predicate is explicitly present in the catalogue or is one of the two
  gripper_condition open/closed predicates defined in STEP 2.

Do not emit:

* predicates absent from the catalogue, other than the two gripper_condition
  open/closed sequence-state predicates defined in STEP 2;
* failure names absent from the matching catalogue rule;
* compound predicates;
* generic or invented object identifiers;
* stage-completed or previous-stage-satisfied guards unless explicitly defined
  by the catalogue; or
* duplicate instances of the same predicate in the same condition list.
"""


# ---------------------------------------------------------------------------
# Agent 2 prompt template — all condition-specific checks come from the JSON
# ---------------------------------------------------------------------------

REVIEWER_SYSTEM_PROMPT_TEMPLATE = r"""You are Agent 2, a structural reviewer for robot-manipulation behaviour trees.

You receive:

1. Agent 1's generated stages;
2. authoritative per-waypoint gripper and held-object state;
3. valid physical-object identifiers; and
4. the same external verification-condition catalogue used by Agent 1, which
   defines every available failure mode.

Audit the stages, make only necessary corrections, and return ONLY:

{
  "violations": [
    {
      "stage": 0,
      "rule": "R1",
      "detail": "What was wrong.",
      "fix": "What was changed."
    }
  ],
  "stages": []
}

Preserve stage count, stage order, valid names, valid object identifiers, and all
well-formed conditions. You may correct a primitive when authoritative state and
physical waypoint behaviour show that the original classification is wrong.

=== EXTERNAL PREDICATE WHITELIST ===

[[PREDICATE_WHITELIST]]

=== EXTERNAL CONDITION CATALOGUE ===

[[CONDITION_CATALOGUE]]

=== GENERIC REVIEW RULES ===

R1 — Output structure
* Every condition block contains exactly one predicate.
* alternatives is always [].
* Every condition contains exactly one failure_links entry.
* Remove exact duplicate conditions from the same list.

R2 — Catalogue-only predicates
* Every condition must instantiate exactly one predicate_template from the
  external catalogue, with one exception: the two generic gripper_condition
  open/closed sequence-state predicates (gripper_condition = Open,
  gripper_condition = Closed) are permitted even though the catalogue does not
  define them.
* Remove or replace predicates absent from the catalogue, unless they are one of
  those two gripper_condition predicates.
* Do not introduce a condition merely because it was used in another task.

R3 — Primitive classification
* Classify each waypoint from its physical behaviour and authoritative state:
  move is empty-arm free-space motion; grasp establishes holding; transport
  preserves holding during motion; place positions a held object before
  release; release relinquishes it; push uses non-grasping contact.
* Correct a primitive only when it conflicts with these semantics.
* Three checks are mandatory and are not matters of judgement:
  - If robot_action explicitly says the robot pushes or is pushing an object,
    the primitive MUST be push, even when gripper_state or held_object fields
    conflict. Reclassify it and remove conditions requiring the pushed object
    to be held. An approach to push later does not trigger this rule.
  - Pressing is push. If the waypoint description says the robot presses,
    pushes down on, depresses, taps or pokes an object, the primitive MUST be
    push. Reclassify it if it is anything else. A closed gripper never turns a
    press into a grasp, and an object displaced by a press is never transported.
  - held_object gates the primitive. Where held_object is null, only move and
    push are legal; reclassify any grasp, transport, place or release, and
    remove every held-object condition (object_in_gripper and the like) that the
    old primitive authorised. grasp needs held_object to go null -> object;
    transport and place need it named at both ends; release needs object -> null.
    gripper_state alone never establishes holding.
* Reclassification is not complete until the waypoint's ENTIRE condition set
  matches the new primitive. When you change a primitive, re-derive the
  waypoint's conditions from scratch exactly as the generator would for the new
  primitive — do not merely delete the offending conditions:
  - Remove every condition whose only authorised placement was under the old
    primitive.
  - Add every catalogue condition whose placement matches the new primitive and
    whose task-applicability holds at this waypoint — each authorised
    precondition AND each authorised postcondition, including every required /
    every_matching_waypoint / final_matching_waypoint check (see R5), each bound
    to the correct task-context object.
  - Apply the gripper open/closed sequence-state rule for the new primitive and
    re-establish continuity with the neighbouring waypoints (see R9).
  The corrected waypoint must be indistinguishable from what Agent 1 would have
  produced had it classified the waypoint as the new primitive from the start.

R4 — Placement and timing
* Match each condition to one valid placement of its catalogue rule.
* Preconditions are observed at start_of_waypoint.
* Postconditions are observed at end_of_waypoint.
* Remove or move a condition whose primitive, position, or observation moment
  does not match any authorised placement.

R5 — Occurrence
* Enforce the occurrence rule defined by the matching catalogue placement.
* Remove repeated task-global or once-per-interaction checks.
* Add a missing check whenever the catalogue marks it as required, OR its
  matching placement has an occurrence of every_matching_waypoint or
  final_matching_waypoint and that placement applies to this waypoint. An
  every_matching_waypoint placement is authoritative: the condition must appear
  on every waypoint whose primitive and position match it.
* In particular, every release waypoint MUST carry the catalogue's release
  condition (the Grasp predicate whose placement is
  primitive=release, position=postcondition,
  bound to the object that was held going into the waypoint, with that rule's
  exact failure name. This applies
  equally when you reclassified the waypoint into a release under R3.
* A later waypoint that simply has an open gripper is not a second release.
* Check every gripper_condition value against the authoritative per-waypoint
  gripper states: a waypoint's postcondition uses its OWN state, its precondition
  uses the PREVIOUS waypoint's state. Fix any waypoint whose precondition was
  taken from its own state instead of the previous waypoint's.

R6 — Predicate arguments
* Replace placeholders with exact valid task-context identifiers.
* Never invent, shorten, merge, or generalise physical-object names.
* Reject simulator-only or generic identifiers unless explicitly permitted by
  the catalogue template.
* Every object identifier must be one of the task scene object names listed in
  your message. Replace any identifier that is not on that list — a raw
  simulator handle echoed out of the waypoint prose — with the scene object it
  refers to.

R7 — Failure links
* Every predicate must use the exact failure name assigned by its catalogue rule.
* The two gripper_condition open/closed predicates have no catalogue rule; they
  must use the failure name ExecutionSequenceMismatch.
* Do not use a preferred mapping from memory or infer a different failure name.
* Do not invent an unavailable failure mode.

R8 — Catalogue instructions and prohibitions
* Enforce every generation instruction, prohibition, condition role, and
  task-applicability restriction defined by the matching rule.
* When a rule has multiple placements, treat them as authorised uses of one
  condition type, not as duplicate condition definitions.

R9 — State continuity
* Compare consecutive stages with authoritative state and any continuity
  requirement defined in the catalogue.
* Waypoint 0 always begins with the gripper open: if its precondition carries
  gripper_condition = Closed, correct it to gripper_condition = Open (the arm
  starts empty with an open gripper). This rule is absolute and OVERRIDES the
  authoritative per-waypoint gripper state — do NOT change a correct
  gripper_condition = Open on waypoint 0 to Closed just because the authoritative
  gripper block reports waypoint 0 as closed. The authoritative state is not
  trusted for the waypoint 0 precondition; waypoint 0 is always Open.
* Fix only the incorrect endpoint.
* Do not create a new continuity predicate unless the catalogue explicitly
  defines one.

R10 — Role cardinality
* Enforce role/cardinality requirements only when they are provided in the
  data-driven structural requirements.
* Do not hard-code a predicate as an interaction, geometry, grounding, or
  effect condition based on its name alone.

R11 — Minimal correction
* Preserve every catalogue-compliant condition.
* For every actual correction, add one violation entry explaining the catalogue
  or generic rule that was violated and the exact fix.
* If nothing changes, return an empty violations list and preserve the stages.

R12
* Primitive scope is not a judgement call. Before keeping any condition, look up
  its catalogue rule and confirm this waypoint's primitive appears among that
  rule's authorised placements. If it does not, DELETE the condition — even when
  it is physically true at that waypoint, even when it states the task's goal
  effect, and even when no other waypoint in the task can legally carry it.
  A predicate whose only authorised primitive never occurs in this task is
  simply not checked in this task. Never relocate such a condition to the
  nearest semantically plausible waypoint: a push-only effect predicate does not
  belong on a release, place, grasp, or final move waypoint.
* Do not legalise a condition by reclassifying its waypoint. R3 reclassification
  is driven only by the waypoint's observed physical behaviour and authoritative
  gripper state, never by a wish to keep a condition that would otherwise be
  removed.

R13 — Grasp predicate precedence
* For each waypoint independently, inspect `preconditions` and `postconditions` separately.
* If `preconditions` contains a catalogue `"Grasp"` predicate, remove `gripper_condition = Open/Closed` only from that waypoint's `preconditions`.
* If `postconditions` contains a catalogue `"Grasp"` predicate, remove `gripper_condition = Open/Closed` only from that waypoint's `postconditions`.

R14 — Waypoint 0 precondition cleanup
* In waypoint 0 preconditions only, remove:
    every condition from category "Spatial Relationship";
    every condition from category "Object State" whose catalogue occurrence is not once_per_task.

"""


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def _render_template(template: str, rules: dict[str, Any]) -> str:
    return (
        template.replace(
            "[[PREDICATE_WHITELIST]]", render_predicate_whitelist(rules)
        )
        .replace("[[CONDITION_CATALOGUE]]", render_condition_catalogue(rules))
    )


def get_system_prompt(
    task_name: str = "",
    condition_rules_path: str | Path | None = None,
) -> str:
    """Return Agent 1's prompt rendered entirely from the external catalogue."""

    del task_name  # Kept for backward-compatible call sites.
    rules = load_condition_rules(condition_rules_path)
    return _render_template(SYSTEM_PROMPT_TEMPLATE, rules)


def get_reviewer_system_prompt(
    condition_rules_path: str | Path | None = None,
) -> str:
    """Return Agent 2's prompt rendered from the same external catalogue."""

    rules = load_condition_rules(condition_rules_path)
    return _render_template(REVIEWER_SYSTEM_PROMPT_TEMPLATE, rules)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

CONDITION_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["stages"],
    "properties": {
        "stages": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "stage",
                    "name",
                    "primitive",
                    "preconditions",
                    "postconditions",
                ],
                "properties": {
                    "stage": {"type": "integer"},
                    "name": {"type": "string"},
                    "primitive": {
                        "type": "string",
                        "enum": list(WAYPOINT_PRIMITIVES),
                    },
                    "preconditions": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/condition"},
                    },
                    "postconditions": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/condition"},
                    },
                },
            },
        }
    },
    "$defs": {
        "condition": {
            "type": "object",
            "additionalProperties": False,
            "required": ["condition", "alternatives", "failure_links"],
            "properties": {
                "condition": {"type": "string"},
                "alternatives": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 0,
                },
                "failure_links": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["failure", "reason"],
                        "properties": {
                            "failure": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                    },
                },
            },
        }
    },
}

REVIEWER_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["violations", "stages"],
    "properties": {
        "violations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["stage", "rule", "detail", "fix"],
                "properties": {
                    "stage": {"type": "integer"},
                    "rule": {"type": "string"},
                    "detail": {"type": "string"},
                    "fix": {"type": "string"},
                },
            },
        },
        "stages": CONDITION_RESPONSE_SCHEMA["properties"]["stages"],
    },
    "$defs": CONDITION_RESPONSE_SCHEMA["$defs"],
}


# ---------------------------------------------------------------------------
# Rendered constants
# ---------------------------------------------------------------------------
# The reviewer prompt is task-independent, so it is rendered once at import
# time; the generator prompt is per-task and callers use get_system_prompt()
# directly. Rendering here means a missing JSON file raises a clear
# configuration error immediately.

REVIEWER_SYSTEM_PROMPT = get_reviewer_system_prompt()
