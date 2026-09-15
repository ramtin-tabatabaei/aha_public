#!/usr/bin/env python3
"""Generate exactly one waypoint-verification rule per input condition.

The input JSON already carries the authored half of every rule: category,
source_condition_id, failure, predicate_template, an optional predicate_meaning,
and occurrence. Those fields are copied verbatim into the output. stage_rule is
sent to the model as timing evidence but is not itself an output field. The
model is asked for the
derived half only -- per-primitive timing, generation_rule, and prohibitions --
and the program merges the two halves and derives every field that follows
mechanically from the others (the applicable-primitive list and the
scalar/list collapsing of uniform timing).

Every path defaults to a file sitting next to this script. From the repository
root, supply the authored taxonomy input (not bundled in this release):

    python src/aha_publish/behavior_trees/condition_rule_generator.py \
        --input /path/to/condition_taxonomy_input.example.json

Override any of them when needed:

    python src/aha_publish/behavior_trees/condition_rule_generator.py \
        --input condition_taxonomy_input.example.json \
        --output-json generated_condition_rules.json \
        --output-csv generated_condition_rules.csv \
        --output-text generated_condition_rules.txt \
        --model gpt-5.4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


# Defaults resolve next to this file, so the script runs with no arguments from
# any working directory (including an IDE "Run" button).
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "condition_taxonomy_input.example.json"
DEFAULT_OUTPUT_JSON = SCRIPT_DIR / "generated_condition_rules.json"
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "generated_condition_rules.csv"
DEFAULT_OUTPUT_TEXT = SCRIPT_DIR / "generated_condition_rules.txt"


PRIMITIVE_DEFINITIONS = r"""
=== WAYPOINT PRIMITIVES ===

move
* Moves without holding an object.
* Includes approaching, reaching, retracting, or reorienting the empty arm.
* It does not establish a grasp or task-relevant contact.
* A waypoint that only approaches, aligns, hovers over, or positions above a
  contact target without contacting it is move. The push begins only when the
  waypoint actually establishes or maintains contact.

grasp
* Closes the gripper on an object to pick it up.
* The object becomes held as the result of this waypoint.

transport
* Moves, lifts, lowers, or reorients an object while maintaining a grasp.
* The same object is held at the start and end.

place
* Moves a held object onto or into its intended physical target.
* The object remains held. Release occurs in a separate waypoint.

release
* Opens the gripper and relinquishes the held object.
* When applicable, this waypoint establishes the final placement relation.

push
* Manipulates an object through non-grasping contact: the end effector touches
  the object and acts on it without ever closing the gripper around it.
* Contacting, pushing, and pressing are all push, as are sliding, opening, and
  closing. There is no separate contact, press, or slide primitive: every
  waypoint whose purpose is non-grasping end-effector contact with an object is
  a push waypoint.
* The contacted object is never held by the gripper.
* A waypoint without actual contact is move, not push.
* push is the ONLY primitive that establishes or maintains non-grasping contact.
  move, grasp, transport, place, and release never carry it. An object enclosed
  in the gripper is a GRASP state, not a contact state, so a contact predicate
  must never be attached to grasp, transport, place, or release.

Classify each waypoint by what physically occurs in that waypoint, not by the
overall task intent.
""".strip()


ACTION_SEQUENCE = r"""
=== ACTION SEQUENCE ===

Waypoints are not an unordered set. Every task is built from runs of two
canonical chains, executed in order:

  A. grasp-based transfer:      move -> grasp -> transport -> place -> release
  B. contact-based manipulation: move -> push   (push may repeat consecutively)

A task may contain several runs of either chain, in any order, and a run may
omit intermediate steps, but the relative order within a run never changes: no
transport before its grasp, no release before its place, no push before the
move that approaches the contact target.

The chain determines which states exist at which point:
* No object is held before its grasp. The object is held from the end of grasp
  until the start of release. Nothing is held after release.
* Non-grasping contact exists only from the moment a push establishes it, and
  persists only while consecutive pushes maintain it. No other primitive
  establishes or maintains it, so a predicate about end-effector contact
  attaches to push and to nothing else.
* A move changes neither grasp state nor contact state.
* A placement relation exists only once the object has been relinquished.

Before assigning any timing, locate the predicate on this chain: identify the
waypoint that first makes it true and the waypoint at which its validity ends.
Only then decide which primitives it touches and how.
""".strip()


# ---------------------------------------------------------------------------
# Shared vocabularies
# ---------------------------------------------------------------------------

class Primitive(str, Enum):
    move = "move"
    grasp = "grasp"
    transport = "transport"
    place = "place"
    release = "release"
    push = "push"


class Occurrence(str, Enum):
    once_per_task = "once_per_task"
    once_per_new_object_interaction = "once_per_new_object_interaction"
    every_matching_waypoint = "every_matching_waypoint"
    final_matching_waypoint = "final_matching_waypoint"


class CheckType(str, Enum):
    precondition = "precondition"
    postcondition = "postcondition"


# Execution order, used to sort primitives and to render deterministic output.
PRIMITIVE_ORDER = {
    primitive: index
    for index, primitive in enumerate(
        (
            Primitive.move,
            Primitive.grasp,
            Primitive.transport,
            Primitive.place,
            Primitive.release,
            Primitive.push,
        )
    )
}

CHECK_ORDER = {CheckType.precondition: 0, CheckType.postcondition: 1}


# ---------------------------------------------------------------------------
# Input schema: the authored half of every rule
# ---------------------------------------------------------------------------

class ConditionSpecification(BaseModel):
    """One atomic condition, authored by hand and copied verbatim to output."""

    model_config = ConfigDict(extra="forbid")

    source_condition_id: str = Field(min_length=1)
    failure: str = Field(min_length=1)
    predicate_template: str = Field(min_length=1)
    # Optional authoritative definition of what the predicate asserts and which
    # objects it applies to. Supplied per condition; copied verbatim to output.
    predicate_meaning: str | None = Field(default=None, min_length=1)
    occurrence: Occurrence
    stage_rule: str = Field(min_length=1)


class ConditionCategory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str = Field(min_length=1)
    category_explanation: str = Field(min_length=1)
    rules: list[ConditionSpecification] = Field(min_length=1)


class TaxonomyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    taxonomy_name: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    categories: list[ConditionCategory] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_taxonomy(self) -> "TaxonomyInput":
        names = [category.category for category in self.categories]
        duplicate_names = sorted(
            name for name, count in Counter(names).items() if count > 1
        )
        if duplicate_names:
            raise ValueError(f"category names must be unique: {duplicate_names}")

        ids = [condition.source_condition_id for condition in self.conditions]
        duplicates = sorted(
            item for item, count in Counter(ids).items() if count > 1
        )
        if duplicates:
            raise ValueError(f"condition IDs must be unique: {duplicates}")

        return self

    @property
    def conditions(self) -> list[ConditionSpecification]:
        return [
            condition
            for category in self.categories
            for condition in category.rules
        ]


# ---------------------------------------------------------------------------
# Model schema: the derived half only
# ---------------------------------------------------------------------------

class PrimitiveCheck(BaseModel):
    """How one primitive carries the predicate."""

    model_config = ConfigDict(extra="forbid")

    primitive: Primitive
    check_type: list[CheckType] = Field(min_length=1, max_length=2)


class RuleDerivation(BaseModel):
    """Everything the model must add to one authored condition."""

    model_config = ConfigDict(extra="forbid")

    source_condition_id: str = Field(min_length=1)
    primitive_timing: list[PrimitiveCheck] = Field(min_length=1)
    generation_rule: str = Field(min_length=1)
    prohibitions: list[str] = Field(default_factory=list)


class DerivationSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    derivations: list[RuleDerivation] = Field(min_length=1)


# ---------------------------------------------------------------------------
# Output schema: authored half + derived half, merged by the program
# ---------------------------------------------------------------------------

class PrimitiveTiming(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primitive: Primitive
    check_type: list[CheckType]


class WaypointTiming(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Scalar when the rule uses one uniform check type across all applicable
    # primitives; a list holding the union of values otherwise.
    check_type: CheckType | list[CheckType]
    applicable_primitives: list[Primitive]
    # One entry per applicable primitive when timing differs across primitives;
    # null when a single uniform timing applies.
    per_primitive: list[PrimitiveTiming] | None = None
    occurrence: Occurrence


class VerificationRule(BaseModel):
    """Exactly one generated rule for exactly one input condition ID."""

    model_config = ConfigDict(extra="forbid")

    source_condition_id: str
    category: str
    failure: str
    predicate_template: str = Field(min_length=1)
    predicate_meaning: str | None = None
    timing: WaypointTiming
    generation_rule: str = Field(min_length=1)
    prohibitions: list[str] = Field(default_factory=list)


class CategoryRuleSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    category_explanation: str
    rules: list[VerificationRule] = Field(min_length=1)


class GeneratedTaxonomy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    taxonomy_name: str
    categories: list[CategoryRuleSet] = Field(min_length=1)


SYSTEM_PROMPT = f"""
You are a robotics knowledge engineer completing symbolic verification rules for
waypoint-level manipulation monitoring.

=== WHAT IS GIVEN AND WHAT YOU PRODUCE ===

Each input condition is ATOMIC and already carries its authored half:
category, source_condition_id, failure, predicate_template, and occurrence.
Those fields are final. The program copies them into the output verbatim, so
never restate, rename, reword, correct, or re-derive them.

stage_rule is given to you as evidence for deriving timing. It is NOT copied to
the output, so the derived timing must stand on its own: everything the
stage_rule implies about which waypoints are checked has to be expressed through
primitive_timing, and any boundary exception through prohibitions.

A condition may also carry predicate_meaning: an authoritative definition of
what the predicate asserts and which objects it may be instantiated on. It is
also copied verbatim, so do not restate or summarise it. Treat it as ground
truth: it constrains which primitives the predicate can attach to and which
objects generation_rule may name. Never contradict it, and never broaden the
predicate beyond the object class and the states it defines.

You are given exactly ONE condition and return EXACTLY ONE derivation for it,
containing only the missing half: primitive_timing, generation_rule, and
prohibitions.

CRITICAL CARDINALITY RULE
* Never split the condition into separate primitive-specific,
  precondition-specific, postcondition-specific, relation-specific,
  open/closed, or other sub-entries. A condition that touches several
  primitives stays ONE derivation with several primitive_timing entries.
* Echo the given source_condition_id exactly. Never invent another.

{PRIMITIVE_DEFINITIONS}

{ACTION_SEQUENCE}

=== TIMING DERIVATION ===

1. stage_rule and occurrence are your evidence. stage_rule states WHEN the
   predicate must hold, in plain temporal language ("before", "after", "during",
   "from ... until", "at the end of the final ..."). It names no primitives and
   no check types: you must reason those out from the action sequence.
2. Read stage_rule as a validity window. A single instant is a degenerate
   window: "before <action>" is that action's precondition, "after <action>" is
   that action's postcondition.
3. For a spanning window, for each primitive the window touches:
   * the window STARTS at that primitive's completion -> postcondition;
   * the window ENDS at that primitive's start -> precondition;
   * the primitive lies WHOLLY INSIDE the window (the state must hold for the
     entire waypoint) -> BOTH precondition and postcondition.
4. Emit one primitive_timing entry per primitive the window touches, with that
   primitive's check_type(s). List every primitive the window touches and no
   others. Do not include a primitive merely because it is nearby in the chain.
5. A predicate about the empty arm's readiness cannot apply to a primitive that
   presupposes a held object; a predicate about a held object cannot apply to
   a primitive that runs before its grasp; and a predicate about non-grasping
   end-effector contact -- contacting, pushing, or pressing -- applies to push
   and to no other primitive. Check each entry against the action sequence
   before returning it.

=== PROHIBITIONS ===

6. prohibitions record boundary exceptions that check_type and occurrence cannot
   express by themselves. They are not a summary of the timing and not a
   restatement of stage_rule.
7. Apply this test. A predicate that recurs on every matching waypoint may be a
   precondition on a waypoint only if some EARLIER waypoint in the action
   sequence has already established the state. Walk back from the first waypoint
   of the run:
   * If an earlier waypoint -- typically one of a different primitive -- already
     establishes the state, then every waypoint of the run may legitimately carry
     it as a precondition, and NO prohibition is needed.
   * If nothing earlier establishes it, the first waypoint of the run is where
     the state comes into existence. There the predicate is a postcondition only
     and can never be a precondition, and you MUST record that single exception
     in prohibitions.
8. Write each prohibition as one short imperative sentence naming the excluded
   check type and the exact waypoint it is excluded on. Return an empty list when
   the test yields no exception; never invent one to fill the field.

=== REMAINING FIELDS ===

9.  generation_rule: one operational instruction for instantiating the predicate
    on a concrete task. Keep it concise and imperative.
10. Object reference and naming: reference only objects the robot actually
    interacts with, never background, decorative, or untouched objects. Keep
    names simple and short (mug, block, drawer). When the same object type occurs
    more than once, disambiguate each with a distinguishing attribute such as
    colour or size (for example red_mug, small_block); never disambiguate by
    spatial position such as left or right.
11. Verify the true object-level state, not merely the gripper mechanism: confirm
    the object is actually held (not just that the gripper closed) and actually
    relinquished (not just that the gripper opened).
12. Generate a relational placement effect only when the task explicitly
    specifies that relation for the object and target. Do not infer, invent, or
    substitute a relation, a target, or a container the task does not state.

Return only the structured output required by the schema.
""".strip()


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------

def load_input(path: Path) -> TaxonomyInput:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"input file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    try:
        return TaxonomyInput.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"input validation failed:\n{exc}") from exc


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else [value]


def derivation_errors(
    item: RuleDerivation,
    condition: ConditionSpecification,
) -> list[str]:
    """Validate one derivation against the condition it was generated for."""

    errors: list[str] = []
    rid = condition.source_condition_id

    if item.source_condition_id != rid:
        errors.append(
            f"{rid}: source_condition_id must be echoed exactly, received "
            f"{item.source_condition_id!r}"
        )

    covered = [entry.primitive for entry in item.primitive_timing]
    if len(covered) != len(set(covered)):
        repeated = sorted({p.value for p, n in Counter(covered).items() if n > 1})
        errors.append(
            f"{rid}: primitive_timing lists a primitive more than once: "
            f"{repeated}"
        )

    for entry in item.primitive_timing:
        if len(set(entry.check_type)) != len(entry.check_type):
            errors.append(
                f"{rid}: primitive_timing[{entry.primitive.value}] repeats a "
                "check_type"
            )

    if any(not text.strip() for text in item.prohibitions):
        errors.append(f"{rid}: prohibitions must not contain empty strings")

    return errors


# ---------------------------------------------------------------------------
# Merging: authored half + derived half + mechanically derived fields
# ---------------------------------------------------------------------------

def build_timing(condition: ConditionSpecification, item: RuleDerivation) -> WaypointTiming:
    """Turn per-primitive check types into the full timing block.

    Uniform per-primitive timing collapses to a scalar check type with
    per_primitive left null.
    """

    entries = sorted(
        item.primitive_timing, key=lambda e: PRIMITIVE_ORDER[e.primitive]
    )

    def ordered_checks(checks: list[CheckType]) -> list[CheckType]:
        return sorted(set(checks), key=lambda c: CHECK_ORDER[c])

    per_primitive = [
        PrimitiveTiming(
            primitive=entry.primitive,
            check_type=ordered_checks(entry.check_type),
        )
        for entry in entries
    ]

    union_checks = ordered_checks(
        [check for entry in per_primitive for check in entry.check_type]
    )

    uniform = len({tuple(entry.check_type) for entry in per_primitive}) == 1

    return WaypointTiming(
        check_type=union_checks[0] if len(union_checks) == 1 else union_checks,
        applicable_primitives=[entry.primitive for entry in per_primitive],
        per_primitive=None if uniform else per_primitive,
        occurrence=condition.occurrence,
    )


def merge(source: TaxonomyInput, generated: DerivationSet) -> GeneratedTaxonomy:
    """Copy the authored fields verbatim and attach the derived ones."""

    by_id = {item.source_condition_id: item for item in generated.derivations}

    categories = [
        CategoryRuleSet(
            category=category.category,
            category_explanation=category.category_explanation,
            rules=[
                VerificationRule(
                    source_condition_id=condition.source_condition_id,
                    category=category.category,
                    failure=condition.failure,
                    predicate_template=condition.predicate_template,
                    predicate_meaning=condition.predicate_meaning,
                    timing=build_timing(
                        condition, by_id[condition.source_condition_id]
                    ),
                    generation_rule=by_id[
                        condition.source_condition_id
                    ].generation_rule,
                    prohibitions=by_id[
                        condition.source_condition_id
                    ].prohibitions,
                )
                for condition in category.rules
            ],
        )
        for category in source.categories
    ]

    return GeneratedTaxonomy(
        taxonomy_name=source.taxonomy_name,
        categories=categories,
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def build_user_message(
    source: TaxonomyInput,
    category: ConditionCategory,
    condition: ConditionSpecification,
) -> str:
    """The one-condition payload. Category context is included as framing only."""

    return (
        f"Scope: {source.scope}\n"
        f"Category: {category.category}\n"
        f"Category explanation: {category.category_explanation}\n\n"
        "Return exactly one derivation for this single condition "
        f"({condition.source_condition_id}):\n\n"
        + json.dumps(
            condition.model_dump(mode="json"), indent=2, ensure_ascii=False
        )
    )


def generate_one(
    client: Any,
    source: TaxonomyInput,
    category: ConditionCategory,
    condition: ConditionSpecification,
    model: str,
    max_attempts: int,
    reasoning_effort: str | None,
) -> RuleDerivation:
    """Derive one condition, retrying only that condition on validation failure."""

    base_message = build_user_message(source, category, condition)
    correction = ""
    extra: dict[str, Any] = (
        {"reasoning": {"effort": reasoning_effort}} if reasoning_effort else {}
    )

    for attempt in range(1, max_attempts + 1):
        response = client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": base_message + correction},
            ],
            text_format=RuleDerivation,
            **extra,
        )

        generated = response.output_parsed
        if generated is None:
            raise RuntimeError(
                f"{condition.source_condition_id}: the API returned no parsed "
                "structured output"
            )

        errors = derivation_errors(generated, condition)
        if not errors:
            return generated

        if attempt == max_attempts:
            raise RuntimeError(
                f"{condition.source_condition_id}: validation failed after "
                f"{max_attempts} attempts:\n- " + "\n- ".join(errors)
            )

        correction = (
            "\n\nRegenerate the derivation. The previous attempt was invalid:\n- "
            + "\n- ".join(errors)
        )

    raise RuntimeError("unreachable retry state")


def generate_derivations(
    client: Any,
    source: TaxonomyInput,
    model: str,
    max_attempts: int,
    concurrency: int,
    reasoning_effort: str | None = None,
) -> DerivationSet:
    """Fan the conditions out across threads; wall clock is the slowest one."""

    jobs = [
        (category, condition)
        for category in source.categories
        for condition in category.rules
    ]
    results: dict[str, RuleDerivation] = {}
    failures: list[str] = []

    with ThreadPoolExecutor(max_workers=max(1, min(concurrency, len(jobs)))) as pool:
        futures = {
            pool.submit(
                generate_one,
                client,
                source,
                category,
                condition,
                model,
                max_attempts,
                reasoning_effort,
            ): condition.source_condition_id
            for category, condition in jobs
        }

        for done, future in enumerate(as_completed(futures), start=1):
            rid = futures[future]
            try:
                results[rid] = future.result()
            except Exception as exc:  # noqa: BLE001 - reported per condition
                failures.append(f"{rid}: {exc}")
                print(f"  [{done}/{len(jobs)}] {rid} FAILED", file=sys.stderr)
            else:
                print(f"  [{done}/{len(jobs)}] {rid}", file=sys.stderr)

    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(jobs)} conditions failed:\n- "
            + "\n- ".join(sorted(failures))
        )

    return DerivationSet(
        derivations=[
            results[condition.source_condition_id]
            for condition in source.conditions
        ]
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

CSV_COLUMNS = (
    "category",
    "source_condition_id",
    "failure",
    "predicate_template",
    "predicate_meaning",
    "check_type",
    "applicable_primitives",
    "per_primitive",
    "occurrence",
    "generation_rule",
    "prohibitions",
)


def _join(values: list[Any]) -> str:
    return "; ".join(
        value.value if isinstance(value, Enum) else str(value) for value in values
    )


def _per_primitive_cell(timing: WaypointTiming) -> str:
    if not timing.per_primitive:
        return ""
    return " | ".join(
        f"{entry.primitive.value}: {', '.join(c.value for c in entry.check_type)}"
        for entry in timing.per_primitive
    )


def render_csv(generated: GeneratedTaxonomy, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for category_result in generated.categories:
            for rule in category_result.rules:
                timing = rule.timing
                writer.writerow(
                    {
                        "category": rule.category,
                        "source_condition_id": rule.source_condition_id,
                        "failure": rule.failure,
                        "predicate_template": rule.predicate_template,
                        "predicate_meaning": rule.predicate_meaning or "",
                        "check_type": _join(_as_list(timing.check_type)),
                        "applicable_primitives": _join(
                            timing.applicable_primitives
                        ),
                        "per_primitive": _per_primitive_cell(timing),
                        "occurrence": timing.occurrence.value,
                        "generation_rule": rule.generation_rule,
                        "prohibitions": " | ".join(rule.prohibitions),
                    }
                )


def render_text(generated: GeneratedTaxonomy) -> str:
    lines = [
        f"# {generated.taxonomy_name}",
        "",
        PRIMITIVE_DEFINITIONS,
        "",
        ACTION_SEQUENCE,
        "",
        "=== GENERATED VERIFICATION RULES ===",
        "",
    ]

    for result in generated.categories:
        lines.extend(
            [
                f"## {result.category}",
                "",
                result.category_explanation.strip(),
                "",
            ]
        )

        for rule in result.rules:
            timing = rule.timing
            primitives = (
                _join(timing.applicable_primitives)
                if timing.applicable_primitives
                else "task-global"
            )
            lines.extend(
                [
                    f"### {rule.failure}",
                    "",
                    f"* Source: `{rule.source_condition_id}`",
                    f"* Predicate: `{rule.predicate_template}`",
                ]
            )
            if rule.predicate_meaning:
                lines.append(f"* Predicate meaning: {rule.predicate_meaning}")
            lines.extend(
                [
                    f"* Check type: `{_join(_as_list(timing.check_type))}`",
                    f"* Applicable primitives: {primitives}",
                    f"* Occurrence: `{timing.occurrence.value}`",
                    f"* Generation rule: {rule.generation_rule}",
                ]
            )
            if timing.per_primitive:
                lines.append("* Per-primitive timing:")
                for entry in timing.per_primitive:
                    entry_checks = ", ".join(c.value for c in entry.check_type)
                    lines.append(
                        f"  - {entry.primitive.value}: {entry_checks}"
                    )
            if rule.prohibitions:
                lines.append("* Prohibitions:")
                lines.extend(f"  - {item}" for item in rule.prohibitions)
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate exactly one verification rule per input condition."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"condition taxonomy JSON (default: {DEFAULT_INPUT.name})",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
    )
    parser.add_argument(
        "--output-text",
        type=Path,
        default=DEFAULT_OUTPUT_TEXT,
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-5.4"),
    )
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="conditions derived in parallel (default: 8; 1 = sequential)",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high"),
        default=os.getenv("OPENAI_REASONING_EFFORT") or None,
        help=(
            "reasoning budget for reasoning-capable models; lower is faster. "
            "Omitted entirely when unset, so non-reasoning models still work."
        ),
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--show-system-prompt", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.show_system_prompt:
        print(SYSTEM_PROMPT)
        return 0

    try:
        source = load_input(args.input)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.validate_only:
        print(
            f"Valid input: {len(source.categories)} categories and "
            f"{len(source.conditions)} atomic conditions."
        )
        return 0

    if args.max_attempts < 1:
        print("ERROR: --max-attempts must be at least 1", file=sys.stderr)
        return 2

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "ERROR: OPENAI_API_KEY is not set. Store the key in the environment.",
            file=sys.stderr,
        )
        return 2

    try:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "the openai package is not installed; run: "
                "python -m pip install -U openai pydantic"
            ) from exc

        print(
            f"Deriving {len(source.conditions)} conditions with "
            f"{args.model} (concurrency {args.concurrency})...",
            file=sys.stderr,
        )
        derivations = generate_derivations(
            client=OpenAI(),
            source=source,
            model=args.model,
            max_attempts=args.max_attempts,
            concurrency=args.concurrency,
            reasoning_effort=args.reasoning_effort,
        )
        generated = merge(source, derivations)
    except Exception as exc:
        print(f"ERROR: generation failed: {exc}", file=sys.stderr)
        return 1

    for path in (args.output_json, args.output_csv, args.output_text):
        path.parent.mkdir(parents=True, exist_ok=True)

    args.output_json.write_text(
        json.dumps(generated.model_dump(mode="json"), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    render_csv(generated, args.output_csv)
    args.output_text.write_text(render_text(generated), encoding="utf-8")

    print(f"Structured rules: {args.output_json}")
    print(f"Spreadsheet rules: {args.output_csv}")
    print(f"Prompt-ready rules: {args.output_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
