"""Prompt assembly for the BT condition-generation agent."""

from aha_publish import paths

import json

from .cleanup import (
    iter_failure_definition_entries,
    task_context_for_prompt,
)


def build_user_message(
    task_context: str,
    failure_definitions: dict,
    reviewer_guidance: str = "",
    task_name: str = "",
) -> str:
    """Build the task-specific message for the LLM.

    All waypoint classification and condition generation are performed by the
    LLM according to the system prompt.

    This function does not:
    - classify tasks by object count;
    - classify waypoints into primitives;
    - derive gripper predicates;
    - assign preconditions or postconditions;
    - select predicates for specific waypoint types.

    ``task_name`` is retained only for compatibility with existing callers.
    """

    del task_name

    cleaned_task_context = task_context_for_prompt(
        task_context
    )

    # Names only. Every statement about WHEN a condition applies — the primitive
    # it attaches to, precondition vs postcondition, how often — belongs to the
    # catalogue rule in the system prompt. Restating it here produced guidance
    # that contradicted the catalogue and won, because it sat in the user message
    # and read as the more specific instruction.
    failures_text = "\n".join(
        f"- {name}"
        for name, _details in iter_failure_definition_entries(failure_definitions)
    )

    reviewer_section = ""

    if reviewer_guidance.strip():
        reviewer_section = (
            "\n\nREVIEWER GUIDANCE\n\n"
            f"{reviewer_guidance.strip()}"
        )

    return f"""TASK DESCRIPTION

{cleaned_task_context}

VALID FAILURE NAMES

Link each condition to the failure name assigned by its catalogue rule in the
system prompt. These are the only names that exist:

{failures_text}{reviewer_section}

Generate the primitive classification, preconditions, and postconditions for
every waypoint.

Use the waypoint descriptions, gripper states, held-object states, scene
objects, and task relationships provided in the task description.

Follow all primitive definitions, allowed predicates, state-transition rules,
object-grounding requirements, timing constraints, failure-linking rules, and
JSON formatting requirements specified in the system prompt.

The structured gripper_state and held_object fields in the task description are
authoritative and override ambiguous free-text descriptions.

Each condition must be linked to exactly one of the failure names listed above.

Do not invent objects, predicates, waypoints, or failure modes.
"""