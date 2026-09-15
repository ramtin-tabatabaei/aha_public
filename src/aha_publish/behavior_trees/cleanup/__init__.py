"""Deterministic condition cleanup and repair passes.

This package is a drop-in replacement for the monolithic cleanup.py module.
All public names are re-exported so ``from .cleanup import *`` continues to
work exactly as before.
"""

from aha_publish import paths

from .predicates import (
    apply_allowed_predicates_filter,
    strip_color_qualifiers_from_object_found,
)
from .object_scope import (
    identify_task_objects_from_conditions,
    filter_object_not_found_scope,
    hoist_object_not_found_conditions,
    ensure_object_found_for_all_objects,
    fix_placement_condition_timing,
)
from .alignment import (
    stage_allows_wrong_object_selection,
    stage_allows_end_effector_alignment,
    condition_has_end_effector_alignment,
    condition_has_orientation,
    strip_end_effector_alignment,
    strip_end_effector_alignment_for_objects,
    strip_orientation,
    strip_orientation_for_objects,
)
from .selected_object import (
    enforce_grasp_result_preconditions,
    enforce_pre_grasp_conditions,
)
from .gripper import (
    gripper_condition_from_text,
    section_gripper_condition,
    gripper_condition_block,
    remove_gripper_condition_blocks,
    object_in_gripper_block,
    stage_creates_held_object,
    section_has_object_in_gripper,
    objects_in_gripper_from_conditions,
    released_objects_from_conditions,
    strip_object_in_gripper_for_objects,
    section_has_gripper_state,
    default_gripper_state_for_stage,
    gripper_state_condition,
    enforce_release_breaks_holding,
    enforce_gripper_condition_continuity,
)
from .pipeline import (
    filter_stage_condition_blocks,
    apply_condition_cleanup_pipeline,
)

# Pull in everything from semantics so that ``from .cleanup import *`` in
# downstream modules continues to expose the full condition/semantics namespace
# that the old monolithic cleanup.py provided via ``from .semantics import *``.
# No __all__ is defined here so that star-imports pick up all public names,
# matching the original behaviour of the monolithic module.
from ..semantics import *
