"""Prompt templates and naming guidance for task description generation."""

from aha_publish import paths

from .config import *

SYSTEM_PROMPT = """You are an expert robotics analyst. You reconstruct what a robot does in a manipulation task from exactly three inputs:
1. A waypoint/stage grid image — the primary visual evidence.
2. A per-waypoint gripper-sequence table — measured ground truth for the gripper action and which object is rigidly held after each waypoint.
3. A structured TTM scene-inspection report — waypoint parent attachments, world-frame poses, and object placement relations.
Base every claim on these three inputs only; when they are silent or ambiguous, say so instead of guessing.
Treat the image as primary visual evidence, and treat the gripper sequence and TTM report as authoritative structured evidence that OVERRIDES visual guesses about grasping, holding, release, and motion direction."""


USER_PROMPT_TEMPLATE = """Analyze the __TASK_NAME__ task from the supplied inputs.

You have three inputs: the grid image, the gripper-sequence table, and the TTM scene report. Read them as follows.

How to read the grid image (primary visual evidence):
- The input image is one combined grid image.
- Rows are task waypoints/stages ordered top-to-bottom. Each row is a SUBGOAL: a distinct intermediate state the robot must reach on the way to finishing the task.
- The expected waypoint row ids are: __WAYPOINT_IDS_JSON__.
- Each column shows the same row from a different camera angle. The number of camera columns can vary; it is __CAMERA_COUNT_TEXT__.
- Use all camera views in a row together. If row or column labels are visible in the image, use them.
- The image tells you WHAT the objects are and their coarse spatial relations. For grasp state and exact motion direction, defer to the two structured inputs below.

Reason about the task as a whole before describing individual waypoints:
- The robot has ONE overall goal. Infer that goal first, then read each row as a step toward it. Some steps are the goal itself; others are only a MEANS to it (e.g. grasping a tool or object so it can later be used, moving to line up before acting).
- Look across all rows together to see the progression: typically approach → grasp/contact → transport → actuate/place → release/retreat. Not every task has all of these; use the rows you are given.
- Everything you need to explain the goal is present in the scene shown in the images and the two structured inputs. Do not invent steps that no input supports.

How to read the gripper-sequence table (authoritative for grasping):
- It lists, per waypoint, the gripper_action (open/close/none), the state_after (Open/Closed), the gripper_open_amount (1.0 = fingers fully apart, 0.0 = fingers fully shut against each other), and which object(s) are rigidly held afterward (held '(none)' = the table records nothing held).
- The Open/Closed state and the gripper_open_amount are both measured ground truth — take them from the table, not from the image.
- Decide WHAT is held at each waypoint with this exact rule:
  1. Gripper OPEN -> the robot is holding NOTHING (held_object = null), no matter what the image seems to show.
  2. Gripper CLOSED and held_after names an object -> the robot is holding THAT object (held_object = that name).
  3. Gripper CLOSED, held_after empty, and gripper_open_amount BELOW 0.01 -> the fingers have shut all the way against each other, so there is nothing between them. The robot is PRESSING or PUSHING with the closed fingertips (held_object = null) or getting ready to push or press.
  4. Gripper CLOSED, held_after empty, and gripper_open_amount ABOVE 0.01 -> the fingers are being held apart by something, so the robot IS gripping an object. Name that object from the image (use its key_scene_objects name) and set held_object to it. an empty held column is NOT evidence that the gripper is empty. If the image cannot show you what is between the fingers, say uncertain rather than guessing an object.
- A held_object that changes from nothing to an object marks a grasp at that waypoint; a change back to nothing marks a release. An unchanged held object across waypoints means the robot is carrying it.
- After a release, the object remains where it was placed. From that waypoint onward, do not describe the object as held, carried, grasped, or "in the gripper" anywhere, including in robot_action, visual_summary, or approx_distance_to_objects, unless the robot subsequently grasps the object again.
- An OPEN gripper with held_object null is not grasping ANYTHING at that waypoint. That is measured, not inferred, so it overrides how the images look. At such a waypoint robot_action, visual_summary and approx_distance_to_objects must not say the robot grips, holds, grasps, carries, and must not call an object gripped, held, in the gripper, or between the fingers. Say what is actually true instead: the OPEN fingers are near, around, aligned with, or withdrawing from the object, and the object rests where it is.

Infer the manipulation from the gripper state:
- If the robot holds an object across a span of waypoints, the task is almost certainly MANIPULATING THAT OBJECT during that span — carrying it to a target, placing/inserting it, or turning/aiming/using it. Identify what it does with the held object by cross-checking the image and how that object's pose/orientation changes in the TTM report over the same span.
- A CLOSED gripper whose gripper_open_amount is below 0.01 — nothing between the fingers — that moves onto an object is a direct press/push, contact with the fingertips and not a grasp. If instead the fingers are held apart (gripper_open_amount above 0.01), the same motion is a grip on whatever is between them.
- An OPEN gripper is NOT pressing or pushing a button: a button press or push needs the gripper to close on / make contact with the control.
- Use the gripper_state to decide whether the robot is going to press any buttons: whenever the gripper closes (or is already closed) with gripper_open_amount below 0.01 and lowers onto / makes contact with a button, switch, key, or similar control, the robot is PRESSING that control. In that case set that waypoint's gripper_state to "pressing button". A gripper that stays open and merely hovers over a button is NOT pressing it, and a closed gripper whose fingers are held apart is gripping something rather than pressing — do not claim a press without the gripper closing on / contacting the control.
- Actuating an articulated object (door, drawer, lid, knob) does NOT require the simulator's latch: the gripper may be open OR closed while opening/closing it, and the held column stays empty either way. Decide it with the rule above — closed with the fingers apart means the robot has taken hold of that handle, panel or knob and is pulling/pushing/turning it; open, or closed with the fingers shut, means it is nudging or pressing it instead.
- Whenever the robot moves, carries, turns, rotates, pushes, or otherwise displaces ANY object — whether it is rigidly held, pinched to be turned, pressed, or pushed while articulated — you MUST name that specific object (using its established name) in the relevant waypoint's robot_action and in the overall_description. Never describe a rotation or displacement in the abstract ("turns the control", "rotates the hand") without stating which named object is being moved or turned.

How to read the TTM scene report (authoritative for motion and targets):
- It gives each waypoint's PARENT attachment, and its WORLD-frame pose.
- Use the WORLD-frame waypoint poses (and the world-frame orientation) for real motion direction, height change, and end-effector reorientation. In world frame, +z is up. The axes are a tool for YOUR reasoning only — translate whatever you work out into scene-relative words before writing it down, and never name an axis or a frame in the JSON.
- A waypoint parented to a graspable object marks where the robot grasps or acts on it; a waypoint parented to a fixed object marks where it places or uses it. Use these parents to name the grasp target and the place/goal target.
- While an object is held according to the gripper table, treat the end-effector motion across the corresponding waypoints as evidence of the held object's motion. Because the object moves with the gripper, a large change in waypoint orientation indicates that the robot is rotating or reorienting the held object, whereas predominantly translational motion indicates that the robot is carrying the object toward another location. Use this information to describe the manipulation concretely, e.g., "carries the screwdriver toward the socket" or "rotates the screwdriver to align it with the socket.
- The report is ONE snapshot taken when the episode was initialised, before the robot moved anything, so every object pose in it is that object's STARTING pose. It is never updated as the task proceeds.
- Do NOT assume the stand-off is vertical. Work out which axis or axes actually differ, then write the WORD that axis corresponds to, never the axis letter: a larger z means "above" it, a smaller z "below", and a horizontal offset at the SAME z means "beside"/"in front of"/"behind" it — not above and not touching. Approaches to buttons and objects lying on a surface are usually offset vertically; approaches to handles, doors, drawers and anything on a vertical face are usually offset horizontally at the same height.
- These trajectories come in pairs: stand off from the target, move in along the offset axis to act, then withdraw. So when one waypoint stands off from an object it has not yet moved and the NEXT waypoint moves in to that object's position, the first is only lining up — the contact/press/grasp is the second. Say "positioned <above/below/beside/in front of/behind> it, not touching" for the stand-off waypoint and put the press/grasp on the one that reaches the object.
- A close/open gripper command issued at a stand-off waypoint is the gripper pre-shaping before it moves in, NOT a press: never call a waypoint a press just because the gripper closes there.
- Where the coordinates ARE usable (an object that has not moved yet) and they put the gripper clearly off the object, do not write contact wording ("touching", "contacting", "pressing", "on its surface") in robot_action, visual_summary, or approx_distance_to_objects; state which WAY it is offset instead, in words. Where they are stale, leave them out of the reasoning entirely rather than describing an offset you cannot support.
- The coordinates are for YOUR reasoning only. NEVER copy a number out of them into the JSON.

NO NUMBERS AND NO AXIS NAMES IN THE OUTPUT (hard rule):
- No text field may contain a measurement, a coordinate, or any quantity — not as digits and not spelled out. Banned: "8 cm", "0.2 m", "about 20 centimeters", "a few millimetres", "45 degrees", "0.05", "(x, y, z)", "twice as far", "second of three".
- No text field may contain a coordinate-axis label either. Banned: "world -x", "+y side", "along z", "the x direction", "x/y offset", "world frame", "negative y". The axes exist only in the TTM report; a reader of your JSON cannot see them, so they mean nothing there.
- Describe every distance, offset, height and rotation in plain words: "beside", "just off to the side", "well clear of", "close above", "far above", "level with", "aligned with", "touching", "inside", "slightly turned", "rotated to face", "uncertain".
- Describe every DIRECTION relative to something visible in the scene instead of an axis: "toward the far end of the pole", "away from the base", "toward the drawer", "to the opposite side of the apparatus", "straight down onto it", "upward and clear of the table", "back toward its starting end". Up/down and lifts/lowers are fine as words; just never say which axis they correspond to.
- Work the axes out in your head from the TTM coordinates, then WRITE only the scene-relative phrase they translate to.
- The ONLY numbers anywhere in your reply are the integer waypoint ids in the "waypoint" field.

Return ONLY valid JSON, no markdown, in exactly this structure:

{
  "task": "__TASK_NAME__",
  "overall_description": "<Two or three sentences: state the robot's single overall goal, then the main steps used to reach it, distinguishing the end goal from the means (e.g. grasping/carrying an object in order to use it).>",
  "key_scene_objects": [
    {
      "name": "<__NAME_FIELD_HINT__>",
      "original_name": "<the exact simulator/TTM name for this object as given in the supplied scene/object context. Use null only if the object has no supplied name. This preserves the link to simulator state even when 'name' is descriptive.>",
      "role": "<role in the task>"__RELATIONSHIPS_FIELD__
    }
  ],
  "waypoints": [
    {
      "waypoint": 0,
      "visual_summary": "<Concrete description from the grid image row.>",
      "robot_action": "<What the robot is doing at this moment.>",__MOTION_FIELD__
      "approx_distance_to_objects": {
        "<object name>": "<Qualitative spatial relation from the end effector to this object, in words only — no measurements, no numbers>"
      },__WAYPOINT_RELATIONSHIP_FIELD__
      "gripper_state": "<open, closing, closed, holding object, releasing, pressing button, not applicable, or uncertain>",
      "held_object": "<the key_scene_objects name of the single object the gripper is holding at this waypoint, decided with the four gripper rules above, or null when it holds nothing>"__SCENE_CONTEXT_FIELD__
    }
  ]
}

Rules:
- Include exactly __WAYPOINT_COUNT__ waypoint entries with waypoint ids __WAYPOINT_IDS_JSON__.
- Keep the same JSON field names shown above.
- held_object is exactly one key_scene_objects name or null — never a list, never free text, never an object missing from key_scene_objects. It is decided ONLY by the four gripper rules above, so a waypoint whose table state is Open always has held_object null. If the fingers are apart while closed but the images cannot settle WHICH object is between them, set gripper_state to "uncertain" and held_object to null rather than guessing one.
- held_object and gripper_state must agree: name an object only where gripper_state says the robot is holding it, and keep the same held_object across every waypoint of a carry until the release.
- Be action-focused and concrete.
- If the robot rotates or moves any object, name that object explicitly. Any waypoint whose robot_action involves a turn, rotation, carry, push, or other displacement must state the named object being turned or moved; do not leave it implicit.
- Every object you name must be a real, physical object that is actually visible in the images. Do NOT list non-physical or invisible helper entities even when they are named in the supplied scene context — exclude proximity/success/failure sensors, detectors, dummies, waypoints, spawn boundaries, and similar invisible markers. Never invent objects that are not actually there.
- __NAMING_RULE__
- Keep names unique and follow the naming rule above. Whenever two or more visible objects are the same type, every one of them MUST carry a distinguishing visible attribute — prefer color, then size, then position — joined with an underscore. Never use bare numeric suffixes to distinguish visible same-type objects. A lone object of its type needs no attribute. Use lowercase with underscores and no spaces, and reuse the exact same name everywhere it appears.
- In approx_distance_to_objects, use the most relevant task objects as keys. Say qualitative relations like "near", "touching", "above", "inside", "aligned with", or "uncertain". Never a distance, never a number, in this field or any other.
- if the image and the two structured inputs cannot settle something, mark it uncertain rather than overstating.
- Do not add any certainty or quality-score field.
- Do not mention API, prompt, or file handling details in the JSON."""


# --- Optional output fields, off by default ---
# These schema fields are not consumed by any downstream code (bt_maker/detectors
# only read them as raw prompt text), so they are excluded from the generated JSON
# unless explicitly re-enabled. Each placeholder in USER_PROMPT_TEMPLATE is replaced
# with its snippet when the controlling flag is set, or "" when it is not.
# All of these are gated by --extra-fields.
OPTIONAL_OUTPUT_FIELDS = {
    "__MOTION_FIELD__": (
        "include_extra_fields",
        '\n      "approx_motion_from_previous_waypoint": "<Movement from the previous '
        'waypoint described in words only — direction and coarse extent, never a '
        'measurement or number. For the first waypoint, describe the initial relation '
        'to the relevant task object instead.>",',
    ),
    "__WAYPOINT_RELATIONSHIP_FIELD__": (
        "include_extra_fields",
        '\n      "waypoint_relationship_used": "<The relevant waypoint/object relation '
        "using neutral waypoint placeholders, or 'not available'.>\",",
    ),
    "__SCENE_CONTEXT_FIELD__": (
        "include_extra_fields",
        ',\n      "scene_context_used": "<How supplied object/relationship context '
        "changes or confirms the interpretation, or 'not available'.>\"",
    ),
    "__RELATIONSHIPS_FIELD__": (
        "include_extra_fields",
        ',\n      "relationships": "<important spatial or task relationship>"',
    ),
}


def optional_field_replacements(args: argparse.Namespace) -> dict[str, str]:
    """Map each optional-field placeholder to its snippet (if the controlling flag is
    set) or an empty string, so disabled fields vanish from the prompt schema."""
    return {
        placeholder: (snippet if getattr(args, flag, False) else "")
        for placeholder, (flag, snippet) in OPTIONAL_OUTPUT_FIELDS.items()
    }


# --- Object-naming guidance, selected by --naming / DEFAULT_NAMING_MODE ---
# Each mode supplies the JSON "name" field hint (__NAME_FIELD_HINT__) and the
# naming rule bullet (__NAMING_RULE__) that get substituted into the prompt.
NAME_FIELD_HINT = {
    "original": (
        "the object's original name exactly as given in the supplied scene/object-relationship "
        "context (the simulator/TTM object name). Only if a visible object has no "
        "name anywhere in the supplied context, give it a short plain descriptive name. If two "
        "objects share the same original name, append a distinguishing visible attribute joined "
        "by an underscore."
    ),
    "descriptive": (
        "a clear, exact, human-readable name for what the object actually is in the image and the "
        "role it plays in the task, grounded in its visible appearance (shape, color, material) and "
        "what the task does with it. Do not copy a cryptic or "
        "generic simulator name. Lowercase with underscores. IMPORTANT "
        "disambiguation rule: if exactly ONE object of a given type is visible, name it by its plain "
        "type with NO distinguishing attribute. Only if TWO OR MORE objects "
        "of the same type are visible, qualify EACH of them with a distinguishing visible attribute "
        "— prefer color, then size, then position — joined with an underscore, so each can be told "
        "apart in the images"
    ),
}

NAMING_RULE = {
    "original": (
        "Use each object's original name exactly as given in the supplied scene/object-relationship "
        "context (the simulator/TTM object name). Do NOT rename, prettify, translate"
        "or invent alternative names for an object that already has a supplied name — copy the "
        "supplied name verbatim, even if it looks technical or terse. Only when a clearly visible "
        "object has no name anywhere in the supplied context may you give it a short, plain "
        "descriptive name. Keep each name identical everywhere it appears, including in "
        "relationships and approx_distance_to_objects, so downstream steps refer "
        "to the same object by the same name."
    ),
    "descriptive": (
        "Name each object for what it visibly is and the role it plays in the task, read from the "
        "images — not from the cryptic simulator/TTM name. Prefer a clear, everyday, task-relevant name grounded in the object's "
        "visible form (straight vs bent vs curved vs looped vs coiled), color, material, and "
        "function. When the supplied/instruction wording and the "
        "image disagree about an object's shape, trust the image. DISAMBIGUATION: when only ONE "
        "object of a type is present, use its plain type name with NO extra attribute. When TWO OR MORE objects of the same type are present, you MUST give each a "
        "distinguishing visible attribute — prefer color, then size, then position — joined with an "
        "underscore, so the VLM can "
        "tell which specific instance a condition refers to; never leave same-type objects on bare "
        "numeric names. Keep each chosen name consistent everywhere it appears, "
        "including in relationships and approx_distance_to_objects, so downstream "
        "steps refer to the same object by the same name."
    ),
}


def resolve_naming_mode(args: argparse.Namespace) -> str:
    mode = str(getattr(args, "naming", DEFAULT_NAMING_MODE) or DEFAULT_NAMING_MODE).strip().lower()
    return mode if mode in NAMING_RULE else "original"

