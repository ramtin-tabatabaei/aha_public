"""
OpenAI VLM confirmation for slip detections.

This module reuses the camera/image helpers from the collision VLM module, but
uses a slip-specific prompt and telemetry table.
"""

from aha_publish import paths

from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import re


REPO_ROOT = (paths.PROJECT_ROOT)
COLLISION_VLM_PATH = (paths.SOURCE_DIR / 'detectors/collision/vlm_confirm.py')
VLM_TRACE_DIR = (paths.OUTPUT_DIR / 'vlm_traces')
BT_CONDITIONS_DIR = (paths.BT_DIR)

INCLUDE_WAYPOINT_CONTEXT = (
    os.getenv("AHA_VLM_WAYPOINT_CONTEXT", "1").strip().lower()
    in ("1", "true", "yes")
)


def _resolve_confirm_cameras(default):
    """Allow an evaluation or batch run to shrink the camera set sent to the VLM.

    Set AHA_VLM_CONFIRM_CAMERAS to a comma-separated list, for example:
    side_rgb,wrist_rgb,front_rgb

    When unset, the full production camera set is used.
    """
    selected = os.getenv("AHA_VLM_CONFIRM_CAMERAS", "").strip()
    if selected:
        return tuple(camera.strip() for camera in selected.split(",") if camera.strip())
    return default


DEFAULT_CAMERA_NAMES = _resolve_confirm_cameras(
    (
        "wrist_rgb",
        "side_rgb",
        "front_rgb",
    )
)

# Spread samples across the full observation buffer so the VLM sees an older
# grasp state, intermediate states, and the current state. Defined here (above
# the prompts) so USER_PROMPT can describe the real sampling to the model and
# never drift from what is actually sent.
_SLIP_IMAGE_OFFSETS = (5, 0)
_SLIP_FRAME_LABELS = ", ".join(
    "current" if offset == 0 else f"-{offset}"
    for offset in _SLIP_IMAGE_OFFSETS
)

DEFAULT_OPENAI_MODEL = os.getenv(
    "OPENAI_SLIP_VLM_MODEL",
    os.getenv("OPENAI_MODEL", "gpt-5.4"),
)

SYSTEM_PROMPT = (
    "You are a visual slip-failure auditor for an RLBench robot simulation.\n\n"

    "You answer exactly one question: did an object that was securely held by "
    "the gripper become no longer securely held? Use the camera images as the "
    "primary evidence and grip-force telemetry only when the visual evidence "
    "is ambiguous.\n\n"

    "DID THE OBJECT LEAVE THE GRIPPER?\n"
    "1. Identify the object the robot is manipulating.\n"
    "2. Determine whether the earlier images clearly show that the object was "
    "securely held between the gripper fingers. If the earlier images clearly "
    "show that it was never securely held, this is not a slip. If prior holding "
    "is visually ambiguous, do not assume that the object was never held.\n"
    "3. Determine whether that same object is still securely constrained by "
    "the gripper in the current frame.\n\n"

    "Inspect all available camera views. Give greatest weight to views in which "
    "the gripper-object relationship is clearly visible. Occluded or "
    "uninformative views should not override clear evidence from an informative "
    "view. If informative views genuinely conflict, treat the visual evidence "
    "as ambiguous. For grasp evidence, normally prioritize wrist_rgb, followed "
    "by side_rgb and front_rgb, unless another view is clearly more informative.\n\n"

    "The object HAS left the gripper if any of these hold:\n"
    "- It fell, dropped, separated, or was left behind.\n"
    "- It is no longer between and constrained by the fingers.\n"
    "- It is now supported by the table, a fixture, or the target instead of "
    "the gripper.\n"
    "- It is still near or touching the fingers but is no longer securely "
    "pinched or constrained by them.\n"
    "- It was securely held earlier but the gripper is now closed without the "
    "object inside.\n\n"

    "The object is STILL HELD when the clearest available views show that it "
    "remains between and constrained by the fingers. Visual proximity, overlap, "
    "or contact with the gripper alone does not prove that the object is held.\n\n"

    "Do not require a dramatic fall or large separation. Loss of secure grasp "
    "support can occur while the object remains close to or touching the "
    "fingers.\n\n"

    "A closed gripper does not prove that the object is inside it, and an "
    "object near, behind, touching, or overlapping the gripper is not "
    "necessarily held.\n\n"

    "THE ANSWER FOLLOWS DIRECTLY: if the object was securely held earlier and "
    "is no longer securely held, slip_happened=true. If it remains securely "
    "held, or the earlier images clearly show that it was never securely held, "
    "slip_happened=false.\n\n"

    "Do NOT reason about whether a release would have been appropriate at this "
    "point in the task. The detector calls this verifier only while continued "
    "object holding is required; an intentional commanded release does not "
    "reach this verifier. Therefore, do not dismiss a visible loss because the "
    "object is near its destination or because a release might otherwise seem "
    "appropriate.\n\n"

    "Use the camera images as the primary evidence. Use grip-force telemetry "
    "and the detector event only to resolve genuinely ambiguous visual evidence; "
    "a force drop supports, but does not by itself prove, loss of grasp.\n\n"

    "Return strict JSON only:\n"
    "{"
    "\"slip_happened\": true/false, "
    "\"explanation\": \"one concise evidence-based sentence\""
    "}"
)
USER_PROMPT = (
    "A low-level detector flagged a possible slip while continued object holding "
    "was required. Analyse the provided multi-view frames from oldest to current "
    "and determine whether an object that was securely held earlier became no "
    "longer securely held.\n\n"

    "HOW TO READ THE IMAGES\n"
    "The frames are in time order, oldest first, and are normally sampled at "
    f"offsets {_SLIP_FRAME_LABELS} relative to the detector event. Each image "
    "group is introduced by '## Step i of N (...)'. Cameras within one group "
    "show different viewpoints of the same instant, not different moments. "
    "Use the clearest available views to establish whether the object was held "
    "earlier and whether it remains securely constrained in the current frame. "
    "Occluded or uninformative views should not override clear evidence from "
    "another view.\n\n"

    "If the earlier images clearly show that the object was never securely held, "
    "set slip_happened=false. If prior holding is visually ambiguous, do not "
    "assume that it was never held.\n\n"

    "Use images as the primary evidence. Use the detector event and grip-force "
    "telemetry only as supporting evidence when the images are ambiguous.\n\n"

    "Return strict JSON only:\n"
    "{"
    "\"slip_happened\": true/false, "
    "\"explanation\": \"one concise evidence-based sentence\""
    "}"
)


def _load_collision_helpers():
    spec = importlib.util.spec_from_file_location(
        "_aha_collision_vlm_helpers",
        COLLISION_VLM_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_helpers = _load_collision_helpers()
show_camera_sequence = _helpers.show_camera_sequence
collect_camera_images = _helpers.collect_camera_images
collect_display_images = _helpers.collect_display_images


def save_camera_grid(
    recent_obs_list,
    step_offsets,
    camera_names,
    env_wrapper,
    out_dir,
    tag="",
):
    """Save the exact camera montage sent to the slip VLM.

    Rows represent time offsets and columns represent cameras. Returns the saved
    path, or an empty string if there is nothing to save or matplotlib is missing.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return ""

    if step_offsets is None:
        step_offsets = [None] * len(recent_obs_list)

    grid_rows = []
    all_cameras = []

    for observation in recent_obs_list:
        images = collect_display_images(
            observation,
            camera_names=camera_names,
            env_wrapper=env_wrapper,
        )
        grid_rows.append(dict(images))

        for camera_name, _ in images:
            if camera_name not in all_cameras:
                all_cameras.append(camera_name)

    if not all_cameras:
        return ""

    figure, axes = plt.subplots(
        len(grid_rows),
        len(all_cameras),
        figsize=(3.2 * len(all_cameras), 2.6 * len(grid_rows)),
        squeeze=False,
    )

    for row_index, (grid_row, offset) in enumerate(zip(grid_rows, step_offsets)):
        for column_index, camera_name in enumerate(all_cameras):
            axis = axes[row_index][column_index]
            image = grid_row.get(camera_name)

            if image is not None:
                axis.imshow(image)

            axis.axis("off")

            if row_index == 0:
                axis.set_title(camera_name, fontsize=9)

            if column_index == 0:
                if offset is None:
                    label = f"frame {row_index + 1}"
                elif offset == 0:
                    label = "current"
                else:
                    label = f"-{offset}"

                axis.set_ylabel(label, fontsize=10, rotation=0, labelpad=30)

    figure.suptitle(f"Images sent to slip VLM {tag}".strip())
    figure.tight_layout()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%H%M%S_%f")
    path = Path(out_dir) / f"slip_vlm_grid_{timestamp}.png"

    figure.savefig(path, dpi=100)
    plt.close(figure)

    return str(path)


def sample_recent_observations(obs_list):
    return _helpers.sample_recent_observations(
        obs_list,
        offsets=_SLIP_IMAGE_OFFSETS,
    )


def _load_task_description(bt_data, waypoints_description_path=None):
    """Return the multimodal task-description JSON backing a Behaviour Tree."""
    candidates = []

    if waypoints_description_path:
        candidates.append(Path(waypoints_description_path))

    context_path = bt_data.get("task_context_path")
    if context_path:
        candidates.append(REPO_ROOT / context_path)

    for candidate in candidates:
        try:
            if candidate.exists():
                with open(candidate, encoding="utf-8") as file:
                    return json.load(file)
        except Exception:
            continue

    return {}


def _format_conditions(label, conditions):
    """Render one hold-condition block, keeping the detector notes."""
    if not conditions:
        return []

    lines = [f"    {label}:"]

    for entry in conditions:
        text = entry.get("condition", "?")
        definition = entry.get("definition", "")
        detail = f" ({definition})" if definition else ""
        lines.append(f"      - {text}{detail}")

    return lines


def _object_in_gripper_preconditions(stage):
    """Return only the stage's object_in_gripper(...) preconditions.

    This is the one precondition the slip verdict needs: it names the object
    that is already meant to be in the gripper when the stage starts, which is
    exactly what the model has to look for in the frames. Every other
    precondition stays out.
    """
    return [
        entry
        for entry in stage.get("preconditions") or []
        if str(entry.get("condition", ""))
        .strip()
        .lower()
        .startswith("object_in_gripper")
    ]


def build_bt_waypoint_context(
    task_name,
    waypoint_index=None,
    waypoints_description_path=None,
):
    """Describe the scene objects and the requirements of the stage executing.

    Sends only the background needed to identify *what* is being manipulated:
    the scene objects and the current stage's object_in_gripper(...)
    precondition and hold conditions. The task goal is withheld too. Anything
    describing what the stage intends to achieve is withheld, because it let
    the model excuse a visible loss of grip as an authorised release, and so
    are the stage names and waypoint numbers, which locate the run in the task
    sequence and invite exactly the timing reasoning the prompts forbid.
    ``waypoint_index`` therefore selects the stage but is never rendered. Keep
    it that way when extending this.
    """
    if not task_name:
        return ""

    path = BT_CONDITIONS_DIR / f"{task_name}.bt_conditions.json"
    if not path.exists():
        return ""

    try:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)

        stages = data.get("generated", {}).get("stages", [])
        if not stages:
            return ""

        description = _load_task_description(data, waypoints_description_path)

        lines = [f"# Task: {task_name}"]

        objects = description.get("key_scene_objects", [])
        if objects:
            lines.append("Scene objects:")
            for obj in objects:
                role = obj.get("role", "")
                suffix = f" - {role}" if role else ""
                lines.append(f"  - {obj.get('name', '?')}{suffix}")

        current_stage = next(
            (
                stage
                for stage in stages
                if waypoint_index is not None
                and stage.get("stage") == waypoint_index
            ),
            None,
        )

        stage_lines = []
        if current_stage is not None:
            stage_lines.extend(
                _format_conditions(
                    "Object that must already be in the gripper when this "
                    "stage starts",
                    _object_in_gripper_preconditions(current_stage),
                )
            )
            stage_lines.extend(
                _format_conditions(
                    "Hold conditions (must hold throughout the stage)",
                    current_stage.get("hold_conditions", []),
                )
            )

        if stage_lines:
            lines.append("")
            lines.append("# Requirements of the stage being executed:")
            lines.extend(stage_lines)

        lines.append("")
        lines.append(
            "This context is background only: it tells you what the robot is "
            "manipulating. An object_in_gripper(x) line names the object that "
            "is already meant to be in the gripper when this stage starts - "
            "use it to know what to look for in the frames, not to reason "
            "about timing. Do NOT use any of this to decide whether a release "
            "would be appropriate here - the verdict is only whether the "
            "object left the gripper. If the frames show the object leaving "
            "the fingers, that is a slip."
        )

        return "\n".join(lines)

    except Exception:
        return ""


def _coerce_bool(value):
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        lowered = value.strip().lower()

        if lowered in ("true", "yes", "y", "slip", "1"):
            return True

        if lowered in ("false", "no", "n", "no_slip", "0"):
            return False

    return None


def _extract_partial_slip_happened(text):
    match = re.search(
        r'"?slip_happened"?\s*:\s*(true|false|"true"|"false")',
        text,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    return _coerce_bool(match.group(1).strip('"'))


def _load_json_object(text):
    """Parse the model's reply, tolerating prose wrapped around the JSON."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            return None

        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    return data if isinstance(data, dict) else None


def _extract_json(text):
    """Return the two fields both prompts ask for: slip_happened, explanation."""
    stripped = text.strip()
    data = _load_json_object(stripped)

    if data is None:
        return {
            "slip_happened": _extract_partial_slip_happened(stripped),
            "explanation": stripped[:200],
        }

    return {
        "slip_happened": _coerce_bool(data.get("slip_happened")),
        "explanation": str(data.get("explanation", "")).strip(),
    }


def format_telemetry_history(history, recent_steps=80):
    if not history:
        return ""

    window = history[-recent_steps:]

    header = (
        f"{'step':>5}  {'hold_phase':>10}  "
        f"{'grip_force':>10}  "
        f"{'force_drop':>10}  {'slip':>5}"
    )

    lines = [
        f"# Recent slip telemetry (last {len(window)} steps)",
        header,
        "-" * len(header),
    ]

    for row in window:
        lines.append(
            f"{row.get('step', 0):>5}  "
            f"{str(bool(row.get('holding_required_phase', False))):>10}  "
            f"{float(row.get('grip_force', 0.0)):>10.3f}  "
            f"{float(row.get('grip_force_drop', 0.0)):>10.3f}  "
            f"{str(bool(row.get('slip', False))):>5}"
        )

    return "\n".join(lines)


def format_detection_event(row):
    if not row:
        return ""

    return "\n".join(
        [
            "# Detector event being reviewed",
            (
                f"step={row.get('step', 'unknown')}  "
                f"grip_force={float(row.get('grip_force', 0.0)):.3f}  "
                f"grip_force_drop="
                f"{float(row.get('grip_force_drop', 0.0)):.3f}  "
                f"holding_required_phase="
                f"{bool(row.get('holding_required_phase', False))}"
            ),
            (
                "This row only identifies the moment that triggered the lightweight "
                "detector; it is not the final decision. This verifier is called only "
                "while continued object holding is required. Inspect the images first. "
                "If the object was securely held earlier but is no longer securely "
                "constrained by the gripper in the current frame, classify the event as "
                "a slip. An open gripper after an earlier secure grasp is evidence of "
                "slip, and an object that is merely near or touching the fingers but is "
                "no longer securely pinched is also evidence of slip. A closed gripper "
                "that still contains and constrains the object is not a slip. Use the "
                "force-drop signal only as supporting evidence when the visual evidence "
                "is ambiguous."
            ),
        ]
    )


def _write_vlm_trace(
    trace_dir,
    model,
    prompt_text,
    image_manifest,
    response_text,
    parsed_result,
):
    trace_dir = Path(trace_dir or VLM_TRACE_DIR)
    trace_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = trace_dir / f"slip_vlm_trace_{timestamp}.json"

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt_text": prompt_text,
        "image_manifest": image_manifest,
        "raw_response_text": response_text,
        "parsed_result": parsed_result,
    }

    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)

    return str(path)


def confirm_slip_with_openai(
    recent_obs_list,
    step_offsets=None,
    row=None,
    telemetry_history=None,
    env_wrapper=None,
    task_name=None,
    waypoint_index=None,
    waypoints_description_path=None,
    camera_names=DEFAULT_CAMERA_NAMES,
    model=DEFAULT_OPENAI_MODEL,
    max_output_tokens=800,
    preview_images=True,
    trace=False,
    trace_dir=VLM_TRACE_DIR,
):
    if model is None:
        model = DEFAULT_OPENAI_MODEL

    if not recent_obs_list:
        raise RuntimeError(
            "recent_obs_list must contain at least one observation."
        )

    if preview_images:
        show_camera_sequence(
            recent_obs_list,
            step_offsets=step_offsets,
            camera_names=camera_names,
            env_wrapper=env_wrapper,
            title="Images that will be sent to slip VLM",
        )

        try:
            import matplotlib.pyplot as plt

            print("  [vlm] Close the image window to continue...")
            plt.show(block=True)
        except Exception:
            pass

    saved_grid = ""
    grid_dir = os.getenv("AHA_SLIP_VLM_SAVE_GRID", "").strip()

    if grid_dir:
        tag = f"wp{waypoint_index}" if waypoint_index is not None else ""

        saved_grid = save_camera_grid(
            recent_obs_list,
            step_offsets,
            camera_names,
            env_wrapper,
            grid_dir,
            tag=tag,
        )

        if saved_grid:
            print(f"  [vlm:slip] camera grid saved -> {saved_grid}")

    try:
        import openai
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing Python package 'openai'. Install it or run in an "
            "environment where the OpenAI SDK is available."
        ) from exc

    prompt_text = "\n\n".join(
        part
        for part in (
            USER_PROMPT,
            format_detection_event(row),
            format_telemetry_history(telemetry_history),
            (
                build_bt_waypoint_context(
                    task_name,
                    waypoint_index,
                    waypoints_description_path,
                )
                if INCLUDE_WAYPOINT_CONTEXT
                else ""
            ),
        )
        if part
    )

    content = [{"type": "input_text", "text": prompt_text}]
    total = len(recent_obs_list)
    all_camera_names = []
    image_manifest = []

    for index, step_observation in enumerate(recent_obs_list):
        step_offset = (
            step_offsets[index]
            if step_offsets is not None and index < len(step_offsets)
            else total - 1 - index
        )

        label = (
            "current"
            if step_offset == 0
            else f"{step_offset} simulator steps ago"
        )

        content.append(
            {
                "type": "input_text",
                "text": f"## Step {index + 1} of {total} ({label})",
            }
        )

        step_images = collect_camera_images(
            step_observation,
            camera_names,
            env_wrapper=env_wrapper,
        )

        for camera_name, data_url in step_images:
            content.append(
                {
                    "type": "input_text",
                    "text": f"Camera: {camera_name}",
                }
            )
            content.append(
                {
                    "type": "input_image",
                    "image_url": data_url,
                }
            )
            image_manifest.append(
                {
                    "step_index": index + 1,
                    "step_offset": step_offset,
                    "camera_name": camera_name,
                }
            )

        if index == total - 1:
            all_camera_names = [
                camera_name
                for camera_name, _ in step_images
            ]

    if not all_camera_names:
        raise RuntimeError(
            "No camera images were available on the observation."
        )

    client = openai.OpenAI()

    effort = (
        os.getenv("AHA_DETECTOR_VLM_EFFORT", "low")
        .strip()
        .lower()
    )

    # Medium or higher reasoning consumes reasoning tokens from the output-token
    # budget. Give it additional headroom so the JSON response is not truncated.
    output_token_limit = (
        max_output_tokens
        if effort in ("minimal", "low")
        else max(max_output_tokens, 3000)
    )

    response = client.responses.create(
        model=model,
        max_output_tokens=output_token_limit,
        reasoning={"effort": effort},
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": content,
            },
        ],
    )

    usage = getattr(response, "usage", None)
    input_tokens = int(
        getattr(usage, "input_tokens", 0) or 0
    )
    output_tokens = int(
        getattr(usage, "output_tokens", 0) or 0
    )

    reasoning_tokens = 0
    output_details = getattr(
        usage,
        "output_tokens_details",
        None,
    )

    if output_details:
        reasoning_tokens = int(
            getattr(output_details, "reasoning_tokens", 0) or 0
        )

    print(
        f"  [vlm:slip] tokens: {input_tokens} in + "
        f"{output_tokens} out (reasoning: {reasoning_tokens})"
    )

    result = _extract_json(response.output_text)

    trace_path = None
    if trace:
        trace_path = _write_vlm_trace(
            trace_dir=trace_dir,
            model=model,
            prompt_text=prompt_text,
            image_manifest=image_manifest,
            response_text=response.output_text,
            parsed_result=result,
        )

    output = {
        "slip_happened": result["slip_happened"],
        "explanation": result["explanation"],
        "model": model,
        "camera_names": all_camera_names,
    }

    if trace_path:
        output["trace_path"] = trace_path

    if saved_grid:
        output["grid_path"] = saved_grid

    return output
