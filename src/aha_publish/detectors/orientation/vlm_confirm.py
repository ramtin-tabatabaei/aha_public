"""
OpenAI VLM confirmation for orientation detections.

This module reuses the camera/image helpers from the collision VLM module, but
uses an orientation-specific prompt and telemetry table.
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

def _resolve_confirm_cameras(default):
    """Allow an eval/batch run to shrink the camera set sent to the VLM (cost).

    Set AHA_VLM_CONFIRM_CAMERAS to a comma-separated list (e.g.
    ``side_rgb,wrist_rgb,front_rgb``) to override the production default.
    Unset -> the full production camera set is used (no behaviour change)."""
    sel = os.getenv('AHA_VLM_CONFIRM_CAMERAS', '').strip()
    if sel:
        return tuple(c.strip() for c in sel.split(',') if c.strip())
    return default


DEFAULT_CAMERA_NAMES = _resolve_confirm_cameras((
    'wrist_rgb',
    'wrist_depth',
    'side_rgb',
    'front_rgb',
    'overhead_rgb',
))
DEFAULT_OPENAI_MODEL = os.getenv(
    'OPENAI_ORIENTATION_VLM_MODEL',
    os.getenv('OPENAI_MODEL', 'gpt-5.4'),
)

SYSTEM_PROMPT = (
    "You are an orientation-failure auditor for an RLBench robot simulation. "
    "Your job is to decide whether the robot gripper, wrist, tool, or held "
    "object appears misoriented for the current waypoint. The image sequence "
    "starts at the beginning of the current waypoint and ends at the current "
    "flagged frame. Use the image sequence first and detector telemetry as "
    "supporting context.\n\n"

    "You must also decide whether the robot is making an UNNECESSARY "
    "orientation change: any visible wrist/tool/object twist, tilt, roll, "
    "yaw, or regrasp-like rotation that is not required by the waypoint. "
    "Unnecessary rotation is itself an orientation failure even when the "
    "gripper still reaches the correct position.\n\n"

    "How to detect unnecessary rotation across the image sequence:\n"
    "  - Compare wrist_rgb across frames. If the object stays roughly "
    "centered but its visible texture, seams, logo, label, or surface "
    "features rotate in the frame, the wrist is rolling about its approach "
    "axis (z-axis spin). For a stationary object, this means the wrist "
    "itself is spinning.\n"
    "  - Compare side_rgb and front_rgb across frames. If the end-effector "
    "position barely changes but its tilt or yaw shifts, the wrist is "
    "rotating in place.\n"
    "  - Compare overhead_rgb across frames for yaw changes with no "
    "translation.\n"
    "  - A consistent angle trend in telemetry (ttm_angle_increase, "
    "gripper_orientation_delta) combined with any visible rotation in the "
    "cameras is strong evidence of an in-place rotation.\n\n"

    "Rotational symmetry rule (IMPORTANT):\n"
    "  - If the target object is rotationally symmetric about the approach "
    "axis (balls, spheres, cylinders grasped from the end, disks, plates, "
    "cups grasped from the top rim, blocks being pushed, generic small "
    "items being picked from above), then ANY roll about the approach axis "
    "is unnecessary by definition. Mark it as an orientation failure even "
    "if the waypoint description does not explicitly forbid rotation.\n"
    "  - Spheres and balls (basketball, tennis ball, soccer ball, etc.) "
    "have full rotational symmetry. Any wrist roll, tilt, or yaw while "
    "approaching or holding a ball is unnecessary unless the waypoint "
    "explicitly requires aiming or pointing (e.g. a throw with a target "
    "direction). For a plain pick or reach toward a ball, treat any "
    "in-place rotation as a failure.\n\n"

    "When orientation changes ARE legitimate:\n"
    "  - Inserting plugs, keys, screws, or pegs along a specific axis.\n"
    "  - Aligning handles, hooks, lids, caps, or asymmetric parts.\n"
    "  - Pouring, scooping, or directional aiming tasks.\n"
    "  - Approaching grasp points on asymmetric objects where a specific "
    "wrist angle is needed to clear obstacles or match the geometry.\n"
    "  - The waypoint description explicitly mentions an orientation goal.\n\n"

    "Camera-specific guidance:\n"
    "  - wrist_rgb: gripper/tool pose, roll about approach axis (watch for "
    "object features rotating in the frame).\n"
    "  - wrist_depth: approach distance and centering.\n"
    "  - side_rgb: approach axis and tilt.\n"
    "  - front_rgb: overall alignment and tilt.\n"
    "  - overhead_rgb: yaw and lateral orientation.\n\n"

    "Confirm an orientation failure when ANY of the following hold:\n"
    "  (a) The current frame shows a visibly wrong approach angle, rotated "
    "tool/object, or misaligned insertion axis for the described task; or\n"
    "  (b) The waypoint requires a specific orientation and the robot has "
    "failed to rotate to it; or\n"
    "  (c) The robot has performed an in-place rotation that the waypoint "
    "does not need (especially on a symmetric object or a plain "
    "pick/place/reach).\n\n"

    "Do NOT confirm when the orientation change is clearly compatible with "
    "the described task (e.g. aligning a handle), or when evidence is only "
    "a small numeric angle trend with no visible rotation across frames. "
    "Return JSON only."
)

USER_PROMPT = (
    "The orientation detector flagged a possible failure. Review the camera "
    "sequence from the beginning of this waypoint through the current frame, "
    "plus telemetry. Decide whether the CURRENT frame visually supports the "
    "orientation failure, including whether the robot is making an unnecessary "
    "orientation change for this waypoint.\n\n"

    "Before answering, explicitly check:\n"
    "  1. Across wrist_rgb frames, does the object's texture/seams/features "
    "rotate while the object stays centered? If yes, the wrist is rolling.\n"
    "  2. Across side_rgb and front_rgb frames, does the end-effector "
    "position stay roughly fixed while its tilt or yaw changes? If yes, "
    "the wrist is rotating in place.\n"
    "  3. Is the target object rotationally symmetric (ball, sphere, "
    "cylinder, disk, generic small item)? If yes, any in-place rotation "
    "is unnecessary.\n"
    "  4. Does the waypoint description explicitly require a specific "
    "orientation? If no, in-place rotation is unnecessary.\n\n"

    "Use waypoint_path_done, expected_waypoint, "
    "waypoint, waypoint_angle, live_waypoint_angle, ttm_waypoint_angle, "
    "ttm_angle_increase, gripper_orientation_delta, and the waypoint "
    "description as context.\n\n"

    "Return JSON with exactly two keys: `orientation_failure_happened` "
    "(true/false) and `explanation` (one concise sentence citing the "
    "clearest camera evidence first, then telemetry if relevant; mention "
    "unnecessary rotation if that is the reason)."
)

TRACE_USER_PROMPT = (
    "The orientation detector flagged a possible failure. Review the camera "
    "sequence from the beginning of this waypoint through the current frame, "
    "plus telemetry. Decide whether the CURRENT frame visually supports the "
    "orientation failure, including whether the robot is making an unnecessary "
    "orientation change for this waypoint.\n\n"

    "Before answering, explicitly check:\n"
    "  1. Across wrist_rgb frames, does the object's texture/seams/features "
    "rotate while the object stays centered? If yes, the wrist is rolling "
    "about its approach axis.\n"
    "  2. Across side_rgb and front_rgb frames, does the end-effector "
    "position stay roughly fixed while its tilt or yaw changes? If yes, "
    "the wrist is rotating in place.\n"
    "  3. Is the target object rotationally symmetric (ball, sphere, "
    "cylinder, disk, generic small item)? If yes, any in-place rotation "
    "is unnecessary.\n"
    "  4. Does the waypoint description explicitly require a specific "
    "orientation? If no, in-place rotation is unnecessary.\n\n"

    "Use waypoint_path_done, expected_waypoint, "
    "waypoint, waypoint_angle, live_waypoint_angle, ttm_waypoint_angle, "
    "ttm_angle_increase, gripper_orientation_delta, and the waypoint "
    "description as context.\n\n"

    "Return JSON with these keys: `orientation_failure_happened` "
    "(true/false), `explanation` (one concise sentence), `visual_evidence` "
    "(concise visible observations across frames, citing camera names and "
    "specifically noting whether object texture/features rotate in "
    "wrist_rgb), `unnecessary_orientation_assessment` (whether the visible "
    "rotation is required by the waypoint or appears extra/unhelpful, and "
    "whether the target object is rotationally symmetric), "
    "`telemetry_evidence` (concise detector-signal summary), and "
    "`decision_basis` (brief final rationale using observable evidence "
    "only)."
)


def _load_collision_helpers():
    spec = importlib.util.spec_from_file_location(
        '_aha_collision_vlm_helpers',
        COLLISION_VLM_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_helpers = _load_collision_helpers()
show_camera_sequence = _helpers.show_camera_sequence
collect_camera_images = _helpers.collect_camera_images
build_context_text = _helpers.build_context_text


def sample_waypoint_observations(waypoint_obs_list, max_images=5):
    """Sample from waypoint start through the current frame."""
    total = len(waypoint_obs_list)
    if total <= 0:
        return [], []
    if total <= max_images:
        indexes = list(range(total))
    elif max_images <= 1:
        indexes = [total - 1]
    else:
        raw_indexes = [
            round(i * (total - 1) / float(max_images - 1))
            for i in range(max_images)
        ]
        indexes = []
        for index in raw_indexes:
            index = int(index)
            if index not in indexes:
                indexes.append(index)
        if indexes[-1] != total - 1:
            indexes[-1] = total - 1

    sampled = [waypoint_obs_list[index] for index in indexes]
    step_offsets = [total - 1 - index for index in indexes]
    return sampled, step_offsets


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ('true', 'yes', 'y', 'orientation', 'failure', '1'):
            return True
        if lowered in ('false', 'no', 'n', 'no_orientation', 'no_failure', '0'):
            return False
    return None


def _extract_partial_verdict(text):
    match = re.search(
        r'"?orientation_failure_happened"?\s*:\s*(true|false|"true"|"false")',
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return _coerce_bool(match.group(1).strip('"'))


def _extract_json(text):
    stripped = text.strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find('{')
        end = stripped.rfind('}')
        if start == -1 or end == -1 or end <= start:
            return {
                'orientation_failure_happened': _extract_partial_verdict(stripped),
                'explanation': stripped[:200],
            }
        try:
            data = json.loads(stripped[start:end + 1])
        except json.JSONDecodeError:
            return {
                'orientation_failure_happened': _extract_partial_verdict(stripped),
                'explanation': stripped[:200],
            }

    return {
        'orientation_failure_happened': _coerce_bool(
            data.get('orientation_failure_happened')
        ),
        'explanation': str(data.get('explanation', '')).strip(),
        'visual_evidence': data.get('visual_evidence'),
        'unnecessary_orientation_assessment': data.get(
            'unnecessary_orientation_assessment'
        ),
        'telemetry_evidence': data.get('telemetry_evidence'),
        'decision_basis': data.get('decision_basis'),
    }


def _float_text(value):
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "nan"


def format_telemetry_history(history):
    if not history:
        return ''

    lines = [
        "# Orientation telemetry history (all steps)",
        (
            f"{'step':>5}  {'wp':>3}  {'exp':>3}  {'done':>5}  "
            f"{'angle':>8}  {'live':>8}  {'ttm':>8}  {'inc':>8}  "
            f"{'g_delta':>8}  {'fail':>5}"
        ),
        "-" * 82,
    ]
    for row in history:
        lines.append(
            f"{row.get('step', 0):>5}  "
            f"{str(row.get('waypoint', '')):>3}  "
            f"{str(row.get('expected_waypoint', '')):>3}  "
            f"{str(bool(row.get('waypoint_path_done', False))):>5}  "
            f"{_float_text(row.get('waypoint_angle')):>8}  "
            f"{_float_text(row.get('live_waypoint_angle')):>8}  "
            f"{_float_text(row.get('ttm_waypoint_angle')):>8}  "
            f"{_float_text(row.get('ttm_angle_increase')):>8}  "
            f"{_float_text(row.get('gripper_orientation_delta')):>8}  "
            f"{str(bool(row.get('orientation_failure', False))):>5}"
        )
    return "\n".join(lines)


def format_detection_event(row):
    if not row:
        return ''

    return "\n".join([
        "# Detector event being reviewed",
        (
            f"step={row.get('step', 'unknown')}  "
            f"waypoint={row.get('waypoint', 'unknown')}  "
            f"expected_waypoint={row.get('expected_waypoint', 'unknown')}  "
            f"waypoint_path_done={bool(row.get('waypoint_path_done', False))}  "
            f"reason={row.get('orientation_reason', '') or 'unknown'}  "
            f"score={_float_text(row.get('orientation_score'))}  "
            f"source={row.get('waypoint_angle_source', 'unknown')}  "
            f"waypoint_angle={_float_text(row.get('waypoint_angle'))}  "
            f"live_waypoint_angle={_float_text(row.get('live_waypoint_angle'))}  "
            f"ttm_waypoint_angle={_float_text(row.get('ttm_waypoint_angle'))}  "
            f"ttm_angle_increase={_float_text(row.get('ttm_angle_increase'))}  "
            f"gripper_orientation_delta={_float_text(row.get('gripper_orientation_delta'))}  "
            f"waypoint_orientation_gap={_float_text(row.get('waypoint_orientation_gap'))}"
        ),
        (
            "Confirm the detector only if the images support a visible "
            "orientation mismatch for the described waypoint, including an "
            "unnecessary twist/tilt/roll/yaw that the waypoint does not require. "
            "Remember: for rotationally symmetric objects (balls, spheres, "
            "cylinders, disks) any in-place rotation is unnecessary by "
            "definition. If the waypoint context makes the orientation change "
            "useful for alignment (handles, plugs, lids, asymmetric parts), or "
            "if the gripper/object appears aligned with the task despite the "
            "numeric trend, mark the VLM check unconfirmed."
        ),
    ])


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
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    path = trace_dir / f'orientation_vlm_trace_{timestamp}.json'
    payload = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'model': model,
        'system_prompt': SYSTEM_PROMPT,
        'user_prompt_text': prompt_text,
        'image_manifest': image_manifest,
        'raw_response_text': response_text,
        'parsed_result': parsed_result,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return str(path)


def confirm_orientation_with_openai(
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
        raise RuntimeError("recent_obs_list must contain at least one observation.")

    if preview_images:
        show_camera_sequence(
            recent_obs_list,
            step_offsets=step_offsets,
            camera_names=camera_names,
            env_wrapper=env_wrapper,
            title='Images that will be sent to orientation VLM',
        )
        input("  Inspect the camera images, then press Enter to send them to VLM...")

    try:
        import openai
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing Python package 'openai'. Install it or run in an environment "
            "where the OpenAI SDK is available."
        ) from exc

    prompt_text = "\n\n".join(
        part
        for part in (
            TRACE_USER_PROMPT if trace else USER_PROMPT,
            format_detection_event(row),
            format_telemetry_history(telemetry_history),
            build_context_text(
                task_name=task_name,
                waypoint_index=waypoint_index,
                description_path=waypoints_description_path,
            ),
        )
        if part
    )

    content = [{'type': 'input_text', 'text': prompt_text}]
    total = len(recent_obs_list)
    all_camera_names = []
    image_manifest = []
    for i, step_obs in enumerate(recent_obs_list):
        step_offset = (
            step_offsets[i]
            if step_offsets is not None and i < len(step_offsets)
            else total - 1 - i
        )
        label = 'current' if step_offset == 0 else f'{step_offset} simulator steps ago'
        content.append({
            'type': 'input_text',
            'text': f"## Step {i + 1} of {total} ({label})",
        })
        step_images = collect_camera_images(
            step_obs,
            camera_names,
            env_wrapper=env_wrapper,
        )
        for camera_name, data_url in step_images:
            content.append({'type': 'input_text', 'text': f'Camera: {camera_name}'})
            content.append({'type': 'input_image', 'image_url': data_url})
            image_manifest.append({
                'step_index': i + 1,
                'step_offset': step_offset,
                'camera_name': camera_name,
            })
        if i == total - 1:
            all_camera_names = [name for name, _ in step_images]

    if not all_camera_names:
        raise RuntimeError("No camera images were available on the observation.")

    client = openai.OpenAI()
    response = client.responses.create(
        model=model,
        max_output_tokens=max_output_tokens,
        input=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': content},
        ],
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
        'orientation_failure_happened': result['orientation_failure_happened'],
        'explanation': result['explanation'],
        'model': model,
        'camera_names': all_camera_names,
    }
    for key in (
        'visual_evidence',
        'unnecessary_orientation_assessment',
        'telemetry_evidence',
        'decision_basis',
    ):
        if result.get(key):
            output[key] = result[key]
    if trace_path:
        output['trace_path'] = trace_path
    return output