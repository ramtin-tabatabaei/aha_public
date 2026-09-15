"""
OpenAI VLM confirmation for transition detections.

This module reuses the camera/image helpers from the collision VLM module, but
uses a transition-specific prompt and telemetry table.
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
    'OPENAI_TRANSITION_VLM_MODEL',
    os.getenv('OPENAI_MODEL', 'gpt-5.4'),
)

SYSTEM_PROMPT = (
    "You are a transition-failure auditor for an RLBench robot simulation. "
    "Your job is to decide whether the robot appears to be at the wrong "
    "spatial waypoint, taking waypoints out of order, or failing to reach the "
    "expected waypoint position. Use the image sequence first and the detector "
    "telemetry as supporting context. Compare wrist_rgb, side_rgb, front_rgb, "
    "and overhead_rgb across time. side_rgb is especially useful for seeing "
    "whether the gripper is above, beside, or past the intended object or goal. "
    "Do not mark a transition failure for normal approach motion before a "
    "waypoint is complete. Confirm only when the current frame visibly supports "
    "the flagged issue, such as the gripper being clearly offset from the "
    "intended object/goal, moving to the wrong object/goal, skipping a waypoint, "
    "or ending a waypoint far from the described target. Return JSON only."
)

USER_PROMPT = (
    "The transition detector flagged a possible failure. Review the recent "
    "camera sequence and telemetry. Decide whether the CURRENT frame visually "
    "supports the transition failure. Use waypoint_path_done, expected_waypoint, "
    "waypoint, waypoint_distance, live_waypoint_distance, ttm_waypoint_distance, "
    "and the waypoint description as context. Return JSON with exactly two "
    "keys: `transition_failure_happened` (true/false) and `explanation` "
    "(one concise sentence citing the clearest camera evidence first, then "
    "telemetry if relevant)."
)

TRACE_USER_PROMPT = (
    "The transition detector flagged a possible failure. Review the recent "
    "camera sequence and telemetry. Decide whether the CURRENT frame visually "
    "supports the transition failure. Use waypoint_path_done, expected_waypoint, "
    "waypoint, waypoint_distance, live_waypoint_distance, ttm_waypoint_distance, "
    "and the waypoint description as context. Return JSON with these keys: "
    "`transition_failure_happened` (true/false), `explanation` (one concise "
    "sentence), `visual_evidence` (concise visible observations, citing camera "
    "names where relevant), `telemetry_evidence` (concise detector-signal "
    "summary), and `decision_basis` (brief final rationale using observable "
    "evidence only)."
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
sample_recent_observations = _helpers.sample_recent_observations
show_camera_sequence = _helpers.show_camera_sequence
collect_camera_images = _helpers.collect_camera_images
build_context_text = _helpers.build_context_text


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ('true', 'yes', 'y', 'transition', 'failure', '1'):
            return True
        if lowered in ('false', 'no', 'n', 'no_transition', 'no_failure', '0'):
            return False
    return None


def _extract_partial_verdict(text):
    match = re.search(
        r'"?transition_failure_happened"?\s*:\s*(true|false|"true"|"false")',
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
                'transition_failure_happened': _extract_partial_verdict(stripped),
                'explanation': stripped[:200],
            }
        try:
            data = json.loads(stripped[start:end + 1])
        except json.JSONDecodeError:
            return {
                'transition_failure_happened': _extract_partial_verdict(stripped),
                'explanation': stripped[:200],
            }

    return {
        'transition_failure_happened': _coerce_bool(
            data.get('transition_failure_happened')
        ),
        'explanation': str(data.get('explanation', '')).strip(),
        'visual_evidence': data.get('visual_evidence'),
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
        "# Transition telemetry history (all steps)",
        (
            f"{'step':>5}  {'wp':>3}  {'exp':>3}  {'done':>5}  "
            f"{'dist':>8}  {'live':>8}  {'ttm':>8}  {'gap':>8}  "
            f"{'fail':>5}"
        ),
        "-" * 72,
    ]
    for row in history:
        lines.append(
            f"{row.get('step', 0):>5}  "
            f"{str(row.get('waypoint', '')):>3}  "
            f"{str(row.get('expected_waypoint', '')):>3}  "
            f"{str(bool(row.get('waypoint_path_done', False))):>5}  "
            f"{_float_text(row.get('waypoint_distance')):>8}  "
            f"{_float_text(row.get('live_waypoint_distance')):>8}  "
            f"{_float_text(row.get('ttm_waypoint_distance')):>8}  "
            f"{_float_text(row.get('waypoint_pose_gap')):>8}  "
            f"{str(bool(row.get('transition_failure', False))):>5}"
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
            f"reason={row.get('transition_reason', '') or 'unknown'}  "
            f"score={_float_text(row.get('transition_score'))}  "
            f"source={row.get('waypoint_distance_source', 'unknown')}  "
            f"waypoint_distance={_float_text(row.get('waypoint_distance'))}  "
            f"live_waypoint_distance={_float_text(row.get('live_waypoint_distance'))}  "
            f"ttm_waypoint_distance={_float_text(row.get('ttm_waypoint_distance'))}  "
            f"waypoint_pose_gap={_float_text(row.get('waypoint_pose_gap'))}"
        ),
        (
            "Confirm the detector only if the images support wrong waypoint "
            "order or a spatial miss at the current/completed waypoint. If "
            "the robot is still naturally approaching and waypoint_path_done "
            "is false, treat the event as unconfirmed unless the images show "
            "a clear wrong target or skipped transition."
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
    path = trace_dir / f'transition_vlm_trace_{timestamp}.json'
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


def confirm_transition_with_openai(
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
            title='Images that will be sent to transition VLM',
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
        'transition_failure_happened': result['transition_failure_happened'],
        'explanation': result['explanation'],
        'model': model,
        'camera_names': all_camera_names,
    }
    for key in ('visual_evidence', 'telemetry_evidence', 'decision_basis'):
        if result.get(key):
            output[key] = result[key]
    if trace_path:
        output['trace_path'] = trace_path
    return output
