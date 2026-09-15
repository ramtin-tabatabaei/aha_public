"""
OpenAI VLM confirmation for collision detections.

This module is intentionally optional: it only imports the OpenAI SDK when the
user asks to confirm a detector hit with the VLM.
"""

from aha_publish import paths
import base64
from datetime import datetime
import io
import json
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image


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
    'side_rgb',
    'front_rgb',
))

# Frames sent to the VLM, oldest first, as offsets in simulator steps back from
# the detector event. Interpolated into USER_PROMPT below so the prompt's
# description of the sequence can never drift from what is actually sent.
RECENT_IMAGE_OFFSETS = (5, 0)
_FRAME_LABELS = " to ".join(
    'current' if offset == 0 else f'-{offset}'
    for offset in RECENT_IMAGE_OFFSETS
)

REPO_ROOT = (paths.PROJECT_ROOT)
WAYPOINTS_DESCRIPTION_DIR = (paths.DESCRIPTION_DIR)
CALIBRATION_ROOT = Path(os.getenv(
    'AHA_CALIBRATION_ROOT',
    str(paths.CALIBRATION_DIR),
)).expanduser()
VLM_TRACE_DIR = (paths.OUTPUT_DIR / 'vlm_traces')

# --- Method-3 (De Luca generalized-momentum observer) residual context -------
# The detector fires on the momentum-observer residual r (one value per joint,
# an estimate of the EXTERNAL torque on that joint), not on raw torque_norm, so
# that is what the verifier is shown: all seven residuals against the per-joint
# envelope calibrated from clean successful runs. These mirror detector.py's
# DEFAULT_RESIDUAL_* knobs so the thresholds quoted here are the ones the
# detector actually applied.
RESIDUAL_STATS_DIR = Path(os.getenv(
    'AHA_RESIDUAL_STATS_DIR',
    str(CALIBRATION_ROOT / 'residual_stats'),
)).expanduser()
RESIDUAL_FLOOR = float(os.getenv('AHA_COLLISION_RESIDUAL_FLOOR', '3.0'))
RESIDUAL_N_JOINTS = 7


def _residual_joint_mult_for(n_joints):
    """The detector's per-joint threshold multiplier, parsed the same way.

    Kept in step with detector._residual_joint_mult so the verifier is shown the
    threshold the detector actually applied; a mismatch here is invisible in a
    run and silently misinforms the VLM.
    """
    try:
        vals = [float(v) for v in os.getenv(
            'AHA_COLLISION_RESIDUAL_JOINT_MULT',
            '5.0,1.1,1.1,1.1,1.1,2.0,1.1').split(',')]
    except ValueError:
        vals = []
    if len(vals) != RESIDUAL_N_JOINTS or any(v <= 0.0 for v in vals):
        vals = [1.0] * RESIDUAL_N_JOINTS
    return vals[:n_joints] + [1.0] * max(0, n_joints - len(vals))

# Waypoint context is ON by default, but it is deliberately MINIMAL and comes
# entirely from the BT: the stage's primitive, its pre/postconditions, and the
# target object. Nothing from the description JSON reaches this prompt — its
# prose robot_action made the model treat "I cannot see that action complete"
# as collision evidence, and the full JSON also leaks other waypoints'
# expectations. Set AHA_VLM_WAYPOINT_CONTEXT=0 to drop the block entirely.
INCLUDE_WAYPOINT_CONTEXT = os.getenv('AHA_VLM_WAYPOINT_CONTEXT', '1').strip() in ('1', 'true', 'yes')

DEFAULT_OPENAI_MODEL = os.getenv(
    'OPENAI_COLLISION_VLM_MODEL',
    os.getenv('OPENAI_MODEL', 'gpt-5.4'),
)

SYSTEM_PROMPT = (
    "You are a collision auditor for an RLBench robot arm simulation. "
    "Decide whether an unintended collision occurred using camera images and telemetry.\n\n"
    "IMAGE SEQUENCE: Frames are in time order (earliest first). Compare across frames, "
    "not just the final frame. Expected manipulation (moving, grasping, transporting, placing, "
    "releasing, intentional pushing) shows a visible state change: gripper closing around "
    "the target, object being lifted/moved/placed. "
    "Look for unintended contact: collision with non-target objects, the table, fixtures, "
    "or robot body; trajectory discontinuity caused by contact; gripper/fingers pressed "
    "into the table.\n\n"
    "SLIP / FAILED-GRASP EXCLUSION: Do NOT mark collision when the only visible failure is "
    "that the target object was not captured, slipped from the fingers, slid on the table, "
    "or was left behind during a pickup. That is a grasp/slip failure, not a collision, "
    "unless there is separate positive visual evidence that the robot struck the table, "
    "a fixture, a non-target object, its own body, or visibly shoved the target out of place. "
    "Do not infer table contact merely because the target remains on or slides across the "
    "table after a failed grasp.\n\n"
    "RIGID BODIES: Every object in this simulation is rigid. Nothing bends, flexes, "
    "compresses, or deforms, so do not require shape change as evidence. Unintended "
    "contact also does NOT require object displacement: the table, a fixed fixture, or "
    "a blocked object can remain stationary while the robot or a held object presses "
    "against it, grazes it, or is stopped by it. Judge visible contact geometry, "
    "contact-induced stopping or deflection, unexpected position/orientation changes, "
    "and the residual evidence together. Unchanged object poses alone do not establish "
    "that no collision occurred; proximity or lack of motion alone does not establish "
    "that one did. Distinguish unintended contact from contact required by the current "
    "manipulation primitive.\n\n"
    "UNEXPECTED OBJECT DISPLACEMENT: If an object visibly tips, topples, "
    "is knocked over, or is shoved from its resting pose when such motion "
    "is not required by the current manipulation primitive, treat this as "
    "evidence of unintended contact, even if the fingers are "
    "still open and even if you cannot see the exact contact point. Compare the object's "
    "pose across the frame sequence: a target that is upright early and toppled/leaning in "
    "the current frame means the arm rammed it. Do NOT dismiss this as 'aligning' just "
    "because the gripper is open and near the target; alignment does not move the target.\n\n"
    "RESIDUAL (the detector's actual signal): This detector uses a De Luca generalized-"
    "momentum observer. For each of the 7 joints it computes a residual r_i — the external "
    "torque on that joint that the arm's own dynamics (inertia, Coriolis, gravity) cannot "
    "explain. During clean successful executions, including free motion, grasping, "
    "and carrying, the residuals should normally remain within the envelopes "
    "measured during calibration. Unintended external contact can drive one "
    "or more residuals beyond these calibrated thresholds. "
    "Multiple joints exceeding their calibrated thresholds strengthens the evidence "
    "for unintended contact, since an external contact wrench can affect several joints "
    "through the manipulator kinematics. A sufficiently large exceedance on a single joint "
    "can also be meaningful, whereas a single joint only marginally above threshold is weak "
    "evidence on its own. If the images are ambiguous but several joints are clearly over "
    "threshold, or one joint is far above its threshold, treat this as strong evidence of "
    "unintended contact unless the images clearly show a normal grasp/place action or a pure "
    "slip/failed-grasp event with no separate contact with the table, fixtures, robot body, "
    "non-target objects, or unintended target displacement. Weigh the residual evidence "
    "together with the images.\n\n"

    "RETRACTION DISCIPLINE: You are auditing an activation from a specialized collision detector. "
    "Overturn it (collision_happened=false) when the images provide clear positive evidence that "
    "the robot is not making unintended contact. This includes frames that clearly show a normal, "
    "completed manipulation (a clean grasp, place, close, or unobstructed free-space motion), a "
    "pure target slip/failed grasp without separate collision evidence, or a robot/gripper that is "
    "visibly separated from the table, fixtures, non-target objects, and robot body with no object "
    "being struck, pushed, displaced, or blocked by contact. If the multi-view images clearly show "
    "that the robot is moving in free space and is not contacting or hitting anything, retract the "
    "detector activation even if the residual is elevated. Do NOT retract merely because no dramatic "
    "strike, deflection, or ejection is visible when contact itself is occluded or ambiguous: subtle "
    "collisions such as brushing a fixture, lightly nudging an object, grazing the table, or pressing "
    "into a stuck object may produce only small visible effects or no object displacement. "
    "A clear final frame does not erase a collision visible earlier in the sequence. "
    "The sampled frames may omit a brief impact between them; do not treat final-frame "
    "separation or an unchanged scene as proof that the entire interval was contact-free. "
    "For an ambiguous interval, weigh the residual evidence using the rules above "
    "without inventing an unseen contact.\n\n"

    "IMPORTANT FAILURE CUES:\n"
    "- Failed pickup at table height is NOT collision by itself. Mark collision only if "
    "the fingers visibly press into/graze the table, hit a fixture/non-target/robot body, "
    "or shove/topple the target beyond ordinary slipping or rolling.\n"
    "- HELD-OBJECT COLLISION: If the robot is already holding the target object and the held "
    "object strikes, pushes, tips, displaces, or otherwise makes unintended contact with another "
    "object, fixture, table, or robot body, classify this as a collision. The collision does not "
    "need to occur directly through the gripper or robot links; contact caused by the object being "
    "transported counts as an unintended collision. Do not confuse this with normal placement or "
    "intentional object-object contact required by the task.\n"
    "- Do not require the exact contact point to be visible when there is clear secondary "
    "evidence such as a fixture or non-target knocked out of place, the target being "
    "toppled/shoved, or several joint residuals far over threshold paired with non-slip "
    "visual cues. Pure target slipping/sliding is not enough.\n"
    "- Do not dismiss a collision as normal alignment just because the gripper is open or "
    "near the target. Alignment should not tip, topple, eject, or shove fixtures/non-targets and should "
    "not drive joint residuals past their clean-run thresholds.\n\n"
    "CAMERA PRIORITY: wrist_rgb (direct gripper contact) → side_rgb (lateral) "
    "→ front_rgb (approach).\n\n"
    "Return strict JSON only: "
    "{\"collision_happened\": true/false, "
    "\"explanation\": \"one sentence citing image evidence first, then the residuals if relevant\"}"
)

USER_PROMPT = (
    "Camera images, the per-joint momentum-observer residuals, their calibrated "
    "clean-run thresholds, and recent residual history are provided below. "
    f"Compare the image sequence ({_FRAME_LABELS}); each frame is labelled with "
    "its age relative to the detector event. Evaluate the candidate collision "
    "according to the collision-auditing rules in the system instruction. "
    "Return JSON: {\"collision_happened\": true/false, "
    "\"explanation\": \"one sentence citing image evidence first, then residual evidence if relevant\"}"
)


def _load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _find_residual_stats_path(task_name, stats_dir=RESIDUAL_STATS_DIR):
    if not task_name:
        return None
    suffix = '_v2' if os.getenv('AHA_COLLISION_MOMENTUM_VERSION', '1').strip() == '2' else ''
    path = Path(stats_dir) / f'{task_name}_residual_stats{suffix}.json'
    return path if path.exists() else None


def load_residual_stats_context(task_name=None, stats_dir=RESIDUAL_STATS_DIR):
    """Clean-run per-joint residual envelope + the thresholds derived from it.

    Mirrors detector.residual_threshold_vector: thr_i = max(FLOOR, clean_i) * MULT_i.
    Returns None when the task has no calibrated residual stats, in which case
    the detector fell back to a scalar warmup threshold and the prompt says so.
    """
    path = _find_residual_stats_path(task_name, stats_dir=stats_dir)
    if path is None:
        return None

    try:
        data = _load_json(path)
    except (OSError, json.JSONDecodeError):
        return None

    if os.getenv('AHA_COLLISION_MOMENTUM_VERSION', '1').strip() == '2' and (
        data.get('observer_version') != '2'
        or data.get('gain') != float(os.getenv('AHA_COLLISION_MOMENTUM_GAIN', '25.0'))
        or data.get('dt') != float(os.getenv('AHA_SIM_DT', '0.05'))
    ):
        return None

    clean = [float(v) for v in (data.get('per_joint_max') or [])]
    if not clean:
        return None

    return {
        'source_path': str(path),
        'gain': data.get('gain'),
        'dt': data.get('dt'),
        'clean_per_joint_max': clean,
        'per_joint_threshold': [
            max(RESIDUAL_FLOOR, v) * m
            for v, m in zip(clean, _residual_joint_mult_for(len(clean)))
        ],
        'floor': RESIDUAL_FLOOR,
        'mult': list(_residual_joint_mult_for(len(clean))),
    }


def _residual_vector(row):
    """The 7 per-joint residuals |r_i| on a detector row, or None."""
    if not row:
        return None
    vec = row.get('mo_residual')
    if vec is None:
        return None
    try:
        values = [abs(float(v)) for v in np.asarray(vec).ravel()]
    except (TypeError, ValueError):
        return None
    return values or None


def build_residual_context_text(task_name=None, row=None):
    """All seven momentum-observer residuals against their clean baselines.

    This is the collision detector's actual firing signal under method 3, so it
    replaces the old torque_norm-vs-torque-stats block: showing the verifier a
    torque baseline while the detector fired on a residual meant the two were
    reasoning about different quantities."""
    stats = load_residual_stats_context(task_name=task_name)
    values = _residual_vector(row)

    lines = [
        "# Collision signal: De Luca generalized-momentum residual (method 3)",
        "r_i is the observer's estimate of the EXTERNAL torque on joint i (Nm) — "
        "the torque that the arm's own dynamics (inertia, Coriolis, gravity) do "
        "NOT explain. On a clean successful run every |r_i| stays inside the "
        "calibrated envelope below; contact with the world is what pushes a "
        "joint past it. This residual, not raw joint torque, is what fired the "
        "detector.",
    ]

    if stats:
        lines.append(
            f"Observer: gain K_I={stats['gain']} 1/s, dt={stats['dt']} s. "
            f"Per-joint threshold = max(floor {stats['floor']:.2f} Nm, clean_max) "
            f"x per-joint multiplier "
            f"[{' '.join(f'{m:g}' for m in stats['mult'])}]."
        )
        lines.append(f"Baseline source (clean successful runs): {stats['source_path']}")
        thresholds = stats['per_joint_threshold']
        clean = stats['clean_per_joint_max']
    else:
        thresholds = None
        clean = None
        scalar = row.get('residual_threshold') if row else None
        lines.append(
            "Baseline: no calibrated per-joint residual stats for this task; the "
            "detector used a scalar warmup threshold"
            + (f" of {float(scalar):.2f} Nm." if scalar is not None else ".")
        )

    header = f"{'joint':>6}  {'|r_i| now':>10}"
    if clean is not None:
        header += f"  {'clean_max':>10}  {'threshold':>10}  {'x_thr':>7}  {'over':>5}"
    lines.append(header)
    lines.append("-" * len(header))

    n = max(
        len(values) if values else 0,
        len(clean) if clean else 0,
        RESIDUAL_N_JOINTS if not values and not clean else 0,
    )
    for i in range(n):
        value = values[i] if values and i < len(values) else None
        line = f"{i + 1:>6}  " + (f"{value:>10.2f}" if value is not None else f"{'n/a':>10}")
        if clean is not None:
            clean_i = clean[i] if i < len(clean) else float('nan')
            thr_i = thresholds[i] if i < len(thresholds) else float('nan')
            ratio = (value / thr_i) if (value is not None and thr_i) else None
            line += (
                f"  {clean_i:>10.2f}  {thr_i:>10.2f}  "
                + (f"{ratio:>7.2f}" if ratio is not None else f"{'n/a':>7}")
                + "  "
                + (f"{('YES' if value > thr_i else 'no'):>5}"
                   if value is not None else f"{'n/a':>5}")
            )
        lines.append(line)

    if row:
        lines.append(
            f"residual_norm ||r|| = {float(row.get('residual_norm', 0.0)):.2f} Nm   "
            f"max_i |r_i| = {float(row.get('residual_score', 0.0)):.2f} Nm   "
            f"joints over threshold = {int(row.get('residual_joints_over', 0))}"
            f" of {len(values) if values else RESIDUAL_N_JOINTS}"
        )

    lines.append(
        "Interpretation: a joint marked 'over' carries external torque that the "
        "clean-run envelope cannot explain. A real end-effector collision loads "
        "the kinematic chain, so SEVERAL joints usually cross at once and x_thr "
        "runs well above 1 — that is strong collision evidence even when the "
        "contact point is subtle or occluded in the frames. One joint barely "
        "over (x_thr close to 1) is weak on its own. Residuals near or below "
        "their thresholds mean the motion is explained by the arm's own "
        "dynamics: normal free motion, grasping, or carrying. Note the wrist "
        "joints (6, 7) have small envelopes, so they cross on light contact. "
        "Image evidence still has priority."
    )
    return "\n".join(lines)


def _find_waypoints_description_path(task_name, description_path=None):
    if description_path:
        path = Path(description_path)
        return path if path.exists() else None
    if not task_name:
        return None

    preferred = [
        WAYPOINTS_DESCRIPTION_DIR
        / f'{task_name}_ALL_WAYPOINTS_COMBINED.openai.multimodal_analysis.json',
        WAYPOINTS_DESCRIPTION_DIR
        / f'{task_name}_ALL_WAYPOINTS_COMBINED.openai.analysis.json',
    ]
    for path in preferred:
        if path.exists():
            return path

    matches = sorted(
        WAYPOINTS_DESCRIPTION_DIR.glob(
            f'{task_name}_ALL_WAYPOINTS_COMBINED*.json'
        )
    )
    return matches[0] if matches else None


def _waypoint_entry(data, waypoint_index):
    if waypoint_index is None:
        return None
    for entry in data.get('waypoints', []):
        if entry.get('waypoint') == int(waypoint_index):
            return entry
    return None


def load_waypoints_context(task_name=None, waypoint_index=None, description_path=None):
    path = _find_waypoints_description_path(task_name, description_path)
    if path is None:
        return None

    data = _load_json(path)
    current_waypoint = _waypoint_entry(data, waypoint_index)
    context = {
        'source_path': str(path),
        'current_waypoint_index': waypoint_index,
        'current_waypoint_description': current_waypoint,
        'waypoints_description': data,
    }
    if current_waypoint is None:
        context['available_waypoints'] = [
            entry.get('waypoint') for entry in data.get('waypoints', [])
        ]
    return context


def build_context_text(task_name=None, waypoint_index=None, description_path=None):
    """Full waypoint-description dump. Still used by the orientation/transition
    verifiers, which import these helpers from this module. The collision
    verifier uses build_intent_context_text instead (minimal, BT-grounded)."""
    if not task_name and waypoint_index is None and not description_path:
        return ''

    lines = [
        "# Task and waypoint context",
        f"task_name: {task_name or 'unknown'}",
        f"current_waypoint: {waypoint_index if waypoint_index is not None else 'unknown'}",
    ]

    context = load_waypoints_context(
        task_name=task_name,
        waypoint_index=waypoint_index,
        description_path=description_path,
    )
    if context is None:
        lines.append("waypoints_description: not found")
    else:
        lines.append(
            "waypoints_description_json:\n"
            + json.dumps(context, indent=2, sort_keys=True)
        )

    return "\n".join(lines)


def _condition_lines(conditions):
    """Non-empty, de-duplicated predicate strings from a BT condition section.

    The caller (``stage_context_for_stage`` in live_detectors) has already
    flattened each condition block down to its ``condition`` string."""
    out = []
    for item in conditions or []:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


def build_intent_context_text(waypoint_index=None, target_object=None,
                              stage_context=None):
    """Minimal intent block: the BT stage's primitive + its pre/postconditions.

    The VLM is told the stage CONTRACT in predicate form, never a prose
    narration of what the robot is supposed to be doing. The narration used to
    come from the description JSON's ``robot_action`` and it backfired: told
    "the robot releases the ball", the model reported "I do not see a clean
    release" and counted that absence as collision evidence. Preconditions state
    what held when the stage started and postconditions what the stage is trying
    to establish — neither is evidence about what the frames actually show."""
    stage_context = stage_context or {}
    primitive = str(stage_context.get('primitive') or '').strip()
    pre = _condition_lines(stage_context.get('preconditions'))
    post = _condition_lines(stage_context.get('postconditions'))
    target = str(target_object).strip() if target_object else ''

    if not (primitive or pre or post or target):
        return ''

    lines = [
        "# Current step intent (behaviour-tree stage contract)",
        f"current_waypoint: {waypoint_index if waypoint_index is not None else 'unknown'}",
        f"primitive: {primitive or 'unknown'}",
        f"target_object: {target or 'unknown'}",
    ]
    for header, conditions in (
        ("preconditions (held when this stage started):", pre),
        ("postconditions (what this stage is trying to establish):", post),
    ):
        lines.append(header)
        lines.extend(f"  - {text}" for text in conditions or ["none"])
    lines.append(
        "Interpretation: 'primitive' is the motion class this stage runs and the "
        "conditions are its contract in predicate form. Contact between the "
        "gripper and target_object while running this primitive is INTENDED and "
        "is not a collision; contact with anything else — the table, fixtures, "
        "other objects, the robot's own body — is unintended. The conditions "
        "describe the stage's GOAL, not what happened: a postcondition you "
        "cannot see satisfied is NOT evidence of collision, and 'I do not see "
        "this stage complete' is never a reason to report unintended contact. "
        "Judge collision only from contact evidence in the frames and the momentum-observer residuals."
    )
    return "\n".join(lines)


def _to_uint8_rgb(image):
    if image is None:
        return None

    arr = np.asarray(image)
    if arr.size == 0:
        return None
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim != 3:
        return None

    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if float(np.nanmax(arr)) <= 1.5:
            arr *= 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _to_depth_rgb(depth):
    if depth is None:
        return None

    arr = np.asarray(depth, dtype=np.float32)
    if arr.size == 0:
        return None
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        return None

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None

    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        normalized = np.zeros_like(arr, dtype=np.float32)
    else:
        normalized = np.clip((arr - low) / (high - low), 0.0, 1.0)

    image = (normalized * 255.0).astype(np.uint8)
    return np.repeat(image[..., None], 3, axis=2)


def _capture_side_rgb(env_wrapper):
    if env_wrapper is None:
        return None

    scene = getattr(getattr(env_wrapper, '_env', None), '_scene', None)
    if scene is None:
        return None

    camera = getattr(scene, '_aha_side_camera', None)
    if camera is None:
        return None

    try:
        return np.clip((camera.capture_rgb() * 255.0).astype(np.uint8), 0, 255)
    except Exception:
        return None


def _source_to_image(obs, source_name, env_wrapper=None):
    if source_name == 'side_rgb':
        image = _to_uint8_rgb(getattr(obs, source_name, None))
        if image is not None:
            return image
        return _to_uint8_rgb(_capture_side_rgb(env_wrapper))
    if source_name.endswith('_depth'):
        return _to_depth_rgb(getattr(obs, source_name, None))
    return _to_uint8_rgb(getattr(obs, source_name, None))


def _env_int(name, default):
    try:
        value = int(os.getenv(name, "").strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _image_to_data_url(image, max_side=None, jpeg_quality=None):
    """Encode one camera frame for the VLM request.

    Tunable so image quality can be ruled in or out as a cause of a misread:
    AHA_VLM_IMAGE_MAX_SIDE raises the cap that downscales large renders (768),
    AHA_VLM_JPEG_QUALITY raises JPEG quality (90), and AHA_VLM_IMAGE_FORMAT=png
    sends lossless frames instead. This matters most for the wrist close-up,
    where the evidence separating a clamped grip from an open one is a gap only
    a few pixels wide and JPEG ringing on a synthetic render can smear it.
    """
    if max_side is None:
        max_side = _env_int("AHA_VLM_IMAGE_MAX_SIDE", 768)
    if jpeg_quality is None:
        jpeg_quality = _env_int("AHA_VLM_JPEG_QUALITY", 90)

    arr = _to_uint8_rgb(image)
    if arr is None:
        return None

    pil_image = Image.fromarray(arr, mode='RGB')
    width, height = pil_image.size
    longest = max(width, height)
    if longest > max_side:
        scale = max_side / float(longest)
        resampling = getattr(Image, 'Resampling', Image).LANCZOS
        pil_image = pil_image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            resampling,
        )

    buffer = io.BytesIO()
    if os.getenv("AHA_VLM_IMAGE_FORMAT", "").strip().lower() == "png":
        pil_image.save(buffer, format='PNG')
        mime = 'png'
    else:
        pil_image.save(buffer, format='JPEG', quality=jpeg_quality)
        mime = 'jpeg'
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    return f'data:image/{mime};base64,{encoded}'


def collect_display_images(
    obs,
    camera_names=DEFAULT_CAMERA_NAMES,
    env_wrapper=None,
):
    images = []
    for camera_name in camera_names:
        image = _source_to_image(obs, camera_name, env_wrapper=env_wrapper)
        if image is not None:
            images.append((camera_name, image))
    return images


def sample_recent_observations(
    recent_obs_list,
    offsets=RECENT_IMAGE_OFFSETS,
):
    sampled = []
    sampled_offsets = []
    total = len(recent_obs_list)
    for offset in offsets:
        index = total - 1 - int(offset)
        if index < 0:
            continue
        sampled.append(recent_obs_list[index])
        sampled_offsets.append(int(offset))
    return sampled, sampled_offsets


def show_camera_sequence(
    recent_obs_list,
    step_offsets=None,
    camera_names=DEFAULT_CAMERA_NAMES,
    env_wrapper=None,
    title='Images sent to collision VLM',
):
    if step_offsets is None:
        step_offsets = [None] * len(recent_obs_list)

    rows = []
    all_camera_names = []
    for obs in recent_obs_list:
        images = collect_display_images(
            obs,
            camera_names=camera_names,
            env_wrapper=env_wrapper,
        )
        rows.append(dict(images))
        for camera_name, _ in images:
            if camera_name not in all_camera_names:
                all_camera_names.append(camera_name)

    if not all_camera_names:
        print("  [vlm] no camera images available to preview")
        return

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("  [vlm] matplotlib is not installed, skipping image preview")
        return

    fig, axes = plt.subplots(
        len(rows),
        len(all_camera_names),
        figsize=(3.2 * len(all_camera_names), 2.6 * len(rows)),
        squeeze=False,
    )

    for row_index, (row, offset) in enumerate(zip(rows, step_offsets)):
        for col_index, camera_name in enumerate(all_camera_names):
            ax = axes[row_index][col_index]
            image = row.get(camera_name)
            if image is not None:
                ax.imshow(image)
            ax.axis('off')
            if row_index == 0:
                ax.set_title(camera_name, fontsize=9)
            if col_index == 0:
                if offset is None:
                    label = f'frame {row_index + 1}'
                else:
                    label = 'current' if offset == 0 else f'-{offset}'
                ax.set_ylabel(label, fontsize=10, rotation=0, labelpad=30)

    fig.suptitle(title)
    fig.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def save_camera_grid(recent_obs_list, step_offsets, camera_names, env_wrapper,
                     out_dir, tag=''):
    """Save the exact camera montage sent to the collision VLM (rows = time
    offsets, cols = cameras) as a PNG in out_dir. Returns the path, or '' if
    nothing to save / matplotlib missing. Mirrors show_camera_sequence to disk."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return ''
    if step_offsets is None:
        step_offsets = [None] * len(recent_obs_list)
    grid_rows, all_cams = [], []
    for obs in recent_obs_list:
        images = collect_display_images(obs, camera_names=camera_names,
                                        env_wrapper=env_wrapper)
        grid_rows.append(dict(images))
        for cam, _ in images:
            if cam not in all_cams:
                all_cams.append(cam)
    if not all_cams:
        return ''
    fig, axes = plt.subplots(len(grid_rows), len(all_cams),
                             figsize=(3.2 * len(all_cams), 2.6 * len(grid_rows)),
                             squeeze=False)
    for r, (grow, offset) in enumerate(zip(grid_rows, step_offsets)):
        for c, cam in enumerate(all_cams):
            ax = axes[r][c]
            img = grow.get(cam)
            if img is not None:
                ax.imshow(img)
            ax.axis('off')
            if r == 0:
                ax.set_title(cam, fontsize=9)
            if c == 0:
                label = (f'frame {r + 1}' if offset is None
                         else 'current' if offset == 0 else f'-{offset}')
                ax.set_ylabel(label, fontsize=10, rotation=0, labelpad=30)
    fig.suptitle(f'Images sent to collision VLM {tag}'.strip())
    fig.tight_layout()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%H%M%S_%f')
    path = Path(out_dir) / f'collision_vlm_grid_{ts}.png'
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return str(path)


def show_camera_images(obs, camera_names=DEFAULT_CAMERA_NAMES, env_wrapper=None):
    images = collect_display_images(obs, camera_names, env_wrapper=env_wrapper)
    if not images:
        print("  [vlm] no camera images available to preview")
        return

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("  [vlm] matplotlib is not installed, skipping image preview")
        return

    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    if len(images) == 1:
        axes = [axes]

    for ax, (camera_name, image) in zip(axes, images):
        ax.imshow(image)
        ax.set_title(camera_name)
        ax.axis('off')

    fig.suptitle('Current images sent to collision VLM')
    fig.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    input("  Inspect the camera images, then press Enter to send them to VLM...")


def _extract_json(text):
    stripped = text.strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find('{')
        end = stripped.rfind('}')
        if start == -1 or end == -1 or end <= start:
            collision_happened = _extract_partial_collision_happened(stripped)
            return {
                'collision_happened': collision_happened,
                'explanation': stripped[:200],
            }
        try:
            data = json.loads(stripped[start:end + 1])
        except json.JSONDecodeError:
            collision_happened = _extract_partial_collision_happened(stripped)
            return {
                'collision_happened': collision_happened,
                'explanation': stripped[:200],
            }

    return {
        'collision_happened': _coerce_collision_happened(
            data.get('collision_happened')
        ),
        'explanation': str(data.get('explanation', '')).strip(),
        'visual_evidence': data.get('visual_evidence'),
        'torque_evidence': data.get('torque_evidence'),
        'decision_basis': data.get('decision_basis'),
    }


def _coerce_collision_happened(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ('true', 'yes', 'y', 'collision', '1'):
            return True
        if lowered in ('false', 'no', 'n', 'no_collision', '0'):
            return False
    return None


def _extract_partial_collision_happened(text):
    match = re.search(
        r'"?collision_happened"?\s*:\s*(true|false|"true"|"false")',
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return _coerce_collision_happened(match.group(1).strip('"'))


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
    path = trace_dir / f'collision_vlm_trace_{timestamp}.json'
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


def collect_camera_images(obs, camera_names=DEFAULT_CAMERA_NAMES, env_wrapper=None):
    images = []
    for camera_name in camera_names:
        image = _source_to_image(obs, camera_name, env_wrapper=env_wrapper)
        data_url = _image_to_data_url(image)
        if data_url is not None:
            images.append((camera_name, data_url))
    return images


def format_telemetry_history(history, recent_steps=25):
    """Recent momentum-observer residual per step, as a compact table.

    ``max|r_i|`` is the value the per-joint thresholds are read against, so the
    history shows how the residual built up (a step change at contact) rather
    than only its value at the reviewed frame."""
    if not history:
        return ""

    window = history[-recent_steps:]
    header = (
        f"{'step':>5}  {'||r||':>8}  {'max|r_i|':>9}  {'joints_over':>11}"
    )
    lines = [
        f"# Recent residual history (last {len(window)} steps)",
        header,
        "-" * len(header),
    ]
    for r in window:
        lines.append(
            f"{r.get('step', 0):>5}  "
            f"{float(r.get('residual_norm', 0.0)):>8.2f}  "
            f"{float(r.get('residual_score', 0.0)):>9.2f}  "
            f"{int(r.get('residual_joints_over', 0)):>11}"
        )
    return "\n".join(lines)


def format_detection_event(row):
    if not row:
        return ""

    return "\n".join([
        "# Detector event being reviewed",
        (
            f"step={row.get('step', 'unknown')}  "
            f"reason={row.get('collision_reason', '') or 'unknown'}  "
            f"score={float(row.get('collision_score', 0.0)):.3f}  "
            f"max|r_i|={float(row.get('residual_score', 0.0)):.2f} Nm  "
            f"residual_norm={float(row.get('residual_norm', 0.0)):.2f} Nm  "
            f"joints_over={int(row.get('residual_joints_over', 0))}"
        ),
        (
            "reason=momentum_residual means the per-joint residual gate below "
            "fired. Use this event to decide which image row is most important, "
            "but base the final verdict primarily on the image sequence. "
            "If the images show an open gripper in earlier frames and fingers "
            "closing around, capturing, or losing only the target object, treat "
            "that as expected manipulation or slip/grasp failure unless there is "
            "separate visual evidence of unintended contact (including a target "
            "that is visibly tipped, toppled, or shoved out of place). Read the "
            "residual table below: several joints over "
            "their calibrated thresholds, or one joint far above (x_thr well "
            "over 1), is collision evidence, while residuals inside the clean "
            "envelope are explained by the arm's own dynamics and are not "
            "collision by themselves. When the residuals are clearly over "
            "threshold and the images do not show a normal capture/place action "
            "or pure target slip/failed grasp, mark collision even if the exact "
            "contact point is subtle or occluded."
        ),
    ])


def confirm_collision_with_openai(
    recent_obs_list,
    step_offsets=None,
    row=None,
    telemetry_history=None,
    env_wrapper=None,
    task_name=None,
    waypoint_index=None,
    target_object=None,
    stage_context=None,
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

    obs = recent_obs_list[-1]

    if preview_images:
        show_camera_sequence(
            recent_obs_list,
            step_offsets=step_offsets,
            camera_names=camera_names,
            env_wrapper=env_wrapper,
        )
        input("  Inspect the camera images, then press Enter to send them to VLM...")

    # Save the montage of images that go into the VLM (env-gated, e.g. for evals).
    saved_grid = ''
    grid_dir = os.getenv('AHA_COLLISION_VLM_SAVE_GRID', '').strip()
    if grid_dir:
        tag = f'wp{waypoint_index}' if waypoint_index is not None else ''
        saved_grid = save_camera_grid(recent_obs_list, step_offsets, camera_names,
                                      env_wrapper, grid_dir, tag=tag)
        if saved_grid:
            print(f"  [vlm:collision] camera grid saved -> {saved_grid}")

    try:
        import openai
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing Python package 'openai'. Install it or run in an environment "
            "where the OpenAI SDK is available."
        ) from exc

    history_text = ""
    if telemetry_history:
        history_text = format_telemetry_history(telemetry_history)

    event_text = format_detection_event(row)

    context_text = build_intent_context_text(
        waypoint_index=waypoint_index,
        target_object=target_object,
        stage_context=stage_context,
    ) if INCLUDE_WAYPOINT_CONTEXT else ''
    residual_text = build_residual_context_text(task_name, row)

    prompt_text = "\n\n".join(
        part
        for part in (
            USER_PROMPT,
            event_text,
            residual_text,
            history_text,
            context_text,
        )
        if part
    )

    content = [
        {
            'type': 'input_text',
            'text': prompt_text,
        }
    ]

    total = len(recent_obs_list)
    all_camera_names = []
    image_manifest = []
    for i, step_obs in enumerate(recent_obs_list):
        step_offset = (
            step_offsets[i]
            if step_offsets is not None and i < len(step_offsets)
            else total - 1 - i
        )
        if step_offset == 0:
            step_label = f"## Step {i + 1} of {total} (current)"
        else:
            step_label = (
                f"## Step {i + 1} of {total} "
                f"({step_offset} simulator steps ago)"
            )
        content.append({'type': 'input_text', 'text': step_label})
        step_images = collect_camera_images(step_obs, camera_names, env_wrapper=env_wrapper)
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
    effort = os.getenv('AHA_DETECTOR_VLM_EFFORT', 'low').strip().lower()
    # 'medium' reasoning spends reasoning tokens out of max_output_tokens; give
    # headroom so the JSON answer is not starved/truncated.
    out_tokens = max_output_tokens if effort in ('minimal', 'low') else max(max_output_tokens, 3000)
    response = client.responses.create(
        model=model,
        max_output_tokens=out_tokens,
        reasoning={'effort': effort},
        input=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': content},
        ],
    )

    usage = getattr(response, 'usage', None)
    input_tokens = int(getattr(usage, 'input_tokens', 0) or 0)
    output_tokens = int(getattr(usage, 'output_tokens', 0) or 0)
    output_details = getattr(usage, 'output_tokens_details', None)
    reasoning_tokens = 0
    if output_details:
        reasoning_tokens = int(
            getattr(output_details, 'reasoning_tokens', 0) or 0
        )
    print(
        f"  [vlm:collision] tokens: {input_tokens} in + "
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
        'collision_happened': result['collision_happened'],
        'explanation': result['explanation'],
        'model': model,
        'camera_names': all_camera_names,
    }
    for key in ('visual_evidence', 'torque_evidence', 'decision_basis'):
        if result.get(key):
            output[key] = result[key]
    if trace_path:
        output['trace_path'] = trace_path
    if saved_grid:
        output['grid_path'] = saved_grid
    return output
