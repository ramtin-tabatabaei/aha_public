
from aha_publish import paths
import json
import os

import numpy as np

REPO_ROOT = str(paths.PROJECT_ROOT)

# Frames observed before detections are allowed.
WARMUP_STEPS = 5

# Same camera set the collision/slip VLM confirmation uses, minus wrist_rgb
# (the wrist cam rides on the arm, so it is excluded from the freeze stillness
# check). side_rgb is a custom camera only the main BT runner stamps onto the
# observation (attach_side_rgb); standalone harnesses that read raw RLBench obs
# won't have it, so the min-visible gate below tolerates one missing camera.
DEFAULT_CAMERA_NAMES = (
    'side_rgb',
    'front_rgb',
    'overhead_rgb',
)

# A freeze should look still in both proprioception and camera observations.
DEFAULT_JOINT_POSITION_DELTA_THRESHOLD = 1e-3
DEFAULT_JOINT_VELOCITY_NORM_THRESHOLD = 0.03
DEFAULT_CAMERA_MOTION_THRESHOLD = 3e-3
DEFAULT_MIN_VISIBLE_CAMERAS = len(DEFAULT_CAMERA_NAMES) - 1
DEFAULT_CONSECUTIVE_FREEZE_FRAMES = 8


def _safe_array(val, length):
    if val is None:
        return np.zeros(length)
    arr = np.asarray(val, dtype=float).flatten()
    out = np.zeros(length)
    out[:min(len(arr), length)] = arr[:length]
    return out


def _safe_float(val, default=0.0):
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _as_rgb_float(image, stride=4):
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.size == 0:
        return None
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        return None
    arr = arr[::stride, ::stride, :3].astype(np.float32)
    if arr.size == 0:
        return None
    if float(np.max(arr)) > 1.5:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0)


def _image_delta(curr, prev):
    if curr is None or prev is None:
        return np.nan
    if curr.shape != prev.shape:
        h = min(curr.shape[0], prev.shape[0])
        w = min(curr.shape[1], prev.shape[1])
        c = min(curr.shape[2], prev.shape[2])
        if h == 0 or w == 0 or c == 0:
            return np.nan
        curr = curr[:h, :w, :c]
        prev = prev[:h, :w, :c]
    return float(np.mean(np.abs(curr - prev)))


def obs_to_row(obs, step, camera_names=DEFAULT_CAMERA_NAMES):
    joint_positions = _safe_array(getattr(obs, 'joint_positions', None), 7)
    joint_velocities = _safe_array(getattr(obs, 'joint_velocities', None), 7)
    gripper_joint_positions = _safe_array(
        getattr(obs, 'gripper_joint_positions', None), 2
    )
    gripper_open = _safe_float(getattr(obs, 'gripper_open', None))

    camera_images = {}
    for name in camera_names:
        camera_images[name] = _as_rgb_float(getattr(obs, name, None))

    return {
        'step': step,
        'gripper_open': gripper_open,
        'joint_position_delta': 0.0,
        'joint_velocity_norm': float(np.linalg.norm(joint_velocities)),
        'gripper_joint_delta': 0.0,
        'camera_motion': np.nan,
        'visible_camera_count': 0,
        'freezing': False,
        'threshold_crossed': False,
        'freezing_reason': '',
        'suppression_reason': '',
        'freezing_score': 0.0,
        'joint_position_still': False,
        'joint_velocity_still': False,
        'camera_still': False,
        '_joint_positions': joint_positions.copy(),
        '_joint_velocities': joint_velocities.copy(),
        '_gripper_joint_positions': gripper_joint_positions.copy(),
        '_camera_images': camera_images,
    }


def update_deltas(logs):
    for i in range(1, len(logs)):
        curr = logs[i]
        prev = logs[i - 1]
        curr['joint_position_delta'] = float(
            np.linalg.norm(curr['_joint_positions'] - prev['_joint_positions'])
        )
        curr['gripper_joint_delta'] = float(
            np.linalg.norm(
                curr['_gripper_joint_positions']
                - prev['_gripper_joint_positions']
            )
        )

        camera_deltas = []
        for name, curr_image in curr['_camera_images'].items():
            prev_image = prev['_camera_images'].get(name)
            delta = _image_delta(curr_image, prev_image)
            if np.isfinite(delta):
                camera_deltas.append(delta)
                curr[f'{name}_delta'] = delta
            else:
                curr[f'{name}_delta'] = np.nan

        curr['visible_camera_count'] = len(camera_deltas)
        # Aggregate per-camera mean-abs pixel deltas with a MEAN, not an L2 norm.
        # Each element of camera_deltas is already a per-camera mean-abs delta, so
        # np.linalg.norm scaled the aggregate by ~sqrt(n_cameras) and pushed a
        # genuinely-still scene above the per-camera camera_motion threshold,
        # suppressing real freezes. The mean keeps the aggregate on the same
        # scale the threshold was calibrated against.
        curr['camera_motion'] = (
            float(np.mean(camera_deltas)) if camera_deltas else np.nan
        )


def make_threshold_overrides(
    joint_position_delta=None,
    joint_velocity_norm=None,
    camera_motion=None,
):
    overrides = {}
    if joint_position_delta is not None:
        overrides['joint_position_delta'] = float(joint_position_delta)
    if joint_velocity_norm is not None:
        overrides['joint_velocity_norm'] = float(joint_velocity_norm)
    if camera_motion is not None:
        overrides['camera_motion'] = float(camera_motion)
    return overrides


def default_threshold_overrides():
    return make_threshold_overrides(
        joint_position_delta=DEFAULT_JOINT_POSITION_DELTA_THRESHOLD,
        joint_velocity_norm=DEFAULT_JOINT_VELOCITY_NORM_THRESHOLD,
        camera_motion=DEFAULT_CAMERA_MOTION_THRESHOLD,
    )


DEFAULT_FREEZING_STATS_DIR = str(paths.CALIBRATION_DIR / 'freezing_stats')
DEFAULT_USE_TASK_FREEZING_STATS_THRESHOLDS = True


def default_detector_settings():
    return {
        'threshold_overrides': default_threshold_overrides(),
        'camera_names': list(DEFAULT_CAMERA_NAMES),
        'min_visible_cameras': DEFAULT_MIN_VISIBLE_CAMERAS,
        'consecutive_freeze_frames': DEFAULT_CONSECUTIVE_FREEZE_FRAMES,
        'use_task_freezing_stats_thresholds': DEFAULT_USE_TASK_FREEZING_STATS_THRESHOLDS,
        'freezing_stats_dir': DEFAULT_FREEZING_STATS_DIR,
    }


def task_freezing_stats_thresholds(task_name, stats_dir=DEFAULT_FREEZING_STATS_DIR):
    """Per-task stillness thresholds from a clean-run freezing-stats file.

    Mirrors the collision/slip per-task calibration: the calibration step
    (detectors/freezing calibration) measures each task's clean joint-velocity /
    joint-position-delta distribution and stores a per-task stillness threshold
    so a freeze (joints at near-zero residual motion) is flagged while the task's
    own slow/paused motion is not. Returns (overrides, info); overrides is empty
    when no stats file exists (the detector then keeps the flat defaults)."""
    path = os.path.join(stats_dir, f'{task_name}.json')
    if not os.path.exists(path):
        return {}, {'status': 'no_stats_file', 'path': path}
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as exc:
        return {}, {'status': f'error: {exc}', 'path': path}
    rec = data.get('recommended_thresholds', {})
    overrides = {}
    for key in ('joint_velocity_norm', 'joint_position_delta', 'camera_motion'):
        if key in rec and rec[key] is not None:
            overrides[key] = float(rec[key])
    return overrides, {'status': 'loaded', 'path': path, 'recommended': rec}


def apply_task_freezing_stats_thresholds(settings, task_name, stats_dir=None):
    """Fold a task's calibrated stillness thresholds into ``settings`` in place.

    No-op (keeps the flat defaults) when disabled or when no stats file exists,
    so uncalibrated tasks behave exactly as before."""
    if not settings.get('use_task_freezing_stats_thresholds', False):
        return settings
    stats_dir = stats_dir or settings.get(
        'freezing_stats_dir', DEFAULT_FREEZING_STATS_DIR)
    overrides, info = task_freezing_stats_thresholds(task_name, stats_dir=stats_dir)
    settings['task_freezing_stats'] = info
    if overrides:
        settings.setdefault('threshold_overrides', {})
        settings['threshold_overrides'].update(overrides)
    return settings


def apply_threshold_overrides(thresholds, overrides=None):
    if not overrides:
        return thresholds
    for key, value in overrides.items():
        if value is not None:
            thresholds[key] = float(value)
    return thresholds


def freeze_thresholds(baseline, threshold_overrides=None):
    """Lock thresholds for low-motion freezing detection."""
    thr = {
        'joint_position_delta': DEFAULT_JOINT_POSITION_DELTA_THRESHOLD,
        'joint_velocity_norm': DEFAULT_JOINT_VELOCITY_NORM_THRESHOLD,
        'camera_motion': DEFAULT_CAMERA_MOTION_THRESHOLD,
    }
    apply_threshold_overrides(thr, threshold_overrides)
    print(
        f"\n  [freezing thresholds locked at step {WARMUP_STEPS}]  "
        f"joint_delta<={thr['joint_position_delta']:.6f}  "
        f"velocity_norm<={thr['joint_velocity_norm']:.6f}  "
        f"camera_motion<={thr['camera_motion']:.6f}\n"
    )
    return thr


def _stillness_ratio(threshold, value):
    if threshold <= 0.0:
        return 0.0
    return threshold / max(float(value), 1e-12)


def check_freezing(row, thr, min_visible_cameras=DEFAULT_MIN_VISIBLE_CAMERAS):
    """Return (is_freezing, reason_string) for one telemetry row."""
    joint_delta = row['joint_position_delta']
    velocity_norm = row['joint_velocity_norm']
    camera_motion = row['camera_motion']
    camera_count = row['visible_camera_count']

    joint_position_still = joint_delta <= thr['joint_position_delta']
    joint_velocity_still = velocity_norm <= thr['joint_velocity_norm']
    camera_available = camera_count >= int(min_visible_cameras)
    camera_still = (
        camera_available
        and np.isfinite(camera_motion)
        and camera_motion <= thr['camera_motion']
    )

    row['joint_position_still'] = joint_position_still
    row['joint_velocity_still'] = joint_velocity_still
    row['camera_still'] = camera_still
    _score_terms = [
        _stillness_ratio(thr['joint_position_delta'], joint_delta),
        _stillness_ratio(thr['joint_velocity_norm'], velocity_norm),
    ]
    # Only fold in the camera term when cameras are actually available; forcing
    # it to 0.0 otherwise collapsed the whole min() to 0 (a meaningless score)
    # on every camera-less frame even when the joints were perfectly still.
    if camera_available and np.isfinite(camera_motion):
        _score_terms.append(_stillness_ratio(thr['camera_motion'], camera_motion))
    row['freezing_score'] = min(_score_terms)

    reasons = []
    if joint_position_still:
        reasons.append('joint_positions_still')
    if joint_velocity_still:
        reasons.append('joint_velocities_still')
    if camera_still:
        reasons.append('cameras_still')
    elif not camera_available:
        reasons.append('not_enough_cameras')

    frozen = joint_position_still and joint_velocity_still and camera_still
    return frozen, '+'.join(reasons)
