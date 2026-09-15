
from aha_publish import paths
import json
import os
import numpy as np

REPO_ROOT = str(paths.PROJECT_ROOT)


# Frames observed before detections are allowed.
WARMUP_STEPS = 5

# Gripper touch forces come from RLBench's gripper touch sensors. Slip detection
# should be driven by loss of grasp contact, not arm torque spikes.
# Fallback holding threshold for tasks with NO per-task grip-force calibration
# (task absent from ALL_TASKS_success_grip_force_stats.json, or its calibrated
# value collapsed to <=0). Set to the calibration floor (0.1 N, = AHA_SLIP_GRIP_FLOOR)
# rather than 1.0 N: many tasks grip well below 1.0 N (e.g. put_knife_in_knife_block
# peaks ~0.83 N), so a 1.0 N fallback made is_holding never true -> held_peak stayed 0
# -> slip was undetectable. 0.1 N lets a real grasp register a hold on uncalibrated tasks.
DEFAULT_GRIP_FORCE_THRESHOLD = 0.1
DEFAULT_GRIP_FORCE_DROP_THRESHOLD = 1
DEFAULT_REQUIRED_PRIOR_HOLDING_STEPS = int(os.getenv('AHA_SLIP_REQUIRED_PRIOR', '1'))
DEFAULT_HOLD_LOST_GRACE_STEPS = int(os.getenv('AHA_SLIP_HOLD_LOST_GRACE', '2'))
DEFAULT_CONSECUTIVE_SLIP_FRAMES = int(os.getenv('AHA_SLIP_CONSECUTIVE_FRAMES', '2'))
DEFAULT_ENABLE_GRASP_NOT_ESTABLISHED_CHECK = False
DEFAULT_WAYPOINTS_DESCRIPTION_DIR = str(paths.DESCRIPTION_DIR)
DEFAULT_USE_GRIP_FORCE_STATS_THRESHOLDS = True
DEFAULT_CALIBRATION_ROOT = os.getenv(
    'AHA_CALIBRATION_ROOT',
    str(paths.CALIBRATION_DIR),
)
DEFAULT_GRIP_FORCE_STATS_PATH = os.getenv(
    'AHA_GRIP_FORCE_STATS_PATH',
    os.path.join(
        DEFAULT_CALIBRATION_ROOT, 'grip_force_stats',
        'ALL_TASKS_success_grip_force_stats.json',
    ),
)
DEFAULT_GRIP_FORCE_STATS_K = 4.0
DEFAULT_GRIP_FORCE_STATS_FIELD = 'average_of_episode_stats'

HOLD_REQUIRED_STATE_TOKENS = (
    'holding',
    'held',
    'closed',
    'carrying',
    'grasped',
    'transport',
    'lifting',
)
GRASP_START_STATE_TOKENS = (
    'closing',
    'close',
    'grasp',
)
RELEASE_STATE_TOKENS = (
    'releasing',
    'release',
    'open_gripper',
    'opening',
    'drop',
)


def _safe_array(val, length):
    if val is None:
        return np.zeros(length)
    arr = np.asarray(val, dtype=float).flatten()
    out = np.zeros(length)
    out[:min(len(arr), length)] = arr[:length]
    return out


def decompose_finger_force(force3, normal_dir):
    """Split one finger's tri-axial contact force into (normal, tangential).

    ``force3`` is a fingertip force vector [fx, fy, fz] in that touch sensor's
    own local frame (RLBench/PyRep ForceSensor.read()[0]); ``normal_dir`` is the
    per-finger unit "press" direction estimated at runtime (calibration-free --
    no knowledge of which sensor axis is the object normal is assumed).

    Returns (f_normal, f_tangential): f_normal is the signed projection onto the
    press direction; f_tangential is the magnitude of the shear component in the
    plane orthogonal to it. When no normal direction is known yet, the whole
    force is reported as normal (tangential 0), so a finger only contributes a
    shear signal once its baseline press direction is established.

    This is the tri-axial "tangential force" f_xy = sqrt(fx^2 + fy^2) of
    Wong & Zhu (2026), generalised so the shear plane is defined by the runtime
    press direction instead of a hardcoded sensor z-axis.
    """
    f = np.asarray(force3, dtype=float).reshape(-1)[:3]
    if f.size < 3:
        f = np.zeros(3)
    if normal_dir is None:
        return float(np.linalg.norm(f)), 0.0
    nd = np.asarray(normal_dir, dtype=float).reshape(-1)[:3]
    nn = float(np.linalg.norm(nd))
    if nn <= 1e-12:
        return float(np.linalg.norm(f)), 0.0
    nd = nd / nn
    f_normal = float(np.dot(f, nd))
    f_tang_vec = f - f_normal * nd
    return f_normal, float(np.linalg.norm(f_tang_vec))


def obs_to_row(obs, step):
    grip_forces = _safe_array(getattr(obs, 'gripper_touch_forces', None), 6)
    joint_forces = _safe_array(getattr(obs, 'joint_forces', None), 7)
    gripper_joint_positions = _safe_array(
        getattr(obs, 'gripper_joint_positions', None), 2
    )
    total_f = float(
        np.linalg.norm(grip_forces[0:3]) + np.linalg.norm(grip_forces[3:6])
    )
    left_f = float(np.linalg.norm(grip_forces[0:3]))
    right_f = float(np.linalg.norm(grip_forces[3:6]))

    return {
        'step': step,
        'grip_force': total_f,
        'left_grip_force': left_f,
        'right_grip_force': right_f,
        'grip_force_delta': 0.0,
        'grip_force_drop': 0.0,
        'gripper_joint_delta': 0.0,
        'torque_norm': float(np.linalg.norm(joint_forces)),
        'is_holding': total_f > DEFAULT_GRIP_FORCE_THRESHOLD,
        'prior_holding_streak': 0,
        'steps_since_holding': None,
        'recently_holding': False,
        'slip': False,
        'slip_candidate_started': False,
        'slip_candidate_active': False,
        'low_force_streak': 0,
        'threshold_crossed': False,
        'slip_reason': '',
        'suppression_reason': '',
        'slip_score': 0.0,
        'force_released': False,
        'force_drop_crossed': False,
        # v2 (tangential-shear / Wong-Zhu) signals. Filled in by the live
        # detector, which owns the stateful per-finger normal-direction and
        # tangential baselines; defaulted here so the row schema is stable even
        # when v2 is inactive or before contact establishes a baseline.
        'left_normal': left_f,
        'right_normal': right_f,
        'left_tangential': 0.0,
        'right_tangential': 0.0,
        'tangential_jump': 0.0,
        'shear_surge': False,
        'normal_rising': False,
        '_grip_forces': grip_forces.copy(),
        '_gripper_joint_positions': gripper_joint_positions.copy(),
        '_joint_forces': joint_forces.copy(),
    }


def _lower_text(value):
    if value is None:
        return ''
    if isinstance(value, dict):
        value = ' '.join(str(v) for v in value.values())
    elif isinstance(value, (list, tuple)):
        value = ' '.join(str(v) for v in value)
    return str(value).lower()


def _contains_any(text, tokens):
    return any(token in text for token in tokens)


def _waypoint_number(entry):
    value = entry.get('waypoint')
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def find_waypoints_description_path(
    task_name,
    description_dir=DEFAULT_WAYPOINTS_DESCRIPTION_DIR,
):
    if not task_name:
        return None
    preferred = [
        os.path.join(
            description_dir,
            f'{task_name}_ALL_WAYPOINTS_COMBINED.openai.multimodal_analysis.json',
        ),
        os.path.join(
            description_dir,
            f'{task_name}_ALL_WAYPOINTS_COMBINED.openai.analysis.json',
        ),
    ]
    for path in preferred:
        if os.path.exists(path):
            return path

    if not os.path.isdir(description_dir):
        return None
    prefix = f'{task_name}_ALL_WAYPOINTS_COMBINED'
    matches = sorted(
        os.path.join(description_dir, name)
        for name in os.listdir(description_dir)
        if name.startswith(prefix) and name.endswith('.json')
    )
    return matches[0] if matches else None


def load_holding_requirement_from_waypoints(
    task_name,
    description_path=None,
    description_dir=DEFAULT_WAYPOINTS_DESCRIPTION_DIR,
):
    path = description_path or find_waypoints_description_path(
        task_name, description_dir=description_dir
    )
    if not path or not os.path.exists(path):
        return {
            'enabled': False,
            'path': path,
            'hold_required_waypoints': [],
            'release_waypoints': [],
            'waypoint_reasons': {},
            'reason': 'waypoints description JSON not found',
        }

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    hold_required = set()
    release_waypoints = set()
    waypoint_reasons = {}

    for entry in data.get('waypoints', []):
        waypoint = _waypoint_number(entry)
        if waypoint is None:
            continue

        state_text = _lower_text(entry.get('gripper_state'))
        action_text = _lower_text(entry.get('robot_action'))
        summary_text = _lower_text(entry.get('visual_summary'))
        object_text = _lower_text(entry.get('approx_distance_to_objects'))
        code_text = _lower_text(entry.get('code_context_used'))
        combined = ' '.join([
            state_text,
            action_text,
            summary_text,
            object_text,
            code_text,
        ])

        reasons = []
        if _contains_any(state_text, HOLD_REQUIRED_STATE_TOKENS):
            hold_required.add(waypoint)
            reasons.append(f"gripper_state={entry.get('gripper_state')}")
        elif _contains_any(combined, HOLD_REQUIRED_STATE_TOKENS):
            hold_required.add(waypoint)
            reasons.append('description indicates held/carried object')

        if _contains_any(state_text, GRASP_START_STATE_TOKENS):
            hold_required.add(waypoint)
            reasons.append(f"grasp starts here ({entry.get('gripper_state')})")
        elif _contains_any(code_text, ('close_gripper',)):
            hold_required.add(waypoint)
            reasons.append('code_context closes gripper')

        if _contains_any(state_text, RELEASE_STATE_TOKENS) or 'open' in state_text.split():
            release_waypoints.add(waypoint)
            reasons.append(f"release waypoint ({entry.get('gripper_state')})")
        elif _contains_any(code_text, ('open_gripper',)):
            release_waypoints.add(waypoint)
            reasons.append('code_context opens gripper')
        elif _contains_any(action_text, RELEASE_STATE_TOKENS):
            release_waypoints.add(waypoint)
            reasons.append('robot_action describes release/drop')

        if reasons:
            waypoint_reasons[str(waypoint)] = '; '.join(reasons)

    return {
        'enabled': True,
        'path': path,
        'hold_required_waypoints': sorted(hold_required),
        'release_waypoints': sorted(release_waypoints),
        'waypoint_reasons': waypoint_reasons,
        'reason': '',
    }


def waypoint_requires_holding(waypoint, phase_info, waypoint_path_done=False):
    # NOTE: release_waypoints is deliberately NOT consulted here. It was derived
    # from waypoint description text (gripper_state / robot_action keywords), which
    # false-positives on object nouns -- e.g. close_jar wp4 "lower the lid onto the
    # jar *opening*" matched the 'opening' release token, tagging a held-throughout
    # waypoint as a release. That flipped holding_required_phase to False the moment
    # the path completed and wiped a genuine slip candidate. Commanded releases are
    # already handled downstream by the live detector's _grasp_active gate, which
    # tracks the ACTUAL gripper open/close commands, so holding is required exactly
    # when the waypoint is a hold-required waypoint -- no path_done / release logic.
    if waypoint is None:
        return False
    try:
        waypoint = int(waypoint)
    except (TypeError, ValueError):
        return False

    hold_required = set(phase_info.get('hold_required_waypoints') or [])
    return waypoint in hold_required


def update_deltas(logs):
    for i, row in enumerate(logs):
        if i > 0:
            prev = logs[i - 1]
            force_delta = row['grip_force'] - prev['grip_force']
            row['grip_force_delta'] = float(abs(force_delta))
            row['grip_force_drop'] = float(max(0.0, -force_delta))
            curr_joints = row.get(
                '_gripper_joint_positions', np.zeros(2, dtype=float)
            )
            prev_joints = prev.get(
                '_gripper_joint_positions', np.zeros(2, dtype=float)
            )
            row['gripper_joint_delta'] = float(
                np.linalg.norm(curr_joints - prev_joints)
            )

    update_holding_states(logs)


def update_holding_states(logs, thresholds=None):
    thresholds = thresholds or {}
    grip_force_threshold = thresholds.get(
        'grip_force_threshold', DEFAULT_GRIP_FORCE_THRESHOLD
    )
    holding_streak = 0
    last_holding_step = None
    for row in logs:
        row['prior_holding_streak'] = holding_streak
        row['steps_since_holding'] = (
            None
            if last_holding_step is None
            else int(row['step'] - last_holding_step)
        )
        row['is_holding'] = row['grip_force'] > grip_force_threshold
        if row['is_holding']:
            holding_streak += 1
            last_holding_step = row['step']
            row['steps_since_holding'] = 0
        else:
            holding_streak = 0


def make_threshold_overrides(
    grip_force_threshold=None,
    grip_force_drop_threshold=None,
):
    overrides = {}
    if grip_force_threshold is not None:
        overrides['grip_force_threshold'] = float(grip_force_threshold)
    if grip_force_drop_threshold is not None:
        overrides['grip_force_drop_threshold'] = float(grip_force_drop_threshold)
    return overrides


def default_threshold_overrides():
    return make_threshold_overrides(
        grip_force_threshold=DEFAULT_GRIP_FORCE_THRESHOLD,
        grip_force_drop_threshold=DEFAULT_GRIP_FORCE_DROP_THRESHOLD,
    )


def default_detector_settings():
    return {
        'threshold_overrides': default_threshold_overrides(),
        'use_grip_force_stats_thresholds': (
            DEFAULT_USE_GRIP_FORCE_STATS_THRESHOLDS
        ),
        'grip_force_stats_path': DEFAULT_GRIP_FORCE_STATS_PATH,
        'grip_force_stats_k': DEFAULT_GRIP_FORCE_STATS_K,
        'grip_force_stats_field': DEFAULT_GRIP_FORCE_STATS_FIELD,
        'task_grip_force_stats_thresholds': None,
        'required_prior_holding_steps': DEFAULT_REQUIRED_PRIOR_HOLDING_STEPS,
        'hold_lost_grace_steps': DEFAULT_HOLD_LOST_GRACE_STEPS,
        'consecutive_slip_frames': DEFAULT_CONSECUTIVE_SLIP_FRAMES,
        'enable_grasp_not_established_check': (
            os.getenv('AHA_SLIP_ENABLE_GRASP_NOT_ESTABLISHED', '0').lower()
            in ('1', 'true', 'yes', 'on')
        ),
    }


def apply_threshold_overrides(thresholds, overrides=None):
    if not overrides:
        return thresholds
    for key, value in overrides.items():
        if value is not None:
            thresholds[key] = float(value)
    return thresholds


def _median_episode_stat(metric_data, key):
    vals = []
    for episode in metric_data.get('episodes', []) or []:
        stats = episode.get('stats') or {}
        if stats.get(key) is not None:
            vals.append(float(stats[key]))
    if not vals:
        return None
    return float(np.median(vals))


def _stats_threshold_from_metric(metric_stats, metric, k, metric_data=None):
    mean = float(metric_stats['mean'])
    std = float(metric_stats['std'])
    if metric == 'grip_force':
        # Robust held-force threshold: a fixed number of std below the task's
        # mean holding force, floored so it never collapses toward 0. The old
        # `median_episode_min - std` formula collapsed for light/noisy grips and
        # fell back to a flat 1.0 N for every task (the FP/FN source). mean/std
        # come from this task's clean baseline, so it is per-object. Tunable via
        # env for offline sweeps without recalibrating.
        # Grasp bar = max(mean - k*std, floor) of the clean-run baseline. In the
        # proportional method (method 1) this is the "a grasp really happened"
        # gate, so it should reflect the true baseline hold force, not be capped.
        # The ceiling is off by default (kept as an optional knob).
        grip_k = float(os.getenv('AHA_SLIP_GRIP_K', '2.0'))
        floor = float(os.getenv('AHA_SLIP_GRIP_FLOOR', '0.1'))
        ceiling = float(os.getenv('AHA_SLIP_GRIP_CEILING', 'inf'))
        return float(min(ceiling, max(floor, mean - grip_k * std)))
    if metric == 'grip_force_drop':
        return float(mean + float(k) * std)
    raise ValueError(f"Unsupported grip-force stats metric: {metric}")


def _task_entry_from_all_grip_force_stats(data, task_name):
    for task in data.get('tasks', []):
        if task.get('task') == task_name:
            return task
    if data.get('task') == task_name:
        return data
    return None


def task_grip_force_stats_thresholds(
    task_name,
    stats_path=DEFAULT_GRIP_FORCE_STATS_PATH,
    k=DEFAULT_GRIP_FORCE_STATS_K,
    stats_field=DEFAULT_GRIP_FORCE_STATS_FIELD,
):
    task_path = os.path.join(DEFAULT_CALIBRATION_ROOT, 'grip_force_stats', f'{task_name}_success_grip_force_stats.json')
    if stats_path == DEFAULT_GRIP_FORCE_STATS_PATH and os.path.exists(task_path):
        stats_path = task_path
    if not os.path.exists(stats_path):
        return {}, {
            'enabled': False,
            'reason': f'grip force stats JSON not found: {stats_path}',
        }

    with open(stats_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    task_data = _task_entry_from_all_grip_force_stats(data, task_name)
    if task_data is None:
        return {}, {
            'enabled': False,
            'path': stats_path,
            'reason': f'task not found in grip force stats JSON: {task_name}',
        }

    metrics = task_data.get('metrics', {})
    grip_metric = metrics.get('grip_force', {})
    drop_metric = metrics.get('grip_force_drop', {})
    grip_stats = grip_metric.get(stats_field)
    drop_stats = drop_metric.get(stats_field)
    if not grip_stats or not drop_stats:
        return {}, {
            'enabled': False,
            'path': stats_path,
            'task': task_name,
            'reason': f'missing metric field: {stats_field}',
        }

    grip_force_threshold = _stats_threshold_from_metric(
        grip_stats, 'grip_force', k, metric_data=grip_metric
    )
    grip_force_drop_threshold = _stats_threshold_from_metric(
        drop_stats, 'grip_force_drop', k
    )
    return make_threshold_overrides(
        grip_force_threshold=grip_force_threshold,
        grip_force_drop_threshold=grip_force_drop_threshold,
    ), {
        'enabled': True,
        'path': stats_path,
        'task': task_name,
        'k': float(k),
        'stats_field': stats_field,
        'grip_force_threshold': grip_force_threshold,
        'grip_force_drop_threshold': grip_force_drop_threshold,
        'grip_force_formula': (
            'max(floor, mean - grip_k * std)  '
            '[grip_k=AHA_SLIP_GRIP_K (def 2.0), floor=AHA_SLIP_GRIP_FLOOR (def 0.1)]'
        ),
        'grip_force_median_episode_min': _median_episode_stat(
            grip_metric, 'min'
        ),
        'grip_force_drop_formula': 'mean + k * std',
    }


def apply_task_grip_force_stats_thresholds(
    settings,
    task_name,
    manual_threshold_keys=None,
):
    if not settings.get('use_grip_force_stats_thresholds', False):
        return settings

    manual_threshold_keys = set(manual_threshold_keys or ())
    overrides, info = task_grip_force_stats_thresholds(
        task_name,
        stats_path=settings.get(
            'grip_force_stats_path', DEFAULT_GRIP_FORCE_STATS_PATH
        ),
        k=settings.get('grip_force_stats_k', DEFAULT_GRIP_FORCE_STATS_K),
        stats_field=settings.get(
            'grip_force_stats_field', DEFAULT_GRIP_FORCE_STATS_FIELD
        ),
    )
    settings['task_grip_force_stats_thresholds'] = info
    if not overrides:
        return settings

    for key, value in overrides.items():
        if key not in manual_threshold_keys:
            # grip_force_threshold collapses to 0 when the episode-min is
            # dominated by non-gripping frames (e.g. tasks with very light
            # grip where min_holding ≈ std). A 0 threshold means
            # force_released never fires. Skip the override and keep the
            # default (0.1 N floor) so the detector remains functional.
            if key == 'grip_force_threshold' and (value is None or value <= 0):
                info = settings.get('task_grip_force_stats_thresholds', {})
                info['grip_force_threshold_skipped'] = (
                    'calibrated value <= 0; keeping default')
                continue
            settings['threshold_overrides'][key] = value
    return settings


def freeze_thresholds(baseline, threshold_overrides=None):
    """Lock slip detector thresholds.

    The defaults are intentionally absolute because early episode warmup often
    happens before the robot grasps anything, so warmup-only force statistics
    are not a reliable slip baseline.
    """
    thr = {
        'grip_force_threshold': DEFAULT_GRIP_FORCE_THRESHOLD,
        'grip_force_drop_threshold': DEFAULT_GRIP_FORCE_DROP_THRESHOLD,
    }
    apply_threshold_overrides(thr, threshold_overrides)
    print(
        f"\n  [slip thresholds locked at step {WARMUP_STEPS}]  "
        f"grip_force_threshold={thr['grip_force_threshold']:.3f}  "
        f"grip_force_drop_threshold={thr['grip_force_drop_threshold']:.3f}\n"
    )
    return thr


def _ratio(value, threshold):
    if threshold <= 0.0:
        return 0.0
    return float(value) / float(threshold)


def check_slip(
    row,
    thr,
    required_prior_holding_steps=DEFAULT_REQUIRED_PRIOR_HOLDING_STEPS,
    hold_lost_grace_steps=DEFAULT_HOLD_LOST_GRACE_STEPS,
):
    """Return (slip_candidate_started, reason_string) for one telemetry row.

    A slip candidate starts when grip force falls below the hold threshold after
    a valid hold. Sustained low grip force is checked by the caller because it
    is a stateful condition across rows.
    """
    prior_holding = (
        row['prior_holding_streak'] >= int(required_prior_holding_steps)
    )
    steps_since_holding = row.get('steps_since_holding')
    recently_holding = (
        steps_since_holding is not None
        and steps_since_holding <= int(hold_lost_grace_steps)
    )
    force_released = row['grip_force'] <= thr['grip_force_threshold']
    force_drop_crossed = (
        row['grip_force_drop'] >= thr['grip_force_drop_threshold']
    )

    row['force_released'] = force_released
    row['force_drop_crossed'] = force_drop_crossed
    row['recently_holding'] = recently_holding

    reasons = []
    if not (prior_holding or recently_holding):
        row['slip_score'] = 0.0
        return False, 'not_recently_holding'

    # Detection method (AHA_SLIP_METHOD):
    #   1 (default) = proportional collapse: fire when grip falls by a set
    #                 fraction of its held peak (row['proportional_collapse'],
    #                 computed by the live detector which tracks the peak). Falls
    #                 back to the grip-level test if the field is absent.
    #   2 (previous) = force drop: fire when the frame-to-frame grip drop crosses
    #                  the drop threshold.
    #   3 (v2, Wong-Zhu) = tangential shear surge: fire when the per-finger
    #                 tangential (shear) force jumps -- an INCIPIENT slip, caught
    #                 before the grip magnitude collapses. row['shear_surge'] is
    #                 computed by the live detector from the calibration-free
    #                 per-finger tangential baseline.
    method = os.getenv('AHA_SLIP_METHOD', '1').strip()
    if method == '2':
        slip_started = force_drop_crossed
    elif method == '3':
        slip_started = bool(row.get('shear_surge', False))
    else:
        slip_started = bool(row.get('proportional_collapse', force_released))

    if force_released:
        reasons.append('force_released')
    if force_drop_crossed:
        reasons.append('force_drop_start')
    if row.get('shear_surge'):
        reasons.append(
            f"tangential_shear_surge(jump={row.get('tangential_jump', 0.0):.3f})")
    if recently_holding and not prior_holding:
        reasons.append(f"recently_held_{steps_since_holding}_steps_ago")

    row['slip_score'] = max(
        _ratio(row['grip_force_drop'], thr['grip_force_drop_threshold']),
        (
            _ratio(thr['grip_force_threshold'], max(row['grip_force'], 1e-12))
            if force_released
            else 0.0
        ),
    )

    if not slip_started:
        if force_released or force_drop_crossed:
            reasons.append('waiting_for_low_force')
        else:
            reasons.append('thresholds_not_crossed')

    return bool(slip_started), '+'.join(reasons)
