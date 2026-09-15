
from aha_publish import paths
import json
import math
import os
import re

import numpy as np

REPO_ROOT = str(paths.PROJECT_ROOT)


# Frames observed before detections are allowed.
WARMUP_STEPS = 5

DEFAULT_WAYPOINTS_DESCRIPTION_DIR = str(paths.DESCRIPTION_DIR)
DEFAULT_TTM_CONTEXT_DIR = str(paths.TTM_CONTEXT_DIR)

# Orientation defaults are intentionally conservative. This detector watches the
# TTM-recalculated waypoint angle trend, not live object poses during execution.
DEFAULT_TTM_ANGLE_INCREASE_THRESHOLD = float(
    os.getenv('AHA_ORIENTATION_ANGLE_THRESHOLD', '0.3'))
DEFAULT_CONSECUTIVE_ORIENTATION_FRAMES = 9
DEFAULT_CHECK_ON_WAYPOINT_DONE_ONLY = True
DEFAULT_CHECK_SEQUENCE_ORDER = True
DEFAULT_CHECK_WAYPOINT_ORIENTATION = True
DEFAULT_CHECK_GRIPPER_STATE = False
DEFAULT_OPEN_GRIPPER_THRESHOLD = 0.5

OPEN_STATE_TOKENS = (
    'open',
    'opening',
    'releasing',
    'release',
)
CLOSED_STATE_TOKENS = (
    'closed',
    'closing',
    'holding',
    'held',
    'grasped',
    'carrying',
)


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


def _lower_text(value):
    if value is None:
        return ''
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            parts.append(str(key))
            parts.append(str(item))
        value = ' '.join(parts)
    elif isinstance(value, (list, tuple)):
        value = ' '.join(str(v) for v in value)
    return str(value).lower()


def _contains_any(text, tokens):
    return any(token in text for token in tokens)


def _waypoint_number(entry):
    value = entry.get('waypoint', entry.get('stage_number'))
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


def find_ttm_context_path(task_name, context_dir=DEFAULT_TTM_CONTEXT_DIR):
    if not task_name:
        return None
    # Prefer the lean JSON emitted by inspect_ttm.py; fall back to the legacy .md.
    for ext in ('json', 'md'):
        exact = os.path.join(context_dir, f'{task_name}.llm_context.{ext}')
        if os.path.exists(exact):
            return exact
    if not os.path.isdir(context_dir):
        return None
    for ext in ('json', 'md'):
        matches = sorted(
            os.path.join(context_dir, name)
            for name in os.listdir(context_dir)
            if name.endswith(f'_{task_name}.llm_context.{ext}')
        )
        if matches:
            return matches[0]
    return None


def _extract_first_json_block(text):
    match = re.search(r'```json\s*(.*?)\s*```', text, flags=re.DOTALL)
    if not match:
        return None
    return json.loads(match.group(1))


def load_ttm_waypoint_equations(
    task_name,
    context_path=None,
    context_dir=DEFAULT_TTM_CONTEXT_DIR,
):
    path = context_path or find_ttm_context_path(task_name, context_dir)
    if not path or not os.path.exists(path):
        return {
            'enabled': False,
            'path': path,
            'waypoints': {},
            'parent_names': [],
            'reason': 'TTM context not found',
        }

    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    if str(path).endswith('.json'):
        try:
            data = json.loads(text)
        except ValueError:
            data = None
    else:
        data = _extract_first_json_block(text)
    if not data:
        return {
            'enabled': False,
            'path': path,
            'waypoints': {},
            'parent_names': [],
            'reason': 'TTM context JSON block not found',
        }

    waypoints = {}
    parent_names = set()
    for entry in data.get('waypoints', []):
        name = entry.get('name')
        parent = entry.get('orientation_parent', entry.get('parent'))
        position_parent = entry.get('position_parent', parent)
        # inspect_ttm.py stores each waypoint's pose in its PARENT's frame, the
        # quaternion included. Those values are episode-invariant, so they are
        # taken as given here; the only runtime work is composing them with the
        # live parent pose.
        offset = entry.get('local_position_xyz_m')
        local_quat = entry.get('local_quaternion_xyzw')
        match = re.fullmatch(r'waypoint(\d+)', str(name or ''))
        if not match or parent is None:
            continue
        index = int(match.group(1))
        wp_entry = {'name': name, 'parent': str(parent),
                    'position_parent': str(position_parent)}
        if (offset is not None and local_quat is not None
                and not entry.get('reference_unavailable')
                and index not in data.get('dynamic_waypoints', [])):
            wp_entry['offset'] = np.asarray(offset, dtype=float)
            wp_entry['local_quat'] = np.asarray(local_quat, dtype=float)
        waypoints[index] = wp_entry
        for anchor in (parent, position_parent):
            if anchor is not None and not str(anchor).startswith('waypoint'):
                parent_names.add(str(anchor))

    return {
        'enabled': True,
        'path': path,
        'waypoints': waypoints,
        'parent_names': sorted(parent_names),
        'reason': '',
    }


def _normalize_quat(q):
    q = np.asarray(q, dtype=float).flatten()
    if len(q) < 4:
        return np.asarray([0.0, 0.0, 0.0, 1.0])
    q = q[:4]
    norm = np.linalg.norm(q)
    if norm <= 1e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0])
    return q / norm


def _quat_to_matrix(q):
    x, y, z, w = _normalize_quat(q)
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _quat_multiply(a, b):
    ax, ay, az, aw = _normalize_quat(a)
    bx, by, bz, bw = _normalize_quat(b)
    return _normalize_quat(np.asarray([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]))


def _quat_angle(a, b):
    if a is None or b is None:
        return np.nan
    qa = _normalize_quat(a)
    qb = _normalize_quat(b)
    dot = abs(float(np.dot(qa, qb)))
    dot = max(-1.0, min(1.0, dot))
    return float(2.0 * math.acos(dot))


def _pose_position(pose):
    arr = np.asarray(pose, dtype=float).flatten()
    if len(arr) < 3:
        return None
    return arr[:3]


def _pose_quaternion(pose):
    arr = np.asarray(pose, dtype=float).flatten()
    if len(arr) >= 7:
        return _normalize_quat(arr[3:7])
    return None


def recalculate_ttm_waypoint_poses(ttm_info, parent_poses):
    """Compose waypoint world poses from the TTM report's parent-local poses.

    The report gives each waypoint's offset and quaternion in its parent's frame,
    so the only runtime work is walking the parent chain:
    T_world(node) = T_world(parent) . T_local(node). Parent object poses are
    sampled once after reset; they are not updated during execution.
    """
    equations = ttm_info.get('waypoints', {}) if ttm_info else {}
    resolved = {}
    unresolved = {}

    def resolve(index, stack=None):
        if index in resolved:
            return resolved[index]
        stack = set(stack or ())
        if index in stack:
            unresolved[index] = 'cycle in waypoint parent graph'
            return None
        equation = equations.get(index)
        if equation is None:
            unresolved[index] = 'missing TTM equation'
            return None
        if 'offset' not in equation:
            unresolved[index] = 'no parent-local pose in TTM context'
            return None

        parent = equation['parent']
        offset = equation['offset']
        local_quat = equation['local_quat']
        def parent_pose_for(anchor):
            if anchor.startswith('waypoint'):
                match = re.fullmatch(r'waypoint(\d+)', anchor)
                if not match:
                    return None
                return resolve(int(match.group(1)), stack | {index})
            return parent_poses.get(anchor)

        parent_pose = parent_pose_for(parent)
        position_parent = equation.get('position_parent', parent)
        position_pose = parent_pose_for(position_parent)
        parent_quat = _pose_quaternion(parent_pose)
        position = _pose_position(position_pose)
        position_quat = _pose_quaternion(position_pose)
        if position is None or position_quat is None or parent_quat is None:
            unresolved[index] = f'parent pose unavailable: {position_parent}, {parent}'
            return None

        world_position = position + _quat_to_matrix(position_quat).dot(equation['offset'])
        world_quat = _quat_multiply(parent_quat, equation['local_quat'])
        resolved[index] = np.concatenate([world_position, world_quat])
        return resolved[index]

    for index in sorted(equations):
        resolve(index)

    return {
        'poses': resolved,
        'unresolved': unresolved,
    }


def _expected_gripper_state(text):
    text = _lower_text(text)
    if _contains_any(text, OPEN_STATE_TOKENS):
        return 'open'
    if _contains_any(text, CLOSED_STATE_TOKENS):
        return 'closed'
    return None


def _waypoint_summary(entry):
    return (
        entry.get('robot_action')
        or entry.get('visual_summary')
        or entry.get('summary')
        or ''
    )


def load_orientation_specs_from_waypoints(
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
            'waypoints': {},
            'reason': 'waypoints description JSON not found',
        }

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    entries = data.get('waypoints')
    if entries is None:
        entries = data.get('stages', [])

    specs = {}
    for entry in entries:
        waypoint = _waypoint_number(entry)
        if waypoint is None:
            continue

        gripper_state_text = entry.get('gripper_state')
        if gripper_state_text is None:
            gripper_state_text = _waypoint_summary(entry)

        specs[waypoint] = {
            'waypoint': waypoint,
            'summary': _waypoint_summary(entry),
            'gripper_state': entry.get('gripper_state', ''),
            'expected_gripper_state': _expected_gripper_state(
                gripper_state_text
            ),
            'raw_entry': entry,
        }

    return {
        'enabled': True,
        'path': path,
        'waypoints': specs,
        'reason': '',
    }


def make_threshold_overrides(
    ttm_angle_increase=None,
    waypoint_angle=None,
    open_gripper=None,
):
    overrides = {}
    if ttm_angle_increase is not None:
        overrides['ttm_angle_increase'] = float(ttm_angle_increase)
    if waypoint_angle is not None:
        # Backward-compatible alias for older CLI/code paths.
        overrides['ttm_angle_increase'] = float(waypoint_angle)
    if open_gripper is not None:
        overrides['open_gripper'] = float(open_gripper)
    return overrides


def default_threshold_overrides():
    return make_threshold_overrides(
        ttm_angle_increase=DEFAULT_TTM_ANGLE_INCREASE_THRESHOLD,
        open_gripper=DEFAULT_OPEN_GRIPPER_THRESHOLD,
    )


def default_detector_settings():
    return {
        'threshold_overrides': default_threshold_overrides(),
        'waypoints_description_path': None,
        'ttm_context_path': None,
        'ttm_waypoint_equations': None,
        'waypoint_orientation_specs': None,
        'consecutive_orientation_frames': (
            DEFAULT_CONSECUTIVE_ORIENTATION_FRAMES
        ),
        'check_on_waypoint_done_only': DEFAULT_CHECK_ON_WAYPOINT_DONE_ONLY,
        'check_sequence_order': DEFAULT_CHECK_SEQUENCE_ORDER,
        'check_waypoint_orientation': DEFAULT_CHECK_WAYPOINT_ORIENTATION,
        'check_gripper_state': DEFAULT_CHECK_GRIPPER_STATE,
    }


def apply_threshold_overrides(thresholds, overrides=None):
    if not overrides:
        return thresholds
    for key, value in overrides.items():
        if value is not None:
            thresholds[key] = float(value)
    return thresholds


def freeze_thresholds(baseline, threshold_overrides=None):
    """Lock orientation detector thresholds.

    Named like the other detector modules so the interactive runners can share
    a familiar structure. Thresholds are description-driven, not learned from
    the warmup baseline.
    """
    thr = default_threshold_overrides()
    apply_threshold_overrides(thr, threshold_overrides)
    print(
        f"\n  [orientation thresholds locked at step {WARMUP_STEPS}]  "
        f"ttm_angle_increase>={thr['ttm_angle_increase']:.3f}rad\n"
    )
    return thr


def obs_to_row(
    obs,
    step,
    waypoint=None,
    expected_waypoint=None,
    waypoint_path_done=False,
    waypoint_pose=None,
    current_waypoint_pose=None,
    ttm_waypoint_pose=None,
    waypoint_started=False,
    started_waypoint=None,
):
    grip_forces = _safe_array(getattr(obs, 'gripper_touch_forces', None), 6)
    gripper_pose = _safe_array(getattr(obs, 'gripper_pose', None), 7)
    gripper_open = _safe_float(getattr(obs, 'gripper_open', None))
    total_f = float(
        np.linalg.norm(grip_forces[0:3]) + np.linalg.norm(grip_forces[3:6])
    )
    gripper_quat = _pose_quaternion(gripper_pose)

    expected_waypoint_quat = _pose_quaternion(waypoint_pose)
    expected_live_waypoint_angle = _quat_angle(
        gripper_quat, expected_waypoint_quat
    )

    live_waypoint_quat = expected_waypoint_quat
    if current_waypoint_pose is not None:
        current_waypoint_quat = _pose_quaternion(current_waypoint_pose)
        if current_waypoint_quat is not None:
            live_waypoint_quat = current_waypoint_quat

    live_waypoint_angle = _quat_angle(gripper_quat, live_waypoint_quat)

    ttm_waypoint_quat = _pose_quaternion(ttm_waypoint_pose)
    ttm_waypoint_angle = _quat_angle(gripper_quat, ttm_waypoint_quat)

    waypoint_angle = (
        ttm_waypoint_angle
        if np.isfinite(ttm_waypoint_angle)
        else expected_live_waypoint_angle
    )
    waypoint_angle_source = (
        'ttm_recalculated'
        if np.isfinite(ttm_waypoint_angle)
        else 'expected_live_waypoint'
    )
    waypoint_orientation_gap = _quat_angle(
        expected_waypoint_quat,
        ttm_waypoint_quat,
    )

    row = {
        'step': step,
        'waypoint': waypoint,
        'expected_waypoint': expected_waypoint,
        'waypoint_path_done': bool(waypoint_path_done),
        'waypoint_started': bool(waypoint_started),
        'started_waypoint': started_waypoint,
        'gripper_open': gripper_open,
        'grip_force': total_f,
        'is_holding': total_f > 0.1 and gripper_open < 0.5,
        'waypoint_angle': waypoint_angle,
        'waypoint_angle_source': waypoint_angle_source,
        'live_waypoint_angle': live_waypoint_angle,
        'expected_live_waypoint_angle': expected_live_waypoint_angle,
        'ttm_waypoint_angle': ttm_waypoint_angle,
        'waypoint_angle_delta': np.nan,
        'live_waypoint_angle_delta': np.nan,
        'ttm_waypoint_angle_delta': np.nan,
        'ttm_angle_increasing_flag': False,
        'ttm_angle_increase_start': np.nan,
        'ttm_angle_increase': np.nan,
        'waypoint_orientation_gap': waypoint_orientation_gap,
        'orientation_failure': False,
        'threshold_crossed': False,
        'orientation_reason': '',
        'suppression_reason': '',
        'orientation_score': 0.0,
        'wrong_waypoint_order': False,
        'waypoint_orientation_error': False,
        'gripper_state_error': False,
        '_gripper_quat': gripper_quat.copy(),
        '_waypoint_quat': (
            expected_waypoint_quat.copy()
            if expected_waypoint_quat is not None else None
        ),
        '_live_waypoint_quat': (
            live_waypoint_quat.copy()
            if live_waypoint_quat is not None else None
        ),
        '_ttm_waypoint_quat': (
            ttm_waypoint_quat.copy()
            if ttm_waypoint_quat is not None else None
        ),
    }
    for prefix, quat in (
        ('gripper', gripper_quat),
        ('live_waypoint', live_waypoint_quat),
        ('expected_live_waypoint', expected_waypoint_quat),
        ('ttm_waypoint', ttm_waypoint_quat),
    ):
        if quat is None:
            continue
        row[f'{prefix}_qx'] = float(quat[0])
        row[f'{prefix}_qy'] = float(quat[1])
        row[f'{prefix}_qz'] = float(quat[2])
        row[f'{prefix}_qw'] = float(quat[3])

    return row


def update_deltas(logs):
    for i in range(1, len(logs)):
        curr = logs[i]
        prev = logs[i - 1]
        curr['gripper_orientation_delta'] = _quat_angle(
            curr['_gripper_quat'], prev['_gripper_quat']
        )
        for key in (
            'waypoint_angle',
            'live_waypoint_angle',
            'expected_live_waypoint_angle',
            'ttm_waypoint_angle',
        ):
            curr_value = curr.get(key, np.nan)
            prev_value = prev.get(key, np.nan)
            delta_key = f'{key}_delta'
            if np.isfinite(curr_value) and np.isfinite(prev_value):
                curr[delta_key] = float(curr_value) - float(prev_value)
            else:
                curr[delta_key] = np.nan

    flag = False
    start_angle = np.nan
    for row in logs:
        delta = row.get('ttm_waypoint_angle_delta', np.nan)
        angle = row.get('ttm_waypoint_angle', np.nan)

        if np.isfinite(delta):
            if delta > 0.0:
                if not flag:
                    flag = True
                    start_angle = (
                        float(angle) - float(delta)
                        if np.isfinite(angle) else np.nan
                    )
            elif delta < 0.0:
                flag = False
                start_angle = np.nan

        row['ttm_angle_increasing_flag'] = flag
        row['ttm_angle_increase_start'] = start_angle
        if flag and np.isfinite(angle) and np.isfinite(start_angle):
            row['ttm_angle_increase'] = float(angle) - float(start_angle)
        else:
            row['ttm_angle_increase'] = np.nan


def _ratio(value, threshold):
    if threshold <= 0.0 or not np.isfinite(value):
        return 0.0
    return float(value) / float(threshold)


def _inverse_ratio(threshold, value):
    if threshold <= 0.0 or not np.isfinite(value):
        return 0.0
    return float(threshold) / max(float(value), 1e-12)


def _check_gripper_state(row, spec, thr):
    expected = spec.get('expected_gripper_state')
    if expected == 'open':
        ok = row['gripper_open'] >= thr['open_gripper']
        score = _inverse_ratio(thr['open_gripper'], row['gripper_open'])
        return ok, 'gripper_not_open', score
    if expected == 'closed':
        ok = row['gripper_open'] < thr['open_gripper']
        score = _ratio(row['gripper_open'], thr['open_gripper'])
        return ok, 'gripper_not_closed', score
    return True, '', 0.0


def check_orientation_failure(
    row,
    waypoint_spec,
    thr,
    check_sequence_order=DEFAULT_CHECK_SEQUENCE_ORDER,
    check_waypoint_orientation=DEFAULT_CHECK_WAYPOINT_ORIENTATION,
    check_gripper_state=DEFAULT_CHECK_GRIPPER_STATE,
    check_on_waypoint_done_only=DEFAULT_CHECK_ON_WAYPOINT_DONE_ONLY,
):
    """Return (orientation_failure, reason_string) for one telemetry row."""
    reasons = []
    scores = []

    if (
        check_sequence_order
        and row.get('expected_waypoint') is not None
        and row.get('waypoint') is not None
        and row.get('expected_waypoint') != row.get('waypoint')
    ):
        row['wrong_waypoint_order'] = True
        reasons.append(
            f"wrong_waypoint_order_expected_{row.get('expected_waypoint')}"
            f"_got_{row.get('waypoint')}"
        )
        scores.append(1.0)

    ttm_angle_increase = row.get('ttm_angle_increase', np.nan)
    if (
        check_waypoint_orientation
        and row.get('ttm_angle_increasing_flag', False)
        and np.isfinite(ttm_angle_increase)
        and ttm_angle_increase >= thr['ttm_angle_increase']
    ):
        row['waypoint_orientation_error'] = True
        reasons.append(
            "ttm_waypoint_angle_increase:"
            f"delta={row.get('ttm_waypoint_angle_delta', np.nan):.3f},"
            f"increase={ttm_angle_increase:.3f}"
            f">={thr['ttm_angle_increase']:.3f}"
        )
        scores.append(_ratio(ttm_angle_increase, thr['ttm_angle_increase']))

    if (
        check_on_waypoint_done_only
        and not row.get('waypoint_path_done', False)
        and not row.get('wrong_waypoint_order', False)
        and not reasons
    ):
        row['orientation_score'] = max(scores or [0.0])
        return False, 'waiting_for_waypoint_completion'

    if waypoint_spec is None:
        row['orientation_score'] = max(scores or [0.0])
        if reasons:
            return True, '+'.join(reasons)
        return False, 'missing_waypoint_description'

    if check_gripper_state:
        ok, reason, score = _check_gripper_state(row, waypoint_spec, thr)
        scores.append(score)
        if not ok:
            row['gripper_state_error'] = True
            reasons.append(reason)

    row['orientation_score'] = max(scores or [0.0])
    return bool(reasons), '+'.join(r for r in reasons if r)
