
from aha_publish import paths
import json
import os
import re

import numpy as np

REPO_ROOT = str(paths.PROJECT_ROOT)


# Frames observed before detections are allowed.
WARMUP_STEPS = 5

DEFAULT_WAYPOINTS_DESCRIPTION_DIR = str(paths.DESCRIPTION_DIR)
DEFAULT_TTM_CONTEXT_DIR = str(paths.TTM_CONTEXT_DIR)

# Spatial defaults are intentionally conservative. This detector checks whether
# the end effector reaches the expected waypoint pose; it does not inspect live
# object poses.
DEFAULT_WAYPOINT_DISTANCE_THRESHOLD = 0.12
DEFAULT_CONSECUTIVE_TRANSITION_FRAMES = 1
DEFAULT_CHECK_ON_WAYPOINT_DONE_ONLY = True
DEFAULT_CHECK_SEQUENCE_ORDER = True
DEFAULT_CHECK_WAYPOINT_POSE = True
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
        # inspect_ttm.py stores each waypoint's pose in its PARENT's frame. Those
        # values are episode-invariant, so they are taken as given here; the only
        # runtime work is composing them with the live parent pose.
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


def _pose_position(pose):
    arr = np.asarray(pose, dtype=float).flatten()
    if len(arr) < 3:
        return None
    return arr[:3]


def _pose_quaternion(pose):
    arr = np.asarray(pose, dtype=float).flatten()
    if len(arr) >= 7:
        return arr[3:7]
    return None


def recalculate_ttm_waypoint_positions(ttm_info, parent_poses):
    """Compose waypoint world positions from the TTM report's parent-local poses.

    The report gives each waypoint's offset and quaternion in its parent's frame,
    so the only runtime work is walking the parent chain:
    T_world(node) = T_world(parent) . T_local(node). parent_poses is a snapshot
    collected once after reset; the object poses are not updated during execution.
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
        # The quaternion is carried only to chain through waypoint parents; this
        # detector reports positions.
        'positions': {index: pose[:3] for index, pose in resolved.items()},
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


def load_transition_specs_from_waypoints(
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

        spec = {
            'waypoint': waypoint,
            'summary': _waypoint_summary(entry),
            'gripper_state': entry.get('gripper_state', ''),
            'expected_gripper_state': _expected_gripper_state(
                gripper_state_text
            ),
            'raw_entry': entry,
        }
        specs[waypoint] = spec

    return {
        'enabled': True,
        'path': path,
        'waypoints': specs,
        'reason': '',
    }


def make_threshold_overrides(
    waypoint_distance=None,
    open_gripper=None,
):
    overrides = {}
    if waypoint_distance is not None:
        overrides['waypoint_distance'] = float(waypoint_distance)
    if open_gripper is not None:
        overrides['open_gripper'] = float(open_gripper)
    return overrides


def default_threshold_overrides():
    return make_threshold_overrides(
        waypoint_distance=DEFAULT_WAYPOINT_DISTANCE_THRESHOLD,
        open_gripper=DEFAULT_OPEN_GRIPPER_THRESHOLD,
    )


def default_detector_settings():
    return {
        'threshold_overrides': default_threshold_overrides(),
        'waypoints_description_path': None,
        'ttm_context_path': None,
        'ttm_waypoint_equations': None,
        'waypoint_transition_specs': None,
        'consecutive_transition_frames': (
            DEFAULT_CONSECUTIVE_TRANSITION_FRAMES
        ),
        'check_on_waypoint_done_only': DEFAULT_CHECK_ON_WAYPOINT_DONE_ONLY,
        'check_sequence_order': DEFAULT_CHECK_SEQUENCE_ORDER,
        'check_waypoint_pose': DEFAULT_CHECK_WAYPOINT_POSE,
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
    """Lock transition detector thresholds.

    Named like the other detector modules so the interactive runners can share
    a familiar structure. Thresholds are description-driven, not learned from
    the warmup baseline.
    """
    thr = default_threshold_overrides()
    apply_threshold_overrides(thr, threshold_overrides)
    print(
        f"\n  [transition thresholds locked at step {WARMUP_STEPS}]  "
        f"waypoint_distance<={thr['waypoint_distance']:.3f}\n"
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
    gripper_xyz = gripper_pose[:3].astype(float)

    expected_waypoint_xyz = None
    expected_live_waypoint_distance = np.nan
    if waypoint_pose is not None:
        waypoint_arr = np.asarray(waypoint_pose, dtype=float).flatten()
        if len(waypoint_arr) >= 3:
            expected_waypoint_xyz = waypoint_arr[:3]
            expected_live_waypoint_distance = float(
                np.linalg.norm(gripper_xyz - expected_waypoint_xyz)
            )

    live_waypoint_xyz = expected_waypoint_xyz
    if current_waypoint_pose is not None:
        current_waypoint_arr = np.asarray(
            current_waypoint_pose, dtype=float
        ).flatten()
        if len(current_waypoint_arr) >= 3:
            live_waypoint_xyz = current_waypoint_arr[:3]

    live_waypoint_distance = np.nan
    if live_waypoint_xyz is not None:
        live_waypoint_distance = float(
            np.linalg.norm(gripper_xyz - live_waypoint_xyz)
        )

    ttm_waypoint_xyz = None
    ttm_waypoint_distance = np.nan
    if ttm_waypoint_pose is not None:
        ttm_arr = np.asarray(ttm_waypoint_pose, dtype=float).flatten()
        if len(ttm_arr) >= 3:
            ttm_waypoint_xyz = ttm_arr[:3]
            ttm_waypoint_distance = float(
                np.linalg.norm(gripper_xyz - ttm_waypoint_xyz)
            )

    waypoint_distance = (
        ttm_waypoint_distance
        if np.isfinite(ttm_waypoint_distance)
        else expected_live_waypoint_distance
    )
    waypoint_distance_source = (
        'ttm_recalculated'
        if np.isfinite(ttm_waypoint_distance)
        else 'expected_live_waypoint'
    )
    waypoint_pose_gap = np.nan
    if expected_waypoint_xyz is not None and ttm_waypoint_xyz is not None:
        waypoint_pose_gap = float(
            np.linalg.norm(expected_waypoint_xyz - ttm_waypoint_xyz)
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
        'gripper_x': float(gripper_xyz[0]),
        'gripper_y': float(gripper_xyz[1]),
        'gripper_z': float(gripper_xyz[2]),
        'waypoint_distance': waypoint_distance,
        'waypoint_distance_source': waypoint_distance_source,
        'live_waypoint_distance': live_waypoint_distance,
        'expected_live_waypoint_distance': expected_live_waypoint_distance,
        'ttm_waypoint_distance': ttm_waypoint_distance,
        'waypoint_distance_delta': np.nan,
        'live_waypoint_distance_delta': np.nan,
        'ttm_waypoint_distance_delta': np.nan,
        'waypoint_pose_gap': waypoint_pose_gap,
        'transition_failure': False,
        'threshold_crossed': False,
        'transition_reason': '',
        'suppression_reason': '',
        'transition_score': 0.0,
        'wrong_waypoint_order': False,
        'waypoint_pose_error': False,
        'gripper_state_error': False,
        '_gripper_xyz': gripper_xyz.copy(),
        '_waypoint_xyz': (
            expected_waypoint_xyz.copy()
            if expected_waypoint_xyz is not None else None
        ),
        '_live_waypoint_xyz': (
            live_waypoint_xyz.copy() if live_waypoint_xyz is not None else None
        ),
        '_ttm_waypoint_xyz': (
            ttm_waypoint_xyz.copy() if ttm_waypoint_xyz is not None else None
        ),
    }
    for prefix, xyz in (
        ('live_waypoint', live_waypoint_xyz),
        ('expected_live_waypoint', expected_waypoint_xyz),
        ('ttm_waypoint', ttm_waypoint_xyz),
    ):
        if xyz is None:
            continue
        row[f'{prefix}_x'] = float(xyz[0])
        row[f'{prefix}_y'] = float(xyz[1])
        row[f'{prefix}_z'] = float(xyz[2])

    return row


def update_deltas(logs):
    for i in range(1, len(logs)):
        curr = logs[i]
        prev = logs[i - 1]
        curr['gripper_motion'] = float(
            np.linalg.norm(curr['_gripper_xyz'] - prev['_gripper_xyz'])
        )
        for key in (
            'waypoint_distance',
            'live_waypoint_distance',
            'expected_live_waypoint_distance',
            'ttm_waypoint_distance',
        ):
            curr_value = curr.get(key, np.nan)
            prev_value = prev.get(key, np.nan)
            delta_key = f'{key}_delta'
            if np.isfinite(curr_value) and np.isfinite(prev_value):
                curr[delta_key] = float(curr_value) - float(prev_value)
            else:
                curr[delta_key] = np.nan


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


def check_transition_failure(
    row,
    waypoint_spec,
    thr,
    check_sequence_order=DEFAULT_CHECK_SEQUENCE_ORDER,
    check_waypoint_pose=DEFAULT_CHECK_WAYPOINT_POSE,
    check_gripper_state=DEFAULT_CHECK_GRIPPER_STATE,
    check_on_waypoint_done_only=DEFAULT_CHECK_ON_WAYPOINT_DONE_ONLY,
):
    """Return (transition_failure, reason_string) for one telemetry row."""
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

    if (
        check_on_waypoint_done_only
        and not row.get('waypoint_path_done', False)
        and not row.get('wrong_waypoint_order', False)
    ):
        row['transition_score'] = max(scores or [0.0])
        return False, 'waiting_for_waypoint_completion'

    if waypoint_spec is None:
        row['transition_score'] = max(scores or [0.0])
        if reasons:
            return True, '+'.join(reasons)
        return False, 'missing_waypoint_description'

    waypoint_distance = row.get('waypoint_distance', np.nan)
    if (
        check_waypoint_pose
        and np.isfinite(waypoint_distance)
        and waypoint_distance > thr['waypoint_distance']
    ):
        row['waypoint_pose_error'] = True
        reasons.append(
            f"waypoint_pose_error:source={row.get('waypoint_distance_source')},"
            f"dist={waypoint_distance:.3f}"
            f">{thr['waypoint_distance']:.3f}"
        )
        scores.append(_ratio(waypoint_distance, thr['waypoint_distance']))

    if check_gripper_state:
        ok, reason, score = _check_gripper_state(row, waypoint_spec, thr)
        scores.append(score)
        if not ok:
            row['gripper_state_error'] = True
            reasons.append(reason)

    row['transition_score'] = max(scores or [0.0])
    return bool(reasons), '+'.join(r for r in reasons if r)
