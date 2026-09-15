
from aha_publish import paths
import json
import os

import numpy as np

REPO_ROOT = str(paths.PROJECT_ROOT)


# Frames observed before thresholds are locked.
# Steps before this mark are never flagged and feed the baseline.
WARMUP_STEPS = 5

# Do not edit these three names unless you also update the code below.
# To switch detector behavior, edit DEFAULT_TORQUE_RULE.
TORQUE_RULE_EITHER = 'either'
TORQUE_RULE_BOTH = 'both'
TORQUE_RULE_WEIGHTED = 'weighted'
TORQUE_RULES = (TORQUE_RULE_EITHER, TORQUE_RULE_BOTH, TORQUE_RULE_WEIGHTED)

# User-editable detector defaults.
#
# Switch torque detection mode by changing this one line:
#   DEFAULT_TORQUE_RULE = TORQUE_RULE_EITHER
#       collision if torque_delta OR torque_norm is above threshold
#   DEFAULT_TORQUE_RULE = TORQUE_RULE_BOTH
#       collision only if torque_delta AND torque_norm are above threshold
#   DEFAULT_TORQUE_RULE = TORQUE_RULE_WEIGHTED
#       collision if weighted score is above DEFAULT_TORQUE_SCORE_THRESHOLD
DEFAULT_TORQUE_RULE = TORQUE_RULE_EITHER

# Automatically load torque thresholds from:
#   <repo>/aha_output/aha_calibration/torque_stats/
#       <task>_success_torque_stats.json
#
# Edit DEFAULT_TORQUE_STATS_K to make thresholds looser/tighter.
# The threshold for each metric (torque_norm and torque_delta) is:
#   threshold = max(summary_mean + K * summary_std, summary_max + summary_std)
DEFAULT_USE_TASK_TORQUE_STATS_THRESHOLDS = True
DEFAULT_CALIBRATION_ROOT = os.getenv(
    'AHA_CALIBRATION_ROOT',
    str(paths.CALIBRATION_DIR),
)
DEFAULT_TORQUE_STATS_DIR = os.getenv(
    'AHA_TORQUE_STATS_DIR',
    os.path.join(DEFAULT_CALIBRATION_ROOT, 'torque_stats'),
)
DEFAULT_TORQUE_STATS_K = float(os.getenv("AHA_COLLISION_TORQUE_K", "3.0"))
# Use per-episode clean summaries, taking the median across clean episodes. This
# avoids a single long or spiky clean episode dominating the pooled all-frame
# envelope.
DEFAULT_TORQUE_STATS_FIELD = 'median_of_episode_stats'
DEFAULT_TORQUE_STATS_USE_MAX_FLOOR = os.getenv("AHA_COLLISION_MAX_FLOOR", "1") == "1"

# Used only when DEFAULT_TORQUE_RULE = TORQUE_RULE_WEIGHTED.
# Example:
#   score = 0.7 * (torque_norm / norm_threshold)
#         + 0.3 * (torque_delta / delta_threshold)
# If score >= DEFAULT_TORQUE_SCORE_THRESHOLD, it is a torque collision.
DEFAULT_TORQUE_NORM_WEIGHT = 0.5
DEFAULT_TORQUE_DELTA_WEIGHT = 0.5
DEFAULT_TORQUE_SCORE_THRESHOLD = 1.0

# How many consecutive detected collision frames are required before flagging.
# The live BT wrapper can still let the scene advance a few frames before VLM
# confirmation so the verifier sees the contact after it develops.
DEFAULT_CONSECUTIVE_COLLISION_FRAMES = 1

# Detection method (AHA_COLLISION_METHOD), the collision analog of
# AHA_SLIP_METHOD:
#   3 (default) = MOMENTUM OBSERVER. The De Luca (IROS 2006) generalized-momentum
#                 residual: a per-joint estimate of the external (collision) joint
#                 torque, thresholded per joint against a clean-run calibration.
#                 See the block below. This is the default collision detector.
#   1 = PROPORTIONAL SPIKE. Track a running "normal-motion" torque baseline and
#                 fire when torque_norm jumps to >= (1 + RISE_FRAC) * baseline (a
#                 sudden, proportionally large increase). Self-scaling per task,
#                 mirrors slip's proportional collapse: slip watches a fractional
#                 DROP below a held peak, collision watches a fractional RISE.
#   2 = absolute thresholds via DEFAULT_TORQUE_RULE (either/both/weighted) on
#       torque_norm / torque_delta (the previous behavior).
DEFAULT_COLLISION_METHOD = os.getenv('AHA_COLLISION_METHOD', '3').strip()
# Minimum fractional rise above baseline to call it a sudden spike (1.0 = +100%).
DEFAULT_TORQUE_RISE_FRAC = float(os.getenv('AHA_COLLISION_RISE_FRAC', '1.0'))
# Trailing frames used to estimate the baseline (robust median, causal so the
# live and offline pipelines agree frame-for-frame).
DEFAULT_TORQUE_BASELINE_WINDOW = int(os.getenv('AHA_COLLISION_BASELINE_WINDOW', '10'))

# --- Method 3: generalized-momentum observer (De Luca et al., IROS 2006) ------
# "Collision Detection and Safe Reaction with the DLR-III Lightweight
# Manipulator Arm." A model-based residual r that estimates the external
# (collision) joint torque from proprioception alone, WITHOUT joint acceleration:
#
#   p = M(q) qd                                  (generalized momentum)
#   r = K_I [ p - integral(tau + C^T qd - g + r) dt - p(0) ]
#
# so that r_dot = K_I (tau_ext - r): each r_i is a first-order low-pass estimate
# of the true external torque at joint i. A collision is declared when the
# residual magnitude clears a (per-task-calibratable) threshold. The Panda
# dynamic model M, C^T qd, g is provided self-contained in panda_dynamics.py
# (validated to machine precision against roboticstoolbox's Panda). Because the
# simulator's true dynamics differ slightly from the nominal model, the clean
# residual is a small non-zero offset; the threshold is set adaptively above the
# warmup/clean baseline (like methods 1 and 2) so a collision must exceed it.
#
# Observer gain K_I (1/s), diagonal, applied to every joint. Higher = faster
# tracking / tighter residual but noisier.
DEFAULT_MOMENTUM_GAIN = float(os.getenv('AHA_COLLISION_MOMENTUM_GAIN', '25.0'))
# Control-step duration used to integrate the observer (shared AHA convention).
DEFAULT_MOMENTUM_DT = float(os.getenv('AHA_SIM_DT', '0.05'))
# V1 preserves the original update. Opt into backward Euler with
# AHA_COLLISION_MOMENTUM_VERSION=2 (still collision method 3).
DEFAULT_MOMENTUM_VERSION = '1'


def momentum_observer_version():
    version = os.getenv('AHA_COLLISION_MOMENTUM_VERSION', DEFAULT_MOMENTUM_VERSION).strip()
    if version not in ('1', '2'):
        raise ValueError('AHA_COLLISION_MOMENTUM_VERSION must be 1 or 2')
    return version


# Minimum residual magnitude (Nm) that ever counts as a collision, so quiet
# waypoints with a near-zero clean residual still need a real physical torque.
DEFAULT_RESIDUAL_FLOOR = float(os.getenv('AHA_COLLISION_RESIDUAL_FLOOR', '3.0'))
# Headroom multiplier on the residual when locking the method-3 threshold from
# the WARMUP envelope: thr = max(FLOOR, RESIDUAL_K * warmup_base). This is the
# scalar fallback path only (no calibrated per-joint stats for the task). The
# calibrated per-joint path takes its headroom entirely from
# DEFAULT_RESIDUAL_JOINT_MULT -- applying K there too meant two stacked
# multipliers and a threshold that no longer matched the per-joint weights.
DEFAULT_RESIDUAL_K = float(os.getenv('AHA_COLLISION_RESIDUAL_K', '1.1'))


def _residual_joint_mult(spec):
    """Parse the per-joint threshold multiplier vector (7 comma-separated floats)."""
    try:
        vals = [float(v) for v in str(spec).split(',')]
    except ValueError:
        vals = []
    if len(vals) != 7 or any(v <= 0.0 for v in vals):
        return np.ones(7)
    return np.asarray(vals, dtype=float)


# Per-joint multiplier applied on top of the calibrated threshold vector:
#     thr_i = max(FLOOR, clean_max_i) * MULT_i
# The joints are not equally informative. Measured over the collision runs of
# basketball_in_hoop / beat_the_buzz / change_channel / close_jar, j0, j2 and j5
# cross their calibrated threshold in nearly every burst -- including bursts the
# VLM did not confirm -- so at 1.0 they carry most of the false positives, while
# the wrist joints j4/j6 only ever cross on real contacts. Replaying the recorded
# traces of those tasks (25 labelled bursts, MIN_JOINTS=1) against the previous
# vector [5, 1.1, 5, 1.1, 1.1, 2, 1.1]:
#     all 1.0 : TP=13 FP=7 FN=3   65% precision / 81% recall
#     previous: TP=13 FP=4 FN=3   76% precision / 81% recall
# i.e. three false positives removed at no cost in detections. j5 is flat over
# 1.8-2.5 (identical results); 3.0 is a cliff -- basketball_in_hoop drops from 3
# true positives to 1, because its surviving detections peak on j5 at 2.5-3.4x.
# NOTE: fitted in-sample on those four tasks; a leave-one-task-out check showed
# per-joint multipliers do not transfer well across tasks, and the 5x on j0/j2
# was derived from close_jar's false positives -- it costs basketball_in_hoop 2
# of its 5 detections. Treat as a starting point and re-check on new tasks. Set
# all-ones to disable. Only applies to the calibrated per-joint path, not the
# warmup scalar fallback.
# Joint 3 (plot r3, zero-based index 2) now uses 1.1 instead of 5.0.
# The historical metrics above do not evaluate this updated vector.
DEFAULT_RESIDUAL_JOINT_MULT = _residual_joint_mult(
    os.getenv('AHA_COLLISION_RESIDUAL_JOINT_MULT', '5.0,1.1,1.1,1.1,1.1,2.0,1.1'))
# The observer's residual has a transient while its integral catches the initial
# momentum; these first frames are marked ``_mo_settling`` and excluded from the
# warmup baseline so they cannot inflate the fallback threshold.
DEFAULT_RESIDUAL_SETTLE = int(os.getenv('AHA_COLLISION_RESIDUAL_SETTLE', '3'))
# How many joints must simultaneously exceed their per-joint residual threshold to
# declare a collision. A real end-effector collision loads the whole kinematic
# chain, so several joints cross at once; requiring >= this many rejects a lone
# noisy joint. Applies only when per-joint calibrated thresholds are available
# (the scalar warmup fallback keeps the single-condition rule).
DEFAULT_RESIDUAL_MIN_JOINTS = int(os.getenv('AHA_COLLISION_RESIDUAL_MIN_JOINTS', '1'))

try:
    from . import panda_dynamics as _panda_dyn
except (ImportError, ValueError):  # executed as a bare module (bundle exec)
    try:
        import panda_dynamics as _panda_dyn
    except ImportError:
        _panda_dyn = None


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


def obs_to_row(obs, step):
    joint_forces = _safe_array(getattr(obs, 'joint_forces', None), 7)
    joint_positions = _safe_array(getattr(obs, 'joint_positions', None), 7)
    joint_velocities = _safe_array(getattr(obs, 'joint_velocities', None), 7)
    grip_forces = _safe_array(getattr(obs, 'gripper_touch_forces', None), 6)
    gripper_open = _safe_float(getattr(obs, 'gripper_open', None))
    total_f = float(
        np.linalg.norm(grip_forces[0:3]) + np.linalg.norm(grip_forces[3:6])
    )
    return {
        'step': step,
        'is_holding': total_f > 0.1 and gripper_open < 0.5,
        'gripper_open': gripper_open,
        'grip_force': total_f,
        'grip_force_delta': 0.0,
        'torque_norm': float(np.linalg.norm(joint_forces)),
        'torque_delta': 0.0,
        'collision': False,
        'threshold_crossed': False,
        'collision_reason': '',
        'suppression_reason': '',
        'collision_score': 0.0,
        'torque_norm_crossed': False,
        'torque_delta_crossed': False,
        'torque_weighted_score': 0.0,
        'grip_force_crossed': False,
        'grip_delta_crossed': False,
        # Momentum-observer (collision method 3) fields; filled by
        # update_momentum_observer once positions/velocities are available.
        'residual_norm': 0.0,
        'residual_score': 0.0,
        'residual_crossed': False,
        '_joint_forces': joint_forces.copy(),
        '_joint_positions': joint_positions.copy(),
        '_joint_velocities': joint_velocities.copy(),
    }


def update_deltas(logs):
    for i in range(1, len(logs)):
        raw_delta = float(
            np.linalg.norm(
                logs[i]['_joint_forces'] - logs[i - 1]['_joint_forces']
            )
        )
        # torque_delta is a collision signal only when the overall torque
        # magnitude is rising. A drop in torque_norm (relaxing / releasing a
        # grip) still produces a large joint-force difference, but it must not
        # trip the delta threshold, so zero it out on any non-increase.
        rising = logs[i]['torque_norm'] > logs[i - 1]['torque_norm']
        logs[i]['torque_delta'] = raw_delta if rising else 0.0
        logs[i]['grip_force_delta'] = abs(
            logs[i]['grip_force'] - logs[i - 1]['grip_force']
        )
    update_torque_baseline(logs)
    update_momentum_observer(logs)


def update_momentum_observer(logs, gain=None, dt=None):
    """Integrate the De Luca (IROS 2006) generalized-momentum residual (method 3).

    For each row, using q, qd (joint_positions / joint_velocities) and the
    measured joint torque tau (joint_forces):

        p_k          = M(q_k) qd_k
        integrand_k  = tau_k + C(q_k, qd_k)^T qd_k - g(q_k) + r_{k-1}
        acc_k        = acc_{k-1} + integrand_k * dt
        r_k          = K_I ( p_k - p_0 - acc_k )

    V2 uses backward Euler instead:
        b_k = tau_k + C(q_k, qd_k)^T qd_k - g(q_k)
        r_k = K_I * (p_k - p_0 - acc_{k-1} - dt*b_k) / (1 + dt*K_I)
        acc_k = acc_{k-1} + dt * (b_k + r_k)
    This solves the implicit residual feedback without joint acceleration.
    For positive gain/dt its homogeneous pole is 1/(1 + dt*K_I).

    r_0 = 0 by construction (acc_0 = 0). The recursion is causal and carries its
    running state (p_0, acc, r) on each row, so it advances incrementally as new
    frames arrive and survives the live detector trimming old rows from the front
    of `logs`. residual_norm = ||r||_2, residual_score = max_i |r_i| (Nm); the
    per-joint max is what the De Luca thresholds compare against.
    """
    if _panda_dyn is None:
        return
    # The observer costs ~15 dynamics evaluations per frame; only run it when the
    # momentum-observer method is actually selected so methods 1/2 stay fast.
    if os.getenv('AHA_COLLISION_METHOD', DEFAULT_COLLISION_METHOD).strip() != '3':
        return
    gain = DEFAULT_MOMENTUM_GAIN if gain is None else float(gain)
    dt = DEFAULT_MOMENTUM_DT if dt is None else float(dt)
    version = momentum_observer_version()
    if not np.isfinite(gain) or gain <= 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Momentum observer gain and dt must be finite and positive')
    if not logs:
        return

    # Find the last row that already carries observer state; resume from there so
    # only new frames are integrated (and truncation of old rows is harmless).
    seed = None
    start = 0
    for i in range(len(logs) - 1, -1, -1):
        if logs[i].get('_mo_computed'):
            seed = logs[i]
            if seed.get('_mo_config', ('1', gain, dt)) != (version, gain, dt):
                raise ValueError('Momentum observer configuration changed; start a fresh log')
            start = i + 1
            break

    if seed is None:
        # First-ever frame: p_0 anchors the observer, residual is zero.
        first = logs[0]
        q0 = first['_joint_positions']
        qd0 = first['_joint_velocities']
        p0 = _panda_dyn.mass_matrix(q0) @ qd0
        acc = np.zeros(_panda_dyn.N)
        r = np.zeros(_panda_dyn.N)
        _store_residual(first, p0, acc, r, np.zeros(_panda_dyn.N))
        first['_mo_config'] = (version, gain, dt)
        first['_mo_settling'] = int(first.get('step', 0)) < DEFAULT_RESIDUAL_SETTLE
        start = 1
        seed = first

    p0 = seed['_mo_p0']
    acc = seed['_mo_acc']
    r = seed['mo_residual']
    for i in range(start, len(logs)):
        row = logs[i]
        q = row['_joint_positions']
        qd = row['_joint_velocities']
        tau = row['_joint_forces']
        g = _panda_dyn.gravity_torque(q)
        ct_qd = _panda_dyn.coriolis_transpose_qd(q, qd)
        p = _panda_dyn.mass_matrix(q) @ qd
        if version == '2':
            b = tau + ct_qd - g
            r = gain * (p - p0 - acc - dt * b) / (1.0 + dt * gain)
            acc = acc + dt * (b + r)
        else:
            integrand = tau + ct_qd - g + r
            acc = acc + integrand * dt
            r = gain * (p - p0 - acc)
        _store_residual(row, p0, acc, r, r)
        row['_mo_config'] = (version, gain, dt)
        # The absolute frame index (step) marks the observer settle window; the
        # first DEFAULT_RESIDUAL_SETTLE frames carry the startup transient.
        row['_mo_settling'] = int(row.get('step', i)) < DEFAULT_RESIDUAL_SETTLE


def _store_residual(row, p0, acc, r, residual_vec):
    row['_mo_p0'] = p0
    row['_mo_acc'] = acc
    row['mo_residual'] = r
    row['_mo_computed'] = True
    row['residual_norm'] = float(np.linalg.norm(residual_vec))
    row['residual_score'] = float(np.max(np.abs(residual_vec))) if len(residual_vec) else 0.0


def update_torque_baseline(logs, window=None):
    """Track a running 'normal-motion' torque baseline and the rise above it.

    baseline[i] = median torque_norm over the trailing `window` frames strictly
    before i. It is causal (past frames only) so the live detector -- which
    recomputes over its growing log each step -- and the offline pipeline agree
    frame-for-frame. From it:

        torque_rise[i]      = torque_norm[i] - baseline[i]      (absolute jump)
        torque_rise_frac[i] = torque_rise[i] / baseline[i]      (proportional)

    These feed the proportional-spike method in check_collision. This is the
    collision mirror of the slip detector's held-peak / proportional-collapse:
    slip tracks a peak and watches for a fractional DROP; collision tracks a
    baseline and watches for a fractional RISE.
    """
    if window is None:
        window = DEFAULT_TORQUE_BASELINE_WINDOW
    window = max(1, int(window))
    norms = [float(r['torque_norm']) for r in logs]
    for i, row in enumerate(logs):
        prior = norms[max(0, i - window):i]
        baseline = float(np.median(prior)) if prior else norms[i]
        rise = norms[i] - baseline
        row['torque_baseline'] = baseline
        row['torque_rise'] = float(rise)
        row['torque_rise_frac'] = float(rise / baseline) if baseline > 1e-9 else 0.0


def robust_threshold(values, minimum, scale=6.0):
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return minimum
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    return max(float(minimum), median + scale * 1.4826 * mad)


def make_threshold_overrides(
    torque_norm=None,
    torque_delta=None,
    grip_force=None,
    grip_delta=None,
):
    overrides = {}
    if torque_norm is not None:
        overrides['tq_norm_free'] = float(torque_norm)
        overrides['tq_norm_hold'] = float(torque_norm)
    if torque_delta is not None:
        overrides['tq_delta_free'] = float(torque_delta)
        overrides['tq_delta_hold'] = float(torque_delta)
    if grip_force is not None:
        overrides['grip_thr'] = float(grip_force)
    if grip_delta is not None:
        overrides['grip_d_thr'] = float(grip_delta)
    return overrides


def _metric_threshold_from_task_stats(
    metric_stats,
    k=DEFAULT_TORQUE_STATS_K,
    use_max_floor=DEFAULT_TORQUE_STATS_USE_MAX_FLOOR,
):
    # Default threshold = max(summary_mean + K * summary_std,
    # summary_max + summary_std). With median_of_episode_stats, these are
    # median(mean), median(std), and median(max) across clean episodes.
    mean = float(metric_stats['mean'])
    std = float(metric_stats['std'])
    threshold = mean + float(k) * std
    if use_max_floor and metric_stats.get('max') is not None:
        threshold = max(threshold, float(metric_stats['max']) + std)
    return float(threshold)


def _median_of_episode_stats(metric):
    episodes = metric.get('episodes') or []
    out = {'count': len(episodes)}
    for key in ('mean', 'std', 'max', 'min'):
        vals = []
        for episode in episodes:
            stats = episode.get('stats') or {}
            if stats.get(key) is not None:
                vals.append(float(stats[key]))
        if vals:
            out[key] = float(np.median(vals))
    if 'mean' not in out or 'std' not in out:
        return None
    return out


def _metric_stats_for_field(metric, stats_field):
    if stats_field == 'median_of_episode_stats':
        return _median_of_episode_stats(metric)
    return metric.get(stats_field)


def task_torque_stats_thresholds(
    task_name,
    stats_dir=DEFAULT_TORQUE_STATS_DIR,
    k=DEFAULT_TORQUE_STATS_K,
    stats_field=DEFAULT_TORQUE_STATS_FIELD,
    use_max_floor=DEFAULT_TORQUE_STATS_USE_MAX_FLOOR,
):
    path = os.path.join(stats_dir, f'{task_name}_success_torque_stats.json')
    if not os.path.exists(path):
        return {}, {
            'enabled': False,
            'reason': f'torque stats JSON not found: {path}',
        }

    with open(path) as f:
        data = json.load(f)

    metrics = data.get('metrics', {})
    norm_stats = _metric_stats_for_field(
        metrics.get('torque_norm', {}),
        stats_field,
    )
    delta_stats = _metric_stats_for_field(
        metrics.get('torque_delta', {}),
        stats_field,
    )
    if not norm_stats or not delta_stats:
        return {}, {
            'enabled': False,
            'path': path,
            'reason': f'missing metric field: {stats_field}',
        }

    torque_norm = _metric_threshold_from_task_stats(
        norm_stats, k=k, use_max_floor=use_max_floor
    )
    torque_delta = _metric_threshold_from_task_stats(
        delta_stats, k=k, use_max_floor=use_max_floor
    )
    return make_threshold_overrides(
        torque_norm=torque_norm,
        torque_delta=torque_delta,
    ), {
        'enabled': True,
        'path': path,
        'k': float(k),
        'stats_field': stats_field,
        'use_max_floor': bool(use_max_floor),
        'torque_norm_threshold': torque_norm,
        'torque_delta_threshold': torque_delta,
    }


def apply_task_torque_stats_thresholds(
    settings,
    task_name,
    manual_threshold_keys=None,
):
    if not settings.get('use_task_torque_stats_thresholds', False):
        return settings

    manual_threshold_keys = set(manual_threshold_keys or ())
    overrides, info = task_torque_stats_thresholds(
        task_name,
        stats_dir=settings.get('torque_stats_dir', DEFAULT_TORQUE_STATS_DIR),
        k=settings.get('torque_stats_k', DEFAULT_TORQUE_STATS_K),
        stats_field=settings.get(
            'torque_stats_field', DEFAULT_TORQUE_STATS_FIELD
        ),
        use_max_floor=settings.get(
            'torque_stats_use_max_floor',
            DEFAULT_TORQUE_STATS_USE_MAX_FLOOR,
        ),
    )
    settings['task_torque_stats_thresholds'] = info
    if not overrides:
        return settings

    for key, value in overrides.items():
        if key not in manual_threshold_keys:
            settings['threshold_overrides'][key] = value
    return settings


def default_threshold_overrides():
    return {}


def default_detector_settings():
    return {
        'threshold_overrides': default_threshold_overrides(),
        'use_task_torque_stats_thresholds': (
            DEFAULT_USE_TASK_TORQUE_STATS_THRESHOLDS
        ),
        'torque_stats_dir': DEFAULT_TORQUE_STATS_DIR,
        'torque_stats_k': DEFAULT_TORQUE_STATS_K,
        'torque_stats_field': DEFAULT_TORQUE_STATS_FIELD,
        'torque_stats_use_max_floor': DEFAULT_TORQUE_STATS_USE_MAX_FLOOR,
        'task_torque_stats_thresholds': None,
        'torque_rule': DEFAULT_TORQUE_RULE,
        'torque_norm_weight': DEFAULT_TORQUE_NORM_WEIGHT,
        'torque_delta_weight': DEFAULT_TORQUE_DELTA_WEIGHT,
        'torque_score_threshold': DEFAULT_TORQUE_SCORE_THRESHOLD,
        'consecutive_collision_frames': DEFAULT_CONSECUTIVE_COLLISION_FRAMES,
        'use_touch_force': False,
    }


def apply_threshold_overrides(thresholds, overrides=None):
    if not overrides:
        return thresholds
    for key, value in overrides.items():
        if value is not None:
            thresholds[key] = float(value)
    return thresholds


def freeze_thresholds(baseline, threshold_overrides=None, residual_stats=None):
    """Compute detection thresholds once from the warmup baseline.

    residual_stats: optional clean-run per-joint residual max (from
    load_residual_stats). When given, the method-3 threshold is the De Luca
    per-joint vector max(FLOOR, clean_max_i) * MULT_i; otherwise it falls
    back to a scalar taken from the (settle-trimmed) warmup residual envelope.
    """
    free_rows = [r for r in baseline if not r['is_holding']]
    holding_rows = [r for r in baseline if r['is_holding']]

    def _thr(rows, key, minimum):
        vals = [r[key] for r in rows] if rows else [r[key] for r in baseline]
        return robust_threshold(vals, minimum=minimum)

    # Method-3 (momentum observer). Preferred: per-joint thresholds calibrated
    # from a clean run. Fallback: the warmup residual envelope, dropping the
    # observer's first-frame startup transients so they do not inflate it.
    residual_thr_vec = None
    if residual_stats is not None:
        residual_thr_vec = residual_threshold_vector(residual_stats)
        residual_thr = float(np.max(residual_thr_vec))
    else:
        settled = [r for r in baseline
                   if not r.get('_mo_settling', False)]
        residual_scores = [r.get('residual_score', 0.0)
                           for r in (settled or baseline)]
        residual_base = robust_threshold(residual_scores, minimum=0.0)
        residual_thr = max(DEFAULT_RESIDUAL_FLOOR, DEFAULT_RESIDUAL_K * residual_base)

    thr = {
        'tq_norm_free': _thr(free_rows, 'torque_norm', 45.0),
        'tq_delta_free': _thr(free_rows, 'torque_delta', 5.0),
        'tq_norm_hold': _thr(holding_rows, 'torque_norm', 45.0),
        'tq_delta_hold': _thr(holding_rows, 'torque_delta', 5.0),
        'grip_thr': _thr(free_rows, 'grip_force', 1.5),
        'grip_d_thr': _thr(free_rows, 'grip_force_delta', 0.75),
        'residual_thr': residual_thr,
        'residual_thr_vec': residual_thr_vec,
    }
    apply_threshold_overrides(thr, threshold_overrides)
    # Log the per-joint vector, not just its max: the scalar alone cannot say
    # which calibration (or which multipliers) a run actually used, which makes
    # after-the-fact analysis of a trace guesswork.
    if thr.get('residual_thr_vec') is not None:
        vec_txt = ('  residual_vec=['
                   + ' '.join(f'{v:.1f}' for v in thr['residual_thr_vec'])
                   + ']  joint_mult=['
                   + ' '.join(f'{v:g}' for v in DEFAULT_RESIDUAL_JOINT_MULT)
                   + f']  min_joints={DEFAULT_RESIDUAL_MIN_JOINTS}')
    else:
        vec_txt = '  residual_vec=none (warmup scalar fallback)'
    print(
        f"\n  [thresholds locked at step {WARMUP_STEPS}]  "
        f"tq_norm_free={thr['tq_norm_free']:.2f}  "
        f"tq_delta_free={thr['tq_delta_free']:.2f}  "
        f"tq_norm_hold={thr['tq_norm_hold']:.2f}  "
        f"tq_delta_hold={thr['tq_delta_hold']:.2f}  "
        f"grip={thr['grip_thr']:.2f}  "
        f"residual={thr.get('residual_thr', float('nan')):.2f}"
        f"{vec_txt}\n"
    )
    return thr


def rise_frac_gates_path(task_name, stats_dir=None):
    """Where the per-(task, waypoint) rise_frac gates JSON lives."""
    stats_dir = stats_dir or DEFAULT_TORQUE_STATS_DIR
    return os.path.join(stats_dir, f'{task_name}_rise_frac_gates.json')


def save_rise_frac_gates(task_name, gates, stats_dir=None):
    """gates: {waypoint_ordinal(int): gate(float)}. Written keyed by string."""
    path = rise_frac_gates_path(task_name, stats_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        'task': task_name,
        'floor': DEFAULT_TORQUE_RISE_FRAC,
        'gates': {str(int(k)): float(v) for k, v in gates.items()},
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def load_rise_frac_gates(task_name, stats_dir=None):
    """Return {waypoint_ordinal(int): gate(float)} for a task, or {} if absent.

    Used by the live/offline detector to pick the proportional-spike gate for
    whichever waypoint is currently executing. A missing waypoint falls back to
    DEFAULT_TORQUE_RISE_FRAC in check_collision.
    """
    path = rise_frac_gates_path(task_name, stats_dir)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return {
            int(k): max(DEFAULT_TORQUE_RISE_FRAC, float(v))
            for k, v in data.get('gates', {}).items()
        }
    except (ValueError, TypeError, OSError):
        return {}


# Headroom added above the largest clean rise_frac so a collision must clearly
# EXCEED what a clean run ever produced at that waypoint (not just tie it):
#   gate = max(FLOOR, clean_max * (1 + margin) + std_k * clean_std)
# margin = proportional headroom (0.05 = +5%); std_k scales headroom by the
# clean spread (0 = off). Both env-tunable.
DEFAULT_RISE_FRAC_GATE_MARGIN = float(os.getenv('AHA_COLLISION_GATE_MARGIN', '0.05'))
DEFAULT_RISE_FRAC_GATE_STD_K = float(os.getenv('AHA_COLLISION_GATE_STD_K', '0.0'))


def _finite(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return v if np.isfinite(v) else 0.0


def rise_frac_gate(max_clean_rise_frac, std_clean_rise_frac=0.0,
                   margin=None, std_k=None):
    """Per-waypoint proportional-spike gate from clean-run rise_frac stats.

    gate = max(FLOOR, clean_max * (1 + margin) + std_k * clean_std)

    FLOOR (DEFAULT_TORQUE_RISE_FRAC) keeps a sensible minimum for quiet
    waypoints where the clean max is tiny.
    """
    margin = DEFAULT_RISE_FRAC_GATE_MARGIN if margin is None else float(margin)
    std_k = DEFAULT_RISE_FRAC_GATE_STD_K if std_k is None else float(std_k)
    mx = _finite(max_clean_rise_frac)
    sd = _finite(std_clean_rise_frac)
    return max(DEFAULT_TORQUE_RISE_FRAC, mx * (1.0 + margin) + std_k * sd)


# --- Method-3 residual thresholds: per-task calibration from a clean run -------
# Mirrors the collision rise_frac gates / torque_stats. A clean episode records
# the per-joint residual envelope; the De Luca per-joint threshold is then
#     thr_i = max(RESIDUAL_FLOOR, clean_max_i) * MULT_i
# and a collision is any joint with |r_i| > thr_i. Falls back to the warmup
# baseline when no calibration file exists.
DEFAULT_RESIDUAL_STATS_DIR = os.getenv(
    'AHA_RESIDUAL_STATS_DIR',
    os.path.join(DEFAULT_CALIBRATION_ROOT, 'residual_stats'),
)


def residual_stats_path(task_name, stats_dir=None):
    stats_dir = stats_dir or DEFAULT_RESIDUAL_STATS_DIR
    suffix = '_v2' if momentum_observer_version() == '2' else ''
    return os.path.join(stats_dir, f'{task_name}_residual_stats{suffix}.json')


def save_residual_stats(task_name, per_joint_max, stats_dir=None):
    """per_joint_max: array-like of the clean-run max |r_i| for each joint."""
    path = residual_stats_path(task_name, stats_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        'task': task_name,
        'observer_version': momentum_observer_version(),
        'gain': DEFAULT_MOMENTUM_GAIN,
        'dt': DEFAULT_MOMENTUM_DT,
        'per_joint_max': [float(v) for v in np.asarray(per_joint_max).ravel()],
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    return path


def load_residual_stats(task_name, stats_dir=None):
    """Return the clean per-joint residual max (np.ndarray) or None if absent."""
    path = residual_stats_path(task_name, stats_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if (
            data.get('observer_version') != momentum_observer_version()
            or data.get('gain') != DEFAULT_MOMENTUM_GAIN
            or data.get('dt') != DEFAULT_MOMENTUM_DT
        ):
            return None
        vals = data.get('per_joint_max')
        return np.asarray(vals, dtype=float) if vals else None
    except (ValueError, TypeError, OSError):
        return None


def residual_threshold_vector(clean_per_joint_max, floor=None, mult=None):
    """De Luca per-joint thresholds from the clean-run residual envelope.

        thr_i = max(FLOOR, clean_max_i) * MULT_i

    mult: per-joint multiplier applied after the floor, re-weighting the joints
    by how informative they actually are (see DEFAULT_RESIDUAL_JOINT_MULT). It is
    now the ONLY headroom over the clean envelope: the former global K factor
    stacked on top of it, so every joint carried two multipliers at once and the
    per-joint weights could not be read off the resulting threshold.
    DEFAULT_RESIDUAL_K survives only for the scalar warmup fallback below, which
    has no per-joint vector to weight.
    """
    floor = DEFAULT_RESIDUAL_FLOOR if floor is None else float(floor)
    mult = DEFAULT_RESIDUAL_JOINT_MULT if mult is None else np.asarray(mult, dtype=float)
    mx = np.asarray(clean_per_joint_max, dtype=float).ravel()
    thr = np.maximum(floor, mx)
    return thr * mult if mult.shape == thr.shape else thr


def _weighted_torque_score(norm_ratio, delta_ratio, norm_weight, delta_weight):
    norm_weight = max(0.0, float(norm_weight))
    delta_weight = max(0.0, float(delta_weight))
    total_weight = norm_weight + delta_weight
    if total_weight <= 0.0:
        return 0.0
    return (
        norm_ratio * norm_weight + delta_ratio * delta_weight
    ) / total_weight


def check_collision(
    row,
    thr,
    torque_rule=TORQUE_RULE_EITHER,
    torque_norm_weight=0.5,
    torque_delta_weight=0.5,
    torque_score_threshold=1.0,
    use_touch_force=False,
    torque_rise_gate=None,
):
    """Return (is_collision, reason_string) for one telemetry row.

    torque_rise_gate: the proportional-spike gate for the CURRENT waypoint
    (method 1). None -> fall back to the global DEFAULT_TORQUE_RISE_FRAC. This is
    how the per-(task, waypoint) gates calibrated from clean runs are applied.
    """
    if torque_rule not in TORQUE_RULES:
        raise ValueError(
            f"Unknown torque rule '{torque_rule}'. Expected one of {TORQUE_RULES}."
        )

    holding = row['is_holding']
    tq_norm_thr = thr['tq_norm_hold'] if holding else thr['tq_norm_free']
    tq_delta_thr = thr['tq_delta_hold'] if holding else thr['tq_delta_free']
    norm_ratio = row['torque_norm'] / tq_norm_thr if tq_norm_thr else 0.0
    delta_ratio = row['torque_delta'] / tq_delta_thr if tq_delta_thr else 0.0
    norm_crossed = row['torque_norm'] > tq_norm_thr
    delta_crossed = row['torque_delta'] > tq_delta_thr
    weighted_score = _weighted_torque_score(
        norm_ratio, delta_ratio, torque_norm_weight, torque_delta_weight
    )

    # Proportional-spike fields (method 1). A sudden increase counts when it is
    # proportionally large: rise_frac >= the current waypoint's gate.
    rise_gate = (float(torque_rise_gate) if torque_rise_gate is not None
                 else DEFAULT_TORQUE_RISE_FRAC)
    rise_frac = float(row.get('torque_rise_frac', 0.0))
    prop_spike = bool(rise_frac >= rise_gate)

    row['torque_norm_crossed'] = norm_crossed
    row['torque_delta_crossed'] = delta_crossed
    row['torque_weighted_score'] = weighted_score
    row['torque_rise_gate'] = rise_gate
    row['torque_proportional_spike'] = prop_spike
    row['grip_force_crossed'] = (
        not holding and row['grip_force'] > thr['grip_thr']
    )
    row['grip_delta_crossed'] = (
        not holding and row['grip_force_delta'] > thr['grip_d_thr']
    )

    reasons = []

    # Method-3 (momentum observer) residual gate. De Luca declares a collision
    # per joint (|r_i| > thr_i). With a calibrated per-joint threshold vector we
    # require at least DEFAULT_RESIDUAL_MIN_JOINTS joints over threshold at once
    # (a whole-chain load, not a lone noisy joint); without it we fall back to a
    # single scalar on max_i|r_i|.
    residual_thr = float(thr.get('residual_thr', DEFAULT_RESIDUAL_FLOOR))
    residual_score = float(row.get('residual_score', 0.0))
    residual_thr_vec = thr.get('residual_thr_vec')
    r_vec = row.get('mo_residual')
    if residual_thr_vec is not None and r_vec is not None:
        n_joints_over = int(np.sum(np.abs(r_vec) > residual_thr_vec))
        residual_crossed = bool(n_joints_over >= DEFAULT_RESIDUAL_MIN_JOINTS)
    else:
        n_joints_over = int(residual_score > residual_thr)
        residual_crossed = bool(residual_score > residual_thr)
    row['residual_joints_over'] = n_joints_over
    row['residual_crossed'] = residual_crossed
    row['residual_threshold'] = residual_thr

    method = os.getenv('AHA_COLLISION_METHOD', DEFAULT_COLLISION_METHOD).strip()
    if method == '3':
        # De Luca generalized-momentum observer: the residual estimates the
        # external joint torque; a collision is any joint whose residual clears
        # the calibrated threshold.
        if residual_crossed:
            reasons.append('momentum_residual')
    elif method == '1':
        # Proportional method: a sudden, proportionally-large rise in torque_norm
        # above its running baseline.
        if prop_spike:
            reasons.append('torque_proportional_spike')
    elif torque_rule == TORQUE_RULE_EITHER:
        if delta_crossed:
            reasons.append('torque_spike')
        if norm_crossed:
            reasons.append('high_torque')
    elif torque_rule == TORQUE_RULE_BOTH:
        if norm_crossed and delta_crossed:
            reasons.append('torque_both')
    elif weighted_score >= float(torque_score_threshold):
        reasons.append('torque_weighted')

    if use_touch_force and not holding:
        if row['grip_force_crossed']:
            reasons.append('high_touch_force')
        if row['grip_delta_crossed']:
            reasons.append('touch_force_spike')

    score_parts = [
        delta_ratio,
        norm_ratio,
        weighted_score,
    ]
    if method == '1':
        # For the proportional method, the collision "confidence" is how far the
        # rise cleared the fractional gate (1.0 == exactly at the threshold).
        score_parts.append(
            rise_frac / rise_gate
            if rise_gate > 0.0
            else 0.0
        )
    if method == '3':
        # Method-3 confidence: how far the residual cleared its threshold
        # (1.0 == exactly at the threshold).
        score_parts.append(
            residual_score / residual_thr if residual_thr > 0.0 else 0.0
        )
    if use_touch_force and not holding:
        score_parts.extend([
            (
                row['grip_force'] / thr['grip_thr']
                if thr['grip_thr'] else 0.0
            ),
            (
                row['grip_force_delta'] / thr['grip_d_thr']
                if thr['grip_d_thr'] else 0.0
            ),
        ])
    row['collision_score'] = max(score_parts)
    return bool(reasons), '+'.join(reasons)
