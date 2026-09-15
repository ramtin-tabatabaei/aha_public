"""Validate one tuning file and carry its parameters into live detector runs."""
import json
import math
from pathlib import Path
from aha_publish import paths

DEFAULT_SETTINGS = paths.PROJECT_ROOT / 'config/thresholds.json'


def load_settings(filename=DEFAULT_SETTINGS):
    defaults = json.loads(DEFAULT_SETTINGS.read_text())
    config = json.loads(Path(filename).read_text())
    if not isinstance(config, dict) or config.keys() != defaults.keys():
        raise ValueError('Threshold config must contain transition, orientation, collision, and slip.')
    for group, values in config.items():
        if not isinstance(values, dict) or values.keys() != defaults[group].keys():
            raise ValueError(f'Unexpected or missing settings in {group}. See config/thresholds.json.')
        for key, value in values.items():
            if key == 'residual_joint_multipliers':
                if (not isinstance(value, list) or len(value) != 7
                        or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in value)):
                    raise ValueError('collision.residual_joint_multipliers must contain seven finite, positive numbers')
                continue
            if key == 'max_floor':
                if type(value) is not bool:
                    raise ValueError('collision.max_floor must be true or false')
            elif key == 'grip_ceiling' and value is None:
                continue
            elif type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{group}.{key} must be a finite, nonnegative number')
            if key in ('prop_window', 'baseline_window') and (type(value) is not int or value < 1):
                raise ValueError(f'{group}.{key} must be a positive integer')
            if key in ('prop_floor', 'arrival_floor', 'arrival_scale', 'rise_scale', 'prop_ratio_scale') and value <= 0:
                raise ValueError(f'{group}.{key} must be positive')
            if key in ('momentum_gain', 'time_step', 'residual_floor', 'residual_warmup_multiplier') and value <= 0:
                raise ValueError(f'{group}.{key} must be positive')
            if key == 'residual_settle_steps' and type(value) is not int:
                raise ValueError('collision.residual_settle_steps must be a nonnegative integer')
    c = config['collision']
    if type(c['method']) is not int or c['method'] not in (1, 2, 3):
        raise ValueError('collision.method must be 1 (rise), 2 (absolute torque), or 3 (De Luca).')
    if type(c['momentum_version']) is not int or c['momentum_version'] not in (1, 2):
        raise ValueError('collision.momentum_version must be 1 or 2')
    if type(c['residual_min_joints']) is not int or not 1 <= c['residual_min_joints'] <= 7:
        raise ValueError('collision.residual_min_joints must be an integer from 1 to 7')
    slip = config['slip']
    if slip['grip_ceiling'] is not None and slip['grip_ceiling'] < slip['grip_floor']:
        raise ValueError('slip.grip_ceiling must be null or >= grip_floor')
    return config


def compute_arguments(config):
    args = []
    for group, prefix in [('transition', 'trans'), ('orientation', 'ori')]:
        for key, value in config[group].items():
            args += [f'--{prefix}-{key.replace("_", "-")}', str(value)]
    collision, slip = config['collision'], config['slip']
    args += ['--torque-k', str(collision['torque_k']), '--torque-max-floor' if collision['max_floor'] else '--no-torque-max-floor']
    for key, flag in [('grip_k', 'grip-k'), ('grip_floor', 'grip-floor'), ('grip_ceiling', 'grip-ceiling'), ('min_force', 'grip-min-force')]:
        args += ['--' + flag, str(slip[key]) if slip[key] is not None else 'inf']
    return args


def detector_environment(config):
    c, s = config['collision'], config['slip']
    return {
        'AHA_COLLISION_METHOD': str(c['method']),
        'AHA_COLLISION_MOMENTUM_VERSION': str(c['momentum_version']),
        'AHA_COLLISION_MOMENTUM_GAIN': str(c['momentum_gain']),
        'AHA_SIM_DT': str(c['time_step']),
        'AHA_COLLISION_RESIDUAL_FLOOR': str(c['residual_floor']),
        'AHA_COLLISION_RESIDUAL_JOINT_MULT': ','.join(map(str, c['residual_joint_multipliers'])),
        'AHA_COLLISION_RESIDUAL_K': str(c['residual_warmup_multiplier']),
        'AHA_COLLISION_RESIDUAL_SETTLE': str(c['residual_settle_steps']),
        'AHA_COLLISION_RESIDUAL_MIN_JOINTS': str(c['residual_min_joints']),
        'AHA_TRANSITION_ARRIVAL_THRESHOLD': str(config['transition']['arrival_floor']),
        'AHA_ORIENTATION_ARRIVAL_THRESHOLD': str(config['orientation']['arrival_floor']),
        'AHA_COLLISION_TORQUE_K': str(c['torque_k']),
        'AHA_COLLISION_MAX_FLOOR': str(int(c['max_floor'])),
        'AHA_COLLISION_BASELINE_WINDOW': str(c['baseline_window']),
        'AHA_COLLISION_RISE_FRAC': str(c['rise_fraction']),
        'AHA_COLLISION_GATE_MARGIN': str(c['gate_margin']),
        'AHA_COLLISION_GATE_STD_K': str(c['gate_std_k']),
        'AHA_SLIP_METHOD': '1',
        'AHA_SLIP_GRIP_K': str(s['grip_k']),
        'AHA_SLIP_GRIP_FLOOR': str(s['grip_floor']),
        'AHA_SLIP_GRIP_CEILING': str(s['grip_ceiling']) if s['grip_ceiling'] is not None else 'inf',
    }


def runtime_environment(env, tasks, dry_run=False):
    selected = None
    for task in tasks:
        manifest = paths.CALIBRATION_DIR / 'manifests' / f'{task}.json'
        if dry_run:
            continue
        if not manifest.exists():
            raise ValueError(f'No calibration manifest for {task}. Run calibration/main.py first.')
        data = json.loads(manifest.read_text())
        settings = data['runtime_environment']
        if selected is not None and selected != settings:
            raise ValueError('Selected tasks use different calibration settings; run them separately or recalibrate together.')
        selected = settings
        for artifact in data['artifacts']:
            if not (paths.CALIBRATION_DIR / artifact).is_file():
                raise ValueError(f'Missing calibration artifact: {artifact}; recompute this task.')
    env.update(selected or {})
    return env
