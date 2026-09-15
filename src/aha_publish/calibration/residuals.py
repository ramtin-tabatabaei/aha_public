"""Collect and recompute De Luca thresholds from successful clean episodes."""
import csv
import json
from pathlib import Path

import numpy as np

from aha_publish import paths


def observer_parameters(detector):
    return {
        'observer_version': detector.momentum_observer_version(),
        'gain': detector.DEFAULT_MOMENTUM_GAIN,
        'dt': detector.DEFAULT_MOMENTUM_DT,
        'settle_steps': detector.DEFAULT_RESIDUAL_SETTLE,
        'warmup_steps': detector.WARMUP_STEPS,
    }


def expected_parameters(collision):
    from aha_publish.detectors.collision.detector import WARMUP_STEPS
    return {
        'observer_version': str(collision['momentum_version']),
        'gain': collision['momentum_gain'],
        'dt': collision['time_step'],
        'settle_steps': collision['residual_settle_steps'],
        'warmup_steps': WARMUP_STEPS,
    }


def write_episode(raw_csv, task, episode, logs, detector):
    """Store the settled clean maximum |r_i|, excluding observer startup frames."""
    values = [row['mo_residual'] for row in logs
              if row.get('mo_residual') is not None
              and not row.get('_mo_settling', False)
              and int(row.get('step', 0)) >= detector.WARMUP_STEPS]
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 7 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f'{task}: no valid settled De Luca residuals; check the dynamics model and clean run.')
    payload = dict(observer_parameters(detector), task=task, episode=episode,
                   frames=len(array), per_joint_max=np.abs(array).max(axis=0).tolist())
    filename = Path(raw_csv).with_suffix('.residual.json')
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text(json.dumps(payload, indent=2) + '\n')


def load_envelopes(raw_dir, task, expected):
    """Require a matching residual sidecar for every raw clean episode."""
    files = sorted((Path(raw_dir) / task).glob('ep*.csv'))
    if not files:
        raise ValueError(f'{task}: no clean episodes in {raw_dir}')
    maxima = []
    for raw_csv in files:
        filename = raw_csv.with_suffix('.residual.json')
        if not filename.exists():
            raise ValueError(f'Missing De Luca calibration data: {filename}. Run calibration/main.py all --task {task} --force.')
        data = json.loads(filename.read_text())
        if any(data.get(key) != value for key, value in expected.items()):
            raise ValueError(f'{filename}: observer parameters changed; collect fresh clean episodes with --force.')
        if data.get('task') != task or data.get('episode') != int(raw_csv.stem[2:]):
            raise ValueError(f'{filename}: residual data does not match this task/episode')
        values = np.asarray(data.get('per_joint_max'), dtype=float)
        if values.shape != (7,) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f'{filename}: expected seven finite, nonnegative residual maxima')
        maxima.append(values)
    return np.max(maxima, axis=0), len(files)


def compute(task, raw_dir, collision):
    """Apply the configured joint multipliers once and export a live-read baseline."""
    from aha_publish.detectors.collision.detector import residual_threshold_vector
    parameters = expected_parameters(collision)
    maxima, episodes = load_envelopes(raw_dir, task, parameters)
    thresholds = residual_threshold_vector(maxima, floor=collision['residual_floor'],
                                           mult=collision['residual_joint_multipliers'])
    suffix = '_v2' if collision['momentum_version'] == 2 else ''
    relative = f'residual_stats/{task}_residual_stats{suffix}.json'
    filename = paths.CALIBRATION_DIR / relative
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text(json.dumps(dict(parameters, task=task, episodes_completed=episodes,
                                       per_joint_max=maxima.tolist()), indent=2) + '\n')
    report = paths.CALIBRATION_DIR / 'threshold_report' / f'{task}_collision_residual.csv'
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['task', 'joint', 'clean_max_nm', 'floor_nm', 'multiplier', 'threshold_nm', 'episodes'])
        for index, (maximum, multiplier, threshold) in enumerate(zip(maxima, collision['residual_joint_multipliers'], thresholds), 1):
            writer.writerow([task, index, maximum, collision['residual_floor'], multiplier, threshold, episodes])
    return relative
