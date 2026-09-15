#!/usr/bin/env python3
"""Convert raw freezing-calibration telemetry -> per-task freezing baseline.

Reads the raw clean/freeze telemetry collected by collect.py
(outputs/backend_data/freezing_calibration/<task>.json) and writes the published baseline
(aha_output/freezing_stats/<task>.json) that the freezing detector reads.

The baseline is computed from the CLEAN run only: per-frame distributions of
joint_velocity_norm (jvn) and joint_position_delta (jpd), plus recommended
stillness thresholds. The freeze run is collected for validation/separation
checks; it is not part of the published stats.

Threshold design (reverse-engineered from the original committed baselines and
verified to reproduce them):
    jvn_thresh = clamp(p10(jvn) * 0.4, 0.045, 0.10)
    jpd_thresh = clamp(p10(jpd) * 0.4, 0.004, 0.008)
    consecutive_freeze_frames = 8
"""

from aha_publish import paths
import argparse
import json
from pathlib import Path

import numpy as np

RAW_DIR = paths.BACKEND_DATA_DIR / 'freezing_calibration'
STATS_DIR = (paths.OUTPUT_DIR / 'freezing_stats')

JVN_FLOOR, JVN_CEIL = 0.045, 0.10
JPD_FLOOR, JPD_CEIL = 0.004, 0.008
SCALE = 0.4
CONSECUTIVE_FREEZE_FRAMES = 8


def _clean(values):
    a = np.asarray(values, dtype=float)
    return a[~np.isnan(a)]


def _dist(values):
    a = _clean(values)
    return {
        'mean': float(a.mean()),
        'std': float(a.std()),
        'p5': float(np.percentile(a, 5)),
        'p10': float(np.percentile(a, 10)),
        'p50': float(np.percentile(a, 50)),
        'min': float(a.min()),
        'max': float(a.max()),
    }


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def build_stats(raw):
    clean = raw.get('clean', [])
    jvn = _dist([r['jvn'] for r in clean])
    jpd = _dist([r['jpd'] for r in clean])
    return {
        'task': raw['task'],
        'n_clean_episodes': 1,
        'joint_velocity_norm': jvn,
        'joint_position_delta': jpd,
        'recommended_thresholds': {
            'joint_velocity_norm': round(_clamp(jvn['p10'] * SCALE, JVN_FLOOR, JVN_CEIL), 5),
            'joint_position_delta': round(_clamp(jpd['p10'] * SCALE, JPD_FLOOR, JPD_CEIL), 5),
            'consecutive_freeze_frames': CONSECUTIVE_FREEZE_FRAMES,
        },
        'design': 'T=clamp(p10*0.4, floor, ceil); floors above absolute freeze level',
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tasks', nargs='*', help='tasks to analyze (default: all raw files)')
    ap.add_argument('--raw-dir', default=str(RAW_DIR))
    ap.add_argument('--out-dir', default=str(STATS_DIR))
    ap.add_argument('--overwrite', action='store_true',
                    help='rewrite baselines that already exist')
    args = ap.parse_args()

    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = args.tasks or sorted(p.stem for p in raw_dir.glob('*.json'))

    wrote = skipped = empty = 0
    for t in tasks:
        rp = raw_dir / f'{t}.json'
        if not rp.exists():
            print(f'  {t}: no raw telemetry, skipped')
            continue
        raw = json.loads(rp.read_text())
        if not raw.get('clean'):
            print(f'  {t}: empty clean run, skipped')
            empty += 1
            continue
        op = out_dir / f'{t}.json'
        if op.exists() and not args.overwrite:
            skipped += 1
            continue
        op.write_text(json.dumps(build_stats(raw), indent=2))
        wrote += 1
        print(f'  {t}: wrote baseline')
    print(f'done: wrote={wrote} skipped_existing={skipped} empty={empty}')


if __name__ == '__main__':
    main()
