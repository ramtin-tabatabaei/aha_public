#!/usr/bin/env python3
"""Collect clean telemetry once; recompute and apply thresholds from one config."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from aha_publish import paths
from aha_publish.commands import entry, environment, parser, require, run
from aha_publish.calibration.settings import DEFAULT_SETTINGS, compute_arguments, detector_environment, load_settings
from aha_publish.common.tasks import validate_task


def main():
    p = parser('Calibrate detectors: collect clean episodes, compute thresholds, or do both.')
    p.add_argument('action', choices=['collect', 'compute', 'all'])
    p.add_argument('--task', action='append', help='Task name; repeat to select several. Default: all available tasks.')
    p.add_argument('--episodes', type=int, default=10, help='Clean episodes per task for collect/all.')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--force', action='store_true', help='Collect fresh clean episodes, replacing this task\'s previous baseline.')
    p.add_argument('--config', type=Path, default=DEFAULT_SETTINGS)
    p.add_argument('--raw-dir', type=Path, help='Read previously collected clean_data from another location (compute only).')
    args = p.parse_args()
    if args.episodes < 1 or args.workers < 1:
        p.error('--episodes and --workers must be positive')
    if args.raw_dir and args.action != 'compute':
        p.error('--raw-dir is only supported with compute')
    if args.force and args.action == 'compute':
        p.error('--force is only supported with collect/all')
    config = load_settings(args.config)
    raw = args.raw_dir or paths.CALIBRATION_DIR / 'clean_data'
    if args.task:
        tasks = list(dict.fromkeys(validate_task(t) for t in args.task))
    elif args.action == 'compute':
        tasks = sorted(d.name for d in raw.glob('*') if d.is_dir() and any(d.glob('ep*.csv')))
    else:
        tasks = sorted(f.stem for f in paths.CONFIGS_DIR.glob('*.yaml'))
    if not tasks:
        p.error('No tasks found; supply --task and check the configured paths.')
    env = environment()
    env.update(detector_environment(config))
    if args.action in ('collect', 'all'):
        for task in tasks:
            require(paths.TTM_CONTEXT_DIR / f'{task}.llm_context.json', args.dry_run)
            require(paths.CONFIGS_DIR / f'{task}.yaml', args.dry_run)
            require(paths.RLBENCH_ROOT / 'rlbench/task_ttms' / f'{task}.ttm', args.dry_run)
        options = ['--episodes', args.episodes, '--workers', args.workers, '--out-root', paths.CALIBRATION_DIR,
                   '--force' if args.force else '--skip-completed']
        for task in tasks:
            options += ['--task', task]
        if args.force and not args.dry_run:
            for task in tasks:
                (paths.CALIBRATION_DIR / 'manifests' / f'{task}.json').unlink(missing_ok=True)
        run('calibration.calibrate_all_clean', options, args.dry_run, env)
    if not args.dry_run:
        from aha_publish.calibration.artifacts import validate_raw
        for task in tasks:
            validate_raw(raw, task, args.episodes if args.action in ('all', 'collect') else 1)
            if config['collision']['method'] == 3:
                from aha_publish.calibration.residuals import expected_parameters, load_envelopes
                load_envelopes(raw, task, expected_parameters(config['collision']))
    if args.action in ('compute', 'all'):
        options = ['--out-root', paths.CALIBRATION_DIR, '--raw-dir', raw, *compute_arguments(config)]
        for task in tasks:
            options += ['--task', task]
        # Invalidate old manifests before replacing calibration files, so a
        # partially failed computation cannot be used by a later run.
        if not args.dry_run:
            for task in tasks:
                (paths.CALIBRATION_DIR / 'manifests' / f'{task}.json').unlink(missing_ok=True)
        run('calibration.compute_thresholds', options, args.dry_run, env)
        if not args.dry_run:
            from aha_publish.calibration.artifacts import finalize
            for task in tasks:
                finalize(task, raw, config)
            print(f'Calibration and runtime settings saved in {paths.CALIBRATION_DIR}')


if __name__ == '__main__':
    entry(main)
