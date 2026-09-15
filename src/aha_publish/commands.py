"""Small shared CLI helpers; simulator and API modules are loaded in workers."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
from aha_publish import paths
from aha_publish.common.tasks import validate_task


def parser(description, tasks=False):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--dry-run', action='store_true', help='Print commands without running them or writing outputs.')
    if tasks:
        selection = p.add_mutually_exclusive_group(required=True)
        selection.add_argument('--task', action='append', help='RLBench task name; repeat to select several.')
        selection.add_argument('--all', action='store_true', help='Select every task YAML in AHA_FAILGEN_ROOT/failgen/configs.')
    return p


def selected_tasks(args):
    names = args.task or sorted(p.stem for p in paths.CONFIGS_DIR.glob('*.yaml'))
    if not names:
        raise ValueError(f'No task configs in {paths.CONFIGS_DIR}; set AHA_FAILGEN_ROOT.')
    return list(dict.fromkeys(validate_task(name) for name in names))


def require(path, dry_run=False):
    if not dry_run and not Path(path).exists():
        raise ValueError(f'Missing input: {path}')


def environment():
    env = os.environ.copy()
    # Workers only search this release and its explicitly configured backends.
    # A shell used for the research checkout may carry old script/data paths.
    env['PYTHONPATH'] = os.pathsep.join([str(paths.SOURCE_DIR.parent), str(paths.FAILGEN_ROOT), str(paths.RLBENCH_ROOT)])
    env['AHA_CALIBRATION_ROOT'] = str(paths.CALIBRATION_DIR)
    env['AHA_OUTPUT_ROOT'] = str(paths.OUTPUT_DIR)
    env['AHA_FAILGEN_ROOT'] = str(paths.FAILGEN_ROOT)
    env['AHA_WP_CHAIN_DIR'] = str(paths.TTM_CONTEXT_DIR)
    env['AHA_RLBENCH_TASKS_DIR'] = str(paths.RLBENCH_ROOT / 'rlbench/tasks')
    for variable, folder in (
        ('AHA_TORQUE_STATS_DIR', 'torque_stats'),
        ('AHA_RESIDUAL_STATS_DIR', 'residual_stats'),
        ('AHA_TRANSITION_STATS_DIR', 'transition_arrival_stats'),
        ('AHA_ORIENTATION_STATS_DIR', 'orientation_arrival_stats'),
    ):
        env[variable] = str(paths.CALIBRATION_DIR / folder)
    env['AHA_GRIP_FORCE_STATS_PATH'] = str(paths.CALIBRATION_DIR / 'grip_force_stats/ALL_TASKS_success_grip_force_stats.json')
    env.pop('BT_TASK_CONTEXT_PATH', None)
    env['RLBENCH_ROOT'] = str(paths.RLBENCH_ROOT)
    env['COPPELIASIM_ROOT'] = str(paths.COPPELIASIM_ROOT)
    env['LD_LIBRARY_PATH'] = os.pathsep.join(filter(None, [str(paths.COPPELIASIM_ROOT), env.get('LD_LIBRARY_PATH', '')]))
    env.setdefault('QT_QPA_PLATFORM_PLUGIN_PATH', str(paths.COPPELIASIM_ROOT))
    return env


def run(module, arguments=(), dry_run=False, env=None):
    command = [sys.executable, '-m', 'aha_publish.' + module, *command_arguments(arguments)]
    execute(command, dry_run, env)


def command_arguments(arguments):
    # Preserve paths supplied relative to the caller before changing worker cwd.
    return [str(value.expanduser().resolve()) if isinstance(value, Path) else str(value)
            for value in arguments]


def run_stage(stage, arguments=(), dry_run=False):
    command = [sys.executable, str(paths.PROJECT_ROOT / stage / 'main.py'),
               *command_arguments(arguments)]
    execute(command, dry_run)


def execute(command, dry_run=False, env=None):
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, env=env or environment(), cwd=paths.PROJECT_ROOT,
                       check=True, stdin=subprocess.DEVNULL)


def entry(main):
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(2)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)
