"""Portable CLI, scoring, and offline-to-live calibration regression checks."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
# These tests copy portable source, not machine-specific environments or assets.
SOURCE_COPY_IGNORE = shutil.ignore_patterns(
    '__pycache__', 'outputs', '.git', '.conda', '.cache', '.venv', 'external',
)


class PublicWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.work = Path(self.temp.name)
        self.env = dict(os.environ, AHA_OUTPUT_ROOT=str(self.work / 'outputs'),
                        AHA_CALIBRATION_ROOT=str(self.work / 'outputs/calibration'),
                        PYTHONPATH=str(ROOT / 'src'), MPLBACKEND='Agg',
                        MPLCONFIGDIR=str(self.work / 'matplotlib'))
        self.env.pop('AHA_GRIP_FORCE_STATS_PATH', None)

    def tearDown(self):
        self.temp.cleanup()

    def command(self, script, *arguments, ok=True):
        result = subprocess.run([sys.executable, str(ROOT / script), *map(str, arguments)],
                                cwd=self.work, env=self.env, capture_output=True, text=True)
        if ok:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def raw_episodes(self):
        folder = self.work / 'raw/example_task'
        folder.mkdir(parents=True)
        columns = ['waypoint', 'local_step', 'path_done', 'distance_m', 'angle_rad', 'torque_norm', 'torque_delta', 'grip_force', 'grip_force_drop']
        # Two occurrences of waypoint 0 must be treated as independent segments.
        for episode in range(2):
            with (folder / f'ep{episode}.csv').open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(columns)
                for i in range(32):
                    writer.writerow([0, i % 16, 0, 0.04 if i % 16 else 0.1,
                                     0.2, 10 + i % 3, 1 + i % 2, 4 + i % 3, i % 2])
        return folder.parent

    def test_help_and_dry_run_work_without_simulator_or_outputs(self):
        stages = ['01_ttm_context', '02_descriptions', '03_behavior_trees', '04_running', '05_scoring', 'calibration']
        for stage in stages:
            self.command(stage + '/main.py', '--help')
            args = ['compute', '--task', 'example_task'] if stage == 'calibration' else ([] if stage == '05_scoring' else ['--task', 'example_task'])
            self.command(stage + '/main.py', *args, '--dry-run')
        self.assertFalse((self.work / 'outputs').exists())

    def test_missing_input_fails_before_running(self):
        result = self.command('03_behavior_trees/main.py', '--task', 'example_task', ok=False)
        self.assertIn('Missing input:', result.stderr)
        self.assertFalse((self.work / 'outputs').exists())

    def test_invalid_config_writes_nothing(self):
        config = json.loads((ROOT / 'config/thresholds.json').read_text())
        config['transition']['prop_window'] = 0
        path = self.work / 'bad.json'
        path.write_text(json.dumps(config))
        self.command('calibration/main.py', 'compute', '--task', 'example_task', '--config', path, ok=False)
        self.assertFalse((self.work / 'outputs').exists())

    def test_empty_episode_is_not_a_successful_calibration(self):
        folder = self.work / 'raw/example_task'
        folder.mkdir(parents=True)
        (folder / 'ep0.csv').write_text('waypoint\n')
        self.command('calibration/main.py', 'compute', '--task', 'example_task', '--raw-dir', folder.parent, ok=False)
        self.assertFalse((self.work / 'outputs').exists())

    def test_calibration_collects_without_generated_behavior_trees(self):
        spec = importlib.util.spec_from_file_location('calibration_cli', ROOT / 'calibration/main.py')
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        local = SimpleNamespace(
            CALIBRATION_DIR=self.work / 'calibration',
            TTM_CONTEXT_DIR=self.work / 'context',
            CONFIGS_DIR=self.work / 'configs',
            RLBENCH_ROOT=self.work / 'RLBench',
            BT_DIR=self.work / 'missing_behavior_trees',
        )
        inputs = [local.TTM_CONTEXT_DIR / 'example_task.llm_context.json',
                  local.CONFIGS_DIR / 'example_task.yaml',
                  local.RLBENCH_ROOT / 'rlbench/task_ttms/example_task.ttm']
        for path in inputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}')
        config = json.loads((ROOT / 'config/thresholds.json').read_text())
        config['collision']['method'] = 2
        settings = self.work / 'settings.json'
        settings.write_text(json.dumps(config))
        raw = self.raw_episodes()

        def collect(module, *args):
            self.assertEqual(module, 'calibration.calibrate_all_clean')
            shutil.copytree(raw, local.CALIBRATION_DIR / 'clean_data')

        argv = ['calibration/main.py', 'collect', '--task', 'example_task',
                '--episodes', '2', '--config', str(settings)]
        with mock.patch.object(cli, 'paths', local), \
             mock.patch.object(sys, 'argv', argv), \
             mock.patch.object(cli, 'run', side_effect=collect) as worker:
            cli.main()
            worker.assert_called_once()
            self.assertFalse(local.BT_DIR.exists())
            for path in inputs[1:]:
                with self.subTest(missing=path.name):
                    path.unlink()
                    worker.reset_mock()
                    with self.assertRaisesRegex(ValueError, 'Missing input:'):
                        cli.main()
                    worker.assert_not_called()
                    path.write_text('{}')

    def test_calibration_reports_match_live_detectors_with_custom_settings(self):
        raw = self.raw_episodes()
        config = json.loads((ROOT / 'config/thresholds.json').read_text())
        config['collision'].update(method=2, torque_k=5.0, max_floor=False)
        config['slip'].update(grip_k=1.0, grip_floor=0.4, min_force=4.1)
        settings = self.work / 'settings.json'
        settings.write_text(json.dumps(config))
        self.command('calibration/main.py', 'compute', '--task', 'example_task', '--raw-dir', raw, '--config', settings)
        calibration = self.work / 'outputs/calibration'
        manifest = json.loads((calibration / 'manifests/example_task.json').read_text())
        self.assertEqual(manifest['parameters'], config)
        self.assertEqual(manifest['episodes'], 2)
        env = dict(self.env, **manifest['runtime_environment'])
        code = '''import json
from aha_publish.detectors.slip.detector import default_detector_settings, apply_task_grip_force_stats_thresholds
from aha_publish.detectors.collision.detector import default_detector_settings as collision_settings, apply_task_torque_stats_thresholds
slip = apply_task_grip_force_stats_thresholds(default_detector_settings(), 'example_task')
collision = apply_task_torque_stats_thresholds(collision_settings(), 'example_task')
assert slip['task_grip_force_stats_thresholds']['enabled']
assert collision['task_torque_stats_thresholds']['enabled']
print(json.dumps({'slip': slip['threshold_overrides'], 'collision': collision['threshold_overrides']}))
'''
        result = subprocess.run([sys.executable, '-c', code], env=env, cwd=self.work, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        live = json.loads(result.stdout)
        with (calibration / 'threshold_report/slip.csv').open() as handle:
            slip = next(csv.DictReader(handle))
        self.assertAlmostEqual(live['slip']['grip_force_threshold'], float(slip['threshold']), places=4)
        with (calibration / 'threshold_report/collision.csv').open() as handle:
            for row in csv.DictReader(handle):
                self.assertAlmostEqual(live['collision'][{'torque_norm': 'tq_norm_free', 'torque_delta': 'tq_delta_free'}[row['metric']]], float(row['threshold']), places=4)
        # Runtime automatically applies the manifest even when shell defaults differ.
        code = "from aha_publish.calibration.settings import runtime_environment; e=runtime_environment({}, ['example_task']); assert e['AHA_COLLISION_METHOD']=='2' and e['AHA_SLIP_GRIP_K']=='1.0'"
        result = subprocess.run([sys.executable, '-c', code], env=self.env, cwd=self.work, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_repeat_waypoints_do_not_join_signal_histories(self):
        from aha_publish.calibration.compute_thresholds import per_waypoint_arrays
        episode = [{'waypoint': '0', 'local_step': str(i), 'path_done': '0', 'signal': str(v)} for i, v in [(0, 100), (1, 1), (0, 50), (1, 2)]]
        arrival, runup, _ = per_waypoint_arrays([episode], 'signal', 5, 0.01)
        self.assertEqual(arrival[0], [2.0])
        self.assertEqual(runup[0], [0.0])

    def test_de_luca_default_and_live_thresholds_match_offline_report(self):
        from aha_publish.calibration.residuals import expected_parameters
        config = json.loads((ROOT / 'config/thresholds.json').read_text())
        c = config['collision']
        self.assertEqual(c['method'], 3)
        self.assertEqual(c['residual_joint_multipliers'], [5.0, 1.1, 1.1, 1.1, 1.1, 2.0, 1.1])
        raw = self.raw_episodes()
        for episode in range(2):
            data = dict(expected_parameters(c), task='example_task', episode=episode,
                        per_joint_max=[float(j + episode) for j in range(1, 8)])
            (raw / 'example_task' / f'ep{episode}.residual.json').write_text(json.dumps(data))
        self.command('calibration/main.py', 'compute', '--task', 'example_task', '--raw-dir', raw)
        calibration = self.work / 'outputs/calibration'
        manifest = json.loads((calibration / 'manifests/example_task.json').read_text())
        self.assertIn('residual_stats/example_task_residual_stats.json', manifest['artifacts'])
        env = dict(self.env, **manifest['runtime_environment'])
        code = '''import json
from aha_publish.detectors.collision import detector as d
from aha_publish.running import detector_vlm_config as config
from aha_publish.calibration.calibrate_all_clean import CD
assert CD._panda_dyn is not None
assert config.apply_detector_vlm_env({})['AHA_COLLISION_METHOD'] == '3'
assert d.DEFAULT_COLLISION_METHOD == '3'
stats = d.load_residual_stats('example_task')
assert stats is not None
print(json.dumps(d.residual_threshold_vector(stats).tolist()))
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=self.work, env=env,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        live = json.loads(result.stdout)
        expected = [max(3, j + 2) * m for j, m in enumerate(c['residual_joint_multipliers'])]
        self.assertEqual(live, expected)
        with (calibration / 'threshold_report/example_task_collision_residual.csv').open() as handle:
            self.assertEqual([float(row['threshold_nm']) for row in csv.DictReader(handle)], expected)

    def test_de_luca_requires_matching_clean_observer_data(self):
        raw = self.raw_episodes()
        result = self.command('calibration/main.py', 'compute', '--task', 'example_task', '--raw-dir', raw, ok=False)
        self.assertIn('Missing De Luca calibration data', result.stderr)
        self.assertFalse((self.work / 'outputs/calibration/manifests/example_task.json').exists())
        from aha_publish.calibration.residuals import expected_parameters
        c = json.loads((ROOT / 'config/thresholds.json').read_text())['collision']
        for episode in range(2):
            data = dict(expected_parameters(c), task='example_task', episode=episode,
                        per_joint_max=[1.0] * 7)
            data['gain'] += 1
            (raw / 'example_task' / f'ep{episode}.residual.json').write_text(json.dumps(data))
        result = self.command('calibration/main.py', 'compute', '--task', 'example_task', '--raw-dir', raw, ok=False)
        self.assertIn('observer parameters changed', result.stderr)

    def test_de_luca_collection_excludes_startup_and_unsettled_residuals(self):
        from aha_publish.calibration.residuals import write_episode
        detector = SimpleNamespace(momentum_observer_version=lambda: '1', DEFAULT_MOMENTUM_GAIN=25.0,
                                   DEFAULT_MOMENTUM_DT=0.05, DEFAULT_RESIDUAL_SETTLE=3, WARMUP_STEPS=5)
        rows = [dict(step=0, mo_residual=[1000] * 7),
                dict(step=5, mo_residual=[2000] * 7, _mo_settling=True),
                dict(step=6, mo_residual=[-2, 1, -4, 3, -6, 5, -8]),
                dict(step=7, mo_residual=[1, -3, 2, -5, 4, -7, 6])]
        raw = self.work / 'ep0.csv'
        write_episode(raw, 'example_task', 0, rows, detector)
        data = json.loads(raw.with_suffix('.residual.json').read_text())
        self.assertEqual(data['per_joint_max'], [2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(data['frames'], 2)

    def test_invalid_joint_multipliers_are_rejected(self):
        config = json.loads((ROOT / 'config/thresholds.json').read_text())
        for value in ([1] * 6, [1] * 6 + [0], [1] * 6 + [float('nan')]):
            config['collision']['residual_joint_multipliers'] = value
            filename = self.work / 'settings.json'
            filename.write_text(json.dumps(config))
            result = self.command('calibration/main.py', 'compute', '--task', 'example_task', '--config', filename, ok=False)
            self.assertIn('seven finite, positive numbers', result.stderr)

    def test_scoring_an_existing_csv_is_offline(self):
        sample = self.work / 'samples.csv'
        with sample.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['task', 'failure_condition', 'waypoint', 'gt', 'B2', 'A1', 'A2', 'A3', 'A4'])
            writer.writerow(['example_task', 'clean', 0, 'success', '', 'success', 'success', 'success', 'success'])
            writer.writerow(['example_task', 'collision_wp1', 1, 'failure,collision', '', 'success', 'failure,collision', 'failure,collision', 'failure,collision'])
        self.command('05_scoring/main.py', '--samples', sample, '--methods', 'A4')
        files = list((self.work / 'outputs/scores').glob('*.csv'))
        self.assertTrue(files)
        self.assertFalse((self.work / 'outputs/runs').exists())

    def test_folder_can_be_copied_away_from_original_repository(self):
        destination = self.work / 'standalone'
        shutil.copytree(ROOT, destination, ignore=SOURCE_COPY_IGNORE)
        env = dict(self.env, PYTHONPATH='')
        result = subprocess.run([sys.executable, str(destination / '03_behavior_trees/main.py'), '--task', 'example_task', '--dry-run'], env=env, cwd=self.work, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('aha_publish.behavior_trees.cli', result.stdout)

    def test_pipeline_generates_all_stages_without_reusing_evidence(self):
        result = self.command('run_pipeline.py', '--task', 'example_task', '--dry-run')
        commands = result.stdout.splitlines()
        self.assertEqual(len(commands), 6)
        for line, stage in zip(commands, ['01_ttm_context', '02_descriptions', '03_behavior_trees', 'calibration', '04_running', '05_scoring']):
            self.assertIn(stage + '/main.py', line)
        self.assertIn('--force', commands[3])
        self.assertIn('--redo', commands[4])
        self.assertNotIn('--reuse-evidence', result.stdout)
        self.assertNotIn('--raw-dir', result.stdout)
        self.assertNotIn('--samples', result.stdout)
        self.assertFalse((self.work / 'outputs').exists())

    def test_workers_ignore_the_original_checkout_and_old_artifact_overrides(self):
        destination = self.work / 'standalone'
        shutil.copytree(ROOT, destination, ignore=SOURCE_COPY_IGNORE)
        legacy = self.work / 'old_checkout'
        legacy.mkdir()
        (legacy / 'legacy_module_marker.py').write_text('raise AssertionError("old checkout imported")\n')
        probe = destination / 'src/aha_publish/_isolation_probe.py'
        probe.write_text('''import importlib.util
import os
from pathlib import Path
import sys
from aha_publish import paths
forbidden = Path(os.environ['AHA_TEST_FORBIDDEN_ROOT'])
# The local Conda interpreter and TMPDIR may both live inside that checkout.
# Allow the runtime and relocated copy, while still blocking original sources.
allowed = (paths.PROJECT_ROOT.resolve(), Path(sys.prefix).resolve())
def audit(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        filename = Path(os.fsdecode(args[0])).resolve()
        if filename.is_relative_to(forbidden) and not any(filename.is_relative_to(p) for p in allowed):
            raise AssertionError('Read from the original checkout: ' + str(filename))
sys.addaudithook(audit)
assert Path.cwd() == paths.PROJECT_ROOT
assert importlib.util.find_spec('legacy_module_marker') is None
from aha_publish.descriptions.config import DEFAULT_GRIPPER_SEQUENCE_DIR
from aha_publish.descriptions.gripper_ground_truth import DESC_DIR
from aha_publish.behavior_trees.logic import main
from aha_publish.detectors.slip.detector import DEFAULT_GRIP_FORCE_STATS_PATH
from aha_publish.running.live_detectors.orientation import ARRIVAL_STATS_DIR
from aha_publish.common.waypoint_chain import DEFAULT_CHAIN_DIR
assert DEFAULT_GRIPPER_SEQUENCE_DIR == paths.GRIPPER_DIR
assert DESC_DIR == paths.DESCRIPTION_DIR
assert Path(DEFAULT_CHAIN_DIR) == paths.TTM_CONTEXT_DIR
assert Path(os.environ['AHA_WP_CHAIN_DIR']) == paths.TTM_CONTEXT_DIR
assert Path(DEFAULT_GRIP_FORCE_STATS_PATH).is_relative_to(paths.CALIBRATION_DIR)
assert ARRIVAL_STATS_DIR.is_relative_to(paths.CALIBRATION_DIR)
assert 'BT_TASK_CONTEXT_PATH' not in os.environ
for name, module in list(sys.modules.items()):
    if name.startswith('aha_publish') and getattr(module, '__file__', None):
        assert Path(module.__file__).is_relative_to(paths.SOURCE_DIR)
''')
        env = dict(self.env, PYTHONPATH=os.pathsep.join([str(destination / 'src'), str(legacy)]),
                   AHA_WP_CHAIN_DIR=str(legacy), AHA_ORIENTATION_STATS_DIR=str(legacy),
                   AHA_GRIP_FORCE_STATS_PATH=str(legacy / 'old.json'), BT_TASK_CONTEXT_PATH=str(legacy / 'old.json'),
                   AHA_TEST_FORBIDDEN_ROOT=str(ROOT.parent))
        result = subprocess.run([sys.executable, '-c', "from aha_publish.commands import run; run('_isolation_probe')"],
                                env=env, cwd=legacy, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_new_gripper_evidence_is_written_before_description_generation(self):
        from aha_publish.descriptions import runner
        from aha_publish.descriptions.context_blocks import gripper_sequence_block
        sequence = self.work / 'new_outputs/gripper_sequences/example_task.json'
        description = self.work / 'new_outputs/descriptions/example_task.json'
        task_paths = SimpleNamespace(task_name='example_task', grid_image_path=self.work / 'grid.png',
                                     gripper_sequence_path=sequence, output_json_path=description)
        args = SimpleNamespace(extract_gripper_sequence=True, include_gripper_sequence=True,
                               add_held_object=True, dry_run=True)
        capture = {'task': 'example_task', 'n_waypoints': 1, 'tip_travel_m': 0.2,
                   'waypoints': [{'index': 0, 'action': 'open', 'state_after': 'Open',
                                  'gripper_open_amount': 1.0, 'held_after': []}]}
        def generate(task, options, client):
            self.assertEqual(json.loads(sequence.read_text()), capture)
            self.assertIn('| wp0 | open | Open | 1.000 |', gripper_sequence_block(task, options))
            return {'waypoints': [{'waypoint': 0, 'gripper_state': 'Closed', 'held_object': 'ball'}]}, runner.Usage()
        with mock.patch.object(runner, 'build_task_paths', return_value=task_paths), \
             mock.patch('aha_publish.descriptions.extract_gripper_sequence.extract', return_value=capture), \
             mock.patch.object(runner, 'analyze_task', side_effect=generate):
            runner.run_one('example_task', args, None)
        self.assertIsNone(json.loads(description.read_text())['waypoints'][0]['held_object'])

    def test_failed_gripper_capture_stops_before_api_call(self):
        from aha_publish.descriptions import runner
        task_paths = SimpleNamespace(task_name='example_task', grid_image_path=self.work / 'grid.png',
                                     gripper_sequence_path=self.work / 'sequence.json')
        with mock.patch.object(runner, 'build_task_paths', return_value=task_paths), \
             mock.patch('aha_publish.descriptions.extract_gripper_sequence.extract', side_effect=RuntimeError('capture failed')), \
             mock.patch.object(runner, 'analyze_task') as analyze:
            with self.assertRaisesRegex(RuntimeError, 'capture failed'):
                runner.run_one('example_task', SimpleNamespace(extract_gripper_sequence=True), None)
            analyze.assert_not_called()

    def test_missing_gripper_evidence_is_not_silently_ignored(self):
        from aha_publish.descriptions.context_blocks import gripper_sequence_block
        with self.assertRaisesRegex(ValueError, 'Missing or empty gripper sequence'):
            gripper_sequence_block(SimpleNamespace(gripper_sequence_path=self.work / 'absent.json'),
                                   SimpleNamespace(include_gripper_sequence=True))


if __name__ == '__main__':
    unittest.main()
