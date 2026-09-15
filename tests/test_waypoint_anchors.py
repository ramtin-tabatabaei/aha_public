"""Regression checks for independently anchored position and orientation.

Run: python -m unittest discover -s aha_scripts -p test_waypoint_anchors.py
"""

import copy
import unittest

import numpy as np

from aha_publish.ttm.inspect_ttm import build_llm_report_json
from aha_publish.common.waypoint_anchors import infer_waypoint_anchors
from aha_publish.common.waypoint_chain import (
    WaypointChain, compose, pose_error, rpy_to_quat,
    spec_from_inspection_report, validate_spec,
)


def pose(xyz, rpy=(0, 0, 0)):
    return np.r_[xyz, rpy_to_quat(rpy)]


WAYPOINTS = [
    {'name': 'waypoint0', 'parent': 'base'},
    {'name': 'waypoint1', 'parent': 'waypoint0'},
    {'name': 'waypoint2', 'parent': 'base'},
]


def episode(seed):
    rng = np.random.default_rng(seed)
    base = pose(rng.uniform(-1, 1, 3), (0, 0, rng.uniform(-3, 3)))
    obj = compose(base, pose(rng.uniform(-.3, .3, 3), (0, 0, rng.uniform(-3, 3))))
    wp0 = compose(base, pose([0, 0, 0], (np.pi, 0, 0)))
    wp0[:3] = compose(obj, pose([0, 0, .1]))[:3]
    return {'base': base, 'object': obj, 'waypoint0': wp0,
            'waypoint1': compose(wp0, pose([0, 0, .08], (0, 0, .9))),
            'waypoint2': compose(base, pose([.4, .3, .2]))}


def report(entries):
    return {'task_name': 'synthetic', 'waypoints': entries}


class AnchorTests(unittest.TestCase):
    def setUp(self):
        self.episodes = [episode(i) for i in range(5)]
        self.entries = infer_waypoint_anchors(WAYPOINTS, self.episodes)
        self.spec = spec_from_inspection_report(report(self.entries))

    def test_split_anchors_generalize_to_unseen_resets(self):
        self.assertEqual(self.entries[0]['position_parent'], 'object')
        self.assertEqual(self.entries[0]['orientation_parent'], 'base')
        self.assertEqual(self.entries[1]['position_parent'], 'waypoint0')
        self.assertEqual(validate_spec(self.spec), [])
        chain = WaypointChain(self.spec)
        for seed in range(5, 30):
            scene = episode(seed)
            actual = chain.resolve({n: scene[n] for n in chain.root_objects}, strict=True)
            for i in range(3):
                dp, dq = pose_error(actual[i], scene[f'waypoint{i}'])
                self.assertLess(dp, 1e-8)
                self.assertLess(dq, 1e-7)

    def test_reference_stays_frozen_when_target_or_object_moves(self):
        scene = episode(10)
        chain = WaypointChain(self.spec)
        chain.resolve({n: scene[n] for n in chain.root_objects})
        reference = np.array(chain.pose(0))
        scene['object'][:3] += [.2, 0, 0]
        shifted_target = scene['waypoint0'].copy()
        shifted_target[0] += .2
        np.testing.assert_array_equal(chain.pose(0), reference)
        self.assertAlmostEqual(pose_error(chain.pose(0), shifted_target)[0], .2)

    def test_missing_position_root_does_not_use_stale_parent_offset(self):
        chain = WaypointChain(self.spec)
        result = chain.resolve({'base': episode(10)['base']})
        self.assertEqual(set(result), {2})
        self.assertEqual(chain.unresolved(), [0, 1])

    def test_no_valid_anchor_abstains_with_descendants(self):
        episodes = copy.deepcopy(self.episodes)
        for i, scene in enumerate(episodes):
            scene['waypoint0'][0] += i * .1
            scene['waypoint1'] = compose(scene['waypoint0'], pose([0, 0, .08]))
        entries = infer_waypoint_anchors(WAYPOINTS, episodes)
        self.assertIn('reference_unavailable', entries[0])
        spec = spec_from_inspection_report(report(entries))
        chain = WaypointChain(spec)
        chain.resolve({n: episodes[0][n] for n in chain.root_objects})
        self.assertEqual(chain.unresolved(), [0, 1])
        self.assertEqual(spec['dynamic_waypoints'], [0])

    def test_explicit_dynamic_guard_is_preserved(self):
        data = report(self.entries)
        data['dynamic_waypoints'] = [0]
        chain = WaypointChain(spec_from_inspection_report(data))
        chain.resolve({n: self.episodes[0][n] for n in chain.root_objects})
        self.assertEqual(chain.unresolved(), [0, 1])

    def test_reset_inference_never_certifies_live_hooks_or_repeats(self):
        from aha_publish.common.rlbench_dynamic_waypoints import analyze_source, unresolved_dynamic_waypoints
        source = '''
class Example:
    def init_episode(self, index):
        w = Dummy('waypoint0')
        w.set_position([0, 0, index])
    def init_task(self):
        self.register_waypoint_ability_start(1, self.move)
    def move(self, waypoint):
        waypoint.get_waypoint_object().set_position([0, 0, 1])
'''
        analysis = analyze_source(source, 3)
        self.assertEqual(analysis['runtime_dynamic'], [1])
        self.assertEqual(unresolved_dynamic_waypoints(analysis, report(self.entries)), [1])
        source += "\n    def repeat(self):\n        self.register_waypoints_should_repeat(self.again)\n"
        analysis = analyze_source(source, 3)
        self.assertEqual(unresolved_dynamic_waypoints(analysis, report(self.entries)), [0, 1, 2])

    def test_mixed_frame_cycle_is_rejected(self):
        spec = copy.deepcopy(self.spec)
        spec['nodes']['waypoint0']['position_parent'] = 'waypoint1'
        self.assertTrue(any('cyclic' in p for p in validate_spec(spec)))
        with self.assertRaisesRegex(ValueError, 'cyclic'):
            WaypointChain(spec).resolve({'base': pose([0, 0, 0])}, strict=True)

    def test_legacy_single_parent_report(self):
        entries = [{'name': 'waypoint0', 'parent': 'base',
                    'local_position_xyz_m': [.1, .2, .3],
                    'local_quaternion_xyzw': [0, 0, 0, 1]}]
        chain = WaypointChain(spec_from_inspection_report(report(entries)))
        base = episode(7)['base']
        chain.resolve({'base': base}, strict=True)
        np.testing.assert_allclose(chain.pose(0), compose(base, pose([.1, .2, .3])))

    def test_variation_mismatch_clears_previous_references(self):
        spec = dict(self.spec, reference_variation_index=2)
        chain = WaypointChain(spec)
        chain.resolve({n: self.episodes[0][n] for n in chain.root_objects}, variation_index=2)
        with self.assertRaisesRegex(ValueError, 'variation'):
            chain.resolve({}, variation_index=3)
        self.assertIsNone(chain.pose(0))

    def test_report_writer_preserves_component_frames(self):
        import json
        data = json.loads(build_llm_report_json(
            'synthetic', 'unused', WAYPOINTS,
            {'live_waypoint_world_poses': self.entries, 'reference_variation_index': 2}))
        self.assertEqual(data['reference_variation_index'], 2)
        self.assertEqual(data['waypoints'][0]['position_parent'], 'object')
        self.assertEqual(data['waypoints'][0]['orientation_parent'], 'base')

    def test_standalone_detectors_use_the_same_frames(self):
        import json
        import tempfile
        from aha_publish.detectors.orientation import detector as orientation
        from aha_publish.detectors.transition import detector as transition
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
            json.dump(report(self.entries), f)
            f.flush()
            scene = episode(12)
            roots = {n: scene[n] for n in self.spec['root_objects']}
            for module, fn, key in (
                (orientation, orientation.recalculate_ttm_waypoint_poses, 'poses'),
                (transition, transition.recalculate_ttm_waypoint_positions, 'positions'),
            ):
                info = module.load_ttm_waypoint_equations('synthetic', context_path=f.name)
                self.assertIn('object', info['parent_names'])
                calculated = fn(info, roots)
                self.assertEqual(calculated['unresolved'], {})
                for i, value in calculated[key].items():
                    np.testing.assert_allclose(value[:3], scene[f'waypoint{i}'][:3], atol=1e-8)


if __name__ == '__main__':
    unittest.main()
