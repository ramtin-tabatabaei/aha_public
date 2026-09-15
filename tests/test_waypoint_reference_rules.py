"""Reference equations must use inspection geometry, never live waypoints."""

import unittest
from types import SimpleNamespace

import numpy as np

from aha_publish.common.waypoint_chain import WaypointChain, compose, rpy_to_quat, spec_from_inspection_report
from aha_publish.common.waypoint_reference_rules import ReferenceRuleEvaluator, export_hook_reference_rules


class NameOnlyObject:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def get_pose(self, *args, **kwargs):
        raise AssertionError('simulator pose access is forbidden')

    get_position = get_orientation = get_quaternion = get_pose


SOURCE = '''
class Task:
    def init_task(self):
        self.register_waypoint_ability_start(0, self.prepare)
    def prepare(self, waypoint):
        w = Dummy('waypoint0')
        x, y, z = self.targets[self.count].get_position()
        ox, oy, _ = w.get_orientation()
        _, _, yaw = self.targets[self.count].get_orientation()
        w.set_position([x, y, z + 0.1])
        w.set_orientation([ox, oy, -yaw])
'''


class RuleTests(unittest.TestCase):
    def make(self, source=SOURCE):
        rules = export_hook_reference_rules(source)
        self.assertEqual(rules['errors'], [])
        report = {'task_name': 'synthetic', 'dynamic_waypoints': [0, 1],
                  'rule_resolved_waypoints': [0, 1], 'hook_reference_rules': rules,
                  'reference_state': {'offset': [0, 0, .23]},
                  'waypoints': [
                      {'name': 'waypoint0', 'parent': 'base',
                       'local_position_xyz_m': [0, 0, 0],
                       'local_quaternion_xyzw': rpy_to_quat([np.pi, 0, 0]).tolist()},
                      {'name': 'waypoint1', 'parent': 'waypoint0',
                       'local_position_xyz_m': [0, 0, .1],
                       'local_quaternion_xyzw': [0, 0, 0, 1]}]}
        chain = WaypointChain(spec_from_inspection_report(report))
        base = [0, 0, 0, 0, 0, 0, 1]
        chain.resolve({'base': base})
        objects = {'base': base, 'a': [1, 2, 3, *rpy_to_quat([0, 0, .7])],
                   'b': [2, 3, 4, *rpy_to_quat([0, 0, -.4])]}
        reads = []
        def read(name):
            self.assertNotIn('waypoint', name)
            reads.append(name)
            return objects[name]
        task = SimpleNamespace(count=0, targets=[NameOnlyObject('a'), NameOnlyObject('b')],
                               w=NameOnlyObject('waypoint0'), offset=[999, 999, 999])
        return chain, ReferenceRuleEvaluator(chain, task, read), task, reads

    def test_repeated_targets_and_orientation_from_inspection_chain(self):
        chain, engine, task, reads = self.make()
        for count in (0, 1):
            task.count = count
            engine.apply('start', 0)
            np.testing.assert_allclose(chain.pose(0)[:3], [1 + count, 2 + count, 3.1 + count])
            expected = rpy_to_quat([np.pi, 0, -.7 if count == 0 else .4])
            np.testing.assert_allclose(chain.pose(0)[3:], expected, atol=1e-8)
            np.testing.assert_allclose(chain.pose(1), compose(chain.pose(0), [0, 0, .1, 0, 0, 0, 1]))
        self.assertTrue(reads)
        self.assertEqual(set(reads), {'base', 'a', 'b'})

    def test_inspection_constant_overrides_task_geometry(self):
        source = SOURCE[:SOURCE.index('    def prepare')] + '''    def prepare(self, waypoint):
        self.w.set_position(self.offset, relative_to=self.targets[self.count])
'''
        chain, engine, task, _ = self.make(source)
        engine.apply('start', 0)
        np.testing.assert_allclose(chain.pose(0)[:3], [1, 2, 3.23])
        self.assertEqual(task.offset, [999, 999, 999])

    def test_state_writes_are_shadowed_and_conditionals_supported(self):
        source = SOURCE[:SOURCE.index('    def prepare')] + '''    def prepare(self, waypoint):
        self.count += 1
        if self.count == 1:
            self.offset[0] = 0.2
            self.w.set_position(self.offset, self.targets[self.count])
'''
        chain, engine, task, _ = self.make(source)
        engine.apply('start', 0)
        self.assertEqual(task.count, 0)
        self.assertEqual(task.offset, [999, 999, 999])
        expected = compose([2, 3, 4, *rpy_to_quat([0, 0, -.4])], [.2, 0, .23, 0, 0, 0, 1])
        np.testing.assert_allclose(chain.pose(0)[:3], expected[:3])

    def test_reparenting_changes_only_reference_graph(self):
        source = SOURCE[:SOURCE.index('    def prepare')] + '''    def prepare(self, waypoint):
        self.w.set_parent(self.targets[self.count])
        self.w.set_position(self.offset, relative_to=self.targets[self.count])
'''
        chain, engine, task, _ = self.make(source)
        engine.apply('start', 0)
        self.assertEqual(chain.spec['nodes']['waypoint0']['parent'], 'a')
        np.testing.assert_allclose(chain.pose(0)[:3], [1, 2, 3.23])

    def test_missing_inspection_geometry_cannot_fall_back_to_task(self):
        source = SOURCE[:SOURCE.index('    def prepare')] + '''    def prepare(self, waypoint):
        self.w.set_position(self.offset)
'''
        chain, engine, _, _ = self.make(source)
        chain.spec['reference_state'] = {}
        # Use float array to identify geometric state, not an integer selector.
        engine.task.offset = [999., 999., 999.]
        before = chain.pose(0)
        with self.assertRaisesRegex(ValueError, 'missing from inspection'):
            engine.apply('start', 0)
        np.testing.assert_array_equal(chain.pose(0), before)

    def test_unrestricted_python_is_never_executed(self):
        source = SOURCE[:SOURCE.index('    def prepare')] + '''    def prepare(self, waypoint):
        value = __import__('os').getcwd()
        self.w.set_position(value)
'''
        chain, engine, _, _ = self.make(source)
        with self.assertRaises(ValueError):
            engine.apply('start', 0)


if __name__ == '__main__':
    unittest.main()
