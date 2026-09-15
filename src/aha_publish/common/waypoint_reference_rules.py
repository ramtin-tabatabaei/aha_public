"""Inspection-exported pose equations for dynamic waypoint hooks.

The inspector compiles task hooks into restricted, declarative assignments and
pose writes. Runtime evaluates this JSON against task state, cached scene-object
poses and the inspection's waypoint chain. It never reads a simulator waypoint.
No eval/exec, simulator setters, or arbitrary task methods are invoked here.
"""

from aha_publish import paths

import ast
import copy
import operator

import numpy as np

from aha_publish.common.waypoint_chain import compose, quat_to_rpy, relative_pose, rpy_to_quat


SETTERS = {'set_position', 'set_orientation', 'set_quaternion', 'set_pose', 'set_parent'}


def export_hook_reference_rules(source):
    """Compile straight-line pose equations; unsupported control flow is explicit."""
    tree = ast.parse(source)
    methods = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    hooks, errors = {}, []
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            continue
        if call.func.attr not in ('register_waypoint_ability_start', 'register_waypoint_ability_end'):
            continue
        try:
            index = ast.literal_eval(call.args[0])
            method = methods[call.args[1].attr]
            phase = 'start' if call.func.attr.endswith('start') else 'end'
            setters = [n for n in ast.walk(method) if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Attribute) and n.func.attr in SETTERS]
            if not setters:
                continue
            def compile_statements(statements):
                instructions = []
                for stmt in statements:
                    if isinstance(stmt, ast.Assign):
                        if len(stmt.targets) != 1:
                            raise ValueError('multiple assignment targets')
                        instructions.append({'assign': ast.unparse(stmt.targets[0]),
                                             'value': ast.unparse(stmt.value)})
                    elif isinstance(stmt, ast.AugAssign):
                        value = ast.BinOp(left=stmt.target, op=stmt.op, right=stmt.value)
                        instructions.append({'assign': ast.unparse(stmt.target),
                                             'value': ast.unparse(value)})
                    elif isinstance(stmt, ast.If):
                        instructions.append({'if': ast.unparse(stmt.test),
                                             'then': compile_statements(stmt.body),
                                             'else': compile_statements(stmt.orelse)})
                    elif (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                          and isinstance(stmt.value.func, ast.Attribute)
                          and stmt.value.func.attr in SETTERS):
                        instructions.append({'write': ast.unparse(stmt.value)})
                    elif isinstance(stmt, (ast.Raise, ast.Pass)):
                        continue
                    elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                        continue
                    elif (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                          and isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == 'print'):
                        continue
                    else:
                        raise ValueError(f'unsupported hook statement {type(stmt).__name__}')
                return instructions
            instructions = compile_statements(method.body)
            params = [a.arg for a in method.args.args if a.arg != 'self']
            hooks[f'{phase}:{index}'] = {'parameter': params[0] if params else None,
                                        'instructions': instructions}
        except (ValueError, KeyError, AttributeError, IndexError) as exc:
            errors.append(str(exc))
    return {'version': 1, 'hooks': hooks, 'errors': errors}


def referenced_object_names(rules, task=None):
    """Scene objects the equations will read, so the caller can cache them first.

    Two ways an equation names an object: a literal constructor -- Dummy('x'),
    Shape('x') -- and a task attribute holding one or a list of them, e.g.
    `self.grasp_points[self.variation_index]`. The second kind cannot be read off
    the source, so the object NAMES (never their geometry) are taken from the live
    task here; the caller still snapshots the poses at the moment it chooses.

    Without this the attribute-selected objects were absent from the pose cache
    and every equation using one died on a KeyError, costing its waypoints their
    reference (place_shape_in_shape_sorter's 'cube_grasp_point').

    Waypoint objects are excluded: the chain owns those, and reading one from the
    simulator is exactly what these rules exist to avoid.
    """
    names, attributes = set(), set()
    for program in (rules.get('hooks') or {}).values():
        def walk(instructions):
            for instruction in instructions:
                if 'if' in instruction:
                    sources = [instruction['if']]
                    walk(instruction.get('then') or [])
                    walk(instruction.get('else') or [])
                else:
                    sources = [instruction.get('value'), instruction.get('write')]
                for source in filter(None, sources):
                    for node in ast.walk(ast.parse(source, mode='eval')):
                        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                                and node.func.id in ('Dummy', 'Shape', 'ProximitySensor')
                                and node.args and isinstance(node.args[0], ast.Constant)):
                            names.add(node.args[0].value)
                        elif (isinstance(node, ast.Attribute)
                              and isinstance(node.value, ast.Name)
                              and node.value.id == 'self'):
                            attributes.add(node.attr)
        walk(program.get('instructions') or [])
    for attribute in sorted(attributes) if task is not None else ():
        value = getattr(task, attribute, None)
        stack = [value]
        while stack:
            item = stack.pop()
            if isinstance(item, (list, tuple, np.ndarray)):
                stack.extend(item)
            elif hasattr(item, 'get_name'):
                try:
                    names.add(item.get_name())
                except Exception:
                    pass
    return sorted(n for n in names
                  if isinstance(n, str) and 'waypoint' not in n.lower())


class ObjectReference:
    def __init__(self, name):
        self.name = name


class ReferenceRuleEvaluator:
    """Evaluate inspection equations with a read-only, name-based object reader."""
    def __init__(self, chain, task, scene_pose):
        self.chain, self.task, self.scene_pose = chain, task, scene_pose
        self.locals = {}
        self._nodes = None
        self._world = {}
        self.state = {}
        self.failed = False

    def _object(self, value):
        if isinstance(value, ObjectReference):
            return value
        # Only the name of a task-owned PyRep object is used. Geometry goes
        # through _pose, which forbids live reads for every waypoint name.
        if hasattr(value, 'get_name'):
            return ObjectReference(value.get_name())
        return value

    def _pose(self, ref):
        ref = self._object(ref)
        if ref.name in self._nodes:
            from aha_publish.common.waypoint_chain import resolve_waypoint_poses
            # Resolve all chain nodes, including decorated waypoint parents.
            spec = dict(self.chain.spec, nodes=self._nodes,
                        waypoints=[{'index': 0, 'name': ref.name}])
            poses = resolve_waypoint_poses(spec, self._world, strict=True)
            return poses[0]
        if 'waypoint' in ref.name.lower():
            raise ValueError(f'waypoint {ref.name} is missing from inspection data')
        pose = np.asarray(self.scene_pose(ref.name), dtype=float)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError(f'invalid scene pose for {ref.name}')
        self._world[ref.name] = pose.copy()
        return pose

    def expression(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id == 'self':
                return self.task
            return self.locals[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self.expression(v) for v in node.elts]
        if isinstance(node, ast.IfExp):
            return self.expression(node.body if self.expression(node.test) else node.orelse)
        if isinstance(node, ast.BoolOp):
            values = [self.expression(v) for v in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.Compare):
            left = self.expression(node.left)
            for op, right_node in zip(node.ops, node.comparators):
                right = self.expression(right_node)
                test = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
                        ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne}[type(op)]
                if not test(left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.Subscript):
            return self.expression(node.value)[self.expression(node.slice)]
        if isinstance(node, ast.UnaryOp):
            return {ast.USub: operator.neg, ast.UAdd: operator.pos}[type(node.op)](self.expression(node.operand))
        if isinstance(node, ast.BinOp):
            op = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                  ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
                  ast.Mod: operator.mod, ast.Pow: operator.pow}[type(node.op)]
            return op(self.expression(node.left), self.expression(node.right))
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == 'self':
                if node.attr in self.state:
                    return self.state[node.attr]
                constants = self.chain.spec.get('reference_state', {})
                if node.attr in constants:
                    self.state[node.attr] = copy.deepcopy(constants[node.attr])
                    return self.state[node.attr]
                value = getattr(self.task, node.attr)
                # Scalars and object selections are live state; geometric arrays
                # must have been exported by inspect_ttm, never read from the task.
                if isinstance(value, (list, tuple, np.ndarray)):
                    flat = np.asarray(value, dtype=object).flatten()
                    if any(isinstance(v, (float, np.floating)) for v in flat):
                        raise ValueError(f'geometric task state {node.attr} missing from inspection')
                    # Copy containers without cloning simulator objects. Writes
                    # to selector arrays in a hook remain local to the evaluator.
                    def clone_container(v):
                        if isinstance(v, (list, tuple, np.ndarray)):
                            return [clone_container(item) for item in v]
                        return v
                    value = clone_container(value)
                    self.state[node.attr] = value
                return value
            if isinstance(node.value, ast.Name) and node.value.id in ('np', 'math') and node.attr == 'pi':
                return np.pi
            raise ValueError('unsupported attribute in reference equation')
        if isinstance(node, ast.Call):
            args = [self.expression(a) for a in node.args]
            kwargs = {k.arg: self.expression(k.value) for k in node.keywords}
            if isinstance(node.func, ast.Name):
                name = node.func.id
                if name in ('Dummy', 'Shape', 'ProximitySensor'):
                    return ObjectReference(args[0])
                if name in ('list', 'float', 'int', 'abs', 'min', 'max'):
                    return {'list': list, 'float': float, 'int': int, 'abs': abs,
                            'min': min, 'max': max}[name](*args)
                raise ValueError(f'unsupported function {name}')
            if isinstance(node.func, ast.Attribute):
                ref = self._object(self.expression(node.func.value))
                method = node.func.attr
                if method == 'get_waypoint_object' and isinstance(ref, ObjectReference):
                    return ref
                if not isinstance(ref, ObjectReference):
                    raise ValueError('only object pose operations are allowed')
                relative = kwargs.get('relative_to', (args[0] if args else None)
                                      if method.startswith('get_') else (args[1] if len(args) > 1 else None))
                if method.startswith('get_'):
                    pose = self._pose(ref)
                    if relative is not None:
                        pose = relative_pose(self._pose(relative), pose)
                    if method == 'get_position':
                        return pose[:3].tolist()
                    if method == 'get_orientation':
                        return quat_to_rpy(pose[3:]).tolist()
                    if method == 'get_quaternion':
                        return pose[3:].tolist()
                    if method == 'get_pose':
                        return pose.tolist()
                elif method in SETTERS:
                    if ref.name not in self._nodes:
                        raise ValueError('reference rules may only write waypoint references')
                    if method == 'set_pose':
                        pose = np.asarray(args[0], dtype=float).copy()
                        if pose.shape != (7,) or not np.isfinite(pose).all():
                            raise ValueError('invalid inspection pose equation')
                        if relative is not None:
                            pose = compose(self._pose(relative), pose)
                        self._world['__reference_world__'] = np.array([0, 0, 0, 0, 0, 0, 1.])
                        self._nodes[ref.name] = {
                            'parent': '__reference_world__',
                            'position_parent': '__reference_world__',
                            'local_position': pose[:3].tolist(),
                            'local_quaternion': pose[3:].tolist(), 'kind': 'waypoint'}
                        return None
                    pose = self._pose(ref).copy()
                    if method == 'set_parent':
                        parent = self._object(args[0])
                        parent_pose = self._pose(parent)
                        entry = self._nodes[ref.name]
                        local = relative_pose(parent_pose, pose)
                        if kwargs.get('keep_in_place', args[1] if len(args) > 1 else True):
                            entry['local_position'] = local[:3].tolist()
                            entry['local_quaternion'] = local[3:].tolist()
                        entry['parent'] = entry['position_parent'] = parent.name
                        return None
                    if relative is not None:
                        pose = relative_pose(self._pose(relative), pose)
                    if method == 'set_position':
                        pose[:3] = args[0]
                    elif method == 'set_orientation':
                        pose[3:] = rpy_to_quat(args[0])
                    elif method == 'set_quaternion':
                        pose[3:] = args[0]
                    else:
                        pose[:] = args[0]
                    if relative is not None:
                        pose = compose(self._pose(relative), pose)
                    entry = self._nodes[ref.name]
                    # Independent position/orientation frames remain explicit.
                    qparent = ObjectReference(entry['parent'])
                    pparent = ObjectReference(entry.get('position_parent', entry['parent']))
                    entry['local_position'] = relative_pose(self._pose(pparent), pose)[:3].tolist()
                    entry['local_quaternion'] = relative_pose(self._pose(qparent), pose)[3:].tolist()
                    return None
                raise ValueError(f'unsupported object operation {method}')
        raise ValueError(f'unsupported reference expression {type(node).__name__}')

    def _assign(self, target, value):
        if isinstance(target, ast.Name):
            self.locals[target.id] = value
        elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
            self.state[target.attr] = copy.deepcopy(value)
        elif isinstance(target, ast.Subscript):
            self.expression(target.value)[self.expression(target.slice)] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            if len(target.elts) != len(value):
                raise ValueError('reference assignment arity mismatch')
            for t, v in zip(target.elts, value):
                self._assign(t, v)
        else:
            raise ValueError('unsupported reference assignment')

    def apply(self, phase, index):
        if self.failed:
            return
        rules = self.chain.spec.get('hook_reference_rules') or {}
        program = rules.get('hooks', {}).get(f'{phase}:{index}')
        if program is None:
            return
        self._nodes = copy.deepcopy(self.chain.spec['nodes'])
        self._world = dict(self.chain.world_poses)
        self.locals = {program.get('parameter'): ObjectReference(f'waypoint{index}')}
        self.state = {}
        def execute(instructions):
            for instruction in instructions:
                if 'if' in instruction:
                    test = self.expression(ast.parse(instruction['if'], mode='eval').body)
                    execute(instruction['then'] if test else instruction['else'])
                elif 'assign' in instruction:
                    value = self.expression(ast.parse(instruction['value'], mode='eval').body)
                    self._assign(ast.parse(instruction['assign'], mode='eval').body, value)
                else:
                    self.expression(ast.parse(instruction['write'], mode='eval').body)
        execute(program['instructions'])
        self.chain.spec['nodes'] = self._nodes
        self.chain.resolve(self._world)
