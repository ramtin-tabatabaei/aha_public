"""Resolve RLBench waypoint world poses from a TTM-derived chain spec.

This is the transform composition that CoppeliaSim normally does for us. In the
simulator, waypoint dummies are parented to other waypoints and ultimately to a
scene object, so ``waypoint.get_pose()`` silently walks that chain and returns a
world pose. Every such call is a live sim read.

Here the chain is derived from ``ttm_inspection_reports/*.llm_context.json``,
which stores every waypoint's and object's WORLD pose after one init_episode.
World poses are episode-specific, so we convert them once into the invariant
parent-local offsets (spec_from_inspection_report). At run time we read
only the world poses of the root objects (``ball``, ``oven``, ...) and compose the
rest ourselves:

    T_world(node) = T_world(parent) . T_local(node)

which in position/quaternion form is

    p_world = p_parent + R(q_parent) @ p_local
    q_world = q_parent * q_local

Poses are 7-element ``[x, y, z, qx, qy, qz, qw]`` throughout -- PyRep's
``get_pose()`` layout and the layout the live detectors expect (see
``live_detectors/transition.py:_pose_quat``, which returns None for anything
shorter than 7 and silently disables the orientation comparison).

Parent-local offsets are invariant under RLBench's per-episode placement
randomization -- randomization moves the ROOT object, and the offsets ride along
-- so one spec per task is valid for every episode.
"""

from aha_publish import paths

import glob
import json
import os

import numpy as np

__all__ = [
    "quat_multiply",
    "quat_conjugate",
    "quat_rotate",
    "compose",
    "relative_pose",
    "pose_error",
    "rpy_to_quat",
    "spec_from_inspection_report",
    "load_chain_spec",
    "validate_spec",
    "resolve_waypoint_poses",
    "WaypointChain",
]


# --------------------------------------------------------------------------- #
# Quaternion / transform math (xyzw order, matching PyRep)
# --------------------------------------------------------------------------- #
def _as_quat(q):
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.size != 4:
        raise ValueError(f"expected a 4-element xyzw quaternion, got {q.size}")
    n = np.linalg.norm(q)
    if n <= 1e-12:
        raise ValueError("degenerate (zero-norm) quaternion")
    return q / n


def quat_multiply(q1, q2):
    """Hamilton product q1 * q2, both xyzw. Applies q2 first, then q1."""
    x1, y1, z1, w1 = _as_quat(q1)
    x2, y2, z2, w2 = _as_quat(q2)
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def quat_rotate(q, v):
    """Rotate vector v by quaternion q (xyzw)."""
    x, y, z, w = _as_quat(q)
    v = np.asarray(v, dtype=float).reshape(3)
    # v + 2 * cross(u, cross(u, v) + w * v), u = (x, y, z). Cheaper and more
    # numerically stable than building the full rotation matrix.
    u = np.array([x, y, z])
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def rpy_to_quat(rpy):
    """CoppeliaSim/PyRep Euler angles (radians) -> xyzw quaternion.

    The convention is INTRINSIC X-Y-Z: rotate about x, then the new y, then the
    new z, i.e. R = Rx(a) @ Ry(b) @ Rz(g). Verified against the parent-local
    offsets read straight out of PyRep: median agreement 6e-7 m over 363
    offsets, with the only outliers being waypoints init_episode repositions.
    Getting this wrong is silent -- Rz@Ry@Rx reproduces identity poses perfectly
    and diverges by up to 180 deg on rotated ones.
    """
    a, b, g = (float(x) for x in np.asarray(rpy, dtype=float).reshape(3))
    ca, sa, cb, sb, cg, sg = (np.cos(a), np.sin(a), np.cos(b),
                              np.sin(b), np.cos(g), np.sin(g))
    r = np.array([
        [cb * cg, -cb * sg, sb],
        [sa * sb * cg + ca * sg, -sa * sb * sg + ca * cg, -sa * cb],
        [-ca * sb * cg + sa * sg, ca * sb * sg + sa * cg, ca * cb],
    ])
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        q = np.array([(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s,
                      (r[1, 0] - r[0, 1]) / s, 0.25 * s])
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = np.array([0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s,
                      (r[2, 1] - r[1, 2]) / s])
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = np.array([(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s,
                      (r[0, 2] - r[2, 0]) / s])
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = np.array([(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s,
                      (r[1, 0] - r[0, 1]) / s])
    return q / np.linalg.norm(q)


def quat_to_rpy(q):
    """Inverse of rpy_to_quat: xyzw quaternion -> intrinsic X-Y-Z Euler radians.

    Only used to write a human-readable companion to the authoritative
    quaternion in the report; nothing reads it back.
    """
    x, y, z, w = _as_quat(q)
    r02 = 2 * (x * z + w * y)
    r12 = 2 * (y * z - w * x)
    r22 = 1 - 2 * (x * x + y * y)
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    beta = np.arcsin(np.clip(r02, -1.0, 1.0))
    if abs(r02) < 1.0 - 1e-9:
        alpha = np.arctan2(-r12, r22)
        gamma = np.arctan2(-r01, r00)
    else:                                     # gimbal lock: fold into alpha
        alpha = np.arctan2(2 * (y * z + w * x), 1 - 2 * (x * x + z * z))
        gamma = 0.0
    return np.array([float(alpha), float(beta), float(gamma)])


def quat_conjugate(q):
    """Inverse rotation of a unit quaternion (xyzw)."""
    x, y, z, w = _as_quat(q)
    return np.array([-x, -y, -z, w])


def compose(parent_pose, local_pose):
    """T_world(child) = T_world(parent) . T_local(child).

    Both poses are 7-element [x, y, z, qx, qy, qz, qw]; returns the same.
    """
    parent = np.asarray(parent_pose, dtype=float).reshape(-1)
    local = np.asarray(local_pose, dtype=float).reshape(-1)
    if parent.size < 7 or local.size < 7:
        raise ValueError("compose() needs 7-element [xyz + xyzw quat] poses")
    position = parent[:3] + quat_rotate(parent[3:7], local[:3])
    quaternion = quat_multiply(parent[3:7], local[3:7])
    return np.concatenate([position, quaternion])


def relative_pose(parent_pose, child_pose):
    """Inverse of compose(): T_local(child) = T_world(parent)^-1 . T_world(child).

    Used when extracting the spec -- turns two world poses into the fixed
    parent-local offset that gets stored in the JSON.
    """
    parent = np.asarray(parent_pose, dtype=float).reshape(-1)
    child = np.asarray(child_pose, dtype=float).reshape(-1)
    if parent.size < 7 or child.size < 7:
        raise ValueError("relative_pose() needs 7-element poses")
    inv_q = quat_conjugate(parent[3:7])
    position = quat_rotate(inv_q, child[:3] - parent[:3])
    quaternion = quat_multiply(inv_q, child[3:7])
    return np.concatenate([position, quaternion])


def pose_error(pose_a, pose_b):
    """(position error in m, orientation error in rad) between two 7-poses.

    Quaternion double cover is handled, so q and -q compare as identical.
    """
    a = np.asarray(pose_a, dtype=float).reshape(-1)
    b = np.asarray(pose_b, dtype=float).reshape(-1)
    if a.size < 7 or b.size < 7:
        return float("nan"), float("nan")
    d_pos = float(np.linalg.norm(a[:3] - b[:3]))
    qa, qb = _as_quat(a[3:7]), _as_quat(b[3:7])
    dot = float(np.clip(abs(np.dot(qa, qb)), -1.0, 1.0))
    return d_pos, float(2.0 * np.arccos(dot))


# --------------------------------------------------------------------------- #
# Spec loading
# --------------------------------------------------------------------------- #
DEFAULT_CHAIN_DIR = str(paths.TTM_CONTEXT_DIR)


def chain_spec_path(task_name, chain_dir=None):
    """Path to a task's inspection report.

    Reports are named ``NNN_<task>.llm_context.json``, so the numeric prefix has
    to be globbed rather than guessed.
    """
    base = chain_dir or os.getenv("AHA_WP_CHAIN_DIR") or DEFAULT_CHAIN_DIR
    exact = os.path.join(base, f"{task_name}.llm_context.json")
    if os.path.isfile(exact):
        return exact
    matches = sorted(glob.glob(
        os.path.join(base, f"*_{task_name}.llm_context.json")))
    return matches[0] if matches else exact


def _index_of(name):
    """Waypoint index, or None when the name is not one RLBench executes.

    RLBench collects exactly ``waypoint%d`` (backend/task.py:_get_waypoints), but
    inspect_ttm selects scene objects by substring, so reports also contain
    decorated dummies like ``waypoint3_``. Those are legitimate parents in the
    tree but are not waypoints -- treating them as such would collide with the
    real ``waypoint3`` on index 3.
    """
    if not name.startswith("waypoint"):
        return None
    suffix = name[len("waypoint"):]
    if not suffix.isdigit():
        return None
    return int(suffix)


def spec_from_inspection_report(report):
    """Build a chain spec from a ttm_inspection_reports entry.

    Each waypoint carries its pose in its PARENT's frame
    (``local_position_xyz_m`` + ``local_quaternion_xyzw``), read from the
    simulator by inspect_ttm.py and invariant across episodes; that is what the
    chain is built from. The report's world poses are episode-specific and are
    used only for the scene objects the chain resolves against, whose
    orientations arrive as Euler rpy and are converted by ``rpy_to_quat``.

    A waypoint with no local pose in the report, or whose parent is absent from
    it, is recorded ungrounded: it gets no local offset and will not resolve.
    Reports written before cartesian-path waypoints were included skip those
    entries, which also strips every waypoint parented below one.
    """
    objects = {}
    scene = report.get("scene_objects") or {}
    for group in ("graspable", "fixed"):
        for obj in scene.get(group) or []:
            pose = obj.get("world_pose_after_init") or {}
            position, rpy = pose.get("position_xyz_m"), pose.get("orientation_rpy_rad")
            if position is None or rpy is None:
                continue
            objects[obj["name"]] = np.concatenate(
                [np.asarray(position, dtype=float).reshape(3), rpy_to_quat(rpy)])
    # The task model base is a frame, not a scene object, so it lives outside
    # scene_objects -- but it IS the parent of waypoints anchored to the task
    # frame rather than to a manipulable object.
    base = report.get("task_boundary_root") or {}
    base_pose = base.get("world_pose_after_init") or {}
    if base.get("name") and base_pose.get("position_xyz_m") is not None:
        objects[base["name"]] = np.concatenate([
            np.asarray(base_pose["position_xyz_m"], dtype=float).reshape(3),
            rpy_to_quat(base_pose["orientation_rpy_rad"])])

    # Waypoints the task itself repositions at run time (init_episode or a
    # register_waypoint_ability_start hook writing set_pose/set_position/...).
    # Their stored parent-local offset described the ONE episode inspect_ttm.py
    # observed and does not describe any other, so composing them yields a pose
    # that is wrong by a fixed transform for the whole episode -- and every
    # waypoint parented below inherits that error. Recorded by
    # rlbench_dynamic_waypoints.py; see that module for how they are found.
    #
    # These are left UNGROUNDED on purpose. The runtime handles a missing
    # reference correctly (waypoint_distance stays NaN, the detector yields no
    # verdict and the banner names the waypoint), whereas a confidently wrong
    # reference makes the arrival detectors fire on every waypoint of a clean
    # run. No reference beats a wrong one.
    dynamic = {int(i) for i in (report.get("dynamic_waypoints") or [])}
    rules = report.get('hook_reference_rules') or {}
    if rules.get('hooks') and not rules.get('errors'):
        dynamic.difference_update(report.get('rule_resolved_waypoints', []))
    dynamic.update(_index_of(w['name']) for w in report.get('waypoints', [])
                   if w.get('reference_unavailable') and _index_of(w['name']) is not None)

    nodes, waypoints, roots = {}, [], set()
    # Every report entry becomes a node -- decorated dummies (waypoint3_) are not
    # waypoints RLBench executes, but they DO appear as parents of ones that are,
    # so the chain has to be able to pass through them.
    for wp in report.get("waypoints") or []:
        name = wp["name"]
        parent = wp.get("orientation_parent", wp.get("parent"))
        position_parent = wp.get("position_parent", parent)
        # The local offset is taken from the report as written by inspect_ttm.py,
        # never derived here: it is read straight from the simulator in the
        # parent's frame, it is the value a human would edit, and it survives
        # even where the world poses do not.
        stored_p = wp.get("local_position_xyz_m")
        stored_q = wp.get("local_quaternion_xyzw")
        if _index_of(name) in dynamic or wp.get("reference_unavailable"):
            nodes[name] = {"parent": None, "local_position": None,
                           "local_quaternion": None, "kind": "dynamic",
                           "missing_parent": parent}
        elif parent and stored_p is not None and stored_q is not None:
            nodes[name] = {
                "parent": parent,
                "local_position": list(stored_p),
                "local_quaternion": list(stored_q),
                "kind": "waypoint",
                "position_parent": position_parent,
            }
            if parent in objects:
                roots.add(parent)
            if position_parent and not position_parent.startswith('waypoint'):
                roots.add(position_parent)
        else:
            nodes[name] = {"parent": None, "local_position": None,
                           "local_quaternion": None, "kind": "ungrounded",
                           "missing_parent": parent}

    # Second pass, once EVERY node exists: walk each waypoint up to the terminus
    # of its chain. That has to wait for the full node set, or a waypoint
    # parented to a dummy the report happens to list later stops short of the
    # real terminus.
    for wp in report.get("waypoints") or []:
        name = wp["name"]
        index = _index_of(name)
        if index is None:
            continue
        node = nodes.get(name) or {}
        chain = [name]
        cursor = node.get("parent") or node.get("missing_parent")
        seen = {name}
        while cursor in nodes and nodes[cursor].get("parent") and cursor not in seen:
            chain.append(cursor)
            seen.add(cursor)
            cursor = nodes[cursor]["parent"]
        if cursor:
            chain.append(cursor)
            # The terminus is a root: its world pose is what resolve() needs
            # supplied. It is NOT required to appear in `objects` -- inspect_ttm.py
            # only exports SHAPEs inside the boundary root, so a terminus that is
            # the task model root ABOVE the boundary (open_oven's 'oven'), a
            # proximity sensor ('success_detector') or a *_visual shape
            # ('charger_visual') never lands there. Gating roots on `objects`
            # membership left those chains ungrounded, which cost the transition
            # and orientation detectors their whole reference target. Nothing is
            # lost by trusting the name: the spec stores no world poses at all,
            # roots are read live from the simulator once per episode -- and for a
            # task base that base_rotation_bounds re-yaws each episode, live is
            # the only correct source anyway.
            # One name is NOT trusted: a terminus that is itself a waypoint
            # object. inspect_ttm selects those by substring, so a waypoint the
            # report does not list is one it DROPPED (a PATH waypoint in a report
            # written before PATH types were kept). Grounding on it would read the
            # object's own frame -- mid-path for a PATH -- and compose a plausible
            # but wrong target, which is worse than none. Leave it ungrounded so
            # check_waypoint_chain.py keeps reporting it.
            if (cursor not in nodes and node.get("kind") == "waypoint"
                    and not cursor.startswith("waypoint")):
                roots.add(cursor)
        parent = node.get("parent")
        waypoints.append({"index": index, "name": name, "scene_name": name,
                          "kind": "point", "root_object": parent if parent in objects
                          else None, "chain": chain})

    return {
        "task_name": report.get("task_name"),
        "reference_variation_index": report.get("reference_variation_index"),
        "hook_reference_rules": rules,
        "reference_scene_objects": sorted(set(objects) | set(report.get('reference_objects', {}))),
        "reference_state": report.get('reference_state', {}),
        "generated_by": "spec_from_inspection_report (ttm_inspection_reports)",
        "convention": {
            "pose_layout": "[x, y, z, qx, qy, qz, qw]",
            "quaternion_order": "xyzw (PyRep/CoppeliaSim)",
            "composition": "T_world(node) = T_world(parent) . T_local(node)",
        },
        "root_objects": sorted(roots),
        "nodes": nodes,
        "waypoints": waypoints,
        # Carried through so the runner and check_waypoint_chain.py can name the
        # waypoints they have no reference for. Before this key was emitted here,
        # every `spec.get("dynamic_waypoints")` in the codebase read an empty
        # list and the warnings guarding this case were unreachable.
        "dynamic_waypoints": sorted(dynamic),
        # The waypoints grounded ONLY because the hook equations recompute them
        # every execution. They are gone from dynamic_waypoints above, so the
        # runner needs them named here to know which references to drop -- and
        # only those -- if a rule turns out to be unevaluatable at run time.
        "rule_resolved_waypoints": sorted(
            {int(i) for i in (report.get('rule_resolved_waypoints') or [])}
            - set(dynamic)),
    }


def load_chain_spec(task_name, chain_dir=None):
    """Load a task's inspection report and convert it to a chain spec."""
    path = chain_spec_path(task_name, chain_dir)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        report = json.load(fh)
    return spec_from_inspection_report(report)


def validate_spec(spec):
    """Structural problems in a hand-authored spec, as a list of strings.

    Specs are maintained by hand, so a typo here is silent: a missing key makes
    a waypoint drop out of the resolve, and a non-unit quaternion skews every
    descendant. Empty list means the structure is sound -- it says nothing about
    whether the NUMBERS are right, which only verify_waypoint_chain.py can tell
    you.
    """
    problems = []
    nodes = spec.get("nodes")
    if not isinstance(nodes, dict) or not nodes:
        return ["'nodes' is missing or empty"]

    for name, node in nodes.items():
        if not isinstance(node, dict):
            problems.append(f"node {name!r} is not an object")
            continue
        # A node the task repositions at run time is stripped of its parent and
        # offsets ON PURPOSE (see spec_from_inspection_report): there is no
        # static pose to validate, and reporting it as a structural defect would
        # bury the real ones. It is surfaced as dynamic_waypoints instead.
        if node.get("kind") == "dynamic":
            continue
        parent = node.get("parent")
        if parent is None:
            problems.append(f"node {name!r} has no parent (ungrounded chain)")
        for key, size in (("local_position", 3), ("local_quaternion", 4)):
            value = node.get(key)
            if value is None:
                problems.append(f"node {name!r} is missing {key}")
                continue
            try:
                arr = np.asarray(value, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                problems.append(f"node {name!r} has a non-numeric {key}")
                continue
            if arr.size != size:
                problems.append(
                    f"node {name!r} {key} has {arr.size} values, expected {size}")
            elif not np.all(np.isfinite(arr)):
                problems.append(f"node {name!r} {key} contains NaN/inf")
            elif key == "local_quaternion":
                norm = float(np.linalg.norm(arr))
                if abs(norm - 1.0) > 1e-3:
                    problems.append(
                        f"node {name!r} local_quaternion is not unit "
                        f"(norm {norm:.6f}); xyzw order, w last")

    seen = set()
    for wp in spec.get("waypoints") or []:
        name = wp.get("name")
        if name not in nodes:
            problems.append(f"waypoint entry {name!r} has no matching node")
        try:
            index = int(wp["index"])
        except (KeyError, TypeError, ValueError):
            problems.append(f"waypoint entry {name!r} has no integer 'index'")
            continue
        if index in seen:
            problems.append(f"duplicate waypoint index {index}")
        seen.add(index)

    # Both component frames must be grounded and acyclic.
    roots = set(spec.get("root_objects") or [])
    checked = set()

    def check_parents(name, visiting):
        if name in visiting:
            problems.append(f"cyclic parent chain at {name!r}")
            return
        if name in checked:
            return
        if name not in nodes:
            if name not in roots:
                problems.append(f"frame {name!r} is not listed in root_objects")
            return
        node = nodes[name]
        if not isinstance(node, dict):
            return
        parent = node.get('parent')
        for anchor in {parent, node.get('position_parent', parent)} - {None}:
            check_parents(anchor, visiting | {name})
        checked.add(name)

    for name in nodes:
        check_parents(name, set())
    return problems


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
def resolve_waypoint_poses(spec, world_poses, strict=False):
    """Compose every waypoint's world pose from the chain spec.

    spec         chain spec dict (see waypoint_chains/*.chain.json for the layout)
    world_poses  {object_name: 7-element world pose} -- normally just the root
                 objects, but ANY node may be supplied and short-circuits the
                 walk above it (useful for a waypoint whose root object the
                 robot has already moved: pass that waypoint's own live pose).
    strict       raise instead of skipping when a chain cannot be grounded.

    Returns {waypoint_index: np.ndarray(7,)}. Waypoints whose chain never
    reaches a supplied pose are omitted (or raise when strict).
    """
    nodes = spec.get("nodes") or {}
    known = {}
    for name, pose in (world_poses or {}).items():
        arr = np.asarray(pose, dtype=float).reshape(-1)
        if arr.size >= 7:
            known[name] = arr[:7]

    resolving = set()

    def world_pose_of(name):
        if name in known:
            return known[name]
        if name in resolving:                     # cyclic parenting in the ttm
            raise ValueError(f"cyclic parent chain at {name!r}")
        node = nodes.get(name)
        if node is None:
            raise KeyError(f"no world pose and no chain entry for {name!r}")
        parent = node.get("parent")
        if not parent:
            raise KeyError(f"chain for {name!r} reaches the scene root ungrounded")
        resolving.add(name)
        try:
            parent_world = world_pose_of(parent)
            position_world = world_pose_of(node.get('position_parent', parent))
            position = (position_world[:3] + quat_rotate(
                position_world[3:7], node['local_position']))
            quaternion = quat_multiply(parent_world[3:7], node['local_quaternion'])
            pose = np.concatenate([position, quaternion])
        finally:
            resolving.discard(name)
        known[name] = pose
        return pose

    out = {}
    for wp in spec.get("waypoints") or []:
        name = wp["name"]
        try:
            out[int(wp["index"])] = world_pose_of(name)
        except (KeyError, ValueError):
            if strict:
                raise
    return out


class WaypointChain:
    """Convenience wrapper: load a task spec once, resolve per episode.

    Typical use -- read the root objects from the sim ONCE at episode start,
    then serve every waypoint target from memory with no further sim access:

        chain = WaypointChain.for_task(task_name)
        chain.resolve({name: obj.get_pose() for name, obj in root_objects})
        pose = chain.pose(waypoint_index)     # 7-element, or None
    """

    def __init__(self, spec):
        self.spec = spec
        self.poses = {}
        self.world_poses = {}

    @classmethod
    def for_task(cls, task_name, chain_dir=None):
        spec = load_chain_spec(task_name, chain_dir)
        return cls(spec) if spec else None

    @property
    def root_objects(self):
        """Names whose world pose must be supplied to resolve()."""
        return list(self.spec.get("root_objects") or [])

    def resolve(self, world_poses, strict=False, variation_index=None):
        expected = self.spec.get('reference_variation_index')
        if expected is not None and variation_index is not None and expected != variation_index:
            self.poses = {}
            raise ValueError(f"reference inspected for variation {expected}, "
                             f"episode uses {variation_index}; regenerate its inspection")
        self.world_poses = dict(world_poses)
        self.poses = resolve_waypoint_poses(self.spec, world_poses, strict=strict)
        return self.poses

    def pose(self, index):
        pose = self.poses.get(int(index))
        return None if pose is None else list(pose)

    def unresolved(self):
        """Waypoint indices the last resolve() could not ground."""
        want = {int(w["index"]) for w in (self.spec.get("waypoints") or [])}
        return sorted(want - set(self.poses))
