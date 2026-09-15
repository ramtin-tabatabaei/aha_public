"""
Inspect an RLBench task TTM: the fixed scene structure from the .ttm file plus live
object/waypoint poses after init_episode, written as an LLM-readable JSON report.

Usage (same conda env + env vars as interactive_failure_runner.py):

    # Single task (writes one report):
    python3 inspect_ttm.py basketball_in_hoop
    python3 inspect_ttm.py /full/path/to/file.ttm
    python3 inspect_ttm.py basketball_in_hoop --save report.llm_context.json

    # Batch over many tasks (each runs in its own subprocess; writes a manifest):
    python3 inspect_ttm.py --all
    python3 inspect_ttm.py --all --output-dir ttm_inspection_reports
    python3 inspect_ttm.py --task basketball_in_hoop --task beat_the_buzz
    python3 inspect_ttm.py --all --skip-existing --stop-on-error

    # Batch by the task description generator's numbering (task_description_maker):
    python3 inspect_ttm.py --list-tasks
    python3 inspect_ttm.py --tasks 31-90
    python3 inspect_ttm.py --tasks 1,3-5,31
"""

from aha_publish import paths

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = (paths.PROJECT_ROOT)
# External RLBench checkout configured with RLBENCH_ROOT.
RLBENCH_ROOT = paths.RLBENCH_ROOT
TTM_DIR    = str(RLBENCH_ROOT / 'rlbench/task_ttms')
BASE_SCENE = str(RLBENCH_ROOT / 'rlbench/task_design.ttt')
sys.path.insert(0, str(paths.PROJECT_ROOT))
sys.path.insert(0, str(RLBENCH_ROOT))

# Batch mode (--all / --task) inspects many tasks by re-running this script as a
# subprocess once per task, so a CoppeliaSim crash only loses that single task.
CONFIGS_PATH = (paths.FAILGEN_ROOT / 'failgen/configs')
DEFAULT_OUTPUT_DIR = (paths.TTM_CONTEXT_DIR)
SELF_SCRIPT = (paths.SOURCE_DIR / 'ttm/inspect_ttm.py')

W = 80

def fmt(v, p=3):
    return '[' + ', '.join(f'{x:+.{p}f}' for x in v) + ']'

def as_float_list(v, p=6):
    return [round(float(x), p) for x in v]

def safe_pose_value(getter, relative_to=None):
    try:
        return getter(relative_to=relative_to)
    except Exception:
        return None

def type_name(obj):
    try:
        return obj.get_type().name
    except Exception:
        return '?'

def parent_name(obj):
    try:
        p = obj.get_parent()
        return p.get_name() if p is not None else '(root)'
    except Exception:
        return '?'

def is_waypoint_object(obj):
    """RLBench executes DUMMY waypoints as Points and PATH waypoints as
    PredefinedPaths (backend/task.py:_get_waypoints); both belong in the report."""
    return ('waypoint' in obj.get_name().lower()
            and type_name(obj) in ('DUMMY', 'PATH'))

def waypoint_world_pose(obj):
    """World pose [x, y, z, qx, qy, qz, qw] of a waypoint object.

    A cartesian-path waypoint's own frame sits mid-path, so its get_pose() is not
    the pose the arm is driven to -- PredefinedPath runs to the END of the path,
    and every other waypoint_pose helper in the repo resolves get_pose_on_path(1.0)
    for the same reason. Dummies are their own target.
    """
    from aha_publish.common.waypoint_chain import rpy_to_quat
    if type_name(obj) == 'PATH':
        from pyrep.objects.cartesian_path import CartesianPath
        position, euler = CartesianPath(obj.get_handle()).get_pose_on_path(1.0)
        return np.concatenate([np.asarray(position, dtype=float),
                               np.asarray(rpy_to_quat(euler), dtype=float)])
    return np.asarray(obj.get_pose(), dtype=float)

def waypoint_index(name):
    """Numeric ordering key from a waypoint name (waypoint0, waypoint10, waypoint-1, waypoint0.5)."""
    digits = ''.join(ch for ch in name if ch.isdigit() or ch in '.-')
    try:
        return float(digits)
    except ValueError:
        return 1e9


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: fixed structure from the .ttm binary
# ─────────────────────────────────────────────────────────────────────────────
def phase1(ttm_path):
    from pyrep import PyRep
    from pyrep.const import ObjectType

    pr = PyRep()
    pr.launch(BASE_SCENE, headless=True)
    pr.start()
    root = pr.import_model(ttm_path)

    all_objs = [root]
    for otype in (ObjectType.SHAPE, ObjectType.DUMMY, ObjectType.JOINT,
                  ObjectType.PROXIMITY_SENSOR, ObjectType.FORCE_SENSOR,
                  ObjectType.CAMERA, ObjectType.PATH):
        try:
            all_objs.extend(root.get_objects_in_tree(object_type=otype, exclude_base=True))
        except Exception:
            pass

    waypoints = [o for o in all_objs if is_waypoint_object(o)]
    waypoint_data = []
    if waypoints:
        print('=' * W)
        print('  WAYPOINTS — parent and fixed pose')
        print('=' * W)
        print()
        from aha_publish.common.waypoint_chain import quat_to_rpy, relative_pose
        for wp in sorted(waypoints, key=lambda o: o.get_name()):
            p     = wp.get_parent()
            pname = parent_name(wp)
            if type_name(wp) == 'PATH' or (p is not None and type_name(p) == 'PATH'):
                # get_position/get_orientation would report the path's own frame,
                # which is neither where the arm ends up nor the frame a child of
                # the path is chained through; use the endpoint pose for both ends.
                world = waypoint_world_pose(wp)
                world_orient = quat_to_rpy(world[3:7])
                rel = relative_pose(waypoint_world_pose(p), world) if p else None
                offset = rel[:3] if rel is not None else None
                orient = quat_to_rpy(rel[3:7]) if rel is not None else None
            else:
                offset = safe_pose_value(wp.get_position, relative_to=p) if p else None
                orient = safe_pose_value(wp.get_orientation, relative_to=p) if p else None
                world_orient = safe_pose_value(wp.get_orientation)
            lpos  = fmt(offset) if offset is not None else 'N/A'
            lori  = fmt(orient) if orient is not None else 'N/A'
            print(f'  {wp.get_name()} = {pname} + pos {lpos}, rpy {lori}')
            waypoint_data.append({
                'name': wp.get_name(),
                'type': type_name(wp),
                'parent': pname,
                'fixed_offset_xyz_m': as_float_list(offset) if offset is not None else None,
                'fixed_orientation_rpy_rad': as_float_list(orient) if orient is not None else None,
                'world_orientation_in_imported_ttm_rpy_rad': (
                    as_float_list(world_orient) if world_orient is not None else None
                ),
            })
        print()

    pr.stop()
    pr.shutdown()
    return waypoint_data


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: live positions after init_episode + placement distribution
# ─────────────────────────────────────────────────────────────────────────────
def phase2(task_name):
    from failgen.env_wrapper import FailGenEnvWrapper
    from pyrep.const import ObjectType

    env = FailGenEnvWrapper(
        task_name=task_name,
        headless=True,
        record=False,
        save_data=False,
        save_path='/tmp',
    )
    env.reset()

    sc    = env._env._scene
    task  = sc.task
    broot = task.boundary_root()
    bname = broot.get_name()
    broot_world_pos = safe_pose_value(broot.get_position)
    broot_world_ori = safe_pose_value(broot.get_orientation)

    # ── placement bounds ───────────────────────────────────────────────────
    ws     = sc._workspace
    ws_pos = np.array(ws.get_position())
    ws_bb  = ws.get_bounding_box()
    t_bb   = (broot.get_model_bounding_box()
               if broot.is_model() else broot.get_bounding_box())

    x_lo  = ws_pos[0] + ws_bb[0] + abs(t_bb[0])
    x_hi  = ws_pos[0] + ws_bb[1] - abs(t_bb[1])
    y_lo  = ws_pos[1] + ws_bb[2] + abs(t_bb[2])
    y_hi  = ws_pos[1] + ws_bb[3] - abs(t_bb[3])
    z_fix = sc._workspace_minz
    rz_lo, rz_hi = task.base_rotation_bounds()[0][2], task.base_rotation_bounds()[1][2]

    # sweep rz to find exact x/y bounds for any offset that rotates with the task
    rz_sweep = np.linspace(rz_lo, rz_hi, 500)
    cos_s    = np.cos(rz_sweep)
    sin_s    = np.sin(rz_sweep)

    def obj_range(obj):
        """World-position Uniform range for obj, accounting for task rotation."""
        try:
            cx, cy, cz = obj.get_position(relative_to=broot)
        except Exception:
            return None
        rot_x = cx * cos_s - cy * sin_s
        rot_y = cx * sin_s + cy * cos_s
        return (x_lo + rot_x.min(), x_hi + rot_x.max(),
                y_lo + rot_y.min(), y_hi + rot_y.max(),
                z_fix + cz)

    def print_range(label, r, note=''):
        if r is None:
            print(f'  {label}: N/A')
            return
        wx_lo, wx_hi, wy_lo, wy_hi, wz = r
        suffix = f'  ({note})' if note else ''
        print(f'  {label}{suffix}')
        print(f'    x ~ Uniform({wx_lo:+.3f},  {wx_hi:+.3f})   spread {wx_hi-wx_lo:.3f} m')
        print(f'    y ~ Uniform({wy_lo:+.3f},  {wy_hi:+.3f})   spread {wy_hi-wy_lo:.3f} m')
        print(f'    z   fixed at {wz:+.3f}')
        print()

    # ── collect graspable ranges + non-graspable scene objects ────────────
    graspable_objs = task.get_graspable_objects()
    graspable_names = {o.get_name() for o in graspable_objs}

    graspable_data = []
    for obj in graspable_objs:
        graspable_data.append({
            'name': obj.get_name(),
            'world_position_range': obj_range(obj),
            'position_from_root': safe_pose_value(obj.get_position, relative_to=broot),
            'orientation_from_root': safe_pose_value(obj.get_orientation, relative_to=broot),
            'world_position': safe_pose_value(obj.get_position),
            'world_orientation': safe_pose_value(obj.get_orientation),
        })

    # other named shapes (non-graspable, non-visual) relative to graspable objects
    from pyrep.const import ObjectType
    all_shapes = broot.get_objects_in_tree(object_type=ObjectType.SHAPE, exclude_base=False)
    non_graspable = [o for o in all_shapes if o.get_name() not in graspable_names
                     and 'visual' not in o.get_name().lower()
                     and 'stop' not in o.get_name().lower()
                     and o.get_name() != bname]

    # for each non-graspable, find its offset from the nearest graspable
    ng_data = []
    for obj in non_graspable:
        obj_pos_world = np.array(obj.get_position())
        best_obj, best_name, best_offset = None, None, None
        best_dist = float('inf')
        for g in graspable_objs:
            g_pos = np.array(g.get_position())
            off   = obj_pos_world - g_pos
            dist  = float(np.linalg.norm(off))
            if dist < best_dist:
                best_dist, best_obj, best_name, best_offset = dist, g, g.get_name(), off
        ng_data.append({
            'name': obj.get_name(),
            'reference_name': best_name,
            'world_delta_from_reference': best_offset,
            'position_from_reference': (
                safe_pose_value(obj.get_position, relative_to=best_obj)
                if best_obj is not None else None
            ),
            'orientation_from_reference': (
                safe_pose_value(obj.get_orientation, relative_to=best_obj)
                if best_obj is not None else None
            ),
            'world_position': safe_pose_value(obj.get_position),
            'world_orientation': safe_pose_value(obj.get_orientation),
        })

    # ── live waypoint world poses (after init_episode; reveals reorientation) ─
    # Phase 1 reads waypoints from the raw .ttm import, where the graspable sits
    # at its canonical pose, so every waypoint's world orientation is constant and
    # any in-task reorientation is invisible. Here the episode is initialized (the
    # graspable is placed + yaw-rotated), so the waypoint dummies carry their real
    # world orientation and turns between waypoints become visible.
    #
    # Each waypoint also gets its pose in its PARENT's frame. The world pose is
    # specific to the episode that produced the report; the parent-local offset is
    # the part that holds across episodes, and it is what the transition and
    # orientation detectors compose their reference target from.
    from aha_publish.common.waypoint_chain import quat_to_rpy, relative_pose

    live_waypoint_world = []
    live_wp_objs = [
        o
        for otype in (ObjectType.DUMMY, ObjectType.PATH)
        for o in broot.get_objects_in_tree(object_type=otype, exclude_base=False)
        if 'waypoint' in o.get_name().lower()
    ]
    for o in sorted(live_wp_objs, key=lambda x: (waypoint_index(x.get_name()), x.get_name())):
        parent = None
        try:
            parent = o.get_parent()
        except Exception:
            parent = None
        # A path waypoint is BOTH targeted at and chained through the end of its
        # path (see waypoint_world_pose), so a child parented to one must have its
        # offset expressed in that same endpoint frame -- get_position(relative_to=)
        # would return it in the path object's own mid-path frame instead, and the
        # chain would then compose the two mismatched frames.
        parent_is_path = parent is not None and type_name(parent) == 'PATH'
        if type_name(o) == 'PATH' or parent_is_path:
            world = waypoint_world_pose(o)
            wpos, wori = world[:3], quat_to_rpy(world[3:7])
            if parent is None:
                lpos = lquat = lori = None
            else:
                rel = relative_pose(waypoint_world_pose(parent), world)
                lpos, lquat, lori = rel[:3], rel[3:7], quat_to_rpy(rel[3:7])
        else:
            wpos = safe_pose_value(o.get_position)
            wori = safe_pose_value(o.get_orientation)
            lpos = safe_pose_value(o.get_position, relative_to=parent) if parent else None
            lquat = safe_pose_value(o.get_quaternion, relative_to=parent) if parent else None
            lori = safe_pose_value(o.get_orientation, relative_to=parent) if parent else None
        live_waypoint_world.append({
            'name': o.get_name(),
            'type': type_name(o),
            'parent': parent_name(o),
            'world_position_xyz_m': as_float_list(wpos) if wpos is not None else None,
            'world_orientation_rpy_rad': as_float_list(wori) if wori is not None else None,
            'local_position_xyz_m': as_float_list(lpos, 9) if lpos is not None else None,
            'local_quaternion_xyzw': as_float_list(lquat, 9) if lquat is not None else None,
            'local_orientation_rpy_rad': as_float_list(lori, 9) if lori is not None else None,
        })

    # Infer each component independently: an object's position can determine a
    # waypoint while its independently randomized yaw must not determine it.
    from aha_publish.common.waypoint_anchors import infer_waypoint_anchors
    ANCHOR_EPISODES = 5

    def tree_poses():
        poses = {}
        for o in broot.get_objects_in_tree(exclude_base=False):
            try:
                # Path waypoints contribute their endpoint, so a re-anchored
                # offset stays about the same point the local_* fields describe.
                poses[o.get_name()] = (waypoint_world_pose(o)
                                       if is_waypoint_object(o)
                                       else np.array(o.get_pose()))
            except Exception:
                pass
        try:
            poses[bname] = np.array(broot.get_pose())
        except Exception:
            pass
        return poses

    # episodes[0] is the episode every world pose in this report came from, so
    # re-anchored offsets are taken from it and stay consistent with them.
    episodes = [tree_poses()]
    # Numeric vector state referenced by pose equations belongs in the report
    # (e.g. an initial relative offset), not in runtime simulator waypoint reads.
    from aha_publish.common.waypoint_reference_rules import export_hook_reference_rules
    import ast
    _source = RLBENCH_ROOT / 'rlbench/tasks' / f'{task_name}.py'
    _rules = export_hook_reference_rules(_source.read_text()) if _source.exists() else {}
    _state_names = set()
    def collect_state(value):
        if isinstance(value, dict):
            for item in value.values():
                collect_state(item)
        elif isinstance(value, list):
            for item in value:
                collect_state(item)
        elif isinstance(value, str):
            try:
                for node in ast.walk(ast.parse(value, mode='eval')):
                    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'self':
                        _state_names.add(node.attr)
            except SyntaxError:
                pass
    collect_state(_rules)
    def numeric_state():
        result = {}
        for name in _state_names:
            value = getattr(task, name, None)
            if isinstance(value, (list, tuple, np.ndarray)):
                try:
                    arr = np.asarray(value, dtype=float)
                    if arr.size and np.isfinite(arr).all() and arr.ndim:
                        result[name] = arr.tolist()
                except (TypeError, ValueError):
                    pass
        return result
    _state_samples = [numeric_state()]
    for _ in range(ANCHOR_EPISODES - 1):
        env.reset()
        episodes.append(tree_poses())
        _state_samples.append(numeric_state())
    _reference_state = {}
    for name, value in _state_samples[0].items():
        try:
            if all(name in sample and np.shape(sample[name]) == np.shape(value)
                   and np.allclose(sample[name], value, atol=1e-5, rtol=0)
                   for sample in _state_samples[1:]):
                _reference_state[name] = value
        except (TypeError, ValueError):
            pass

    live_waypoint_world = infer_waypoint_anchors(live_waypoint_world, episodes)
    reanchored = []
    for w in live_waypoint_world:
        if w.get('reference_unavailable'):
            print(f'  WARNING: {w["name"]}: {w["reference_unavailable"]}')
        for key in ('position_parent', 'orientation_parent'):
            anchor = w.get(key)
            if anchor and anchor != w['parent']:
                reanchored.append((w['name'] + ':' + key, anchor))

    # An anchor the report does not list has no world pose to compose against,
    # so record any that is missing (close_jar's waypoint3 anchors onto
    # `success`, a proximity sensor, which is in neither object group).
    known_objects = ({o['name'] for o in graspable_data}
                     | {o['name'] for o in ng_data} | {bname})
    for _, anchor in reanchored:
        if anchor in known_objects:
            continue
        obj = next((o for o in broot.get_objects_in_tree(exclude_base=False)
                    if o.get_name() == anchor), None)
        if obj is None:
            continue
        ng_data.append({
            'name': anchor,
            'reference_name': None,
            'world_delta_from_reference': None,
            'position_from_reference': None,
            'orientation_from_reference': None,
            'world_position': list(episodes[0][anchor][:3]),
            'world_orientation': list(quat_to_rpy(episodes[0][anchor][3:7])),
        })
        known_objects.add(anchor)

    env.shutdown()

    if reanchored:
        print()
        print('  RE-ANCHORED (parent was not invariant across episodes)')
        for name, anchor in reanchored:
            print(f'    {name} -> {anchor}')

    # ── print ──────────────────────────────────────────────────────────────
    print('=' * W)
    print('  PLACEMENT DISTRIBUTION  (Uniform, never Normal)')
    print('=' * W)
    print()

    print(f'  Graspable objects:')
    print()
    for obj in graspable_data:
        print_range(obj['name'], obj['world_position_range'])
        rel_ori = obj['orientation_from_root']
        world_ori = obj['world_orientation']
        print(f"    root-local rpy {fmt(rel_ori) if rel_ori is not None else 'N/A'}")
        print(f"    live world rpy {fmt(world_ori) if world_ori is not None else 'N/A'}")
        print()

    if ng_data:
        print(f'  Other scene objects  (fixed pose from graspable, no separate distribution):')
        print()
        for obj in ng_data:
            off = obj['world_delta_from_reference']
            rel_pos = obj['position_from_reference']
            rel_ori = obj['orientation_from_reference']
            if off is None:
                print(f"  {obj['name']} = N/A")
                continue
            dx, dy, dz = off
            print(f"  {obj['name']} = {obj['reference_name']} + world-delta [{dx:+.3f}, {dy:+.3f}, {dz:+.3f}]")
            print(f"    reference-local pos {fmt(rel_pos) if rel_pos is not None else 'N/A'}")
            print(f"    reference-local rpy {fmt(rel_ori) if rel_ori is not None else 'N/A'}")
        print()

    return {
        'live_waypoint_world_poses': live_waypoint_world,
        'reference_variation_index': sc._variation_index,
        'reference_state': _reference_state,
        'reference_objects': {name: pose.tolist() for name, pose in episodes[0].items()
                              if 'waypoint' not in name.lower()},
        # The task model base is a frame, not a scene object, so it stays outside
        # scene_objects -- but it IS the parent of the waypoints that are anchored
        # to the task frame rather than to a manipulable object, so the chain math
        # needs its world pose.
        'task_boundary_root': {
            'name': bname,
            'world_pose_after_init': {
                'position_xyz_m': (
                    as_float_list(broot_world_pos) if broot_world_pos is not None else None
                ),
                'orientation_rpy_rad': (
                    as_float_list(broot_world_ori) if broot_world_ori is not None else None
                ),
            }
        },
        'placement_distribution': {
            'type': 'Uniform for x/y; fixed z',
            'units': 'meters',
            'task_boundary_root': bname,
            'task_boundary_root_world_pose_after_init': {
                'position_xyz_m': (
                    as_float_list(broot_world_pos) if broot_world_pos is not None else None
                ),
                'orientation_rpy_rad': (
                    as_float_list(broot_world_ori) if broot_world_ori is not None else None
                ),
            },
            'task_base_rotation_rz_bounds_rad': [
                round(float(rz_lo), 6),
                round(float(rz_hi), 6),
            ],
            'graspable_objects': [
                {
                    'name': obj['name'],
                    'world_position_distribution': {
                        'x_uniform_min_m': round(float(obj['world_position_range'][0]), 6) if obj['world_position_range'] is not None else None,
                        'x_uniform_max_m': round(float(obj['world_position_range'][1]), 6) if obj['world_position_range'] is not None else None,
                        'y_uniform_min_m': round(float(obj['world_position_range'][2]), 6) if obj['world_position_range'] is not None else None,
                        'y_uniform_max_m': round(float(obj['world_position_range'][3]), 6) if obj['world_position_range'] is not None else None,
                        'z_fixed_m': round(float(obj['world_position_range'][4]), 6) if obj['world_position_range'] is not None else None,
                    },
                    'world_pose_after_init': {
                        'position_xyz_m': (
                            as_float_list(obj['world_position'])
                            if obj['world_position'] is not None else None
                        ),
                        'orientation_rpy_rad': (
                            as_float_list(obj['world_orientation'])
                            if obj['world_orientation'] is not None else None
                        ),
                    },
                    'fixed_offset_from_task_boundary_root_xyz_m': (
                        as_float_list(obj['position_from_root'])
                        if obj['position_from_root'] is not None else None
                    ),
                    'fixed_orientation_from_task_boundary_root_rpy_rad': (
                        as_float_list(obj['orientation_from_root'])
                        if obj['orientation_from_root'] is not None else None
                    ),
                }
                for obj in graspable_data
            ],
            'other_scene_objects': [
                {
                    'name': obj['name'],
                    'reference_graspable_object': obj['reference_name'],
                    'fixed_offset_from_reference_xyz_m': (
                        as_float_list(obj['world_delta_from_reference'])
                        if obj['world_delta_from_reference'] is not None else None
                    ),
                    'fixed_offset_from_reference_local_xyz_m': (
                        as_float_list(obj['position_from_reference'])
                        if obj['position_from_reference'] is not None else None
                    ),
                    'fixed_orientation_from_reference_rpy_rad': (
                        as_float_list(obj['orientation_from_reference'])
                        if obj['orientation_from_reference'] is not None else None
                    ),
                    'world_pose_after_init': {
                        'position_xyz_m': (
                            as_float_list(obj['world_position'])
                            if obj['world_position'] is not None else None
                        ),
                        'orientation_rpy_rad': (
                            as_float_list(obj['world_orientation'])
                            if obj['world_orientation'] is not None else None
                        ),
                    },
                    'has_separate_distribution': False,
                }
                for obj in ng_data
            ],
        }
    }


def build_llm_report_json(task_name, ttm_path, waypoint_data, placement_data):
    """Lean, downstream-facing JSON: the fields that inform a task description, plus
    the parent-frame geometry the detectors need.

    Two audiences share one file:

    - The description/BT prompts read the world-frame poses (`world_*_after_init`)
      and `scene_objects`. That is the whole of what they see -- the description
      maker strips everything below before the report reaches the model.
    - The transition/orientation detectors read the per-waypoint `local_*` offset
      (position + xyzw quaternion in the parent frame) and `task_boundary_root` to
      compose a reference target that survives across episodes (the world poses do
      not; they are one episode's sample of the placement distribution).
    """
    live_wp_map = {
        w['name']: w for w in placement_data.get('live_waypoint_world_poses', [])
    }
    pd = placement_data.get('placement_distribution', {})
    base = placement_data.get('task_boundary_root') or {}

    ordered = sorted(waypoint_data, key=lambda w: (waypoint_index(w['name']), w['name']))
    waypoints_out = []
    for wp in ordered:
        lw = live_wp_map.get(wp['name'], {})
        pos = lw.get('world_position_xyz_m')
        ori = lw.get('world_orientation_rpy_rad')
        # The live parent, not phase 1's: a waypoint init_episode re-places is
        # re-anchored there, and local_* below is expressed in THAT frame.
        entry = {'name': wp['name'],
                 'type': lw.get('type') or wp.get('type'),
                 'parent': lw.get('parent') or wp['parent']}
        if pos is not None:
            entry['world_position_after_init_xyz_m'] = pos
        if ori is not None:
            entry['world_orientation_after_init_rpy_rad'] = ori
        local_pos = lw.get('local_position_xyz_m')
        local_quat = lw.get('local_quaternion_xyzw')
        if local_pos is not None and local_quat is not None:
            entry['local_position_xyz_m'] = local_pos
            entry['local_quaternion_xyzw'] = local_quat
        else:
            entry['local_offset_unavailable'] = (
                f"no parent-relative pose could be read for parent {entry['parent']!r}"
            )
        for key in ('position_parent', 'orientation_parent',
                    'anchor_validation_episodes', 'reference_unavailable'):
            if key in lw:
                entry[key] = lw[key]
        waypoints_out.append(entry)

    def _world_pose(obj):
        wp = obj.get('world_pose_after_init', {})
        return {
            'position_xyz_m': wp.get('position_xyz_m'),
            'orientation_rpy_rad': wp.get('orientation_rpy_rad'),
        }

    graspable = [
        {'name': o['name'], 'world_pose_after_init': _world_pose(o)}
        for o in pd.get('graspable_objects', [])
    ]
    fixed = [
        {
            'name': o['name'],
            'reference_graspable_object': o.get('reference_graspable_object'),
            'world_pose_after_init': _world_pose(o),
        }
        for o in pd.get('other_scene_objects', [])
    ]

    data = {
        'task_name': task_name,
        'reference_variation_index': placement_data.get('reference_variation_index', 0),
        'reference_state': placement_data.get('reference_state', {}),
        'reference_objects': placement_data.get('reference_objects', {}),
        'units': 'positions in meters [x, y, z]; orientations rpy Euler radians [roll_x, pitch_y, yaw_z]',
        'waypoints': waypoints_out,
        'scene_objects': {'graspable': graspable, 'fixed': fixed},
    }
    if base.get('name'):
        data['task_boundary_root'] = base
    # Reset-only inference must never certify waypoints changed by live hooks or
    # repeated sequences. Preserve the generic source-analysis guard for those.
    from aha_publish.common.rlbench_dynamic_waypoints import analyze_task, unresolved_dynamic_waypoints
    analysis = analyze_task(task_name, tasks_dir=RLBENCH_ROOT / 'rlbench/tasks',
                            waypoint_count=len(waypoints_out))
    if analysis is not None:
        data['dynamic_waypoints'] = unresolved_dynamic_waypoints(analysis, data)
        data['dynamic_waypoint_reasons'] = {
            str(i): analysis['reasons'][i] for i in data['dynamic_waypoints']}
    # Runtime consumes these equations from this report, not the task source
    # or the simulator's waypoint geometry.
    from aha_publish.common.waypoint_reference_rules import export_hook_reference_rules
    task_source = RLBENCH_ROOT / 'rlbench/tasks' / f'{task_name}.py'
    if task_source.exists():
        rules = export_hook_reference_rules(task_source.read_text())
        if rules['hooks'] or rules['errors']:
            data['hook_reference_rules'] = rules
            if rules['hooks'] and not rules['errors']:
                data['rule_resolved_waypoints'] = data.get('dynamic_waypoints', [])
    return json.dumps(data, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Single-task inspection
# ─────────────────────────────────────────────────────────────────────────────
def run_single(target, save_path=None):
    if os.path.isfile(target):
        ttm_path  = target
        task_name = os.path.basename(target).replace('.ttm', '')
    else:
        ttm_path  = os.path.join(TTM_DIR, f'{target}.ttm')
        task_name = target
        if not os.path.isfile(ttm_path):
            print(f'File not found: {ttm_path}')
            sys.exit(1)

    if save_path is None:
        save_path = safe_report_name(task_name)

    waypoint_data = phase1(ttm_path)
    placement_data = phase2(task_name)

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(build_llm_report_json(
            task_name=task_name,
            ttm_path=ttm_path,
            waypoint_data=waypoint_data,
            placement_data=placement_data,
        ))
    print(f'\n[saved → {save_path}]')


# ─────────────────────────────────────────────────────────────────────────────
# Batch inspection (--all / --task): re-run this script per task as a subprocess
# ─────────────────────────────────────────────────────────────────────────────
def get_all_tasks():
    files = sorted(glob.glob(str(CONFIGS_PATH / '*.yaml')))
    return [Path(f).stem for f in files]


def get_description_tasks():
    """The task list the description generator numbers, in its own order.

    ``--tasks 31-90`` has to mean the same thing here as ``31-90`` typed at the
    task_description_maker menu, so the list is taken from that package rather
    than rebuilt: tasks that have a waypoint grid image in ``waypoints_description``,
    sorted by name (``runner.discover_generator_tasks``). That set is a subset of
    the failgen configs ``--all`` walks, so nothing selectable here is unknown to
    the rest of batch mode.
    """
    sys.path.insert(0, str(SELF_SCRIPT.parent))
    from aha_publish.descriptions.config import DEFAULT_PHOTO_DIR
    from aha_publish.descriptions.utils import extract_task_from_grid_image, find_all_grid_images

    names = {extract_task_from_grid_image(path)
             for path in find_all_grid_images(DEFAULT_PHOTO_DIR)}
    return sorted(names)


def parse_task_selection(spec, tasks):
    """1-based numbers and inclusive ranges ("31-90", "1,3-5") -> task names.

    Same grammar as the description generator's menu prompt, so a selection can
    be copied between the two. Raises ValueError with a usable message.
    """
    selected = {}
    for part in spec.split(','):
        bounds = part.strip().split('-')
        if len(bounds) not in (1, 2) or not all(b.strip().isdecimal() for b in bounds):
            raise ValueError(
                f'{part.strip()!r} is not a number or an ascending range '
                f'(e.g. 1,3-5)')
        start, end = int(bounds[0]), int(bounds[-1])
        if not 1 <= start <= end <= len(tasks):
            raise ValueError(
                f'{part.strip()!r} is out of range: pick numbers or ascending '
                f'ranges between 1 and {len(tasks)}')
        selected.update((index, None) for index in range(start, end + 1))
    return [tasks[index - 1] for index in sorted(selected)]


def print_task_listing(tasks, output_dir):
    print(f'\nTasks known to the description generator ({len(tasks)}):')
    for index, name in enumerate(tasks, start=1):
        done = (output_dir / safe_report_name(name)).exists()
        marker = '  — report already exists (will overwrite)' if done else ''
        print(f'  {index:>3}. {name}{marker}')
    print(f'\nSelect with --tasks (e.g. --tasks 31-{len(tasks)} or --tasks 1,3-5).')


def safe_report_name(task_name):
    """One report per task, named only by the task.

    An ordinal prefix used to be baked in, which made the filename depend on the
    task's position in the run: the same task inspected alone and inspected in a
    full sweep landed in two different files, and downstream lookups (which glob
    ``*_<task>.llm_context.json``) then had to pick between stale duplicates.
    """
    return f'{task_name}.llm_context.json'


def run_one(task_name, index, total, output_dir, log_dir, python_exe):
    report_path = output_dir / safe_report_name(task_name)
    log_path = log_dir / f'{task_name}.log'
    cmd = [
        python_exe,
        str(SELF_SCRIPT),
        task_name,
        '--save',
        str(report_path),
    ]

    started_at = datetime.now().isoformat(timespec='seconds')
    print(f'[{index:03d}/{total:03d}] inspecting {task_name} -> {report_path.name}', flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(paths.PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    finished_at = datetime.now().isoformat(timespec='seconds')
    log_path.write_text(proc.stdout, encoding='utf-8')

    status = 'ok' if proc.returncode == 0 else 'failed'
    return {
        'index': index,
        'task': task_name,
        'status': status,
        'returncode': proc.returncode,
        'report_path': str(report_path),
        'log_path': str(log_path),
        'started_at': started_at,
        'finished_at': finished_at,
    }


def write_manifest(rows, output_dir):
    csv_path = output_dir / 'manifest.csv'
    json_path = output_dir / 'manifest.json'
    fieldnames = [
        'index',
        'task',
        'status',
        'returncode',
        'report_path',
        'log_path',
        'started_at',
        'finished_at',
    ]

    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    return csv_path, json_path


def run_batch(args):
    output_dir = args.output_dir
    log_dir = output_dir / 'logs'
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    known_tasks = get_all_tasks()
    if args.tasks:
        try:
            description_tasks = get_description_tasks()
        except Exception as exc:
            print(f'Could not read the description generator task list: {exc}',
                  file=sys.stderr)
            return 2
        try:
            selection = parse_task_selection(args.tasks, description_tasks)
        except ValueError as exc:
            print(f'--tasks: {exc}', file=sys.stderr)
            return 2
    else:
        selection = []

    if args.task or selection:
        requested = list(args.task) + selection
        unknown = sorted(set(requested) - set(known_tasks))
        if unknown:
            print(f'Unknown task(s): {", ".join(unknown)}', file=sys.stderr)
            return 2
        tasks = sorted(set(requested))
    else:
        tasks = known_tasks

    if args.limit > 0:
        tasks = tasks[:args.limit]

    total = len(tasks)
    rows = []
    skipped = 0
    for index, task_name in enumerate(tasks, start=1):
        report_path = output_dir / safe_report_name(task_name)
        if args.skip_existing and report_path.exists():
            skipped += 1
            rows.append({
                'index': index,
                'task': task_name,
                'status': 'skipped_existing',
                'returncode': 0,
                'report_path': str(report_path),
                'log_path': '',
                'started_at': '',
                'finished_at': '',
            })
            print(f'[{index:03d}/{total:03d}] skipping existing {report_path.name}', flush=True)
            continue

        row = run_one(task_name, index, total, output_dir, log_dir, args.python)
        rows.append(row)
        write_manifest(rows, output_dir)
        if row['status'] != 'ok' and args.stop_on_error:
            break

    csv_path, json_path = write_manifest(rows, output_dir)
    ok = sum(1 for row in rows if row['status'] == 'ok')
    failed = sum(1 for row in rows if row['status'] == 'failed')
    print()
    print(f'Reports directory: {output_dir}')
    print(f'Manifest CSV:      {csv_path}')
    print(f'Manifest JSON:     {json_path}')
    print(f'Completed: {ok} ok, {failed} failed, {skipped} skipped, {len(rows)} recorded')
    return 1 if failed else 0


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description='Inspect RLBench task TTM scene structure + live placement.',
    )
    parser.add_argument(
        'target',
        nargs='?',
        help='Single-task mode: task name or path to a .ttm file.',
    )
    parser.add_argument(
        '-o', '--save',
        help='Single-task mode: output report path. Default: <task>.llm_context.json',
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Batch mode: inspect every task found in the failgen config folder.',
    )
    parser.add_argument(
        '--task',
        action='append',
        default=[],
        help='Batch mode: inspect only this task. Can be passed multiple times.',
    )
    parser.add_argument(
        '--tasks',
        help='Batch mode: 1-based numbers/ranges into the task description '
             'generator\'s task list, e.g. "31-90" or "1,3-5". Same numbering as '
             'the task_description_maker menu; see --list-tasks.',
    )
    parser.add_argument(
        '--list-tasks',
        action='store_true',
        help='Print the numbered description-generator task list and exit.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Batch mode reports directory. Default: {DEFAULT_OUTPUT_DIR}',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=0,
        help='Batch mode: inspect only the first N tasks after sorting.',
    )
    parser.add_argument(
        '--python',
        default=sys.executable,
        help='Batch mode: Python executable used for each single inspection. Default: current Python.',
    )
    parser.add_argument(
        '--stop-on-error',
        action='store_true',
        help='Batch mode: stop after the first failed task.',
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Batch mode: skip tasks whose report file already exists.',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_tasks:
        try:
            print_task_listing(get_description_tasks(), args.output_dir)
        except Exception as exc:
            print(f'Could not read the description generator task list: {exc}',
                  file=sys.stderr)
            return 2
        return 0

    if args.all or args.task or args.tasks:
        return run_batch(args)

    if not args.target:
        print(__doc__)
        sys.exit(1)

    run_single(args.target, args.save)


if __name__ == '__main__':
    raise SystemExit(main())
