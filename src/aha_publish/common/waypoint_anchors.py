"""Infer independent waypoint position/orientation frames from clean resets.

No task names or simulator access: episodes map object names to xyzw poses.
Inference applies to the sampled variation, not to later waypoint ability hooks.
"""

from aha_publish import paths

import copy

import numpy as np

from aha_publish.common.waypoint_chain import pose_error, quat_to_rpy, relative_pose


def infer_waypoint_anchors(waypoints, episodes, position_tol=1e-4,
                           orientation_tol=1e-3):
    """Return report entries with validated frames, or reference_unavailable.

    Keep the scene parent for each component that remains invariant. Otherwise
    search non-waypoint objects, independently for position and orientation.
    Offsets always come from episode zero so the report can round-trip.
    """
    if len(episodes) < 3:
        raise ValueError("anchor inference requires at least three episodes")
    result = copy.deepcopy(waypoints)
    waypoint_names = {w['name'] for w in waypoints}
    candidates = sorted(set.intersection(*(set(e) for e in episodes))
                        - waypoint_names)
    for wp in result:
        name, parent = wp['name'], wp.get('parent')
        relatives = {}
        for anchor in set(candidates) | ({parent} if parent else set()):
            if anchor == name or any(name not in e or anchor not in e for e in episodes):
                continue
            rels = [relative_pose(e[anchor], e[name]) for e in episodes]
            errors = [pose_error(rels[0], r) for r in rels[1:]]
            relatives[anchor] = (rels[0], np.max(errors, axis=0))
        selected = []
        for component, tolerance in enumerate((position_tol, orientation_tol)):
            valid = [a for a in relatives if relatives[a][1][component] <= tolerance]
            if parent in valid:
                selected.append(parent)
            else:
                selected.append(min(valid, key=lambda a: (relatives[a][1][component], a))
                                if valid else None)
        if None in selected:
            wp['reference_unavailable'] = (
                'no invariant position/orientation frames across clean resets')
            continue
        p_parent, q_parent = selected
        wp.pop('reference_unavailable', None)
        wp['position_parent'] = p_parent
        wp['orientation_parent'] = q_parent
        wp['local_position_xyz_m'] = relatives[p_parent][0][:3].round(9).tolist()
        wp['local_quaternion_xyzw'] = relatives[q_parent][0][3:7].round(9).tolist()
        wp['local_orientation_rpy_rad'] = quat_to_rpy(relatives[q_parent][0][3:7]).round(9).tolist()
        wp['anchor_validation_episodes'] = len(episodes)
    return result
