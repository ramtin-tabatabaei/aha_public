"""Task-specific boundary updates for references anchored to moving objects."""

from aha_publish import paths

import numpy as np

from aha_publish.common.waypoint_chain import resolve_waypoint_poses


MOVING_ANCHORS = {
    'take_item_out_of_drawer': {4: 'item', 5: 'item', 6: 'item'},
    'change_channel': {i: 'tv_remote' for i in (0, 1, 2, 6, 7)},
}


def refresh_moving_object_reference(task_name, chain, waypoint, read_pose):
    """Snapshot a task's moving anchor at a boundary, never during motion.

    Drawer opening and remote placement move object-relative targets. Recompose
    only the upcoming target so completed targets retain their own snapshots.
    The caller must invoke this before failure hooks perturb the waypoint.
    """
    anchor = MOVING_ANCHORS.get(task_name, {}).get(waypoint)
    if anchor is None or chain is None:
        return False
    try:
        anchor_pose = np.asarray(read_pose(anchor), dtype=float)
        if anchor_pose.shape != (7,) or not np.all(np.isfinite(anchor_pose)):
            raise ValueError(f'{anchor} pose must contain seven finite values')
        world = dict(chain.world_poses)
        world[anchor] = anchor_pose.copy()
        resolved = resolve_waypoint_poses(chain.spec, world)
        chain.poses[waypoint] = resolved[waypoint].copy()
    except Exception:
        # A stale reference would turn an unavailable measurement into an alarm.
        chain.poses.pop(waypoint, None)
        raise
    return True
