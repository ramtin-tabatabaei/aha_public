
from aha_publish import paths
import argparse
import atexit
from datetime import datetime
import json
import os
import re
import shutil
import sys
import types
from pathlib import Path

PROJECT_ROOT = (paths.PROJECT_ROOT)
SCRIPT_DIR = (paths.SOURCE_DIR / 'running')
FAILGEN_ROOT = (paths.FAILGEN_ROOT)
sys.path.insert(0, str(paths.PROJECT_ROOT))
sys.path.insert(0, str(paths.SOURCE_DIR))
sys.path.insert(0, str(FAILGEN_ROOT))

from aha_publish.common.task_categories import print_task_category


CONFIGS_PATH = (paths.FAILGEN_ROOT / 'failgen/configs')
PREPARED_BTS_DIR = (paths.BT_DIR)
# Condition definitions live next to this main script (this folder).
CONDITION_DEFINITIONS_DIR = SCRIPT_DIR
CONDITION_DEFINITIONS_PATH = CONDITION_DEFINITIONS_DIR / 'condition_definitions.json'
VLM_TRACE_DIR = (paths.OUTPUT_DIR / 'vlm_traces')
# Debugging/offline-tuning aid: when AHA_SAVE_VLM_FRAMES points at a directory,
# every image actually sent to the VLM is also written to disk (exact pixels,
# including the widened wrist FOV). Lets us re-judge saved checkpoints offline
# with candidate prompts without re-running the simulator. Off by default.
SAVE_VLM_FRAMES_DIR = os.getenv('AHA_SAVE_VLM_FRAMES', '').strip()
# Analysis aid: when AHA_DUMP_WAYPOINT_FRAMES is truthy, EVERY sim step of EVERY
# waypoint (its path and gripper action) is written to disk as PNGs for all
# cameras (side, wrist, front). Lets you inspect the whole episode -- including
# exactly what the injected failure looked like -- frame-by-frame. Off by
# default.
DUMP_WAYPOINT_FRAMES = os.getenv(
    'AHA_DUMP_WAYPOINT_FRAMES', '').strip().lower() not in ('', '0', 'false', 'no')
# Frames are captured for EVERY waypoint and, by default, KEPT for every
# waypoint: the whole run stays inspectable frame-by-frame. Set
# AHA_KEEP_ALL_WAYPOINT_FRAMES=0 to restore the old end-of-episode pruning that
# deletes everything except the injection waypoint and the detector-fire
# waypoints (far less disk, but only those waypoints survive).
KEEP_ALL_WAYPOINT_FRAMES = os.getenv(
    'AHA_KEEP_ALL_WAYPOINT_FRAMES', '1').strip().lower() not in ('', '0', 'false', 'no')
DEFAULT_VLM_MODEL = os.getenv('OPENAI_BT_CONDITION_VLM_MODEL', os.getenv('OPENAI_MODEL', 'gpt-5.4'))
VLM_CAMERA_NAMES = ('side_rgb', 'front_rgb', 'wrist_rgb')
# OpenAI image `detail` for the condition-check frames. Unset (or 'auto') keeps
# the API default, which tiles a 256x256 render into one 512 tile at ~255
# tokens; 'low' flattens every image to a flat ~85 tokens. Input tokens are
# prefilled in parallel, so this is an input-cost knob, not a latency one -- and
# 'low' throws away the few pixels in the wrist close-up that separate a clamped
# grip from an open one (see _image_to_data_url in
# aha_scripts/detectors/collision/vlm_confirm.py). Override with
# $AHA_VLM_IMAGE_DETAIL.
VLM_IMAGE_DETAIL = os.getenv('AHA_VLM_IMAGE_DETAIL', '').strip().lower()
# The wrist camera's default field of view looks past the fingertips, so the
# fingers fall outside the frame and open-vs-closed is unreadable. We widen the
# existing wrist camera (no extra camera) at startup so wrist_rgb shows both
# fingers and the gap between them. Override with $AHA_WRIST_VLM_FOV_DEG; set to
# 0 to leave the wrist camera untouched.
WRIST_VLM_FOV_DEG = float(os.getenv('AHA_WRIST_VLM_FOV_DEG', '0'))
# When no absolute FOV is requested, widen the wrist camera's native perspective
# angle by this factor so wrist_rgb is slightly wider (1.05 = 5% wider). Set to
# 1.0 to leave the native FOV untouched. Override with $AHA_WRIST_VLM_FOV_SCALE.
WRIST_VLM_FOV_SCALE = float(os.getenv('AHA_WRIST_VLM_FOV_SCALE', '1.1'))
# The wrist camera looks straight down the gripper approach axis, so the finger
# bodies sit at the very edge of the frame. A small downward pitch tilts the
# view toward the fingers so more of the gripper is visible. Override with
# $AHA_WRIST_VLM_TILT_DEG (degrees, about the camera's local X axis); set to 0
# to leave the wrist camera pointing straight down. Flip the sign if it tilts
# the wrong way.
WRIST_VLM_TILT_DEG = float(os.getenv('AHA_WRIST_VLM_TILT_DEG', '20'))

# --- VLM token-usage accounting for one run (printed once at process exit) ---
VLM_USAGE = {'calls': 0, 'input_tokens': 0, 'output_tokens': 0}


def _price_per_1m(env_name):
    raw = os.getenv(env_name)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def print_vlm_usage_summary():
    u = VLM_USAGE
    if not u['calls']:
        return
    total = u['input_tokens'] + u['output_tokens']
    in_p = _price_per_1m('AHA_INPUT_PRICE_PER_1M')
    out_p = _price_per_1m('AHA_OUTPUT_PRICE_PER_1M')
    line = (
        f"\n[vlm-usage] this run: {u['calls']} VLM call(s), "
        f"{u['input_tokens']} input + {u['output_tokens']} output = {total} tokens"
    )
    if in_p is not None and out_p is not None:
        cost = u['input_tokens'] / 1_000_000 * in_p + u['output_tokens'] / 1_000_000 * out_p
        line += f"; cost ${cost:.4f} (at ${in_p:g}/${out_p:g} per 1M in/out)"
    else:
        line += "; cost n/a (set AHA_INPUT_PRICE_PER_1M and AHA_OUTPUT_PRICE_PER_1M to price it)"
    print(line)


atexit.register(print_vlm_usage_summary)

# ---------------------------------------------------------------------------
# Authoritative gripper-state capture (AHA_DUMP_GRIPPER_STATE=1).
#
# During a real demo run the runner knows the TRUE per-waypoint gripper state:
# whether the jaws ended open/closed and which object (if any) is actually
# grasped (gripper.get_grasped_objects()). A standalone replay can't reproduce
# the grasp reliably, so we capture it here and dump it to
# bt_maker/gripper_sequences/<task>.json — the ground truth fed into the
# description maker so gripper_condition / object_in_gripper come out right.
# ---------------------------------------------------------------------------
DUMP_GRIPPER_STATE = os.getenv(
    'AHA_DUMP_GRIPPER_STATE', '').strip().lower() in ('1', 'true', 'yes', 'on')
_GRIPPER_CAPTURE = {}
_GRIPPER_CAPTURE_META = {}


def capture_gripper_state(ctx, i, wp):
    if not DUMP_GRIPPER_STATE:
        return
    _GRIPPER_CAPTURE_META['task'] = getattr(ctx, 'task_name', 'unknown')
    gripper = ctx.robot.gripper
    held = []
    held_positions = {}
    try:
        for o in gripper.get_grasped_objects():
            nm = o.get_name()
            held.append(nm)
            try:
                held_positions[nm] = [float(v) for v in o.get_position()]
            except Exception:
                pass
    except Exception:
        held = []
    open_val = None
    try:
        amt = gripper.get_open_amount()
        open_val = float(sum(amt) / len(amt)) if amt else None
    except Exception:
        open_val = None
    ext = (waypoint_extension(wp) or '').strip()
    action = ('open' if 'open_gripper' in ext.lower()
              else 'close' if 'close_gripper' in ext.lower() else 'none')
    if action == 'open':
        state_after = 'Open'
    elif action == 'close':
        state_after = 'Closed'
    else:
        prev = _GRIPPER_CAPTURE.get(i - 1)
        state_after = prev['state_after'] if prev else 'Open'
    _GRIPPER_CAPTURE[i] = {
        'index': i,
        'extension': ext,
        'action': action,
        'gripper_open_amount': open_val,
        'state_after': state_after,
        'held_after': held,
        'held_positions': held_positions,
    }
    # Write incrementally: the runner can hang at the final wait-for-space
    # prompt and be killed before atexit runs, so don't rely on atexit alone.
    _dump_gripper_capture()


def _dump_gripper_capture():
    if not DUMP_GRIPPER_STATE or not _GRIPPER_CAPTURE:
        return
    import json as _json
    from pathlib import Path as _Path
    task = _GRIPPER_CAPTURE_META.get('task', 'unknown')
    wps = [_GRIPPER_CAPTURE[k] for k in sorted(_GRIPPER_CAPTURE)]
    out = ((paths.GRIPPER_DIR) / f'{task}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json.dumps({
        'task': task, 'initial_state': 'Open',
        'n_waypoints': len(wps), 'waypoints': wps}, indent=2))
    print(f"[gripper-capture] wrote {out}")


atexit.register(_dump_gripper_capture)
GRIPPER_ACTION_MAX_STEPS = int(os.getenv('AHA_GRIPPER_ACTION_MAX_STEPS', '30'))
# Opening is unobstructed, so let it run until the gripper is completely open
# (amount 1.0). The 30-step cap above is meant for closing onto an object, where
# actuate() may never report done; reusing it for opening could stop the gripper
# only partially open.
GRIPPER_OPEN_MAX_STEPS = int(os.getenv('AHA_GRIPPER_OPEN_MAX_STEPS', '200'))
PATH_ACTION_MAX_STEPS = int(os.getenv('AHA_PATH_ACTION_MAX_STEPS', '1000'))

VLM_SYSTEM_PROMPT = (
    "You are a robotics task-verification auditor for an RLBench robot simulation. "
    "You inspect camera images and decide whether behavior-tree conditions are "
    "visually satisfied at a specific checkpoint. A precondition checkpoint is "
    "before the robot moves for that waypoint. A postcondition checkpoint is "
    "after the waypoint path and any gripper action for that waypoint have "
    "completed. Use only observable visual evidence from the images. Do not use "
    "simulator state values, telemetry, numeric measurements, or hidden state. "
    "An object name ending in a numeric suffix "
    "means there are multiple instances of that object type. The number does "
    "not require a particular instance: either matching instance is acceptable. "
    "In contrast, a positional or ordinal word at the START of an object name "
    "(bottom, top, middle, upper, lower, left, right, front, back, near, far, "
    "first, second, third, fourth, last) DOES pick out one specific instance "
    "among several identical-looking siblings, and you must verify you are "
    "looking at that exact one. Locate all the sibling instances first, order "
    "them along the axis the word names (top/bottom and upper/lower by height, "
    "left/right by the horizontal axis in the view, front/back and near/far by "
    "depth from the robot base, ordinals counting from the top or from the "
    "nearest side unless the definition says otherwise), then judge the "
    "condition only against the instance the name selects. Decide which "
    "instance the robot is manipulating from the front and side RGB views "
    "(front_rgb and side_rgb): they show the gripper's height, depth, and "
    "lateral position against the whole set of siblings. The wrist view is too "
    "close to show the siblings it is not pointing at, so use it only to "
    "confirm that choice, never to make it. If the images do not "
    "let you tell which sibling is which, mark the condition uncertain rather "
    "than judging a different instance. "
    "Focus on the object the robot gripper is holding or interacting with, "
    "rather than another matching object elsewhere in the scene. For placement "
    "or release checks, follow that manipulated object to its destination. "
    "Use the three statuses precisely: mark a condition not_satisfied ONLY when "
    "the images clearly show it is violated; if the relevant evidence is occluded, "
    "out of frame, or ambiguous, mark it uncertain (not not_satisfied) instead of "
    "guessing. Return strict JSON only."
)

CAMERAS = [
    "default",
    "_cam_front",
    "_cam_over_shoulder_left",
    "_cam_over_shoulder_right",
    "_cam_overhead",
    "_cam_wrist",
    "side",
    "cam_cinematic_placeholder",
]

CAMERA_LABELS = {
    "default": "Default simulator camera",
    "_cam_front": "Front camera",
    "_cam_over_shoulder_left": "Left shoulder camera",
    "_cam_over_shoulder_right": "Right shoulder camera",
    "_cam_overhead": "Overhead camera",
    "_cam_wrist": "Wrist camera (robot hand)",
    "side": "Detector side camera",
    "cam_cinematic_placeholder": "Cinematic view",
}

SIDE_CAMERA_NAME = 'aha_side_camera'
SIDE_CAMERA_POSITION = (-0.547, -0.803, 1.209)  # nudged ~0.45 m forward along the look axis
SIDE_CAMERA_ORIENTATION_DEG = (-100, 38.3, -180.0)  # panned left


class LiveCameraView:
    def __init__(self, camera, title, refresh_sec=0.05, initial_frame=None):
        import matplotlib.pyplot as plt
        import numpy as np

        self.plt = plt
        self.np = np
        self.camera = camera
        self.refresh_sec = refresh_sec
        self.latest_frame = None
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8.5, 7.2))
        self.fig.canvas.manager.set_window_title(title)
        self.ax.set_title(title)
        self.ax.axis('off')
        if initial_frame is None:
            initial_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        else:
            initial_frame = np.asarray(initial_frame, dtype=np.uint8)
            self.latest_frame = initial_frame.copy()
        self.image_artist = self.ax.imshow(initial_frame)
        self.plt.ion()
        self.fig.show()
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def update(self):
        frame = capture_camera_rgb(self.camera)
        self.latest_frame = frame.copy()
        self.image_artist.set_data(frame)
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)
        return self.latest_frame


def widen_wrist_camera(scene, fov_deg=WRIST_VLM_FOV_DEG, fov_scale=WRIST_VLM_FOV_SCALE):
    """Widen the existing wrist camera's field of view so wrist_rgb is slightly
    wider (no extra camera is created). An explicit ``fov_deg`` sets the
    perspective angle absolutely; otherwise the native angle is scaled by
    ``fov_scale`` (1.05 = 5% wider). Returns the applied FOV in degrees, or None
    if it was left unchanged."""
    wrist_cam = getattr(scene, '_cam_wrist', None)
    if wrist_cam is None:
        return None
    # PyRep VisionSensor perspective angles are in DEGREES.
    if fov_deg and fov_deg > 0:
        wrist_cam.set_perspective_angle(fov_deg)
        return fov_deg
    if fov_scale and fov_scale != 1.0:
        try:
            current = float(wrist_cam.get_perspective_angle())
        except Exception:
            return None
        new_fov = current * fov_scale
        wrist_cam.set_perspective_angle(new_fov)
        return new_fov
    return None


def tilt_wrist_camera(scene, tilt_deg=WRIST_VLM_TILT_DEG):
    """Pitch the wrist camera down by ``tilt_deg`` degrees (about its own local
    X axis) so wrist_rgb shows more of the gripper body instead of looking
    straight past the fingertips. The camera is parented to the gripper, so a
    rotation in its own frame persists as the arm moves. Returns the applied
    tilt, or None if it was left unchanged."""
    import numpy as np

    if not tilt_deg:
        return None
    wrist_cam = getattr(scene, '_cam_wrist', None)
    if wrist_cam is None:
        return None
    # PyRep Object.rotate takes Euler angles in RADIANS, applied about the
    # object's own reference frame.
    wrist_cam.rotate([float(np.radians(tilt_deg)), 0.0, 0.0])
    return tilt_deg


WRIST_VISUALIZATION_MODES = [
    ("save", "Save PNG images only"),
    ("matplotlib", "Show Matplotlib windows and save PNG images"),
    ("off", "Do not save or show wrist images"),
    ("depth_segmentation", "Segment from wrist depth, show Matplotlib windows, and save PNG images"),
    ("rgb_segmentation", "Segment from wrist RGB, show Matplotlib windows, and save PNG images"),
]


def get_camera_handle(scene, camera_name):
    from pyrep.backend import sim

    if camera_name == "default":
        return sim.simGetObjectHandle("DefaultCamera")
    if camera_name == "side":
        return get_or_create_side_camera(scene).get_handle()
    if camera_name.startswith("_cam"):
        return getattr(scene, camera_name).get_handle()
    return sim.simGetObjectHandle(camera_name)


def get_or_create_side_camera(scene):
    existing = getattr(scene, '_aha_side_camera', None)
    if existing is not None:
        return existing

    import numpy as np
    from pyrep.const import RenderMode
    from pyrep.objects.vision_sensor import VisionSensor

    side_camera = VisionSensor.create([256, 256])
    side_camera.set_name(SIDE_CAMERA_NAME)
    side_camera.set_render_mode(RenderMode.OPENGL3)
    side_camera.set_position(list(SIDE_CAMERA_POSITION))
    side_camera.set_orientation(np.radians(SIDE_CAMERA_ORIENTATION_DEG))
    scene._aha_side_camera = side_camera
    return side_camera


def capture_camera_rgb(camera):
    import numpy as np

    # capture_rgb() returns floats in [0, 1], but a freshly created sensor that
    # has not rendered yet can return NaN/inf or out-of-range values. Sanitize
    # before scaling so we never overflow the multiply or cast invalid values.
    frame = np.asarray(camera.capture_rgb(), dtype=np.float64)
    frame = np.nan_to_num(frame, nan=0.0, posinf=1.0, neginf=0.0)
    frame = np.clip(frame, 0.0, 1.0)
    return (frame * 255.0).astype(np.uint8)


def camera_frame_has_content(frame):
    import numpy as np

    if frame is None:
        return False
    arr = np.asarray(frame)
    return arr.size > 0 and (float(arr.max()) > 8.0 or float(arr.mean()) > 2.0)


def waypoint_frames_base(task_name):
    """Base directory holding the per-waypoint ``wp<N>`` frame folders.

    Prefers the active VLM run-log image folder (so batch ``run_task_all_conditions``
    runs drop the frames right inside the case folder); otherwise falls back to
    ``aha_output/waypoint_frames/<task>``. Returns None when dumping is disabled.
    """
    if not DUMP_WAYPOINT_FRAMES:
        return None
    if SAVE_VLM_FRAMES_DIR:
        return Path(SAVE_VLM_FRAMES_DIR) / 'waypoint_frames'
    return (paths.OUTPUT_DIR / 'waypoint_frames') / task_name


def waypoint_frame_dump_dir(task_name, waypoint_index):
    """Directory for the full per-step frame dump of a single waypoint. Returns
    None when dumping is disabled."""
    base = waypoint_frames_base(task_name)
    if base is None:
        return None
    out = base / f'wp{waypoint_index}'
    out.mkdir(parents=True, exist_ok=True)
    return out


def prune_waypoint_frame_dirs(task_name, keep_waypoints):
    """After the episode, keep only the frame folders worth inspecting: the
    injection waypoint plus every waypoint where a detector fired. All other
    per-waypoint dumps -- captured because we cannot know in advance which
    waypoints a detector will flag -- are deleted."""
    base = waypoint_frames_base(task_name)
    if base is None or not base.exists():
        return
    keep = {int(w) for w in keep_waypoints if w is not None}
    for child in base.iterdir():
        if not child.is_dir():
            continue
        match = re.match(r'wp(\d+)$', child.name)
        if match and int(match.group(1)) not in keep:
            shutil.rmtree(child, ignore_errors=True)


def resolve_dump_cameras(scene, side_camera):
    """Vision sensors captured for the failure-waypoint frame dump: side, wrist,
    and front. Each is None-safe; missing sensors are simply skipped."""
    cams = {}
    if side_camera is not None:
        cams['side'] = side_camera
    for label, attr in (('wrist', '_cam_wrist'), ('front', '_cam_front')):
        sensor = getattr(scene, attr, None)
        if sensor is not None:
            cams[label] = sensor
    return cams


def dump_waypoint_step_frames(out_dir, cams, waypoint_index, phase, step_idx):
    """Write one PNG per camera for a single sim step of the failure waypoint.

    ``capture_rgb`` renders each sensor on demand, so this works even though only
    ``chosen_cam`` is rendered by the sim-step call. Best-effort: a bad frame or
    write is skipped rather than aborting the run.
    """
    if not out_dir or not cams:
        return
    from PIL import Image

    for cam_name, sensor in cams.items():
        try:
            frame = capture_camera_rgb(sensor)
        except Exception:
            continue
        if not camera_frame_has_content(frame):
            continue
        # Each camera gets its own subfolder: wp<N>/<camera>/<phase>_stepNNNN.png
        cam_dir = out_dir / cam_name
        try:
            cam_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame, mode='RGB').save(
                cam_dir / f'{phase}_step{step_idx:04d}.png')
        except Exception:
            pass


def attach_side_rgb(obs, side_camera):
    """Give this observation its OWN fresh side-camera frame.

    The detector VLM falls back to a single live side-camera capture when an
    observation has no ``side_rgb`` (vlm_confirm._source_to_image), which collapses
    the per-step image sequence to one moment. The standalone detector harness
    avoids this by stamping a fresh ``side_rgb`` on every observation; we do the
    same here so the recent-observation buffer shows distinct timepoints.
    """
    if obs is None or side_camera is None:
        return obs
    try:
        setattr(obs, 'side_rgb', capture_camera_rgb(side_camera))
    except Exception:
        pass
    return obs


def warm_up_camera_sensor(scene, camera, steps=12):
    """Step the renderer until a newly created vision sensor has a real frame."""
    latest_frame = None
    for _ in range(max(1, steps)):
        try:
            scene.pyrep.step()
        except Exception:
            try:
                scene.step()
            except Exception:
                pass
        try:
            refresh_camera_sensors(scene)
        except Exception:
            pass
        try:
            latest_frame = capture_camera_rgb(camera)
            if camera_frame_has_content(latest_frame):
                return latest_frame
        except Exception:
            pass
    return latest_frame


def sync_default_camera(scene, camera_name):
    """Make the visible CoppeliaSim GUI camera match the selected RLBench camera."""
    from pyrep.backend import sim

    if camera_name == "default":
        return
    src = get_camera_handle(scene, camera_name)
    if sim.lib.simAdjustView(-1, src, 8, sim.ffi.NULL) >= 0:
        return
    dst = sim.simGetObjectHandle("DefaultCamera")
    pos = sim.simGetObjectPosition(src, -1)
    rot = sim.simGetObjectOrientation(src, -1)
    sim.simSetObjectPosition(dst, -1, pos)
    sim.simSetObjectOrientation(dst, -1, rot)


def refresh_camera_sensors(scene):
    """Refresh RLBench vision sensors that use explicit handling."""
    try:
        return scene.get_observation()
    except Exception:
        return None


def step_with_camera(scene, camera_name):
    scene.step()
    refresh_camera_sensors(scene)
    if camera_name == "default":
        return
    try:
        sync_default_camera(scene, camera_name)
    except Exception:
        pass


def pyrep_step_with_camera(scene, camera_name):
    scene.pyrep.step()
    refresh_camera_sensors(scene)
    if camera_name == "default":
        return
    try:
        sync_default_camera(scene, camera_name)
    except Exception:
        pass


def env_step_with_camera(env_wrapper, scene, camera_name):
    scene.step()
    obs = refresh_camera_sensors(scene)
    if obs is not None:
        env_wrapper.on_env_step(obs)
    if camera_name != "default":
        try:
            sync_default_camera(scene, camera_name)
        except Exception:
            pass
    return obs


def env_pyrep_step_with_camera(env_wrapper, scene, camera_name):
    scene.pyrep.step()
    obs = refresh_camera_sensors(scene)
    if obs is not None:
        env_wrapper.on_env_step(obs)
    if camera_name != "default":
        try:
            sync_default_camera(scene, camera_name)
        except Exception:
            pass
    return obs


def get_all_tasks():
    files = sorted(CONFIGS_PATH.glob('*.yaml'))
    return [path.stem for path in files]


def load_task_config(task_name):
    import yaml

    path = CONFIGS_PATH / f'{task_name}.yaml'
    if not path.exists():
        return {}
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def load_yaml_descriptions(task_name):
    config = load_task_config(task_name)
    descriptions = {}
    for st in config.get('sub-tasks', []):
        processes = st.get('processes', [])
        desc_list = st.get('task_description', [''])
        if len(processes) >= 2:
            wp_from = int(processes[0].replace('waypoint', ''))
            descriptions[wp_from] = desc_list[0] if desc_list else ''
    return descriptions


def pick_task():
    tasks = get_all_tasks()
    return pick_from_list('AVAILABLE TASKS', tasks)


def pick_camera():
    return pick_from_list('AVAILABLE CAMERAS', CAMERAS)


def pick_wrist_visualization_mode():
    print("\n" + "=" * 60)
    print("  WRIST DEPTH / SEGMENTATION VIEW")
    print("=" * 60)
    for i, (_, label) in enumerate(WRIST_VISUALIZATION_MODES):
        print(f"  [{i + 1}] {label}")
    print("=" * 60)
    while True:
        try:
            raw = input("\nEnter visualization number [2]: ").strip()
            choice = 1 if raw == "" else int(raw) - 1
            if 0 <= choice < len(WRIST_VISUALIZATION_MODES):
                return WRIST_VISUALIZATION_MODES[choice][0]
        except ValueError:
            pass
        print("Invalid choice.")


def pick_failure(config):
    from aha_publish.common.failures import get_available_failures

    failures = get_available_failures(config)
    print("\n" + "=" * 80)
    print("  CHOOSE FAILURE TYPE")
    print("=" * 80)
    print("  [ 0] No failure")
    for i, failure in enumerate(failures):
        wps = failure.get('waypoints', [])
        print(f"  [{i + 1:>2}] {failure.get('type', '?'):<20} waypoints: {wps}")
    print("=" * 80)
    while True:
        try:
            choice = int(input("\nEnter failure number: "))
            if choice == 0:
                return None, -1
            if 1 <= choice <= len(failures):
                failure = failures[choice - 1]
                ftype = failure.get('type')
                # wrong_sequence_v2 shuffles the whole waypoint order, so there is
                # no single injection waypoint to choose.
                if ftype == 'wrong_sequence_v2':
                    return ftype, -1
                return ftype, pick_failure_waypoint(failure.get('waypoints', []))
        except ValueError:
            pass
        print("Invalid choice.")


def pick_failure_waypoint(waypoints):
    if not waypoints:
        return -1
    print("\n" + "=" * 60)
    print("  WAYPOINTS FOR THIS FAILURE")
    print("=" * 60)
    for i, waypoint in enumerate(waypoints):
        print(f"  [{i + 1:>2}] Waypoint {waypoint}")
    print("=" * 60)
    while True:
        try:
            choice = int(input("\nEnter waypoint number: "))
            if 1 <= choice <= len(waypoints):
                return int(waypoints[choice - 1])
        except ValueError:
            pass
        print("Invalid choice.")


def resolve_failure_choice(config, args):
    if args.failure is None:
        return pick_failure(config)

    if args.failure in ("", "none", "None", "NO_FAIL", "no_failure"):
        return None, -1

    from aha_publish.common.failures import get_available_failures

    failures = get_available_failures(config)
    valid_types = [failure.get('type') for failure in failures]
    if args.failure not in valid_types:
        raise ValueError(
            f"Failure '{args.failure}' is not available for this task. "
            f"Available: {', '.join(str(item) for item in valid_types)}"
        )

    matching = next(failure for failure in failures if failure.get('type') == args.failure)
    waypoint = args.failure_waypoint
    if args.failure == 'wrong_sequence_v2':
        # Whole-sequence shuffle: no single injection waypoint.
        return args.failure, -1
    if waypoint is None:
        waypoint = pick_failure_waypoint(matching.get('waypoints', []))
    return args.failure, int(waypoint)


def configure_failure(env_wrapper, config, failtype, waypoint):
    from aha_publish.common.failures import get_task_waypoints
    from failgen.fail_freezing import FreezingFailure
    from failgen.fail_wrong_object import WrongObjectFailure
    from failgen.fail_sequence_v2 import WrongSequenceV2Failure

    if not failtype:
        for fail_obj in env_wrapper.manager._failures:
            fail_obj.set_enabled(False)
        print("Failure mode: none")
        return

    if failtype == FreezingFailure.FAILURE_TYPE:
        has_freezing = any(
            fail_obj.failure_type == FreezingFailure.FAILURE_TYPE
            for fail_obj in env_wrapper.manager._failures
        )
        if not has_freezing:
            env_wrapper.manager.add_failure(
                FreezingFailure(
                    robot=env_wrapper.robot,
                    name='freezing_random_midway',
                    waypoints_indices=get_task_waypoints(config),
                    freeze_after_range=(3, 10),
                    freeze_duration_steps=60,
                )
            )

    has_failtype = False
    target_fail_obj = None
    for fail_obj in env_wrapper.manager._failures:
        if fail_obj.failure_type == failtype:
            fail_obj.set_enabled(True)
            has_failtype = True
            target_fail_obj = fail_obj
        else:
            fail_obj.set_enabled(False)

    if not has_failtype:
        raise RuntimeError(f"Failure type '{failtype}' not found for this task.")

    # wrong_object and wrong_sequence_v2 are whole-episode failures: they are not
    # tied to a single injection waypoint (v2 shuffles the entire waypoint order),
    # so don't pin their fail name to a chosen waypoint.
    if (
        target_fail_obj
        and failtype not in (WrongObjectFailure.FAILURE_TYPE, WrongSequenceV2Failure.FAILURE_TYPE)
        and waypoint >= 0
    ):
        target_fail_obj.change_waypoint_fail_name(f"waypoint{waypoint}")

    wp_label = f"waypoint {waypoint}" if waypoint >= 0 else "configured waypoint"
    print(f"Failure mode: {failtype} at {wp_label}")


def wait_for_space(prompt):
    """Wait for one space or Enter key press."""
    print(prompt, end='', flush=True)
    if not sys.stdin.isatty():
        try:
            input()
        except EOFError:
            print("  [auto-close]")
        return

    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch in (' ', '\n', '\r'):
                print("  [accepted]")
                return
            if ch.lower() == 'q':
                print("\nStopped by user.")
                raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def ask_yes_no(prompt, default=False):
    suffix = " [Y/n] " if default else " [y/N] "
    if not sys.stdin.isatty():
        return default
    raw = input(prompt + suffix).strip().lower()
    if raw == "":
        return default
    return raw in ("y", "yes")


def pick_from_list(title, items, label_fn=str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)
    for i, item in enumerate(items):
        print(f"  [{i + 1:>2}] {label_fn(item)}")
    print("=" * 60)
    while True:
        try:
            choice = int(input("\nEnter number: ")) - 1
            if 0 <= choice < len(items):
                return items[choice]
        except ValueError:
            pass
        print("Invalid choice.")


def prepared_bt_candidates(task_name):
    pattern = f'{task_name}.bt_conditions.json'
    return sorted(PREPARED_BTS_DIR.rglob(pattern))


def pick_bt_conditions_path(task_name):
    candidates = prepared_bt_candidates(task_name)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return pick_from_list(
            'AVAILABLE PREPARED BT CONDITION FILES',
            candidates,
            lambda path: str(path.relative_to(PROJECT_ROOT)),
        )

    all_condition_files = sorted(PREPARED_BTS_DIR.rglob('*.bt_conditions.json'))
    if not all_condition_files:
        raise FileNotFoundError(f'No .bt_conditions.json files found in {PREPARED_BTS_DIR}')

    print(f"\nNo prepared BT condition file found for task '{task_name}'.")
    print("Choose another prepared condition file, or press Ctrl+C to stop.")
    return pick_from_list(
        'AVAILABLE PREPARED BT CONDITION FILES',
        all_condition_files,
        lambda path: str(path.relative_to(PROJECT_ROOT)),
    )


def selected_condition_blocks(blocks):
    selected = []
    for block in blocks or []:
        if block.get('selected', True) is False:
            continue
        condition = block.get('condition') or block.get('original_condition')
        if not condition:
            continue
        selected.append(block)
    return selected


def load_bt_stages(path):
    with open(path, 'r') as f:
        data = json.load(f)

    source_name = 'review'
    stages = (data.get('review') or {}).get('stages') or []
    if not stages:
        source_name = 'generated'
        stages = (data.get('generated') or {}).get('stages') or []
    if not stages:
        raise ValueError(f'No review.stages or generated.stages found in {path}')

    normalized = []
    for index, stage in enumerate(stages):
        normalized.append({
            'stage': int(stage.get('stage', index)),
            'name': stage.get('name') or f'Stage {stage.get("stage", index)}',
            'summary': stage.get('summary', ''),
            'preconditions': selected_condition_blocks(stage.get('preconditions')),
            'postconditions': selected_condition_blocks(stage.get('postconditions')),
            'hold_conditions': selected_condition_blocks(stage.get('hold_conditions')),
        })
    return data, source_name, normalized


def stage_for_waypoint(stages, waypoint_index):
    for stage in stages:
        if stage['stage'] == waypoint_index:
            return stage
    if waypoint_index < len(stages):
        return stages[waypoint_index]
    return {
        'stage': waypoint_index,
        'name': f'Waypoint {waypoint_index}',
        'summary': '',
        'preconditions': [],
        'postconditions': [],
    }


def condition_text(block):
    text = block.get('condition') or block.get('original_condition') or '(empty condition)'
    failures = [
        link.get('failure')
        for link in block.get('failure_links', [])
        if link.get('failure')
    ]
    if failures:
        return f"{text}  [{', '.join(failures)}]"
    return text


def print_condition_section(title, blocks):
    print(title)
    if not blocks:
        print("  - (none)")
        return
    for block in blocks:
        print(f"  - {condition_text(block)}")


def condition_strings(blocks):
    return [
        block.get('condition') or block.get('original_condition') or ''
        for block in blocks or []
        if block.get('condition') or block.get('original_condition')
    ]


def condition_dedupe_key(condition):
    key = re.sub(r'\s+', '', condition.strip().lower())
    key = key.replace('==true', '=true').replace('==false', '=false')
    return key


def dedupe_labeled_condition_groups(groups):
    labeled_conditions = []
    skipped_conditions = []
    seen = set()
    for label, blocks in groups:
        for condition in condition_strings(blocks):
            key = condition_dedupe_key(condition)
            labeled = f"{label}: {condition}"
            if key in seen:
                skipped_conditions.append(labeled)
                continue
            seen.add(key)
            labeled_conditions.append(labeled)
    return labeled_conditions, skipped_conditions


def print_preconditions(stage, waypoint_index):
    print("\n" + "-" * 70)
    print(f"PRE-CHECK before waypoint {waypoint_index}: {stage['name']}")
    print_condition_section("Preconditions:", stage['preconditions'])
    print("-" * 70)


def print_postconditions(current_stage, waypoint_index, final_waypoint_index):
    print("\n" + "-" * 70)
    print(f"POST-CHECK after waypoint {waypoint_index}: {current_stage['name']}")
    print_condition_section("Postconditions:", current_stage['postconditions'])
    if waypoint_index >= final_waypoint_index:
        print()
        print("No next waypoint preconditions; this was the final waypoint.")
    print("-" * 70)


def hold_block_detector(block):
    return str(block.get('detector') or block.get('failure') or '').strip().lower()


def print_hold_conditions(stage, waypoint_index):
    """Announce the hold conditions the detectors monitor live during a waypoint."""
    blocks = stage.get('hold_conditions') or []
    if not blocks:
        return
    print("\n" + "-" * 70)
    print(f"HOLD CHECK during waypoint {waypoint_index}: {stage['name']}  "
          f"(monitored in real time by detectors)")
    print("Hold conditions:")
    for block in blocks:
        det = hold_block_detector(block)
        print(f"  - {condition_text(block)}  [{det} detector]")
    print("-" * 70)


def print_hold_results(stage, waypoint_index, monitor):
    """Report each hold condition's live result for the waypoint just finished."""
    blocks = stage.get('hold_conditions') or []
    if not blocks:
        return
    fired = monitor.fired_at(waypoint_index) if monitor is not None else set()
    active = monitor.active_at(waypoint_index) if monitor is not None else set()
    any_fired = bool(fired)
    print("\n" + "-" * 70)
    unevaluated = monitor.unevaluated_at(waypoint_index) if monitor is not None else set()
    verdict = "FAILED" if any_fired else ("NOT EVALUATED" if unevaluated else "held")
    print(f"HOLD RESULT for waypoint {waypoint_index}: {stage['name']}  [{verdict}]")
    for block in blocks:
        cond = condition_text(block)
        det = hold_block_detector(block)
        if monitor is None or det not in active:
            status = "not monitored"
        elif det in monitor.unevaluated_at(waypoint_index):
            status = "not evaluated — no valid target measurements"
        elif det in fired:
            status = f"VIOLATED — {det} detected"
        else:
            status = "OK"
        print(f"  - {cond:<34} {status}")
    print("-" * 70)


def waypoint_extension(wp):
    # Prefer the RLBench Waypoint's tracked extension (get_ext): failures such as
    # GraspFailure clear it via clear_ext() to drop the gripper command. Fall back
    # to the raw sim string only if get_ext is unavailable.
    try:
        return wp.get_ext()
    except Exception:
        pass
    try:
        return wp._waypoint.get_extension_string()
    except Exception:
        return ""


def print_waypoint_header(wp, waypoint_index, final_waypoint_index, stage, yaml_descs):
    # No waypoint pose here: this was the last place that read a waypoint dummy
    # from the sim, and it was only ever printed. Reference poses come from the
    # chain spec; the composed values are logged once by the [wp-chain] banner.
    ext = waypoint_extension(wp)
    default_label = yaml_descs.get(waypoint_index, "")

    print("\n" + "=" * 70)
    print(f"Waypoint {waypoint_index} of {final_waypoint_index}")
    print(f"  Action:      {ext if ext else '(just move)'}")
    print(f"  Description: {default_label or stage['summary'] or '(none)'}")
    print("=" * 70)
    return ext


def capture_checkpoint_observation(scene, side_camera=None):
    obs = scene.get_observation()
    if side_camera is not None:
        try:
            setattr(obs, 'side_rgb', capture_camera_rgb(side_camera))
        except Exception:
            pass
    return obs


def load_vlm_helpers():
    import importlib.util

    helper_path = (paths.SOURCE_DIR / 'detectors/collision/vlm_confirm.py')
    spec = importlib.util.spec_from_file_location('aha_collision_vlm_confirm', helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shell_assignment_value(line, name):
    stripped = line.strip()
    if not stripped or stripped.startswith('#'):
        return None
    if stripped.startswith('export '):
        stripped = stripped[len('export '):].strip()
    prefix = f'{name}='
    if not stripped.startswith(prefix):
        return None

    raw_value = stripped[len(prefix):].strip()
    if not raw_value:
        return None
    if raw_value[0] in ('"', "'"):
        quote = raw_value[0]
        end = raw_value.find(quote, 1)
        if end > 0:
            return raw_value[1:end]
    return raw_value.split('#', 1)[0].strip()


def load_openai_credential_from_local_files():
    """Use explicitly configured credentials."""
    return os.environ.get("OPENAI_API_KEY")


def make_openai_client():
    try:
        import openai
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing Python package 'openai'. Install it or run in an environment "
            "where the OpenAI SDK is available."
        ) from exc

    api_key = load_openai_credential_from_local_files()
    if not api_key:
        raise RuntimeError(
            "Missing OpenAI credentials. Set OPENAI_API_KEY in the shell that "
            "launches this script."
        )
    return openai.OpenAI(api_key=api_key)


def model_supports_reasoning(model):
    """True for OpenAI models that accept a `reasoning` effort parameter."""
    name = (model or '').lower()
    return name.startswith(('gpt-5', 'o1', 'o3', 'o4'))


def _flatten_condition_text(data):
    """Long text fields may be written as a JSON array of lines so the file stays
    readable; join each array back into a single space-separated string so the
    model sees exactly the same text it would from a plain string value."""
    def join(value):
        return ' '.join(str(part) for part in value) if isinstance(value, list) else value

    if not isinstance(data, dict):
        return data
    for key in ('general_guidance', 'fallback_definition'):
        if key in data:
            data[key] = join(data[key])
    for entry in (data.get('predicates') or {}).values():
        if isinstance(entry, dict) and 'definition' in entry:
            entry['definition'] = join(entry['definition'])
    return data


def load_condition_definitions(path):
    """Load the per-predicate condition definitions JSON (single source of truth).

    Long text fields (general_guidance, fallback_definition, and each predicate's
    definition) may be written either as a plain string or as a JSON array of
    lines; array values are joined with spaces at load time, so the file can be
    kept readable without changing what the model is shown."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return _flatten_condition_text(json.load(f))
    except FileNotFoundError:
        print(f"WARNING: condition definitions file not found: {path}")
        return {}
    except json.JSONDecodeError as exc:
        print(f"WARNING: could not parse condition definitions {path}: {exc}")
        return {}


_CONDITION_LABEL_RE = re.compile(r'^\s*(?:pre|post)\b[^:]*:\s*', re.IGNORECASE)


def strip_condition_label(condition):
    """Remove a boundary label prefix like 'post waypoint 0: ' from a condition."""
    return _CONDITION_LABEL_RE.sub('', condition or '')


def condition_predicate_name(condition):
    """Leading identifier of a condition, e.g. object_in_gripper(x) == True -> object_in_gripper."""
    match = re.match(r'\s*([A-Za-z_][A-Za-z0-9_]*)', strip_condition_label(condition))
    return match.group(1) if match else ''


def filter_conditions_for_vlm(conditions, allow=None, deny=None):
    """Filter a list of condition strings by predicate name (lowercased).

    ``allow`` (if truthy) keeps only conditions whose predicate is in it.
    ``deny`` (if truthy) drops conditions whose predicate is in it — used to move
    a predicate (e.g. gripper_oriented_for) out of the pre/post checkpoints so it
    is instead verified at waypoint arrival. Conditions removed here are skipped
    by the VLM and thus treated as satisfied by the BT at this checkpoint."""
    result = list(conditions or [])
    if allow:
        result = [c for c in result if condition_predicate_name(c).lower() in allow]
    if deny:
        result = [c for c in result if condition_predicate_name(c).lower() not in deny]
    return result


def condition_predicate_args(condition):
    """Arguments inside the first parentheses, e.g. on(sponge, desk) -> ['sponge', 'desk']."""
    match = re.search(r'\(([^)]*)\)', strip_condition_label(condition))
    if not match or not match.group(1).strip():
        return []
    return [arg.strip() for arg in match.group(1).split(',') if arg.strip()]


_CONDITION_VALUE_RE = re.compile(r'^\s*[A-Za-z_][A-Za-z0-9_]*\s*={1,2}\s*([A-Za-z_][A-Za-z0-9_]*)\s*$')


def condition_value_name(condition):
    """Right-hand value of a `predicate = Value` condition, e.g. 'gripper_condition = Open' -> 'Open'.

    Returns '' for the usual `predicate(args) == True` form, which has parentheses
    and is keyed on the predicate name alone. Used so that a state-style predicate
    can carry a separate definition per value (gripper_condition = Open vs = Closed)."""
    match = _CONDITION_VALUE_RE.match(strip_condition_label(condition))
    return match.group(1) if match else ''


def get_sequence_predicates(definitions):
    """Predicate names that should be judged from a motion sequence.

    Opt-in only: empty/missing `sequence_predicates` means every condition is
    judged from the single checkpoint frame, like all the other predicates.
    """
    values = (definitions or {}).get('sequence_predicates')
    if not values:
        return set()
    return {str(value).strip().lower() for value in values}


def conditions_with_sequence_predicate(conditions, sequence_predicates):
    """Subset of conditions whose predicate needs a temporal motion sequence."""
    wanted = {str(p).lower() for p in (sequence_predicates or set())}
    return [
        condition
        for condition in conditions or []
        if condition_predicate_name(condition).lower() in wanted
    ]


def slim_frame_from_obs(obs):
    """A lightweight frame holding only the cameras the VLM uses (bounds memory).

    Arrays are copied because PyRep/RLBench can reuse the underlying buffers on
    the next sim step, which would otherwise blank out previously stored frames.
    """
    import numpy as np

    frame = types.SimpleNamespace()
    for name in VLM_CAMERA_NAMES:
        value = getattr(obs, name, None)
        if value is not None:
            value = np.array(value, copy=True)
        setattr(frame, name, value)
    return frame


def frame_has_any_image(frame):
    import numpy as np

    for name in VLM_CAMERA_NAMES:
        arr = getattr(frame, name, None)
        if arr is not None and np.asarray(arr).size > 0:
            return True
    return False


def capture_motion_frame(scene, side_camera):
    return slim_frame_from_obs(capture_checkpoint_observation(scene, side_camera))


def append_motion_frame(buffer, scene, side_camera, cap=120):
    """Append a slim frame to the motion buffer, decimating to stay under `cap`."""
    if buffer is None:
        return
    try:
        frame = capture_motion_frame(scene, side_camera)
    except Exception:
        return
    if not frame_has_any_image(frame):
        return
    buffer.append(frame)
    if len(buffer) > cap:
        buffer[:] = buffer[::2]


def sample_motion_frames(frames, count):
    """Pick `count` evenly spaced frames (including first and last) with time labels."""
    if not frames:
        return [], []
    if len(frames) <= count:
        indices = list(range(len(frames)))
    else:
        indices = sorted({
            round(k * (len(frames) - 1) / (count - 1))
            for k in range(count)
        })
    last = max(len(frames) - 1, 1)
    sampled = [frames[idx] for idx in indices]
    labels = [
        f"timepoint {position + 1}/{len(indices)} (~{int(round(100 * idx / last))}% through the move)"
        for position, idx in enumerate(indices)
    ]
    return sampled, labels


def build_motion_sequence(conditions, motion_buffer, sequence_predicates, checkpoint_obs, args):
    """If a checked condition needs a sequence and frames were recorded, sample them.

    The current checkpoint observation is appended as the final frame so the
    sequence ends at the present state. Returns (frames, labels) or (None, None).
    """
    if not motion_buffer:
        return None, None
    seq_conditions = conditions_with_sequence_predicate(conditions, sequence_predicates)
    if not seq_conditions:
        return None, None

    frames = list(motion_buffer)
    if checkpoint_obs is not None:
        frames.append(slim_frame_from_obs(checkpoint_obs))
    sampled, labels = sample_motion_frames(frames, max(2, args.motion_frames))
    print(
        f"  [vlm] using {len(sampled)}-frame motion sequence for in/sequence condition(s): "
        + ", ".join(seq_conditions)
    )
    return sampled, labels


def render_definition(template, args):
    """Substitute the condition's object args into a definition template's placeholders."""
    text = template or ''
    if '<args>' in text:
        text = text.replace('<args>', ', '.join(args) if args else 'the listed objects')
    distinct = []
    for token in re.findall(r'<[^>]+>', text):
        if token not in distinct:
            distinct.append(token)
    for token, value in zip(distinct, args):
        text = text.replace(token, value)
    return text


def lookup_predicate_definition(definitions, predicate):
    predicates = (definitions or {}).get('predicates', {})
    if predicate in predicates:
        return predicates[predicate]
    for key, value in predicates.items():
        if key.lower() == predicate.lower():
            return value
    return None


def build_condition_definitions_block(conditions, definitions):
    """One rendered definition per distinct condition being checked."""
    if not definitions:
        return ''
    lines = []
    seen = set()
    for condition in conditions or []:
        if not condition or condition in seen:
            continue
        seen.add(condition)
        predicate = condition_predicate_name(condition)
        # State-style predicates may define one entry per value, keyed
        # 'gripper_condition = Open' / 'gripper_condition = Closed', so each check
        # only sees the rules for the state it is actually asserting. Fall back to
        # the bare predicate name when no value-specific entry exists.
        value = condition_value_name(condition)
        entry = None
        if value:
            entry = lookup_predicate_definition(definitions, f'{predicate} = {value}')
        if entry is None:
            entry = lookup_predicate_definition(definitions, predicate)
        args = condition_predicate_args(condition)
        if entry:
            definition = render_definition(entry.get('definition', ''), args)
        else:
            definition = (
                definitions.get('fallback_definition')
                or 'Interpret this predicate literally from the camera images.'
            )
        # For door-like objects the robot physically interacts via the handle or
        # edge — holding the handle of a door still satisfies object_in_gripper
        # for that door.
        if (predicate == 'object_in_gripper'
                and any('door' in arg.lower() for arg in args)):
            definition += (
                " NOTE: this object is a door or door panel. The robot grasps "
                "doors by their handle or edge — if the gripper is visibly "
                "gripping the handle or edge of this door, treat "
                "object_in_gripper as True for the door itself."
            )
        lines.append(f"- {condition}\n    {definition}")
    if not lines:
        return ''
    return "# Condition definitions (how to judge each predicate)\n" + "\n".join(lines)


def guidance_section_keys_for_conditions(conditions, definitions):
    """Distinct predicate-specific guidance section names needed by the
    conditions being checked, in first-seen order.

    The predicate -> guidance-section mapping lives in the definitions file
    under 'guidance_map' (e.g. object_in_gripper -> gripper_and_grasp_guidance);
    predicates absent from the map rely on general_guidance alone."""
    guidance_map = {
        str(key).lower(): value
        for key, value in ((definitions or {}).get('guidance_map') or {}).items()
    }
    keys = []
    for condition in conditions or []:
        predicate = condition_predicate_name(condition).lower()
        section = guidance_map.get(predicate)
        if section and section not in keys:
            keys.append(section)
    return keys


def build_guidance_sections_block(conditions, definitions):
    """Render the predicate-specific guidance sections (gripper/grasp,
    alignment/orientation, spatial) that apply to the conditions in this
    checkpoint, each included once. A section may be written as a JSON array of
    lines; it is joined the same way general_guidance is."""
    sections = []
    for key in guidance_section_keys_for_conditions(conditions, definitions):
        section = (definitions or {}).get(key)
        if isinstance(section, list):
            section = ' '.join(str(part) for part in section)
        if section:
            sections.append(section)
    return "\n\n".join(sections)


def vlm_prompt_text(task_name, waypoint_index, stage, checkpoint_kind, conditions, definitions=None, sequence_timepoints=None, no_pass_explanation=False):
    # The stage name/summary describe the commanded action for this waypoint,
    # which is a strong prior the VLM should NOT use (it must judge from the
    # images only). They are deliberately omitted from the prompt rather than
    # fed-then-forbidden. Only the conditions to evaluate are provided.
    parts = [
        "# BT condition checkpoint",
        "conditions_to_check:\n" + json.dumps(conditions, indent=2),
    ]

    general_guidance = (definitions or {}).get('general_guidance')
    if general_guidance:
        parts.append(general_guidance)

    guidance_sections = build_guidance_sections_block(conditions, definitions)
    if guidance_sections:
        parts.append(guidance_sections)

    definitions_block = build_condition_definitions_block(conditions, definitions)
    if definitions_block:
        parts.append(definitions_block)

    if sequence_timepoints:
        parts.append(
            f"You are given camera images at {sequence_timepoints} timepoints spanning the "
            "robot's motion from the previous waypoint to this checkpoint, in time order "
            "(earliest first; the last timepoint is the current checkpoint state). Use the "
            "whole sequence as evidence. A condition counts as satisfied if it is met at the "
            "moment the condition's definition intends, which is not always the final frame: "
            "for transient or pass-through events (for example an object dropping through a "
            "ring, hoop, or basket), treat the condition as satisfied if the event clearly "
            "happens at any timepoint, even if a later frame shows the object having passed "
            "below or out of the target. For conditions that describe a final resting state "
            "(such as on(...) or a closed container), judge from the latest timepoints. Always "
            "follow the per-condition definitions above to decide which applies."
        )

    parts.append(
        "For each condition, decide whether it is visually satisfied at this exact "
        "checkpoint, using the condition definitions above."
    )
    if no_pass_explanation:
        explanation_spec = (
            "`explanation`. Do NOT write any explanation or evidence when the "
            "checkpoint passes: if every condition is `satisfied`, return "
            "`explanation` as an empty string and set each condition's `evidence` "
            "to an empty string. Only when a condition is `not_satisfied` or "
            "`uncertain`, fill in that condition's `evidence` and give a one-sentence "
            "`explanation` describing the problem."
        )
    else:
        explanation_spec = (
            "`explanation` (one concise sentence)."
        )
    parts.append(
        "Return JSON with exactly these keys: "
        "`checkpoint_passed` (true if every condition is satisfied; false only if "
        "at least one condition is clearly not_satisfied; null if no condition is "
        "violated but one or more are uncertain), "
        "`conditions` (array of objects with `condition`, `status` one of "
        "`satisfied`, `not_satisfied`, `uncertain`, and `evidence`), and "
        + explanation_spec
    )
    return "\n\n".join(parts)


def extract_vlm_json(text):
    stripped = text.strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find('{')
        end = stripped.rfind('}')
        if start == -1 or end <= start:
            return {
                'checkpoint_passed': None,
                'conditions': [],
                'explanation': stripped[:300],
            }
        try:
            data = json.loads(stripped[start:end + 1])
        except json.JSONDecodeError:
            return {
                'checkpoint_passed': None,
                'conditions': [],
                'explanation': stripped[:300],
            }

    conditions = data.get('conditions') or []
    checkpoint_passed = data.get('checkpoint_passed')
    if conditions:
        statuses = [str(item.get('status', '')).lower() for item in conditions]
        violated = {'not_satisfied', 'unsatisfied', 'violated', 'failed', 'fail', 'false'}
        if any(s in violated for s in statuses):
            # At least one condition is clearly violated -> hard failure.
            checkpoint_passed = False
        elif all(s == 'satisfied' for s in statuses):
            checkpoint_passed = True
        else:
            # No violation, but one or more conditions are uncertain/unknown. Keep
            # the result three-valued (None) so verdict_to_status treats it as
            # uncertain (pass-with-warning unless --strict) instead of a false
            # failure. This is what stops "uncertain" from reading as a violation.
            checkpoint_passed = None

    return {
        'checkpoint_passed': checkpoint_passed,
        'conditions': conditions,
        'explanation': str(data.get('explanation', '')).strip(),
    }


def dump_vlm_frames(task_name, waypoint_index, checkpoint_kind, conditions, frames):
    """Write the exact images sent to the VLM to disk for offline prompt tuning.

    `frames` is a list of (label, camera_name, data_url) tuples. Only active when
    AHA_SAVE_VLM_FRAMES names a directory. Also drops a small JSON sidecar listing
    the conditions checked so the offline harness knows what to re-judge.
    """
    if not SAVE_VLM_FRAMES_DIR:
        return
    import base64
    out_dir = Path(SAVE_VLM_FRAMES_DIR) / task_name
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for label, camera_name, data_url in frames:
        try:
            b64 = data_url.split(',', 1)[1] if ',' in data_url else data_url
            raw = base64.b64decode(b64)
        except Exception:
            continue
        tag = (label or 'single').replace('/', '-').replace(' ', '')
        fname = f"{task_name}_wp{waypoint_index}_{checkpoint_kind}_{tag}_{camera_name}.png"
        (out_dir / fname).write_bytes(raw)
        saved.append(fname)
    sidecar = out_dir / f"{task_name}_wp{waypoint_index}_{checkpoint_kind}_conditions.json"
    sidecar.write_text(json.dumps({
        'task_name': task_name,
        'waypoint_index': waypoint_index,
        'checkpoint_kind': checkpoint_kind,
        'conditions': conditions,
        'images': saved,
    }, indent=2))


def capture_condition_frames(scene, side_camera, env_wrapper, task_name,
                             waypoint_index, checkpoint_kind, conditions):
    """Save a checkpoint's camera images to the vlm_images/<task>/ folder WITHOUT
    calling the VLM.

    This is the photo-only path: it grabs exactly the frames a VLM condition check
    *would* have sent (same cameras, same capture), so every pre/arrival/post
    condition gets a picture from the first waypoint to the last -- even the ones
    whose VLM check is skipped (e.g. under --limited before the injection
    waypoint). It makes no API call and touches no live detector, so VLM checking
    and the live detectors keep firing exactly as they do now. No-op unless
    SAVE_VLM_FRAMES_DIR is set (i.e. a --vlm-run-log/--vlm-run-log-dir is active)."""
    if not SAVE_VLM_FRAMES_DIR:
        return
    try:
        helpers = load_vlm_helpers()
        obs = capture_checkpoint_observation(scene, side_camera)
        frames = [
            (None, camera_name, data_url)
            for camera_name, data_url in helpers.collect_camera_images(
                obs, camera_names=VLM_CAMERA_NAMES, env_wrapper=env_wrapper)
        ]
        if frames:
            dump_vlm_frames(task_name, waypoint_index, checkpoint_kind,
                            conditions, frames)
            print(f"  [frames] saved {checkpoint_kind}-condition photo for "
                  f"waypoint {waypoint_index} (no VLM call)")
    except Exception as exc:
        print(f"  [frames] {checkpoint_kind}-condition photo capture failed: {exc}")


def write_vlm_trace(task_name, waypoint_index, checkpoint_kind, model, prompt_text, image_manifest, response_text, parsed):
    VLM_TRACE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    path = VLM_TRACE_DIR / f'bt_conditions_vlm_{task_name}_wp{waypoint_index}_{checkpoint_kind}_{timestamp}.json'
    payload = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'task_name': task_name,
        'waypoint_index': waypoint_index,
        'checkpoint_kind': checkpoint_kind,
        'model': model,
        'system_prompt': VLM_SYSTEM_PROMPT,
        'user_prompt_text': prompt_text,
        'image_manifest': image_manifest,
        'raw_response_text': response_text,
        'parsed_result': parsed,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def evaluate_sim_backed_conditions(scene, conditions):
    """All conditions are forwarded to the VLM.

    object_in_gripper is judged from the camera images only; no proprioceptive
    grasp-sensor signal is injected into the prompt.
    """
    return [], list(conditions or [])


def sim_condition_result(sim_items):
    if not sim_items:
        return None
    if any(item['status'] == 'not_satisfied' for item in sim_items):
        passed = False
    elif any(item['status'] == 'unknown' for item in sim_items):
        passed = None
    else:
        passed = True
    return {
        'checkpoint_passed': passed,
        'conditions': sim_items,
        'explanation': 'Simulator-backed condition check.',
        'trace_path': None,
    }


def merge_condition_results(sim_items, vlm_result):
    if not sim_items:
        return vlm_result
    if vlm_result is None:
        return sim_condition_result(sim_items)
    return {
        **vlm_result,
        'conditions': sim_items + list(vlm_result.get('conditions', [])),
    }


# ---------------------------------------------------------------------------
# Post-failure failure-type diagnosis
#
# When a run ends in a failure, the type of failure is classified
# DETERMINISTICALLY (no LLM call), twice: option 1 has the live-detector fires +
# grasp force sensor as hard evidence; option 2 is blind and reasons only from
# the VLM condition-check transcript. Both apply the same ordered decision
# procedure the earlier LLM prompt described, as plain code over the structured
# evidence: live-detector fires map straight to a type; no_grasp/slip are decided
# only when a pick was expected (an object_in_gripper condition was checked),
# using the grasp force sensor + that condition's history; and the BT condition
# predicates that ended
# not_satisfied (selected_object -> wrong_object, gripper_oriented_for ->
# orientation, aligned_with/on/inside/next_to -> translation, gripper_condition
# -> wrong_sequence) supply the rest. Freezing is only knowable from the freezing
# detector, so it is an allowed answer for option 1 only. See
# run_failure_diagnosis / _diagnose_deterministic below.
# ---------------------------------------------------------------------------

# Canonical failure-type labels the diagnosis must choose from (option 1 set;
# option 2 drops 'freezing').
DIAGNOSIS_FAILURE_TYPES = [
    'collision', 'translation', 'orientation', 'wrong_sequence',
    'wrong_object', 'no_grasp', 'slip', 'freezing',
]
# Grip force above this (from the slip detector's max-grip telemetry) means a
# grasp actually registered on the force sensor. Mirrors SLIP_HOLD_FLOOR (0.1).
DIAGNOSIS_GRASP_FORCE_THRESHOLD = float(os.getenv('AHA_DIAGNOSIS_GRASP_FORCE', '0.1'))


def _diagnosis_detector_evidence(ctx):
    """Option-1 hard evidence: which live detectors fired (and at which
    waypoints), whether a grasp ever registered on the force sensor, and what the
    gripper is holding at the moment the run stopped."""
    fired = {}
    max_grip = None
    monitor = getattr(ctx, 'monitor', None)
    detectors = getattr(monitor, 'detectors', {}) if monitor is not None else {}
    for name, det in detectors.items():
        waypoints = sorted(getattr(det, 'fired_waypoints', None) or set())
        if waypoints:
            fired[name] = waypoints
    slip_det = detectors.get('slip')
    if slip_det is not None:
        try:
            max_grip = float(getattr(slip_det, 'diag', {}).get('max_grip'))
        except (TypeError, ValueError):
            max_grip = None
    held_now = []
    try:
        for obj in ctx.robot.gripper.get_grasped_objects():
            held_now.append(obj.get_name())
    except Exception:
        held_now = []
    grasp_ever = (None if max_grip is None
                  else bool(max_grip > DIAGNOSIS_GRASP_FORCE_THRESHOLD))
    return {
        'detectors_fired': fired,
        'max_grip_force': max_grip,
        'grasp_force_threshold': DIAGNOSIS_GRASP_FORCE_THRESHOLD,
        'grasp_ever_registered': grasp_ever,
        'currently_holding': held_now,
    }


# Condition statuses (VLM verdict strings) that count as a violated condition.
_VIOLATED_STATUSES = frozenset(
    {'not_satisfied', 'unsatisfied', 'violated', 'failed', 'fail', 'false'})

# Live-detector name -> diagnosis failure type (option-1 step 1). Mirrors the
# mapping the old prompt spelled out: the transition detector owns 'translation'.
_DETECTOR_TYPE = {
    'collision': 'collision',
    'freezing': 'freezing',
    'slip': 'slip',
    'orientation': 'orientation',
    'transition': 'translation',
}

# Standalone spatial/effect predicates that evidence a wrong POSITION when they
# end not_satisfied. `\b...\(` so we match the predicate head, not a substring
# (e.g. `open(` but not the word "open" inside prose).
_TRANSLATION_PRED_RE = re.compile(
    r'\b(?:end_effector_aligned_with|aligned_with|inside|on|next_to|closed|open|'
    r'object_for_press|end_effector_in_contact_with|in_contact_with)\s*\(')


def _condition_failure_type(text):
    """Map a BT condition predicate string to the diagnosis failure type it
    evidences when not_satisfied — mirrors the predicate->failure links baked
    into every BT (see ALLOWED_CONDITION_PATTERNS). Returns one of
    'wrong_object', 'orientation', 'grasp', 'sequence', 'translation', or None."""
    t = (text or '').lower()
    if 'selected_object' in t or 'object_found' in t:
        return 'wrong_object'
    if 'oriented_for' in t:                       # gripper_oriented_for(...)
        return 'orientation'
    if 'object_in_gripper' in t or 'gripper_released' in t:
        return 'grasp'                            # grasp lifecycle -> slip/no_grasp
    if 'gripper_condition' in t:                  # open/close ordering
        return 'sequence'
    if _TRANSLATION_PRED_RE.search(t):
        return 'translation'
    return None


def _grasp_lifecycle(check_history):
    """Track object_in_gripper across the run. Returns (present, ever_ok, lost,
    never_ok): whether a grasp condition was ever checked, ever satisfied, was
    satisfied then later became not_satisfied, and was checked but never
    satisfied."""
    present = ever_ok = lost = False
    for entry in check_history or []:
        for cond in entry.get('conditions', []) or []:
            text = (cond.get('condition') or '').lower()
            if 'object_in_gripper' not in text:
                continue
            present = True
            status = (cond.get('status') or '').lower()
            if status == 'satisfied':
                ever_ok = True
            elif status in _VIOLATED_STATUSES and ever_ok:
                lost = True
    return present, ever_ok, lost, (present and not ever_ok)


def _failed_condition_types(check_history):
    """(waypoint, type) for every not_satisfied condition, in transcript order.
    `type` per _condition_failure_type (None-typed conditions are dropped)."""
    out = []
    for entry in check_history or []:
        wp = entry.get('waypoint')
        for cond in entry.get('conditions', []) or []:
            if (cond.get('status') or '').lower() not in _VIOLATED_STATUSES:
                continue
            ctype = _condition_failure_type(cond.get('condition') or '')
            if ctype:
                out.append((wp, ctype))
    return out


def _diagnose_deterministic(check_history, allowed_types, evidence, stop_waypoint):
    """Deterministically classify the run-ending failure — no LLM call.

    ``evidence`` is the option-1 detector/grasp-sensor dict (see
    _diagnosis_detector_evidence); pass None for the blind option-2 pass. Applies
    the same ordered decision procedure the old LLM prompt described. Returns
    {'failure_type', 'reasoning'}."""
    def result(ftype, reasoning):
        return {'failure_type': ftype if ftype in allowed_types else 'unknown',
                'reasoning': reasoning}

    present, _grasp_ok, grasp_lost, grasp_never = _grasp_lifecycle(check_history)
    failed = _failed_condition_types(check_history)
    wrong_object_seen = any(ct == 'wrong_object' for _, ct in failed)
    sequence_seen = any(ct == 'sequence' for _, ct in failed)

    def nearest_spatial():
        """translation/orientation category of the not_satisfied condition
        nearest the waypoint where the run stopped."""
        best = None
        for wp, ctype in failed:
            if ctype not in ('translation', 'orientation'):
                continue
            key = wp if isinstance(wp, int) else -1
            if best is None or key >= best[0]:
                best = (key, ctype)
        return best[1] if best else None

    if evidence is not None:
        # --- Option 1: detector-aware. Live-detector response is authoritative. ---
        fired = evidence.get('detectors_fired') or {}
        if fired:
            stop = stop_waypoint if isinstance(stop_waypoint, int) else 0
            best_name, best_dist = None, None
            for name, wps in fired.items():
                for wp in (wps or [stop]):
                    dist = abs((wp if isinstance(wp, int) else stop) - stop)
                    if best_dist is None or dist < best_dist:
                        best_dist, best_name = dist, name
            return result(
                _DETECTOR_TYPE.get(best_name, best_name),
                f"live '{best_name}' detector fired at waypoint(s) "
                f"{sorted(fired.get(best_name) or [])}; detector response is authoritative.")
        # No detector fired: reason from conditions + grasp force sensor.
        grasp_reg = evidence.get('grasp_ever_registered')
        holding = evidence.get('currently_holding') or []
        # no_grasp only applies when the robot was actually supposed to pick an
        # object up: require an object_in_gripper condition to have been checked
        # (present) and the grasp to have never succeeded. Never infer no_grasp
        # from the force sensor alone on a task that has no pick step.
        if grasp_never:
            return result('no_grasp',
                          "no detector fired and object_in_gripper was never satisfied; "
                          "the grasp never succeeded.")
        if present and grasp_reg is False:
            # Hard evidence wins here: the grasp force sensor never registered a
            # hold, so the object was never actually gripped -> no_grasp, even if
            # the VLM's object_in_gripper briefly read satisfied then not
            # (grasp_lost). Slip is reserved for a grasp that physically happened.
            return result('no_grasp',
                          "no detector fired and the grasp force sensor never "
                          "registered a hold, so the object was never actually gripped.")
        if grasp_lost or (grasp_reg is True and not holding):
            return result('slip',
                          "no detector fired but a grasp registered on the force sensor "
                          "and the object was subsequently lost (currently holding nothing).")
        if sequence_seen:
            return result('wrong_sequence',
                          "a gripper open/close ordering condition ended not_satisfied.")
        if wrong_object_seen:
            return result('wrong_object',
                          "a selected_object/object_found condition ended not_satisfied.")
        sp = nearest_spatial()
        if sp:
            return result(sp, f"nearest-to-stop failed condition evidences {sp}.")
        return result('unknown', "no detector fired and no decisive condition failed.")

    # --- Option 2: blind (no detector/sensor access). Transcript only. ---
    if grasp_lost:
        return result('slip',
                      "an object_in_gripper condition was satisfied earlier and later "
                      "became not_satisfied.")
    if grasp_never:
        return result('no_grasp',
                      "the object_in_gripper postcondition was never satisfied after the "
                      "grasp attempt.")
    if sequence_seen:
        return result('wrong_sequence',
                      "a gripper open/close ordering condition ended not_satisfied.")
    if wrong_object_seen:
        return result('wrong_object',
                      "a selected_object/object_found condition ended not_satisfied.")
    sp = nearest_spatial()
    if sp:
        return result(sp, f"nearest-to-stop failed condition evidences {sp}.")
    return result('unknown', "no decisive condition failed in the transcript.")


def run_failure_diagnosis(ctx):
    """Deterministically classify a run-ending failure into one of
    DIAGNOSIS_FAILURE_TYPES twice: option 1 with the live-detector fires + grasp
    force sensor as evidence, and option 2 blind from only the VLM condition-check
    transcript. No LLM call is made. Both predictions are printed (parsed by the
    batch harness from run.log) and logged to the per-run VLM CSV when active.
    Stored on ``ctx.diagnosis``."""
    if not getattr(ctx, 'vlm_enabled', False) or not getattr(ctx, 'failures', None):
        return None
    check_history = getattr(ctx, 'check_history', []) or []
    evidence = _diagnosis_detector_evidence(ctx)
    # The FIRST recorded failure is the one the diagnosis is about: it is where an
    # aborting run would have stopped. Identical to failures[-1] when the run does
    # abort (there is only one), but under --no-abort / --limited-v2 the run keeps
    # going and later checkpoints pile on failures caused by the first one.
    stop_waypoint = ctx.failures[0].waypoint if ctx.failures else None

    print("\n" + "=" * 70)
    print("FAILURE DIAGNOSIS (post-hoc deterministic classification)")
    print("=" * 70)
    opt1 = _diagnose_deterministic(
        check_history, DIAGNOSIS_FAILURE_TYPES, evidence, stop_waypoint)
    # Option 2 excludes freezing: freezing is only knowable from the detector.
    blind_types = [t for t in DIAGNOSIS_FAILURE_TYPES if t != 'freezing']
    opt2 = _diagnose_deterministic(
        check_history, blind_types, None, stop_waypoint)

    ctx.diagnosis = {'with_detectors': opt1, 'no_detectors': opt2, 'evidence': evidence}
    # These two lines are the machine-readable contract the batch harness parses.
    print(f"  [diagnosis] with_detectors: {opt1['failure_type']} | {opt1['reasoning']}")
    print(f"  [diagnosis] no_detectors: {opt2['failure_type']} | {opt2['reasoning']}")
    print("=" * 70)

    try:
        import aha_publish.running.vlm_run_logger as vlm_run_logger
        lg = vlm_run_logger.active()
        if lg is not None:
            fw = stop_waypoint
            lg.log(channel='diagnosis', event_kind='with_detectors', waypoint=fw,
                   verdict=opt1['failure_type'], explanation=opt1['reasoning'])
            lg.log(channel='diagnosis', event_kind='no_detectors', waypoint=fw,
                   verdict=opt2['failure_type'], explanation=opt2['reasoning'])
    except Exception:
        pass
    return ctx.diagnosis


def vlm_image_part(data_url):
    """One `input_image` content part, honouring $AHA_VLM_IMAGE_DETAIL."""
    part = {'type': 'input_image', 'image_url': data_url}
    if VLM_IMAGE_DETAIL in ('low', 'high', 'auto'):
        part['detail'] = VLM_IMAGE_DETAIL
    return part


def run_vlm_condition_check(
    obs,
    scene,
    env_wrapper,
    task_name,
    waypoint_index,
    stage,
    checkpoint_kind,
    conditions,
    model,
    preview_images=False,
    trace=False,
    definitions=None,
    obs_sequence=None,
    sequence_labels=None,
    reasoning_effort=None,
    no_pass_explanation=False,
):
    if not conditions:
        return None

    helpers = load_vlm_helpers()
    is_sequence = bool(obs_sequence)

    if preview_images:
        if is_sequence:
            helpers.show_camera_sequence(
                obs_sequence,
                camera_names=VLM_CAMERA_NAMES,
                env_wrapper=env_wrapper,
                title=f"{task_name} wp{waypoint_index}: motion frames sent to VLM",
            )
            # Give the (often large) grid time to paint, then block so it stays
            # rendered and you can inspect it before the blocking API call.
            try:
                import matplotlib.pyplot as plt
                plt.pause(0.5)
            except Exception:
                pass
            if sys.stdin.isatty():
                input("  Inspect the motion frames above, then press Enter to send them to the VLM...")
        else:
            helpers.show_camera_images(
                obs,
                camera_names=VLM_CAMERA_NAMES,
                env_wrapper=env_wrapper,
            )

    prompt_text = vlm_prompt_text(
        task_name=task_name,
        waypoint_index=waypoint_index,
        stage=stage,
        checkpoint_kind=checkpoint_kind,
        conditions=conditions,
        definitions=definitions,
        sequence_timepoints=len(obs_sequence) if is_sequence else None,
        no_pass_explanation=no_pass_explanation,
    )
    content = [{'type': 'input_text', 'text': prompt_text}]
    image_manifest = []
    saved_frames = []
    if is_sequence:
        for index, frame in enumerate(obs_sequence):
            label = (
                sequence_labels[index]
                if sequence_labels and index < len(sequence_labels)
                else f"timepoint {index + 1}/{len(obs_sequence)}"
            )
            content.append({'type': 'input_text', 'text': f'--- {label} ---'})
            for camera_name, data_url in helpers.collect_camera_images(
                frame,
                camera_names=VLM_CAMERA_NAMES,
                env_wrapper=env_wrapper,
            ):
                content.append({'type': 'input_text', 'text': f'Camera: {camera_name}'})
                content.append(vlm_image_part(data_url))
                image_manifest.append({'timepoint': label, 'camera_name': camera_name})
                saved_frames.append((label, camera_name, data_url))
    else:
        for camera_name, data_url in helpers.collect_camera_images(
            obs,
            camera_names=VLM_CAMERA_NAMES,
            env_wrapper=env_wrapper,
        ):
            content.append({'type': 'input_text', 'text': f'Camera: {camera_name}'})
            content.append(vlm_image_part(data_url))
            image_manifest.append({'camera_name': camera_name})
            saved_frames.append((None, camera_name, data_url))

    if not image_manifest:
        raise RuntimeError("No camera images were available for the VLM condition check.")

    if SAVE_VLM_FRAMES_DIR:
        try:
            dump_vlm_frames(task_name, waypoint_index, checkpoint_kind, conditions, saved_frames)
        except Exception as exc:
            print(f"  [vlm] frame dump failed: {exc}")

    if is_sequence:
        print(
            f"  [vlm] motion sequence sent: {len(obs_sequence)} timepoints x "
            f"{len(VLM_CAMERA_NAMES)} cameras ({len(image_manifest)} images)"
        )
    else:
        print(
            "  [vlm] images sent: "
            + ", ".join(item['camera_name'] for item in image_manifest)
        )
    client = make_openai_client()
    create_kwargs = dict(
        model=model,
        max_output_tokens=1000,
        input=[
            {'role': 'system', 'content': VLM_SYSTEM_PROMPT},
            {'role': 'user', 'content': content},
        ],
    )
    # Lower reasoning effort is a large latency win on reasoning models. Only
    # attach it for models that support it, and fall back gracefully if the API
    # still rejects the parameter for this model.
    if reasoning_effort and model_supports_reasoning(model):
        create_kwargs['reasoning'] = {'effort': reasoning_effort}
    try:
        response = client.responses.create(**create_kwargs)
    except Exception as exc:
        if 'reasoning' in create_kwargs and 'reasoning' in str(exc).lower():
            create_kwargs.pop('reasoning')
            response = client.responses.create(**create_kwargs)
        else:
            raise

    usage = getattr(response, 'usage', None)
    in_tok = int(getattr(usage, 'input_tokens', 0) or 0)
    out_tok = int(getattr(usage, 'output_tokens', 0) or 0)
    VLM_USAGE['calls'] += 1
    VLM_USAGE['input_tokens'] += in_tok
    VLM_USAGE['output_tokens'] += out_tok
    print(
        f"  [vlm] tokens: {in_tok} in + {out_tok} out "
        f"(run total: {VLM_USAGE['input_tokens']} in + {VLM_USAGE['output_tokens']} out)"
    )

    parsed = extract_vlm_json(response.output_text)
    trace_path = None
    if trace:
        trace_path = write_vlm_trace(
            task_name,
            waypoint_index,
            checkpoint_kind,
            model,
            prompt_text,
            image_manifest,
            response.output_text,
            parsed,
        )

    return {
        'checkpoint_passed': parsed['checkpoint_passed'],
        'conditions': parsed['conditions'],
        'explanation': parsed['explanation'],
        'trace_path': str(trace_path) if trace_path else None,
    }


def print_vlm_result(checkpoint_kind, result):
    if result is None:
        return
    passed = result.get('checkpoint_passed')
    status = 'PASS' if passed is True else 'FAIL' if passed is False else 'UNKNOWN'
    print(f"\n  VLM {checkpoint_kind.upper()} CHECK: {status}")
    for item in result.get('conditions', []):
        condition = item.get('condition', '(condition)')
        item_status = item.get('status', 'unknown')
        evidence = item.get('evidence', '')
        print(f"    - {item_status}: {condition}")
        if evidence:
            print(f"      {evidence}")
    if result.get('explanation'):
        print(f"    Explanation: {result['explanation']}")
    if result.get('trace_path'):
        print(f"    Trace: {result['trace_path']}")


_PREDICATE_DEFAULT = object()


def _log_condition_vlm_event(checkpoint_kind, waypoint_index, result):
    """Record one pre/post/arrival condition-check VLM event to the per-run CSV,
    if a RunLogger is active (no-op otherwise). The exact images sent are saved by
    dump_vlm_frames into the run's vlm_images/<task>/ folder (SAVE_VLM_FRAMES_DIR
    is pointed there when logging starts); reference their sidecar here."""
    try:
        import aha_publish.running.vlm_run_logger as vlm_run_logger
        lg = vlm_run_logger.active()
        if lg is None or result is None:
            return
        kind = str(checkpoint_kind or '')
        if kind in ('pre', 'post', 'arrival'):
            event_kind = kind
        elif kind.startswith('pre') and 'post' in kind:
            event_kind = 'prepost'
        else:
            event_kind = kind or 'condition'
        img = ''
        try:
            sidecar = (lg.images_dir / lg.task /
                       f"{lg.task}_wp{waypoint_index}_{checkpoint_kind}_conditions.json")
            if sidecar.exists():
                img = str(sidecar)
        except Exception:
            pass
        lg.log(channel='condition', event_kind=event_kind,
               waypoint=waypoint_index,
               verdict=result.get('checkpoint_passed'),
               explanation=str(result.get('explanation', '')),
               image_path=img)
    except Exception:
        pass


def maybe_run_vlm_condition_check(
    args,
    enabled,
    checkpoint_kind,
    scene,
    side_camera,
    env_wrapper,
    task_name,
    waypoint_index,
    stage,
    conditions,
    motion_buffer=None,
    sequence_predicates=None,
    predicate_allow=_PREDICATE_DEFAULT,
    predicate_deny=_PREDICATE_DEFAULT,
):
    if not enabled:
        return None
    allow = (
        getattr(args, 'vlm_predicate_filter', None)
        if predicate_allow is _PREDICATE_DEFAULT else predicate_allow
    )
    deny = (
        getattr(args, 'vlm_arrival_predicate_filter', None)
        if predicate_deny is _PREDICATE_DEFAULT else predicate_deny
    )
    conditions = filter_conditions_for_vlm(conditions, allow=allow, deny=deny)
    if not conditions:
        return None
    if args.vlm_checks == 'ask':
        label = 'preconditions' if checkpoint_kind == 'pre' else 'postconditions'
        if not ask_yes_no(
            f"\nCheck {label} for waypoint {waypoint_index} with VLM?",
            default=False,
        ):
            return None
    elif args.vlm_checks not in (checkpoint_kind, 'both'):
        return None
    if not conditions:
        return None

    print(f"\n  [vlm] checking {checkpoint_kind}conditions for waypoint {waypoint_index}...")
    sim_items, conditions = evaluate_sim_backed_conditions(scene, conditions)
    if sim_items:
        sim_result = sim_condition_result(sim_items)
        print_vlm_result(f"{checkpoint_kind} simulator", sim_result)
        if sim_result.get('checkpoint_passed') is not True:
            return sim_result
    if not conditions:
        return sim_condition_result(sim_items)
    try:
        obs = capture_checkpoint_observation(scene, side_camera)
        obs_sequence, sequence_labels = build_motion_sequence(
            conditions, motion_buffer, sequence_predicates, obs, args
        )
        result = run_vlm_condition_check(
            obs=obs,
            scene=scene,
            env_wrapper=env_wrapper,
            task_name=task_name,
            waypoint_index=waypoint_index,
            stage=stage,
            checkpoint_kind=checkpoint_kind,
            conditions=conditions,
            model=args.vlm_model,
            preview_images=args.vlm_preview,
            trace=args.vlm_trace,
            definitions=getattr(args, 'condition_definitions_data', None),
            obs_sequence=obs_sequence,
            sequence_labels=sequence_labels,
            reasoning_effort=getattr(args, 'vlm_reasoning_effort', 'low'),
            no_pass_explanation=getattr(args, 'vlm_no_pass_explanation', False),
        )
        result = merge_condition_results(sim_items, result)
        print_vlm_result(checkpoint_kind, result)
        _log_condition_vlm_event(checkpoint_kind, waypoint_index, result)
        return result
    except Exception as exc:
        print(f"  [vlm] condition check failed: {exc}")
        return None


def maybe_run_vlm_boundary_check(
    args,
    enabled,
    scene,
    side_camera,
    env_wrapper,
    task_name,
    waypoint_index,
    stage,
    next_waypoint_index,
    next_stage,
    motion_buffer=None,
    sequence_predicates=None,
):
    conditions, skipped_conditions = dedupe_labeled_condition_groups([
        (f"pre waypoint {next_waypoint_index}", next_stage['preconditions']),
        (f"post waypoint {waypoint_index}", stage['postconditions']),
    ])
    conditions = filter_conditions_for_vlm(
        conditions,
        allow=getattr(args, 'vlm_predicate_filter', None),
        deny=getattr(args, 'vlm_arrival_predicate_filter', None),
    )
    if not enabled or not conditions:
        return None

    checkpoint_kind = f"pre{next_waypoint_index}_post{waypoint_index}"
    if args.vlm_checks == 'ask':
        if not ask_yes_no(
            (
                f"\nCheck preconditions for waypoint {next_waypoint_index} and "
                f"postconditions for waypoint {waypoint_index} with VLM?"
            ),
            default=False,
        ):
            return None
    elif args.vlm_checks not in ('pre', 'post', 'both'):
        return None

    print(
        f"\n  [vlm] checking preconditions for waypoint {next_waypoint_index} "
        f"and postconditions for waypoint {waypoint_index}..."
    )
    if skipped_conditions:
        print(
            "  [vlm] skipped repeated condition(s): "
            + "; ".join(skipped_conditions)
        )
    sim_items, conditions = evaluate_sim_backed_conditions(scene, conditions)
    if sim_items:
        sim_result = sim_condition_result(sim_items)
        print_vlm_result("PRE+POST simulator", sim_result)
        if sim_result.get('checkpoint_passed') is not True:
            return sim_result
    if not conditions:
        return sim_condition_result(sim_items)
    combined_stage = {
        'name': (
            f"{next_stage['name']} / previous postcheck: {stage['name']}"
        ),
        'summary': (
            f"Pre waypoint {next_waypoint_index}: "
            f"{next_stage.get('summary', '') or 'not provided'}\n"
            f"Post waypoint {waypoint_index}: "
            f"{stage.get('summary', '') or 'not provided'}"
        ),
    }
    try:
        obs = capture_checkpoint_observation(scene, side_camera)
        obs_sequence, sequence_labels = build_motion_sequence(
            conditions, motion_buffer, sequence_predicates, obs, args
        )
        result = run_vlm_condition_check(
            obs=obs,
            scene=scene,
            env_wrapper=env_wrapper,
            task_name=task_name,
            waypoint_index=next_waypoint_index,
            stage=combined_stage,
            checkpoint_kind=checkpoint_kind,
            conditions=conditions,
            model=args.vlm_model,
            preview_images=args.vlm_preview,
            trace=args.vlm_trace,
            definitions=getattr(args, 'condition_definitions_data', None),
            obs_sequence=obs_sequence,
            sequence_labels=sequence_labels,
            reasoning_effort=getattr(args, 'vlm_reasoning_effort', 'low'),
            no_pass_explanation=getattr(args, 'vlm_no_pass_explanation', False),
        )
        result = merge_condition_results(sim_items, result)
        print_vlm_result("PRE+POST", result)
        _log_condition_vlm_event(checkpoint_kind, next_waypoint_index, result)
        return result
    except Exception as exc:
        print(f"  [vlm] boundary condition check failed: {exc}")
        return None


def step_sim_light(scene, camera_name):
    scene.pyrep.step()
    try:
        scene.task.step()
    except Exception:
        pass
    if camera_name == "default":
        return
    try:
        sync_default_camera(scene, camera_name)
    except Exception:
        pass


def actuate_gripper_with_camera(
    gripper,
    amount,
    scene,
    env_wrapper,
    camera_name,
    failure_active,
    side_camera_window=None,
    live_camera=None,
    max_steps=GRIPPER_ACTION_MAX_STEPS,
    label='gripper',
    motion_buffer=None,
    side_camera=None,
    motion_capture_interval=5,
    monitor=None,
    monitor_waypoint=None,
    frame_dump_dir=None,
    frame_dump_cams=None,
):
    done = False
    step_count = 0
    while not done and step_count < max_steps:
        done = gripper.actuate(amount, 0.04)
        grip_obs = None
        if failure_active:
            grip_obs = env_pyrep_step_with_camera(env_wrapper, scene, camera_name)
        else:
            step_sim_light(scene, camera_name)
            if monitor is not None:
                grip_obs = refresh_camera_sensors(scene)
        if monitor is not None and grip_obs is not None:
            attach_side_rgb(grip_obs, side_camera)
            monitor.on_step(grip_obs, monitor_waypoint, True, phase="gripper")
        if motion_buffer is not None and step_count % motion_capture_interval == 0:
            append_motion_frame(motion_buffer, scene, side_camera)
        if frame_dump_dir is not None:
            dump_waypoint_step_frames(
                frame_dump_dir, frame_dump_cams, monitor_waypoint,
                f'gripper_{label}', step_count)
        if step_count % 5 == 0:
            if side_camera_window is not None:
                side_camera_window.update()
            if live_camera is not None:
                live_camera.update()
        if step_count and step_count % 10 == 0:
            print(f"  [gripper] {label} still running ({step_count}/{max_steps})...", flush=True)
        step_count += 1
    if side_camera_window is not None:
        side_camera_window.update()
    if live_camera is not None:
        live_camera.update()
    return done, step_count


# ---------------------------------------------------------------------------
# Behavior-tree action primitives.
#
# These are verbatim extractions of the per-waypoint move/gripper blocks that
# used to live inline in main()'s procedural loop. They are wrapped as leaf
# actions by the py_trees behaviors in bt_tree.py (run-to-completion: one call
# executes the whole path / gripper action). Keeping the bodies identical to the
# original loop preserves the exact sim-stepping cadence, failure hooks, detector
# ticks, and motion capture.
# ---------------------------------------------------------------------------

def _detectors_muted(ctx) -> bool:
    """--limited-v2: once the failure has been detected (ctx.failure_detected),
    stop the live detectors (and their VLM confirmations) for the rest of the run.
    Combined with the failure_detected skip in bt_tree._limited_skip_waypoint (which
    turns off the VLM condition checks), every waypoint after the detection is
    photo-only: the arm still moves through it and its frames are dumped, but no VLM
    call and no detector fire."""
    return bool(getattr(ctx.args, 'limited_v2', False)
                and getattr(ctx, 'failure_detected', False))


def _apply_reference_rules(monitor, phase, i):
    """Recompute the hook-written reference waypoints for waypoint ``i``.

    On failure, only the waypoints the rules own (the report's
    rule_resolved_waypoints, i.e. the ones the source analysis flagged dynamic)
    lose their reference: their inspected reset offset is the value the hook was
    going to overwrite, so keeping it would hand the arrival detectors a
    confidently wrong target. Every other waypoint in the chain is a static
    offset the hook never touches, so it keeps the reference it already had --
    dropping those too cost tasks like place_shape_in_shape_sorter (2 hook-driven
    waypoints out of 5) their whole chain over one unevaluatable equation.
    """
    rules = getattr(monitor, 'reference_rules', None)
    if rules is None:
        return
    try:
        rules.apply(phase, monitor._original_waypoint_at(i))
    except Exception as exc:
        rules.failed = True
        chain = monitor.wp_chain
        owned = ({int(w) for w in (chain.spec.get('rule_resolved_waypoints') or [])}
                 | {int(w) for w in (chain.spec.get('dynamic_waypoints') or [])})
        dropped = [idx for idx in sorted(owned)
                   if chain.poses.pop(idx, None) is not None]
        scope = (f"waypoints {dropped}" if dropped else "no waypoint")
        print(f"  [references] NOT EVALUATED for {scope}: "
              f"inspection rule failed: {exc}")


def bt_run_move(ctx, i, wp, stage):
    """Execute one waypoint's path. Stores the recorded motion buffer on
    ``ctx.motion_buffer`` so the gripper action and the post/boundary VLM check
    can reuse it for in/sequence conditions (e.g. ``inside``)."""
    args = ctx.args
    scene = ctx.scene
    env_wrapper = ctx.env_wrapper
    # After the --limited-v2 detection, mute the detectors (photo-only rest of
    # run); ctx.monitor stays intact so bt_tree.HoldMonitor can still read it.
    monitor = None if _detectors_muted(ctx) else ctx.monitor
    chosen_cam = ctx.chosen_cam
    side_camera = ctx.side_camera
    side_camera_window = ctx.side_camera_window
    live_camera = ctx.live_camera
    failure_active = ctx.failure_active
    n = ctx.n

    # Record a motion sequence during this waypoint's path + gripper action
    # when an in/sequence condition (e.g. inside) is checked afterwards.
    post_seq = conditions_with_sequence_predicate(
        condition_strings(stage['postconditions']), ctx.sequence_predicates
    )
    next_pre_seq = []
    if i < n - 1:
        next_pre_seq = conditions_with_sequence_predicate(
            condition_strings(stage_for_waypoint(ctx.bt_stages, i + 1)['preconditions']),
            ctx.sequence_predicates,
        )
    # No point recording motion frames for a sequence VLM check that will be
    # skipped: after the --limited-v2 detection the boundary/post checks are
    # photo-only, so the buffer would never be consumed.
    record_motion = (ctx.vlm_enabled and bool(post_seq or next_pre_seq)
                     and not _detectors_muted(ctx))
    motion_buffer = [] if record_motion else None
    if record_motion:
        print(
            "  [vlm] recording motion frames for in/sequence condition(s): "
            + ", ".join(post_seq + next_pre_seq)
        )
    ctx.motion_buffer = motion_buffer

    # Full per-step frame dump. Every sim step of this waypoint's path (and its
    # gripper action, below) is written to disk for all cameras so the waypoint
    # can be inspected frame-by-frame. Frames are dumped -- and by default kept
    # -- for EVERY waypoint. Only when AHA_KEEP_ALL_WAYPOINT_FRAMES=0 does
    # prune_waypoint_frame_dirs() run at the end of the episode and delete all
    # but the injection waypoint and the waypoints where a detector fired.
    frame_dump_dir = waypoint_frame_dump_dir(ctx.task_name, i)
    frame_dump_cams = None
    if frame_dump_dir is not None:
        frame_dump_cams = resolve_dump_cameras(scene, side_camera)
        print(
            f"  [frames] dumping every-step frames for waypoint {i} "
            f"-> {frame_dump_dir}"
        )
    ctx.frame_dump_dir = frame_dump_dir
    ctx.frame_dump_cams = frame_dump_cams

    if monitor is not None:
        print_hold_conditions(stage, i)
    print(f"  [move] starting waypoint {i}...")
    # RLBench prepares the legitimate target first; failure hooks perturb it
    # afterward. Detector references come exclusively from inspection data.
    if monitor is not None:
        if ctx.task_name in ('take_item_out_of_drawer', 'change_channel'):
            from aha_publish.running.task_reference_refresh import refresh_moving_object_reference
            from pyrep.objects.object import Object as ReferenceObject
            ref_index = monitor._original_waypoint_at(i)
            try:
                refreshed = refresh_moving_object_reference(
                    ctx.task_name, monitor.wp_chain, ref_index,
                    lambda name: ReferenceObject.get_object(name).get_pose())
                if refreshed:
                    print(f"  [references] waypoint {ref_index}: refreshed object "
                          "anchor; target fixed for this motion")
            except Exception as exc:
                print(f"  [references] NOT EVALUATED for waypoint {ref_index}: "
                      f"object anchor refresh failed: {exc}")
        _apply_reference_rules(monitor, 'start', i)
    wp.start_of_path()
    if monitor is not None:
        monitor.begin_waypoint(i)
    if failure_active and not wp.skip:
        env_wrapper.on_env_waypoint(wp)
    if not wp.skip:
        try:
            print(f"  [move] planning path for waypoint {i}...")
            path = wp.get_path()
            print(f"  [move] executing path for waypoint {i}...", flush=True)
            done = False
            step_count = 0
            while not done and step_count < PATH_ACTION_MAX_STEPS:
                paused = (
                    env_wrapper.on_env_should_pause_path(
                        path=path,
                        path_done=False,
                    )
                    if failure_active
                    else False
                )
                if not paused:
                    done = path.step()
                step_obs = None
                if failure_active:
                    step_obs = env_step_with_camera(env_wrapper, scene, chosen_cam)
                else:
                    step_sim_light(scene, chosen_cam)
                    if monitor is not None:
                        step_obs = refresh_camera_sensors(scene)
                if monitor is not None and step_obs is not None:
                    attach_side_rgb(step_obs, side_camera)
                    monitor.on_step(step_obs, i, done)
                if record_motion and step_count % args.motion_capture_interval == 0:
                    append_motion_frame(motion_buffer, scene, side_camera)
                if frame_dump_dir is not None:
                    dump_waypoint_step_frames(
                        frame_dump_dir, frame_dump_cams, i, 'path', step_count)
                if step_count % 5 == 0:
                    if side_camera_window is not None:
                        side_camera_window.update()
                    if live_camera is not None:
                        live_camera.update()
                if step_count and step_count % 25 == 0:
                    print(
                        f"  [move] waypoint {i} still executing "
                        f"({step_count}/{PATH_ACTION_MAX_STEPS})...",
                        flush=True,
                    )
                step_count += 1
            if not done:
                print(
                    "  [move] waypoint path reached step limit "
                    f"({step_count}); continuing to avoid a hang."
                )
            if failure_active:
                while env_wrapper.on_env_should_pause_path(
                    path=path,
                    path_done=True,
                ):
                    pause_obs = env_pyrep_step_with_camera(
                        env_wrapper, scene, chosen_cam
                    )
                    if monitor is not None and pause_obs is not None:
                        attach_side_rgb(pause_obs, side_camera)
                        monitor.on_step(pause_obs, i, True)
                    if frame_dump_dir is not None:
                        dump_waypoint_step_frames(
                            frame_dump_dir, frame_dump_cams, i, 'path', step_count)
                        step_count += 1
                    if side_camera_window is not None:
                        side_camera_window.update()
                    if live_camera is not None:
                        live_camera.update()
            print(f"  [move] finished waypoint {i} path in {step_count} sim steps.")
        except Exception as e:
            print(f"  ERROR: {e}")
    if not wp.skip:
        if monitor is not None:
            _apply_reference_rules(monitor, 'end', i)
        wp.end_of_path()
    if failure_active:
        env_wrapper.on_env_waypoint_end(wp)
    if ctx.report_wrist:
        ctx.report_wrist_values(
            scene, ctx.task_name, f'wp{i}', ctx.depth_units, ctx.visualization_mode
        )
    if side_camera_window is not None:
        side_camera_window.update()
    if live_camera is not None:
        live_camera.update()


def bt_run_gripper(ctx, i, wp):
    """Execute one waypoint's gripper action (open/close from the waypoint
    extension), or do nothing when the waypoint carries no gripper command.

    Reads the extension fresh: a grasp-failure hook in bt_run_move may have
    cleared it via clear_ext(), and this honors that dropped command."""
    if wp.skip:
        return
    args = ctx.args
    scene = ctx.scene
    env_wrapper = ctx.env_wrapper
    # After the --limited-v2 detection, mute the detectors (photo-only rest of run).
    monitor = None if _detectors_muted(ctx) else ctx.monitor
    chosen_cam = ctx.chosen_cam
    side_camera = ctx.side_camera
    side_camera_window = ctx.side_camera_window
    live_camera = ctx.live_camera
    failure_active = ctx.failure_active
    motion_buffer = ctx.motion_buffer
    frame_dump_dir = getattr(ctx, 'frame_dump_dir', None)
    frame_dump_cams = getattr(ctx, 'frame_dump_cams', None)

    ext = waypoint_extension(wp)
    if ext:
        gripper = ctx.robot.gripper
        if 'open_gripper(' in ext:
            print(f"  [gripper] opening for waypoint {i}...", flush=True)
            # Tag this exact frame as the open-command frame in slip telemetry.
            if monitor is not None and hasattr(monitor, "note_gripper_open_command"):
                monitor.note_gripper_open_command()
            gripper.release()
            done, gripper_steps = actuate_gripper_with_camera(
                gripper=gripper,
                amount=1.0,
                scene=scene,
                env_wrapper=env_wrapper,
                camera_name=chosen_cam,
                failure_active=failure_active,
                side_camera_window=side_camera_window,
                live_camera=live_camera,
                max_steps=GRIPPER_OPEN_MAX_STEPS,
                label='open',
                motion_buffer=motion_buffer,
                side_camera=side_camera,
                motion_capture_interval=args.motion_capture_interval,
                monitor=monitor,
                monitor_waypoint=i,
                frame_dump_dir=frame_dump_dir,
                frame_dump_cams=frame_dump_cams,
            )
            if not done:
                print(
                    "  [gripper] open action reached step limit "
                    f"({gripper_steps}); continuing."
                )
            if ctx.report_wrist:
                ctx.report_wrist_values(
                    scene, ctx.task_name, f'wp{i}_gripper_open',
                    ctx.depth_units, ctx.visualization_mode
                )
            if side_camera_window is not None:
                side_camera_window.update()
            if live_camera is not None:
                live_camera.update()
        elif 'close_gripper(' in ext:
            print(f"  [gripper] closing for waypoint {i}...", flush=True)
            # Tag this frame as the close-command frame in slip telemetry (starts
            # the grasp window / resets the held-peak baseline for method 1).
            if monitor is not None and hasattr(monitor, "note_gripper_close_command"):
                monitor.note_gripper_close_command()
            done, gripper_steps = actuate_gripper_with_camera(
                gripper=gripper,
                amount=0.0,
                scene=scene,
                env_wrapper=env_wrapper,
                camera_name=chosen_cam,
                failure_active=failure_active,
                side_camera_window=side_camera_window,
                live_camera=live_camera,
                label='close',
                motion_buffer=motion_buffer,
                side_camera=side_camera,
                motion_capture_interval=args.motion_capture_interval,
                monitor=monitor,
                monitor_waypoint=i,
                frame_dump_dir=frame_dump_dir,
                frame_dump_cams=frame_dump_cams,
            )
            if not done:
                print(
                    "  [gripper] close action reached step limit "
                    f"({gripper_steps}); attempting grasp and continuing."
                )
            for g_obj in scene.task.get_graspable_objects():
                gripper.grasp(g_obj)
            if ctx.report_wrist:
                ctx.report_wrist_values(
                    scene, ctx.task_name, f'wp{i}_gripper_close',
                    ctx.depth_units, ctx.visualization_mode
                )
            if side_camera_window is not None:
                side_camera_window.update()
            if live_camera is not None:
                live_camera.update()

    # Record the authoritative gripper/grasp state for this waypoint (no-op
    # unless AHA_DUMP_GRIPPER_STATE is set).
    capture_gripper_state(ctx, i, wp)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Interactive waypoint runner that prints BT preconditions before each "
            "move and postconditions after each waypoint."
        )
    )
    parser.add_argument('--task', help='RLBench/failgen task name.')
    parser.add_argument(
        '--mode',
        choices=('manual', 'auto'),
        default='manual',
        help=(
            'Behavior-tree run mode. "manual" (default) asks at each checkpoint '
            'whether to verify the conditions with the VLM (answering no accepts '
            'them as satisfied). "auto" verifies pre/post conditions with the VLM '
            'and hold conditions with the detectors automatically, no prompts.'
        ),
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help=(
            'Auto mode only: treat an uncertain VLM verdict as a FAILURE instead '
            'of accepting the checkpoint (the default is to pass with a warning).'
        ),
    )
    parser.add_argument(
        '--camera',
        choices=CAMERAS,
        help='Camera to show in the CoppeliaSim GUI.',
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run CoppeliaSim headless (no GUI window). Useful for batch/parallel '
             'runs. Defaults to a visible GUI.',
    )
    parser.add_argument(
        '--wrist-visualization',
        choices=[mode for mode, _ in WRIST_VISUALIZATION_MODES],
        help='Optional wrist depth/segmentation report mode. Omitted by default.',
    )
    parser.add_argument(
        '--bt-conditions',
        type=Path,
        help='Path to a prepared .bt_conditions.json file.',
    )
    parser.add_argument(
        '--condition-definitions',
        type=Path,
        default=CONDITION_DEFINITIONS_PATH,
        help='Path to the per-predicate BT condition definitions JSON used in VLM prompts.',
    )
    parser.add_argument(
        '--motion-frames',
        type=int,
        default=4,
        help='Number of timepoints sampled across the motion for in/sequence conditions (e.g. inside).',
    )
    parser.add_argument(
        '--motion-capture-interval',
        type=int,
        default=5,
        help='Capture a motion frame every N sim steps while a sequence condition is pending.',
    )
    parser.add_argument(
        '--failure',
        help="Failure type from the task config. Use 'none' for no failure.",
    )
    parser.add_argument(
        '--failure-waypoint',
        type=int,
        help='Waypoint index where the selected failure should be injected.',
    )
    parser.add_argument(
        '--limited',
        action='store_true',
        help=(
            'For a detector-owned failure injected at waypoint N (collision, '
            'orientation, transition, freezing, slip), run the VLM pre/post '
            'condition checks ONLY in the window N..N+1: pre[N] and the post[N] + '
            'pre[N+1] boundary get the first look and, if the failure is not '
            'caught there, waypoint N+1 gets one more; every waypoint before N and '
            'after N+1 is skipped. Clean and wrong_sequence / wrong_object runs (no '
            'responsible detector) are unaffected and check every waypoint.'
        ),
    )
    parser.add_argument(
        '--limited-v2', '--limited_v2',
        dest='limited_v2',
        action='store_true',
        help=(
            'Exactly like --limited (same VLM pre/post condition checks, same live '
            'detectors, same N..N+1 check window) UNTIL the failure is '
            'detected. The failure is still caught (a detector-owned failure by '
            '~waypoint N+1), but instead of aborting there the run goes photo-only '
            'for the rest of the task: no more VLM checks and the live detectors '
            'muted, while the arm keeps moving through every remaining waypoint and '
            'its frames are dumped, to the last waypoint (only the wall-clock '
            'timeout stops it). With --dump-waypoint-frames the per-step frames are '
            'kept for every waypoint (no pruning to just the injection/detector-fire '
            'waypoints).'
        ),
    )
    parser.add_argument(
        '--no-abort', '--no_abort',
        dest='no_abort',
        action='store_true',
        help=(
            'Never stop the behavior tree on a failure: a failed VLM condition '
            'check or a confirmed fire of the responsible detector is recorded '
            'and the tree keeps going, so the arm executes every waypoint to the '
            'last one (only the wall-clock timeout ends the episode). Unlike '
            '--limited-v2 nothing goes photo-only — every remaining waypoint '
            'still runs its full VLM pre/post condition checks with the live '
            'detectors armed, so later checkpoints are scored too.'
        ),
    )
    parser.add_argument(
        '--repeat-waypoints', '--repeat_waypoints',
        dest='repeat_waypoints',
        action='store_true',
        help=(
            'Run every waypoint repetition the task asks for (RLBench '
            'register_waypoints_should_repeat: stack_blocks, place_cups, '
            'empty_container, push_buttons, setup_checkers, setup_chess, '
            'remove_cups, put_all_groceries_in_cupboard). Those tasks reuse the '
            'same waypoint indices once per item, so waypoints 0..N run again '
            'for every block/cup/button. Off by default: the episode stops after '
            'the first pass, because an injected failure is applied only once, '
            'making later passes clean runs whose detector fires are false '
            'positives.'
        ),
    )
    parser.add_argument(
        '--vlm-checks',
        choices=('ask', 'off', 'pre', 'post', 'both'),
        default='ask',
        help=(
            'Use OpenAI VLM to check BT conditions. Default ask prompts at each '
            'pre/post checkpoint; pre/post/both run automatically.'
        ),
    )
    parser.add_argument(
        '--vlm-predicates',
        default=None,
        help='Comma-separated predicate names to restrict VLM checking to (e.g. '
             '"gripper_oriented_for,end_effector_aligned_with"). Conditions whose '
             'predicate is not listed are skipped (treated as satisfied). Default: '
             'check all conditions.',
    )
    parser.add_argument(
        '--vlm-arrival-predicates',
        default=None,
        help='Comma-separated predicate names to verify at WAYPOINT ARRIVAL '
             '(after the move reaches the waypoint pose, before the gripper acts) '
             'instead of at the pre/post boundary. Use for predicates that are only '
             'judgeable once the gripper is at the grasp pose, e.g. '
             '"gripper_oriented_for". These are removed from the pre/post checks.',
    )
    parser.add_argument(
        '--vlm-arrival-strict',
        dest='vlm_arrival_strict',
        action='store_true',
        help='Treat an UNCERTAIN verdict on an arrival-checked predicate (e.g. '
             'gripper_oriented_for) as a FAILURE, even without the global --strict. '
             'On by default: orientation defaults to strict while other conditions '
             'stay lenient.',
    )
    parser.add_argument(
        '--no-vlm-arrival-strict',
        dest='vlm_arrival_strict',
        action='store_false',
        help='Treat an uncertain arrival-predicate (orientation) verdict as '
             'satisfied (lenient), like the other conditions.',
    )
    parser.set_defaults(vlm_arrival_strict=True)
    parser.add_argument(
        '--vlm-model',
        default=DEFAULT_VLM_MODEL,
        help='OpenAI model for BT condition VLM checks. Use a faster/smaller '
             'model here to speed up checks (e.g. a mini variant).',
    )
    parser.add_argument(
        '--vlm-reasoning-effort',
        choices=('minimal', 'low', 'medium', 'high'),
        default='low',
        help='Reasoning effort for reasoning-capable models (gpt-5 / o-series). '
             'Lower is much faster; "minimal" is fastest. Ignored by models that '
             'do not support reasoning.',
    )
    parser.add_argument(
        '--vlm-preview',
        action='store_true',
        help='Preview images in pop-up windows before each VLM condition check (off by default).',
    )
    parser.add_argument(
        '--no-vlm-preview',
        dest='vlm_preview',
        action='store_false',
        help='Do not preview images before VLM condition checks.',
    )
    parser.set_defaults(vlm_preview=False)
    parser.add_argument(
        '--vlm-no-pass-explanation',
        action='store_true',
        help='Tell the VLM to skip the explanation/evidence text when a checkpoint '
             'passes (every condition satisfied); it still writes an explanation for '
             'not_satisfied/uncertain conditions. Saves output tokens on passing '
             'waypoints (off by default).',
    )
    parser.add_argument(
        '--vlm-trace',
        action='store_true',
        help='Save VLM prompts/responses under aha_output/vlm_traces.',
    )
    parser.add_argument(
        '--vlm-diagnose-failure',
        dest='vlm_diagnose_failure',
        action='store_true',
        help='After a run-ending failure, run a text-only LLM diagnosis that '
             'classifies the failure type (collision/translation/orientation/'
             'wrong_sequence/wrong_object/no_grasp/slip/freezing) twice: once with '
             'the live-detector fires + grasp force sensor as evidence (option 1) '
             'and once blind from only the VLM condition-check transcript (option 2). '
             'On by default.',
    )
    parser.add_argument(
        '--no-vlm-diagnose-failure',
        dest='vlm_diagnose_failure',
        action='store_false',
        help='Disable the post-failure failure-type LLM diagnosis.',
    )
    parser.set_defaults(vlm_diagnose_failure=True)
    parser.add_argument(
        '--vlm-run-log',
        action='store_true',
        help='Write a per-run CSV with one row per VLM event (pre/post condition '
             'checks + slip/collision fire verifiers): injection/detection frame, '
             'verifier type, verdict, explanation, api calls, tokens, cost, and the '
             'saved input-image path. Also saves every VLM input montage into one '
             'per-run folder. Default location: aha_output/bt_runs/<task>__<failure>__<ts>/.',
    )
    parser.add_argument(
        '--vlm-run-log-dir',
        default=None,
        help='Explicit output folder for --vlm-run-log (implies it).',
    )
    parser.add_argument(
        '--camera-window',
        action='store_true',
        help='Open a live Matplotlib image window for the selected camera.',
    )
    parser.add_argument(
        '--no-side-camera-window',
        action='store_true',
        help='Do not open the detector side-camera image window.',
    )
    parser.add_argument(
        '--live-detectors',
        action='store_true',
        help='Run the runtime detectors (collision/freezing/slip) live during '
             'the run and report detections (on by default).',
    )
    parser.add_argument(
        '--no-live-detectors',
        dest='live_detectors',
        action='store_false',
        help='Disable live runtime detector monitoring.',
    )
    parser.set_defaults(live_detectors=True)
    parser.add_argument(
        '--detector-plots',
        action='store_true',
        help='Show each detector live telemetry plot (like the standalone '
             'detectors). Off by default.',
    )
    parser.add_argument(
        '--no-detector-plots',
        dest='detector_plots',
        action='store_false',
        help='Do not open detector telemetry plot windows.',
    )
    parser.set_defaults(detector_plots=False)
    return parser.parse_args()


def main():
    args = parse_args()

    seed_raw = os.getenv("AHA_RUN_SEED", "").strip()
    if seed_raw:
        try:
            seed = int(seed_raw)
            import random
            import numpy as np
            random.seed(seed)
            np.random.seed(seed)
            print(f"Run seed: {seed}")
        except ValueError:
            print(f"WARNING: ignoring invalid AHA_RUN_SEED={seed_raw!r}")

    # Resolve --mode onto --vlm-checks, which the rest of the code keys off.
    # manual -> keep the interactive 'ask' default (answering no accepts the
    # conditions). auto -> verify automatically; default to 'both' unless the
    # user explicitly narrowed coverage with --vlm-checks.
    if args.mode == 'auto' and args.vlm_checks == 'ask':
        args.vlm_checks = 'both'

    # --limited-v2 is EXACTLY --limited (same VLM pre/post condition checks, same
    # live detectors, same N..N+1 check window) UNTIL the failure is
    # detected. The failure is still detected (a detector-owned failure catches by
    # ~waypoint N+1), but instead of aborting there the run latches
    # ctx.failure_detected and goes photo-only for the rest of the task: no more VLM
    # condition checks (bt_tree._limited_skip_waypoint) and the live detectors muted
    # (_detectors_muted), while the arm keeps moving through every remaining
    # waypoint and its frames are dumped, to the last waypoint (only the wall-clock
    # --timeout stops it). Turn --limited on here for the skip-before-injection
    # logic; the no-abort + photo-only behaviour lives in bt_tree and _detectors_muted.
    if getattr(args, 'limited_v2', False):
        args.limited = True
        print("--limited-v2: like --limited until the failure is detected, then "
              "photo-only (VLM checks + detectors off) through every remaining "
              "waypoint to the end — never aborts.")

    # --no-abort: same checks everywhere as a normal run, only the stopping is
    # removed. A failed condition / confirmed detector fire is recorded and the
    # tree keeps ticking (bt_tree._no_abort), with the VLM checks and detectors
    # left ON for every remaining waypoint (unlike --limited-v2's photo-only).
    if getattr(args, 'no_abort', False) and not getattr(args, 'limited_v2', False):
        print("--no-abort: failures are recorded but never stop the tree; every "
              "waypoint runs to the end with its VLM condition checks and the "
              "live detectors still on.")

    # In auto mode there are no prompts anywhere: the detector fire-verifiers
    # (slip/collision) must also auto-confirm instead of asking "Confirm this
    # slip with OpenAI VLM? [y/N]". _bundle.VLM_AUTO reads this env at import, and
    # live_detectors is imported later in this same call, so setting it now (before
    # that import) takes effect. Respect an explicit override if already set.
    if args.mode == 'auto' and 'AHA_DETECTOR_VLM_AUTO' not in os.environ:
        os.environ['AHA_DETECTOR_VLM_AUTO'] = '1'

    # Optional predicate allow-list: only these predicates are checked by the VLM.
    args.vlm_predicate_filter = (
        {p.strip().lower() for p in args.vlm_predicates.split(',') if p.strip()}
        if args.vlm_predicates
        else None
    )
    if args.vlm_predicate_filter:
        print(f"VLM predicate filter (only these checked): "
              f"{', '.join(sorted(args.vlm_predicate_filter))}")

    # Predicates verified at waypoint arrival (post-move, pre-grip) rather than at
    # the pre/post boundary — the moment the gripper is at the grasp pose.
    args.vlm_arrival_predicate_filter = (
        {p.strip().lower() for p in args.vlm_arrival_predicates.split(',') if p.strip()}
        if args.vlm_arrival_predicates
        else None
    )
    if args.vlm_arrival_predicate_filter:
        print(f"VLM arrival-checked predicates (verified at grasp-pose arrival): "
              f"{', '.join(sorted(args.vlm_arrival_predicate_filter))}")

    from failgen.env_wrapper import FailGenEnvWrapper
    from aha_publish.common.wrist_sensor_viewer import close_wrist_visualization, report_wrist_values

    # 1) Choose the task first.
    task_name = args.task or pick_task()
    print_task_category(task_name)
    task_config = load_task_config(task_name)

    # 2) Then choose the failure type, and 3) then which waypoint.
    failtype, failure_waypoint = resolve_failure_choice(task_config, args)
    failure_active = failtype is not None

    chosen_cam = args.camera or "default"
    visualization_mode = args.wrist_visualization
    report_wrist = visualization_mode is not None
    if visualization_mode is None:
        visualization_mode = "off"
    vlm_enabled = args.vlm_checks != 'off'

    # Detector that owns the injected failure (slip->slip, collision->collision,
    # rotation_*->orientation, translation_*->transition, freezing->freezing).
    # Used to stop the run only on THAT detector's confirmed fire (cross-fires
    # from other detectors are still recorded/plotted but do not abort).
    import aha_publish.running.vlm_run_logger as _vrl
    args.responsible_detector = (
        _vrl.responsible_detector_for(failtype) if failure_active else None)

    # Opt-in per-run log bundle. This is intentionally NOT gated on vlm_enabled:
    # the RunLogger also wires up the per-step detector drift CSVs, tees stdout to
    # run.log, and renders the trace plots on finalize -- all detector-only
    # artifacts we still want with --vlm-checks off. When VLM is off, vlm_events.csv
    # is header-only and vlm_images stays empty (nothing calls the VLM), but the
    # drift CSVs / plots / run.log / waypoint frames still land in the case folder.
    if args.vlm_run_log or args.vlm_run_log_dir:
        out_dir = args.vlm_run_log_dir or _vrl.default_out_dir(
            str(paths.PROJECT_ROOT), task_name,
            failtype if failure_active else 'none',
            failure_waypoint if failure_active else None)
        _lg = _vrl.start(
            out_dir, task_name,
            failtype if failure_active else 'none',
            failure_waypoint if failure_active else None,
            args.vlm_model)
        # dump_vlm_frames (pre/post condition images) and --dump-waypoint-frames
        # both read this module global; point it at the run's image folder so
        # those images land there too.
        global SAVE_VLM_FRAMES_DIR
        SAVE_VLM_FRAMES_DIR = str(_lg.images_dir)
        print(f"Run log: {_lg.csv_path}")
        print(f"Images:  {_lg.images_dir}")

    args.condition_definitions_data = (
        load_condition_definitions(args.condition_definitions) if vlm_enabled else {}
    )
    sequence_predicates = get_sequence_predicates(args.condition_definitions_data)
    yaml_descs = load_yaml_descriptions(task_name)
    depth_units = (
        "meters"
        if task_config.get('data', {}).get('depth_in_meters', False)
        else "normalized 0-1"
    )

    # BT condition source is selected after the task/failure choices.
    bt_conditions_path = (
        args.bt_conditions.resolve()
        if args.bt_conditions
        else pick_bt_conditions_path(task_name)
    )

    bt_data, bt_source_name, bt_stages = load_bt_stages(bt_conditions_path)
    bt_task_name = bt_data.get('task_name')
    if bt_task_name and bt_task_name != task_name:
        print(
            f"\nWARNING: BT file task_name is '{bt_task_name}', "
            f"but simulator task is '{task_name}'."
        )

    print(f"\nLaunching simulator for '{task_name}' with camera '{chosen_cam}'...")
    print(f"BT mode: {args.mode}")
    print(f"Failure: {failtype if failtype else 'none'}")
    if failtype and failure_waypoint >= 0:
        print(f"Failure waypoint: {failure_waypoint}")
    print(f"VLM checks: {args.vlm_checks if vlm_enabled else 'off'}")
    if vlm_enabled:
        print(f"VLM model: {args.vlm_model}")
        if model_supports_reasoning(args.vlm_model):
            print(f"VLM reasoning effort: {args.vlm_reasoning_effort}")
        if VLM_IMAGE_DETAIL:
            print(f"VLM image detail: {VLM_IMAGE_DETAIL}")
        print(f"Condition definitions: {args.condition_definitions}")
    print(f"BT conditions: {bt_conditions_path}")
    print(f"Condition source: {bt_source_name}.stages\n")

    env_wrapper = FailGenEnvWrapper(
        task_name=task_name,
        headless=args.headless,
        record=False,
        save_data=True,
        save_path='/tmp/aha_label',
        save_keyframes_only=False,
    )
    configure_failure(env_wrapper, task_config, failtype, failure_waypoint)
    env_wrapper.reset()

    try:
        scene = env_wrapper._env._scene
        scene.robot.arm.set_control_loop_enabled(True)

        try:
            sync_default_camera(scene, chosen_cam)
            pyrep_step_with_camera(scene, chosen_cam)
            label = CAMERA_LABELS.get(chosen_cam, chosen_cam)
            print(f"View set to: {label}")
        except Exception as e:
            print(f"Could not set camera: {e}")

        # Speed knob: skip the side-camera vision sensor entirely when it can't
        # be used (no VLM, no side window). The sensor is rendered every physics
        # step on software GL, so creating it for pose-only detector calibration
        # is pure overhead. Gated behind AHA_NO_SIDE_CAMERA so default runs are
        # unchanged. All downstream side_camera uses are None-safe.
        _skip_side_cam = (
            os.getenv("AHA_NO_SIDE_CAMERA", "").strip() not in ("", "0", "false", "False")
            and not vlm_enabled and args.no_side_camera_window and chosen_cam != 'side')
        if _skip_side_cam:
            side_camera = None
            initial_side_frame = None
            print("Side camera SKIPPED (AHA_NO_SIDE_CAMERA: no VLM / no window)")
        else:
            side_camera = get_or_create_side_camera(scene)
            initial_side_frame = warm_up_camera_sensor(scene, side_camera)
            pos = side_camera.get_position()
            print(
                "Side camera ready "
                f"({SIDE_CAMERA_NAME}) at "
                f"x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}"
            )
            try:
                applied_fov = widen_wrist_camera(scene)
                if applied_fov:
                    print(f"Wrist camera FOV widened to {applied_fov:.0f} deg so wrist_rgb shows the gripper fingers")
            except Exception as e:
                print(f"Could not widen wrist camera FOV: {e}")
            try:
                applied_tilt = tilt_wrist_camera(scene)
                if applied_tilt:
                    print(f"Wrist camera tilted down {applied_tilt:.0f} deg so wrist_rgb shows more of the gripper")
            except Exception as e:
                print(f"Could not tilt wrist camera: {e}")

        selected_camera_sensor = (
            side_camera
            if chosen_cam == 'side'
            else getattr(scene, chosen_cam) if chosen_cam.startswith('_cam') else None
        )
        show_selected_camera_window = (
            args.camera_window
            and selected_camera_sensor is not None
            and (selected_camera_sensor is not side_camera or args.no_side_camera_window)
        )
        side_camera_window = (
            None
            if args.no_side_camera_window
            else LiveCameraView(
                side_camera,
                f"{task_name} - detector side camera",
                initial_frame=initial_side_frame,
            )
        )
        live_camera = (
            LiveCameraView(
                selected_camera_sensor,
                f"{task_name} - {CAMERA_LABELS.get(chosen_cam, chosen_cam)}",
                initial_frame=(
                    initial_side_frame
                    if selected_camera_sensor is side_camera
                    else None
                ),
            )
            if show_selected_camera_window
            else None
        )

        robot = scene.robot
        waypoints = scene.task.get_waypoints()
        n = len(waypoints)
        print(f"Found {n} waypoints")

        monitor = None
        if getattr(args, 'live_detectors', False):
            # Detector targets come exclusively from inspection transforms and
            # optional hook equations. Scene-object roots are cached before
            # failure injection; simulator waypoint geometry is never read here.
            # The robot's CURRENT pose still comes from obs.gripper_pose.
            original_wp_poses = {}
            _wp_chain = None
            try:
                from aha_publish.common.waypoint_chain import WaypointChain
                from pyrep.objects.object import Object as _Object
                _chain = WaypointChain.for_task(task_name)
                if _chain is None:
                    raise FileNotFoundError(
                        f"no inspection report for {task_name}")
                _world = {}
                for _name in _chain.root_objects:
                    try:
                        if 'waypoint' in _name.lower():
                            raise ValueError('simulator waypoint reads are prohibited')
                        _world[_name] = list(
                            _Object.get_object(_name).get_pose())
                    except Exception as _exc:
                        print(f"[wp-chain] root {_name!r} unreadable: {_exc}")
                _resolved = _chain.resolve(
                    _world, variation_index=scene._variation_index)
                original_wp_poses = {int(k): list(v)
                                     for k, v in _resolved.items()}
                print(f"[wp-chain] composed {len(_resolved)} waypoint targets "
                      f"from {len(_world)} object poses read once")
                _missing = _chain.unresolved()
                if _missing:
                    print(f"[wp-chain] WARNING: waypoints {_missing} could not "
                          f"be grounded and have NO reference pose")
                _stale = _chain.spec.get("dynamic_waypoints") or []
                if _stale:
                    # init_episode repositions these, so the stored offset does
                    # not describe this episode. Correcting them meant reading
                    # the dummy, which no longer happens -- the spec has to carry
                    # corrected offsets instead.
                    print(f"[wp-chain] WARNING: {task_name} lists "
                          f"dynamic_waypoints {_stale}; their offsets are "
                          f"known-stale and are NOT corrected at run time")
                _wp_chain = _chain
            except Exception as _exc:
                print(f"[wp-chain] FAILED ({type(_exc).__name__}: {_exc}) -- no "
                      f"reference poses, detectors have nothing to compare "
                      f"against for this run")

            def _original_pose_fn(idx):
                # In-memory inspection reference, including any hook equations
                # evaluated for this execution. Never a simulator dummy read.
                return _wp_chain.pose(idx) if _wp_chain is not None else None
            try:
                from aha_publish.running.live_detectors import LiveDetectorMonitor, enabled_detectors_for_stage, stage_context_map_for_stages, target_object_map_for_stages
                # Which detectors each waypoint checks comes straight from the
                # BT's per-stage hold conditions (deselected ones were already
                # dropped by load_bt_stages), so removing a hold condition in the
                # BT maker turns that detector off at that waypoint here.
                enabled_map = {
                    idx: enabled_detectors_for_stage(stage_for_waypoint(bt_stages, idx))
                    for idx in range(n)
                }
                # The object each stage manipulates, straight from that stage's
                # own conditions. The collision VLM verifier is told it, so
                # contact with THIS object reads as intended manipulation while
                # contact with anything else reads as a collision.
                target_object_map = target_object_map_for_stages(
                    bt_stages, n, stage_lookup=stage_for_waypoint)
                # The same verifier is also told the stage's primitive and its
                # pre/postconditions, so the "what is this step for" context is
                # the BT's predicates instead of the description JSON's prose.
                stage_context_map = stage_context_map_for_stages(
                    bt_stages, n, stage_lookup=stage_for_waypoint)
                if vlm_enabled:
                    # Make sure OpenAI creds are loaded so the detector VLM
                    # confirmation (OpenAI) can run on a detection.
                    try:
                        load_openai_credential_from_local_files()
                    except Exception:
                        pass
                monitor = LiveDetectorMonitor(
                    task_name,
                    # No waypoint_pose_fn: the live (failgen-corrupted) target
                    # channel is gone, so the detectors see only the robot's
                    # current pose and the composed reference.
                    original_pose_fn=_original_pose_fn,
                    enabled_map=enabled_map,
                    target_object_map=target_object_map,
                    stage_context_map=stage_context_map,
                    env_wrapper=env_wrapper,
                    vlm_enabled=vlm_enabled,
                    vlm_model=args.vlm_model,
                    vlm_trace=args.vlm_trace,
                    show_plots=getattr(args, 'detector_plots', True),
                    n_waypoints=n,
                    failure_waypoint=failure_waypoint if failure_active else None,
                )
                monitor.wp_chain = _wp_chain
                monitor.reference_rules = None
                if _wp_chain is not None and _wp_chain.spec.get('hook_reference_rules'):
                    from aha_publish.common.waypoint_reference_rules import ReferenceRuleEvaluator, referenced_object_names
                    _scene_pose_cache = dict(_world)
                    def _scene_reference_pose(name):
                        if 'waypoint' in name.lower():
                            raise ValueError('simulator waypoint reads are prohibited')
                        return _scene_pose_cache[name]
                    # Capture named scene objects BEFORE episode-level failure
                    # setup. Pose equations select among these cached roots.
                    # reference_scene_objects covers what the inspection exported
                    # (SHAPEs inside the boundary root); the equations also select
                    # objects the inspection never listed -- grasp/drop-point
                    # DUMMYs, proximity sensors -- through task attributes, and a
                    # single one of those missing from the cache costs its
                    # waypoints their whole reference.
                    _hook_objects = referenced_object_names(
                        _wp_chain.spec.get('hook_reference_rules') or {}, scene.task)
                    for _name in (list(_wp_chain.spec.get('reference_scene_objects', []))
                                  + _hook_objects):
                        if 'waypoint' in _name.lower():
                            raise ValueError('waypoint cannot be a scene-reference root')
                        if _name not in _scene_pose_cache:
                            try:
                                _scene_pose_cache[_name] = list(_Object.get_object(_name).get_pose())
                            except Exception as exc:
                                print(f"  [references] scene object {_name!r} unavailable: {exc}")
                    monitor.reference_rules = ReferenceRuleEvaluator(
                        _wp_chain, scene.task, _scene_reference_pose)
                monitor.print_banner()
            except Exception as e:
                print(f"Could not start live detectors: {e}")
                monitor = None
        if len(bt_stages) != n:
            print(
                f"WARNING: BT file has {len(bt_stages)} stages, "
                f"but simulator has {n} waypoints."
            )
        print()

        if failure_active:
            env_wrapper.on_env_start(scene.task)

        # Run the episode as a real py_trees behavior tree. The tree reuses the
        # bt_run_move / bt_run_gripper / maybe_run_vlm_* primitives above; see
        # bt_tree.py for the node definitions and tree shape. bt_tree is imported
        # here (not at module top level) so the light-import contract holds: the
        # GUI can `import waypoints_interactive_bt_conditions` without pulling in
        # py_trees or any simulator dependency.
        import aha_publish.running.bt_tree as bt_tree

        ctx = bt_tree.BTContext(
            runner=sys.modules[__name__],
            args=args,
            scene=scene,
            env_wrapper=env_wrapper,
            robot=robot,
            waypoints=waypoints,
            n=n,
            bt_stages=bt_stages,
            bt_source_name=bt_source_name,
            task_name=task_name,
            chosen_cam=chosen_cam,
            side_camera=side_camera,
            side_camera_window=side_camera_window,
            live_camera=live_camera,
            monitor=monitor,
            yaml_descs=yaml_descs,
            depth_units=depth_units,
            vlm_enabled=vlm_enabled,
            sequence_predicates=sequence_predicates,
            failure_active=failure_active,
            failure_waypoint=failure_waypoint,
            responsible_detector=getattr(args, 'responsible_detector', None),
            report_wrist=report_wrist,
            visualization_mode=visualization_mode,
            report_wrist_values=report_wrist_values,
        )

        bt_tree.run_tree(ctx)

        no_abort_run = (getattr(args, 'limited_v2', False)
                        or getattr(args, 'no_abort', False))
        if ctx.failures and not no_abort_run:
            print("\nStopped: the behavior tree reported a failure (see BT VERDICT above).")
        elif ctx.failures:
            mode = '--limited-v2' if getattr(args, 'limited_v2', False) else '--no-abort'
            print(f"\nDone! Robot completed all waypoints; {len(ctx.failures)} "
                  f"checkpoint(s) flagged a failure ({mode}: recorded, did "
                  f"not abort).")
        else:
            print("\nDone! Robot completed all waypoints.")
        # Keep only the frame dumps worth inspecting: the injection waypoint plus
        # every waypoint where a live detector fired (fired_waypoints is final now
        # that any VLM confirmation/retraction resolved during run_tree).
        if DUMP_WAYPOINT_FRAMES and (KEEP_ALL_WAYPOINT_FRAMES
                                     or getattr(args, 'limited_v2', False)):
            # limited-v2 (and AHA_KEEP_ALL_WAYPOINT_FRAMES) want the frames for
            # the WHOLE run, so keep every waypoint's dump instead of pruning to
            # the injection/detector-fire waypoints (there is no detector here,
            # and clean runs have none).
            why = ('AHA_KEEP_ALL_WAYPOINT_FRAMES' if KEEP_ALL_WAYPOINT_FRAMES
                   else '--limited-v2')
            print(f"  [frames] {why}: kept waypoint frame dumps for every "
                  "waypoint (no pruning).")
        elif DUMP_WAYPOINT_FRAMES:
            keep_wps = set()
            if failure_active and failure_waypoint is not None and failure_waypoint >= 0:
                keep_wps.add(failure_waypoint)
            if monitor is not None:
                for _det in monitor.detectors.values():
                    keep_wps |= set(getattr(_det, 'fired_waypoints', None) or set())
            prune_waypoint_frame_dirs(task_name, keep_wps)
            print(f"  [frames] kept waypoint frame dumps for: "
                  f"{sorted(keep_wps) if keep_wps else 'none'}")
        if monitor is not None:
            monitor.print_summary()
            monitor.close()
        try:
            import aha_publish.running.vlm_run_logger as _vrl
            _lg = _vrl.active()
            if _lg is not None:
                _lg.finalize(
                    task_name,
                    failtype if failure_active else 'none',
                    failure_waypoint if failure_active else None)
                print("\n" + _lg.summary_line())
        except Exception:
            pass
        wait_for_space("Press Space or Enter to close the simulator...")
    finally:
        close_wrist_visualization(visualization_mode)
        env_wrapper.shutdown()


if __name__ == '__main__':
    main()
