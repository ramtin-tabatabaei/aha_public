"""
Capture per-waypoint camera screenshots for RLBench / failgen tasks.

For every task, the robot is driven through each waypoint and the front, side and
wrist camera views are saved for that waypoint (front and wrist come from the
RLBench observation; side is the custom aha_side_camera VisionSensor we attach to
the scene). The gripper is ALWAYS actuated (so the robot grasps and
carries objects correctly and later waypoints behave). For a waypoint that opens
or closes the gripper, the canonical wp{i} frame is captured AFTER the actuation
so it reflects the waypoint's end state (matching the authoritative per-waypoint
gripper_state); waypoints without a gripper action are captured at arrival. Only
with --gripper does the script ALSO save a dedicated wp{i}.5 screenshot right
after each gripper open/close.

This merges the former:
  - waypoints_auto_screenshot_waypoints.py        (waypoint screenshots)
  - waypoints_screenshot_waypoints_and_gripper.py (waypoint + gripper screenshots)

Each task runs in its own subprocess so a CoppeliaSim crash skips only that task.

Usage:
    python waypoints_screenshot.py                          # all tasks, waypoint shots only
    python waypoints_screenshot.py --gripper                # all tasks, + gripper-action shots
    python waypoints_screenshot.py --single pick_up_cup --gripper
    python waypoints_screenshot.py --output-dir /tmp/shots
"""

from aha_publish import paths

import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import yaml
from PIL import Image

PROJECT_ROOT = str(paths.PROJECT_ROOT)
FAILGEN_ROOT = str(paths.FAILGEN_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, FAILGEN_ROOT)

CONFIGS_PATH = os.path.join(FAILGEN_ROOT, 'failgen', 'configs')
DEFAULT_OUTPUT_DIR = str(paths.OUTPUT_DIR / 'waypoint_screenshots')

# Cameras read directly from the RLBench observation.
CAM_MAP = {
    'front': 'front_rgb',
    'wrist': 'wrist_rgb',
}

# Custom side view: a VisionSensor we attach to the scene ourselves (matches the
# aha_side_camera defined in detectors/collision/interactive.py).
SIDE_CAMERA_NAME = 'aha_side_camera'
# Render the side view natively at 256x256 so the saved image is exactly that
# size with no cropping or scaling.
SIDE_CAMERA_RESOLUTION = [256, 256]
SIDE_CAMERA_POSITION = (-0.547, -0.803, 1.209)  # nudged ~0.45 m forward along the look axis
SIDE_CAMERA_ORIENTATION_DEG = (-100, 38.3, -180.0)  # panned left

WARMUP_STEPS = 50           # renderer warm-up before the first screenshot
SETTLE_STEPS = 20           # settle when a waypoint has no gripper action
GRIPPER_SETTLE_STEPS = 10   # settle after a gripper open/close action
TASK_TIMEOUT_SEC = 180      # per-task subprocess timeout


def get_all_tasks():
    files = sorted(glob.glob(os.path.join(CONFIGS_PATH, '*.yaml')))
    return [os.path.basename(f).replace('.yaml', '') for f in files]


def load_yaml_descriptions(task_name):
    path = os.path.join(CONFIGS_PATH, f'{task_name}.yaml')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        config = yaml.safe_load(f)
    descriptions = {}
    for st in config.get('sub-tasks', []):
        processes = st.get('processes', [])
        desc_list = st.get('task_description', [''])
        if len(processes) >= 2:
            wp_from = int(processes[0].replace('waypoint', ''))
            descriptions[wp_from] = desc_list[0] if desc_list else ''
    return descriptions


def create_side_camera(scene):
    """Attach the custom aha_side_camera VisionSensor to the scene and return it."""
    from pyrep.const import RenderMode
    from pyrep.objects.vision_sensor import VisionSensor

    side_camera = VisionSensor.create(SIDE_CAMERA_RESOLUTION)
    side_camera.set_name(SIDE_CAMERA_NAME)
    side_camera.set_render_mode(RenderMode.OPENGL3)
    side_camera.set_position(list(SIDE_CAMERA_POSITION))
    side_camera.set_orientation(np.radians(SIDE_CAMERA_ORIENTATION_DEG))
    scene._aha_side_camera = side_camera
    return side_camera


def save_cameras(scene, task_name, label, out_dir, side_camera=None):
    """Save the front, side and wrist views.

    `label` is the filename suffix, e.g. 'wp3' or 'wp3.5_gripper_open'. Front and
    wrist come from the RLBench observation; side is the custom aha_side_camera.
    """
    obs = scene.get_observation()
    for cam_name, attr in CAM_MAP.items():
        img_data = getattr(obs, attr)
        if img_data is not None:
            img = Image.fromarray(img_data.astype(np.uint8))
            fname = f'{task_name}_{label}_{cam_name}.png'
            img.save(os.path.join(out_dir, fname))

    if side_camera is not None:
        rgb = side_camera.capture_rgb()
        img = Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8))
        img.save(os.path.join(out_dir, f'{task_name}_{label}_side.png'))


def actuate_gripper(scene, amount):
    """Drive the gripper to `amount` (1.0 = open, 0.0 = closed)."""
    gripper = scene.robot.gripper
    done = False
    while not done:
        done = gripper.actuate(amount, 0.04)
        scene.pyrep.step()


def step(scene, n):
    for _ in range(n):
        scene.pyrep.step()


def process_task(task_name, capture_gripper, output_dir):
    # Imported lazily so task listing / --help / subprocess dispatch don't need CoppeliaSim.
    from failgen.env_wrapper import FailGenEnvWrapper

    task_out_dir = os.path.join(output_dir, task_name)
    # Start from a clean folder so stale screenshots from earlier runs (old cameras
    # like left_shoulder/overhead, or gripper .5 frames from a previous --gripper
    # run) can't leak into the combined grid.
    if os.path.isdir(task_out_dir):
        for f in glob.glob(os.path.join(task_out_dir, f'{task_name}_*.png')):
            os.remove(f)
    os.makedirs(task_out_dir, exist_ok=True)
    yaml_descs = load_yaml_descriptions(task_name)

    print(f"  Launching: {task_name}")
    env_wrapper = FailGenEnvWrapper(
        task_name=task_name,
        headless=True,
        record=True,
        save_data=True,
        save_path=str(paths.BACKEND_DATA_DIR / 'screenshots'),
        save_keyframes_only=False,
        no_failures=True,
    )
    env_wrapper.reset()
    scene = env_wrapper._env._scene
    scene.robot.arm.set_control_loop_enabled(True)

    side_camera = create_side_camera(scene)

    step(scene, WARMUP_STEPS)

    # initial screenshot
    save_cameras(scene, task_name, 'wp-1', task_out_dir, side_camera)
    print("    WP-1 (initial) saved")

    waypoints = scene.task.get_waypoints()

    for i, wp in enumerate(waypoints):
        try:
            ext = wp._waypoint.get_extension_string()
        except Exception:
            ext = ""
        desc = yaml_descs.get(i, f'waypoint {i}')

        # move to waypoint
        wp.start_of_path()
        if not wp.skip:
            try:
                path = wp.get_path()
                done = False
                while not done:
                    done = path.step()
                    scene.step()
            except Exception as e:
                print(f"    WP{i} path error: {e}")
        wp.end_of_path()

        # gripper action: always actuated so the robot progresses. The canonical
        # wp{i} frame is captured AFTER the action so it reflects the waypoint's
        # end state (a close/open waypoint shows the gripper already closed/open),
        # matching the authoritative per-waypoint gripper_state. Waypoints without
        # a gripper action keep the arrival frame.
        if 'open_gripper(' in ext:
            # Release any grasped object first, or the grasp connection holds the
            # fingers shut and actuate(1.0) returns done without opening.
            scene.robot.gripper.release()
            actuate_gripper(scene, 1.0)
            step(scene, GRIPPER_SETTLE_STEPS)
            save_cameras(scene, task_name, f'wp{i}', task_out_dir, side_camera)
            print(f"    WP{i} ({desc[:40]}) saved (after OPEN)")
            if capture_gripper:
                save_cameras(scene, task_name, f'wp{i + 0.5:.1f}_gripper_open', task_out_dir, side_camera)
                print(f"    WP{i} gripper OPEN screenshot saved")
        elif 'close_gripper(' in ext:
            actuate_gripper(scene, 0.0)
            for g in scene.task.get_graspable_objects():
                scene.robot.gripper.grasp(g)
            step(scene, GRIPPER_SETTLE_STEPS)
            save_cameras(scene, task_name, f'wp{i}', task_out_dir, side_camera)
            print(f"    WP{i} ({desc[:40]}) saved (after CLOSE)")
            if capture_gripper:
                save_cameras(scene, task_name, f'wp{i + 0.5:.1f}_gripper_close', task_out_dir, side_camera)
                print(f"    WP{i} gripper CLOSE screenshot saved")
        else:
            step(scene, SETTLE_STEPS)
            save_cameras(scene, task_name, f'wp{i}', task_out_dir, side_camera)
            print(f"    WP{i} ({desc[:40]}) saved")

    env_wrapper.shutdown()


def run_single(task_name, capture_gripper, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    try:
        process_task(task_name, capture_gripper, output_dir)
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)


def _run_one_task(task, capture_gripper, output_dir, timeout):
    """Run a single task in its own subprocess. Returns (task, status_string)."""
    cmd = [sys.executable, __file__, '--single', task, '--output-dir', output_dir]
    if capture_gripper:
        cmd.append('--gripper')
    try:
        result = subprocess.run(
            cmd, timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        status = "ok" if result.returncode == 0 else "error"
        return task, status, result.stdout
    except subprocess.TimeoutExpired as e:
        return task, "timeout", (e.stdout or "") if isinstance(e.stdout, str) else ""


def run_all(capture_gripper, output_dir, timeout, workers=8):
    os.makedirs(output_dir, exist_ok=True)
    tasks = get_all_tasks()
    workers = max(1, min(workers, len(tasks)))
    print(f"Processing {len(tasks)} tasks with {workers} parallel workers...")
    print(f"Output: {output_dir}")
    print(f"Gripper screenshots: {'on' if capture_gripper else 'off'}\n")

    # Each task already runs in its own subprocess, so a thread pool just supervises
    # up to `workers` of them at a time; a CoppeliaSim crash still skips only its task.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_one_task, task, capture_gripper, output_dir, timeout): task
            for task in tasks
        }
        for future in as_completed(futures):
            task, status, out = future.result()
            done += 1
            note = "" if status == "ok" else f"  ({status})"
            print(f"[{done}/{len(tasks)}] {task}{note}", flush=True)


def prompt_yes_no(question, default=True):
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"{question} {suffix}: ").strip().lower()
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please answer y or n.")


def prompt_task_choice(tasks):
    """Show a numbered task menu where 0 = all. Returns a task name, or None for all."""
    print("\nAvailable tasks:")
    print("   0. ALL tasks")
    for index, name in enumerate(tasks, start=1):
        print(f"  {index:>2}. {name}")
    while True:
        choice = input(f"\nSelect a task [0-{len(tasks)}] (0 = all, q to quit): ").strip()
        if choice.lower() in ("q", "quit", "exit"):
            print("No task selected. Exiting.")
            sys.exit(0)
        if choice.isdigit():
            number = int(choice)
            if number == 0:
                return None
            if 1 <= number <= len(tasks):
                return tasks[number - 1]
        print(f"Please enter a number between 0 and {len(tasks)}.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture per-waypoint camera screenshots for RLBench / failgen tasks."
    )
    parser.add_argument('--single', metavar='TASK',
                        help='Process one task in-process (worker mode). Default: all tasks.')
    parser.add_argument('--gripper', action='store_true',
                        help='Also save a screenshot right after each gripper open/close action.')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                        help=f'Where to write screenshots. Default: {DEFAULT_OUTPUT_DIR}')
    parser.add_argument('--timeout', type=int, default=TASK_TIMEOUT_SEC,
                        help='Per-task subprocess timeout in seconds (all-tasks mode).')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of tasks to process in parallel (all-tasks mode). Default: 8.')
    return parser.parse_args()


def main():
    args = parse_args()
    # Explicit / worker mode: a task named on the CLI (including the per-task
    # subprocesses run_all spawns with --single) runs directly, no prompts.
    if args.single:
        run_single(args.single, args.gripper, args.output_dir)
        return
    # Interactive mode (default, no task given): first ask whether to capture
    # the gripper open/close frames, then which task (0 = all), then capture.
    if sys.stdin.isatty():
        capture_gripper = prompt_yes_no(
            "Capture gripper open/close frames too?", default=True
        )
        task = prompt_task_choice(get_all_tasks())
        if task is None:
            run_all(capture_gripper, args.output_dir, args.timeout, args.workers)
        else:
            run_single(task, capture_gripper, args.output_dir)
        return
    # Non-interactive with no task: keep the original all-tasks behavior.
    run_all(args.gripper, args.output_dir, args.timeout, args.workers)


if __name__ == '__main__':
    main()
