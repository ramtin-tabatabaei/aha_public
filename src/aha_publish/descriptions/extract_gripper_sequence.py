"""Extract the AUTHORITATIVE per-waypoint gripper state for a task by REPLAYING
the demo in the simulator.

The gripper actuation timeline and the grasp/hold state live only in the scene
(per-waypoint ``open_gripper()``/``close_gripper()`` extensions) and in the
physics of whether ``gripper.grasp()`` actually latched an object. The
description generator can't see either, so it guesses -- which produces wrong
gripper_condition AND wrong object_in_gripper conditions downstream.

This replays each waypoint exactly like the BT runner (move the path, then run
the gripper extension, then latch graspables) and records, after each waypoint:
  - action:              open | close | none
  - state_after:         Open | Closed   (gripper jaws)
  - gripper_open_amount
  - held_after:          objects LATCHED by gripper.grasp(), i.e. registered graspables

Opt-in per-waypoint fields, all OFF by default (see EXTRA_FLAGS):
  --held-positions   held_positions:    world xyz of each latched object
  --contacts         contact_after:     respondable task shapes between the fingers
                     contact_positions: world xyz of each of them
  --touch-forces     touch_forces:      finger force-sensor readings
  --tip-positions    tip_position:      arm tip world xyz, plus tip_travel_m

``held_after`` on its own is NOT "what the robot is holding". ``Gripper.grasp(obj)``
is a question about ONE named object, and the only objects ever asked about are the
ones the task handed to ``register_graspable_objects``. Tasks whose interaction is
an articulated part register that part nowhere: change_clock registers nothing at
all yet turns a crank, take_tray_out_of_oven registers only the tray yet opens the
oven door, put_shoes_in_box registers only the shoes yet opens the box lid. All of
them report ``held_after: []`` while the fingers are clamped on something.

CoppeliaSim does know what is between the fingers, so ``--contacts`` asks the
gripper's own proximity sensor about every shape in the task tree, which names the
crank, the door handle and the lid. On change_clock it turns ``held_after: []`` into
``contact_after: ['clock_needle_crank']``, and on take_tray_out_of_oven wp1 into
``contact_after: ['oven_door']``.

Two things to know when reading ``contact_after``:
  - It means "inside the jaws", not "held". Combine it with ``state_after``: closed
    + contact is a grip on that object, open + contact is only the gripper being
    around or next to it.
  - The sensor's own single-object read (``simReadProximitySensor``, whose detected
    handle PyRep's ``ProximitySensor.read()`` discards) is deliberately not used: it
    reports the nearest detected object, which in practice is the gripper's own
    finger. Sweeping the task tree with ``is_detected`` skips robot geometry.

Output: outputs/gripper_sequences/<task>.json

Run inside the aha conda env with the CoppeliaSim env vars set. One task per
process (CoppeliaSim does not tear down cleanly across multiple resets in one
process), so more than one task fans out into subprocesses.

Usage:
    extract_gripper_sequence.py change_clock             # one or more named tasks
    extract_gripper_sequence.py --all --workers 4        # every configured task
    extract_gripper_sequence.py change_clock --contacts  # add an optional field
"""

from aha_publish import paths
import argparse
import concurrent.futures
import glob
import json
import subprocess
import sys
from math import sqrt
from pathlib import Path

REPO_ROOT = (paths.PROJECT_ROOT)
OUT_DIR = (paths.GRIPPER_DIR)
CONFIGS_PATH = (paths.FAILGEN_ROOT / 'failgen/configs')

PATH_MAX_STEPS = 600
GRIP_MAX_STEPS = 150
WARMUP_STEPS = 50
SETTLE_STEPS = 20
GRIP_SETTLE_STEPS = 10

# actuate_settled(): "close enough to the commanded opening", "this step moved
# nothing" and "how many such steps mean a real stall" (0.04 m of finger travel).
OPEN_AMOUNT_TOL = 0.02
STALL_EPS = 0.002
STALL_STEPS = 5

# A run where the tip never moves this far produced no trajectory at all, so its
# whole sequence (every waypoint "closed on nothing") is meaningless.
MIN_TIP_TRAVEL_M = 0.01

# Optional per-waypoint fields, all off by default: the written record stays the
# gripper timeline (action / state_after / gripper_open_amount / held_after).
EXTRA_FLAGS = {
    "contacts": "--contacts",
    "held_positions": "--held-positions",
    "touch_forces": "--touch-forces",
    "tip_positions": "--tip-positions",
}


def get_all_tasks():
    return sorted(Path(p).stem for p in glob.glob(str(CONFIGS_PATH / "*.yaml")))


def classify(ext: str):
    e = (ext or "").lower()
    if "open_gripper" in e:
        return "open"
    if "close_gripper" in e:
        return "close"
    return "none"


def waypoint_ext(wp):
    try:
        return wp.get_ext()
    except Exception:
        pass
    try:
        return wp._waypoint.get_extension_string()
    except Exception:
        return ""


def step_scene(scene, steps):
    for _ in range(steps):
        scene.pyrep.step()


def world_position(obj):
    try:
        return [round(float(v), 6) for v in obj.get_position()]
    except Exception:
        return None


def detected_shapes(gripper, task_shapes):
    """Task shapes the gripper's proximity sensor currently sees between the fingers.

    ``grasp()`` only ever asks about the task's registered graspables, so this asks
    the same sensor about every shape in the task tree instead. Robot geometry is
    excluded for free: the sweep never leaves the task's own model tree.

    Only respondable shapes count as contact -- the fingers pass straight through
    anything else. That also drops the two kinds of ghost the sensor otherwise
    reports: the visual twin sitting on top of each physical shape
    (clock_needle_crank_visual) and success/detector markers (success_visual).
    """
    sensor = gripper._proximity_sensor  # PyRep keeps it private; there is no accessor
    found = []
    for shape in task_shapes:
        try:
            if shape.is_respondable() and sensor.is_detected(shape):
                found.append(shape)
        except Exception:
            continue
    return found


def open_amount(gripper):
    """Mean finger opening, 1.0 fully open and 0.0 fully closed."""
    try:
        amounts = gripper.get_open_amount()
        return float(sum(amounts) / len(amounts)) if amounts else None
    except Exception:
        return None


def actuate_settled(scene, gripper, amount, velocity=0.04):
    """Drive the fingers to ``amount``, ignoring actuate()'s premature "done".

    ``Gripper.actuate`` reports itself finished the moment a joint moves less than
    POSITION_ERROR (1 mm) between two calls, so a single slow step -- friction
    while the fingers slide off an object, the velocity ramp at the start of the
    motion -- stops the gripper half way. The recorded gripper_open_amount then
    describes where the actuation happened to give up (0.48 on a commanded OPEN),
    not the state the waypoint commanded.

    So keep driving until the opening reaches the commanded value or genuinely
    stops changing for several consecutive steps -- the latter being a real stall,
    e.g. fingers closing onto an object.
    """
    stalled = 0
    for _ in range(GRIP_MAX_STEPS):
        before = open_amount(gripper)
        gripper.actuate(amount, velocity)
        scene.pyrep.step()
        current = open_amount(gripper)
        if current is None:
            break
        if abs(current - amount) < OPEN_AMOUNT_TOL:
            break
        stalled = stalled + 1 if abs(current - (before or current)) < STALL_EPS else 0
        if stalled >= STALL_STEPS:
            break


def latch_graspables(scene, gripper, graspables):
    """Attach whichever registered graspables are between the fingers.

    Retried once: the fingers can still be settling when the first attempt fires,
    and a missed detection is unrecoverable (nothing re-latches at later waypoints).
    """
    for attempt in range(2):
        for obj in graspables:
            try:
                gripper.grasp(obj)
            except Exception:
                pass
        if gripper.get_grasped_objects() or attempt:
            break
        step_scene(scene, GRIP_SETTLE_STEPS)


def extract(task_name: str, headless: bool = True, extras=frozenset()) -> dict:
    from failgen.env_wrapper import FailGenEnvWrapper
    from pyrep.const import ObjectType

    env = FailGenEnvWrapper(
        task_name=task_name, headless=headless, record=False,
        save_data=False, save_path=str(paths.BACKEND_DATA_DIR / 'gripper_sequences'), no_failures=True)
    try:
        env.reset()
        scene = env._env._scene
        robot = scene.robot
        gripper = robot.gripper
        task = scene.task

        # The env is built with JointVelocity, whose set_control_mode() disables the
        # arm control loop and locks the motors at zero velocity. path.step() only
        # writes joint TARGET POSITIONS, which are ignored in that state, so the arm
        # never leaves home and every grasp silently misses. RLBench's own
        # get_demos()/get_failures() enable the loop around live rollouts and restore
        # it afterwards; this replay has to do the same.
        control_loop = robot.arm.joints[0].is_control_loop_enabled()
        robot.arm.set_control_loop_enabled(True)
        step_scene(scene, WARMUP_STEPS)

        graspables = task.get_graspable_objects()
        # Read every name while the sim is still up: the result dict is assembled
        # after the finally-block shutdown, where object handles resolve to ''.
        graspable_names = [g.get_name() for g in graspables]
        task_shapes = task.get_base().get_objects_in_tree(
            object_type=ObjectType.SHAPE, exclude_base=False)
        tip = robot.arm.get_tip()

        state = "Open"
        entries = []
        tip_positions = []
        for i, wp in enumerate(task.get_waypoints()):
            # 1) move the path
            wp.start_of_path()
            if not getattr(wp, "skip", False):
                try:
                    path = wp.get_path()
                    done = False
                    steps = 0
                    while not done and steps < PATH_MAX_STEPS:
                        done = path.step()
                        scene.step()
                        steps += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"  [warn] wp{i} path: {exc}", flush=True)
            wp.end_of_path()
            step_scene(scene, SETTLE_STEPS)

            # 2) run the gripper extension exactly like the runner
            ext = waypoint_ext(wp)
            action = classify(ext)
            if action == "open":
                gripper.release()
                actuate_settled(scene, gripper, 1.0)
                state = "Open"
                step_scene(scene, GRIP_SETTLE_STEPS)
            elif action == "close":
                actuate_settled(scene, gripper, 0.0)
                state = "Closed"
                step_scene(scene, GRIP_SETTLE_STEPS)
                latch_graspables(scene, gripper, graspables)

            # 3) record the TRUE held set
            held, held_positions = [], {}
            try:
                for obj in gripper.get_grasped_objects():
                    name = obj.get_name()
                    held.append(name)
                    position = world_position(obj)
                    if position is not None:
                        held_positions[name] = position
            except Exception:
                held, held_positions = [], {}

            entry = {
                "index": i,
                "extension": (ext or "").strip(),
                "action": action,
                "gripper_open_amount": open_amount(gripper),
                "state_after": state,
                "held_after": held,
            }
            if "held_positions" in extras:
                entry["held_positions"] = held_positions
            if "contacts" in extras:
                contacts = detected_shapes(gripper, task_shapes)
                entry["contact_after"] = [s.get_name() for s in contacts]
                entry["contact_positions"] = {
                    s.get_name(): world_position(s)
                    for s in contacts if world_position(s) is not None
                }
            if "touch_forces" in extras:
                try:
                    entry["touch_forces"] = [
                        [round(float(c), 4) for c in f]
                        for f in gripper.get_touch_sensor_forces()
                    ]
                except Exception:  # gripper has no touch sensors
                    entry["touch_forces"] = None

            # Read regardless of the flag: the did-the-arm-move guard needs it. Only
            # the per-waypoint value is optional.
            tip_position = world_position(tip)
            tip_positions.append(tip_position)
            if "tip_positions" in extras:
                entry["tip_position"] = tip_position
            entries.append(entry)

        robot.arm.set_control_loop_enabled(control_loop)
    finally:
        try:
            env.shutdown()
        except Exception:
            pass

    return {
        "task": task_name,
        "initial_state": "Open",
        "n_waypoints": len(entries),
        "graspable_objects": graspable_names,
        "source": "extract_gripper_sequence (replayed trajectory)",
        "tip_travel_m": tip_travel(tip_positions),
        "waypoints": entries,
    }


def tip_travel(positions) -> float:
    """Largest tip displacement from the first waypoint, as a did-the-arm-move check."""
    positions = [p for p in positions if p]
    if len(positions) < 2:
        return 0.0
    start = positions[0]
    return round(max(
        sqrt(sum((p[k] - start[k]) ** 2 for k in range(3))) for p in positions[1:]), 4)


def summarize(data: dict) -> str:
    parts = []
    for e in data["waypoints"]:
        line = (f"wp{e['index']}:{e['action']}/{e['state_after']}"
                f"/held={','.join(e['held_after']) or '-'}")
        if "contact_after" in e:
            line += f"/contact={','.join(e['contact_after']) or '-'}"
        parts.append(line)
    return " ".join(parts)


def capture_task(task_name: str, headless: bool, extras=frozenset()) -> bool:
    try:
        data = extract(task_name, headless=headless, extras=extras)
    except Exception as exc:  # noqa: BLE001
        print(f"{task_name}: EXTRACT FAILED: {exc}", flush=True)
        return False
    if not data["waypoints"]:
        print(f"{task_name}: NO WAYPOINTS -- not written", flush=True)
        return False
    # Guard the failure mode that silently produced a whole generation of files in
    # which nothing is ever held: an arm that never moved closes on empty air at
    # every waypoint, and the result still looks like a well-formed sequence.
    if data["tip_travel_m"] < MIN_TIP_TRAVEL_M:
        print(f"{task_name}: ARM NEVER MOVED (tip travel {data['tip_travel_m']} m) "
              f"-- not written; the capture is invalid, not the task", flush=True)
        return False
    if "tip_positions" not in extras:
        data.pop("tip_travel_m")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{task_name}.json"
    out.write_text(json.dumps(data, indent=2))
    print(f"{task_name}: {summarize(data)}  -> {out}", flush=True)
    return True


def worker_command(task_name: str, args) -> list:
    command = [sys.executable, __file__, task_name]
    if not args.headless:
        command.append("--show-simulator")
    command += [EXTRA_FLAGS[name] for name in sorted(args.extras)]
    return command


def run_in_parallel(tasks, args) -> list:
    workers = max(1, min(args.workers, len(tasks)))
    print(f"Capturing {len(tasks)} tasks with {workers} workers", flush=True)
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(subprocess.run, worker_command(t, args)): t for t in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            task_name = futures[future]
            if future.result().returncode != 0:
                failures.append(task_name)
    return failures


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tasks", nargs="*", help="Task names to capture.")
    parser.add_argument("--all", action="store_true",
                        help="Capture every task with a failgen config.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Tasks to run at once (one subprocess each).")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--show-simulator", action="store_false", dest="headless")
    parser.add_argument("--contacts", action="store_true",
                        help="Also record contact_after/contact_positions: the "
                             "respondable task shapes between the fingers, which is "
                             "the only way unregistered knobs, handles and lids get "
                             "named (held_after structurally cannot show them).")
    parser.add_argument("--held-positions", action="store_true", dest="held_positions",
                        help="Also record the world xyz of each latched object.")
    parser.add_argument("--touch-forces", action="store_true", dest="touch_forces",
                        help="Also record the finger force-sensor readings.")
    parser.add_argument("--tip-positions", action="store_true", dest="tip_positions",
                        help="Also record the per-waypoint arm tip xyz and tip_travel_m.")
    parser.add_argument("--extras", action="store_true",
                        help="Shorthand for every optional field above.")
    args = parser.parse_args()
    if not args.tasks and not args.all:
        parser.error("give one or more task names, or --all")
    args.extras = (set(EXTRA_FLAGS) if args.extras
                   else {name for name in EXTRA_FLAGS if getattr(args, name)})
    return args


def main():
    args = parse_args()
    tasks = get_all_tasks() if args.all else args.tasks

    # CoppeliaSim does not tear down cleanly across resets, so a single task runs
    # here and anything more fans out one subprocess per task.
    if len(tasks) == 1:
        sys.exit(0 if capture_task(tasks[0], args.headless, args.extras) else 1)

    failures = run_in_parallel(tasks, args)
    if failures:
        print("\nFailed tasks:", flush=True)
        for task_name in failures:
            print(f"  - {task_name}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
