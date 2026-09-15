"""Build all clean-run detector baselines from one shared simulation pass.

For each task/episode this runs one clean RLBench demo and records telemetry for:
orientation, transition, slip/grip-force, collision/torque, and freezing. The
same observations are then written into one calibration output tree.

Example:
  python -u \
      aha_scripts/main_bt_run/calibrate_all_clean.py \
      --episodes 10 --workers 10 --force
"""

from aha_publish import paths
import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = (paths.SOURCE_DIR / 'calibration')
ROOT = (paths.PROJECT_ROOT)
FAILGEN_ROOT = (paths.FAILGEN_ROOT)
CONFIGS_PATH = FAILGEN_ROOT / "failgen" / "configs"
COPPELIA = paths.COPPELIASIM_ROOT
DEFAULT_CALIBRATION_ROOT = (paths.CALIBRATION_DIR)

CALIBRATION_ROOT = DEFAULT_CALIBRATION_ROOT
ORIENT_DRIFT = CALIBRATION_ROOT / "orientation_eval" / "drift_logs"
TRANS_DRIFT = CALIBRATION_ROOT / "transition_eval" / "drift_logs"
ORIENT_STATS = CALIBRATION_ROOT / "orientation_arrival_stats"
TRANS_STATS = CALIBRATION_ROOT / "transition_arrival_stats"
GRIP_STATS = CALIBRATION_ROOT / "grip_force_stats"
TORQUE_STATS = CALIBRATION_ROOT / "torque_stats"
FREEZE_STATS = CALIBRATION_ROOT / "freezing_stats"
# Step-1 output: one unified raw per-frame CSV per (task, episode) holding ALL
# four detector signals aligned by frame, so thresholds can be (re)derived
# offline by compute_thresholds.py without re-running the simulator.
RAW_DIR = CALIBRATION_ROOT / "clean_data"

MIN_GRIP_FORCE_FOR_STATS = 0.2


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EO = _load_module("eval_orientation_detector_for_all_clean", (paths.SOURCE_DIR / 'calibration/eval_orientation_detector.py'))
ET = _load_module("eval_transition_detector_for_all_clean", (paths.SOURCE_DIR / 'calibration/eval_transition_detector.py'))
OD = _load_module("orientation_detector_for_all_clean", (paths.SOURCE_DIR / 'detectors/orientation/detector.py'))
TD = _load_module("transition_detector_for_all_clean", (paths.SOURCE_DIR / 'detectors/transition/detector.py'))
SD = _load_module("slip_detector_for_all_clean", (paths.SOURCE_DIR / 'detectors/slip/detector.py'))
from aha_publish.detectors.collision import detector as CD
from aha_publish.calibration import residuals
FD = _load_module("freezing_detector_for_all_clean", (paths.SOURCE_DIR / 'detectors/freezing/detector.py'))
GF = _load_module("grip_force_stats_for_all_clean", (paths.SOURCE_DIR / 'detectors/grip_force_stats/run.py'))
TS = _load_module("torque_stats_for_all_clean", (paths.SOURCE_DIR / 'detectors/torque_stats/run.py'))
FA = _load_module("freezing_analyze_for_all_clean", (paths.SOURCE_DIR / 'detectors/freezing/calibration/analyze.py'))


def configure_output_root(out_root):
    """Point every calibration artifact at one output tree."""
    global CALIBRATION_ROOT, ORIENT_DRIFT, TRANS_DRIFT, ORIENT_STATS
    global TRANS_STATS, GRIP_STATS, TORQUE_STATS, FREEZE_STATS, RAW_DIR

    CALIBRATION_ROOT = Path(out_root).expanduser().resolve()
    ORIENT_DRIFT = CALIBRATION_ROOT / "orientation_eval" / "drift_logs"
    TRANS_DRIFT = CALIBRATION_ROOT / "transition_eval" / "drift_logs"
    ORIENT_STATS = CALIBRATION_ROOT / "orientation_arrival_stats"
    TRANS_STATS = CALIBRATION_ROOT / "transition_arrival_stats"
    GRIP_STATS = CALIBRATION_ROOT / "grip_force_stats"
    TORQUE_STATS = CALIBRATION_ROOT / "torque_stats"
    FREEZE_STATS = CALIBRATION_ROOT / "freezing_stats"
    RAW_DIR = CALIBRATION_ROOT / "clean_data"

    # The helper modules compute baselines from their own module-level paths.
    # Keep those in lockstep with the paths where this script writes drift logs.
    EO.OUT_DIR = str(CALIBRATION_ROOT / "orientation_eval")
    EO.DRIFT_DIR = str(ORIENT_DRIFT)
    EO.STATS_DIR = str(ORIENT_STATS)
    ET.OUT_DIR = str(CALIBRATION_ROOT / "transition_eval")
    ET.DRIFT_DIR = str(TRANS_DRIFT)
    ET.STATS_DIR = str(TRANS_STATS)


def configure_env(headless=True):
    os.environ["COPPELIASIM_ROOT"] = str(COPPELIA)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(COPPELIA)
    if headless:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    parts = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    if str(COPPELIA) not in parts:
        parts.append(str(COPPELIA))
        os.environ["LD_LIBRARY_PATH"] = ":".join(parts)
        if os.environ.get("AHA_ALL_CLEAN_ENV_READY") != "1":
            os.environ["AHA_ALL_CLEAN_ENV_READY"] = "1"
            os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)


def available_tasks():
    return sorted(p.stem for p in CONFIGS_PATH.glob("*.yaml"))


def waypoint_index_from_point(point):
    try:
        name = point._waypoint.get_name()
        if name.startswith("waypoint"):
            return int(name[len("waypoint"):])
    except Exception:
        return None
    return None


def waypoint_pose(index):
    if index is None:
        return None
    try:
        from pyrep.objects.object import Object

        obj = Object.get_object(f"waypoint{int(index)}")
        # Cartesian-path waypoints: the object's pose is the path frame origin
        # (~mid-path), but the arm drives to the path END. Resolve to the last
        # sample (relative_distance=1.0), converting Euler->quat via a scratch
        # dummy so the result matches get_pose()'s [x,y,z,qx,qy,qz,qw] layout.
        try:
            from pyrep.const import ObjectType
            from pyrep.objects.cartesian_path import CartesianPath

            if obj.get_type() == ObjectType.PATH:
                position, euler = CartesianPath(
                    obj.get_handle()).get_pose_on_path(1.0)
                from pyrep.objects.dummy import Dummy

                scratch = Dummy.create()
                try:
                    scratch.set_position(list(position))
                    scratch.set_orientation(list(euler))
                    return list(scratch.get_pose())
                finally:
                    scratch.remove()
        except Exception:
            pass
        return obj.get_pose()
    except Exception:
        return None


def summarize(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan")}
    return {"count": int(arr.size), "mean": float(arr.mean()),
            "std": float(arr.std()), "min": float(arr.min()),
            "max": float(arr.max())}


def mean_of_episode_stats(stats):
    if not stats:
        return summarize([])
    return {
        "count": len(stats),
        "mean": float(np.mean([s["mean"] for s in stats])),
        "std": float(np.mean([s["std"] for s in stats])),
        "min": float(np.mean([s["min"] for s in stats])),
        "max": float(np.mean([s["max"] for s in stats])),
    }


def metric_task_json(task, episodes_requested, episode_stats, aggregate_rows,
                     metric_names, aggregate_key):
    metrics = {}
    completed = 0
    for metric in metric_names:
        eps = [
            {"episode": i, "stats": stats[metric]}
            for i, stats in enumerate(episode_stats)
            if metric in stats
        ]
        completed = max(completed, len(eps))
        metrics[metric] = {
            "episodes": eps,
            "average_of_episode_stats": mean_of_episode_stats(
                [e["stats"] for e in eps]),
            aggregate_key: summarize([row[metric] for row in aggregate_rows]),
        }
    return {
        "task": task,
        "episodes_requested": episodes_requested,
        "episodes_completed": completed,
        "metrics": metrics,
    }


def grip_task_json(task, episodes_requested, episode_stats, aggregate_rows):
    data = metric_task_json(
        task, episodes_requested, episode_stats, aggregate_rows,
        GF.METRICS, "aggregate_filtered_frames")
    data["filter"] = {
        "description": (
            "Only rows with grip_force greater than min_grip_force_for_stats "
            "are included in statistics."
        ),
        "min_grip_force_for_stats": MIN_GRIP_FORCE_FOR_STATS,
    }
    return data


def torque_task_json(task, episodes_requested, episode_stats, aggregate_rows):
    return metric_task_json(
        task, episodes_requested, episode_stats, aggregate_rows,
        TS.METRICS, "aggregate_all_frames")


def freezing_task_json(task, episodes_requested, clean_rows):
    raw = {
        "task": task,
        "clean": [
            {"jvn": row["joint_velocity_norm"],
             "jpd": row["joint_position_delta"],
             "cam": row.get("camera_motion", float("nan"))}
            for row in clean_rows
        ],
    }
    data = FA.build_stats(raw)
    data["n_clean_episodes"] = int(episodes_requested)
    return data


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_drift_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


RAW_HEADER = [
    "task", "episode", "waypoint", "local_step", "global_step", "path_done",
    "distance_m",           # transition signal: distance to waypoint pose (m)
    "distance_runup",       # cumulative positive rise of distance in this segment
    "angle_rad",            # orientation signal: angle to waypoint pose (rad)
    "angle_runup",          # cumulative positive rise of angle in this segment
    "torque_norm",          # collision signal
    "torque_delta",
    "grip_force",           # slip signal
    "grip_force_drop",
]


def write_raw_csv(path, task, episode, transition_rows, orientation_rows,
                  torque_logs, grip_logs):
    """Merge the four per-frame signal streams (all logged from the same
    on_env_step frames, keyed by global_step) into one raw CSV. Frames where a
    signal was NaN/absent are left blank so nothing is silently imputed; the
    offline threshold step decides how to treat them."""
    frames = {}

    def slot(gstep):
        return frames.setdefault(int(gstep), {
            "waypoint": "", "local_step": "", "path_done": "",
            "distance_m": "", "distance_runup": "",
            "angle_rad": "", "angle_runup": "",
            "torque_norm": "", "torque_delta": "",
            "grip_force": "", "grip_force_drop": "",
        })

    # transition_rows / orientation_rows: [task, wp, local_step, gstep, val,
    #   runup, positive, extra, path_done]
    for r in transition_rows:
        s = slot(r[3])
        s["waypoint"], s["local_step"], s["path_done"] = r[1], r[2], r[8]
        s["distance_m"], s["distance_runup"] = r[4], r[5]
    for r in orientation_rows:
        s = slot(r[3])
        if s["waypoint"] == "":
            s["waypoint"], s["local_step"], s["path_done"] = r[1], r[2], r[8]
        s["angle_rad"], s["angle_runup"] = r[4], r[5]
    for r in torque_logs:
        s = slot(r["step"])
        s["torque_norm"] = r.get("torque_norm", "")
        s["torque_delta"] = r.get("torque_delta", "")
    for r in grip_logs:
        s = slot(r["step"])
        s["grip_force"] = r.get("grip_force", "")
        s["grip_force_drop"] = r.get("grip_force_drop", "")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(RAW_HEADER)
        for gstep in sorted(frames):
            s = frames[gstep]
            writer.writerow([
                task, episode, s["waypoint"], s["local_step"], gstep,
                s["path_done"], s["distance_m"], s["distance_runup"],
                s["angle_rad"], s["angle_runup"], s["torque_norm"],
                s["torque_delta"], s["grip_force"], s["grip_force_drop"],
            ])


def run_task(task, episodes, max_attempts, show_sim=False):
    if str(FAILGEN_ROOT) not in sys.path:
        sys.path.insert(0, str(FAILGEN_ROOT))
    from failgen.env_wrapper import FailGenEnvWrapper

    configure_env(headless=not show_sim)
    for path in (ORIENT_DRIFT, TRANS_DRIFT, ORIENT_STATS, TRANS_STATS,
                 GRIP_STATS, TORQUE_STATS, FREEZE_STATS):
        path.mkdir(parents=True, exist_ok=True)
    for old in list(ORIENT_DRIFT.glob(f"{task}.clean.ep*.csv")):
        old.unlink()
    for old in list(TRANS_DRIFT.glob(f"{task}.clean.ep*.csv")):
        old.unlink()
    # Remove this task's previous raw dumps so a re-run never mixes episodes.
    raw_task_dir = RAW_DIR / task
    if raw_task_dir.exists():
        for old in raw_task_dir.glob("ep*.csv"):
            old.unlink()
        for old in raw_task_dir.glob("ep*.residual.json"):
            old.unlink()

    env_wrapper = FailGenEnvWrapper(
        task_name=task,
        headless=not show_sim,
        record=False,
        save_data=False,
        no_failures=True,
        save_path=str(paths.BACKEND_DATA_DIR / 'calibration'),
        save_keyframes_only=True,
        max_failure_attempts=max_attempts,
    )

    all_grip_rows = []
    all_torque_rows = []
    all_freeze_rows = []
    grip_episode_stats = []
    torque_episode_stats = []
    completed = 0
    target = max(1, int(episodes))
    attempts = 0
    # Some demos fail to produce a successful trajectory; keep retrying until the
    # task reaches `target` successful clean episodes (capped so a task that can
    # never succeed cannot loop forever).
    attempt_cap = target * max(3, int(max_attempts))

    try:
        while completed < target and attempts < attempt_cap:
            attempts += 1
            episode = completed
            state = {
                "waypoint": None,
                "original_pose": None,
                "current_pose": None,
                "next_waypoint": 0,
                "global_step": 0,
                "local_step": 0,
                "orientation_logs": [],
                "transition_logs": [],
                "orientation_prev": float("nan"),
                "transition_prev": float("nan"),
                "orientation_runup": 0.0,
                "transition_runup": 0.0,
            }
            orientation_rows = []
            transition_rows = []
            grip_logs = []
            torque_logs = []
            freeze_logs = []

            original_step = env_wrapper.on_env_step
            original_waypoint = env_wrapper.on_env_waypoint
            original_waypoint_end = env_wrapper.on_env_waypoint_end

            def reset_waypoint(idx):
                state["waypoint"] = idx
                state["local_step"] = 0
                state["orientation_logs"] = []
                state["transition_logs"] = []
                state["orientation_prev"] = float("nan")
                state["transition_prev"] = float("nan")
                state["orientation_runup"] = 0.0
                state["transition_runup"] = 0.0

            def feed(obs, path_done=False):
                idx = state["waypoint"]
                if idx is None:
                    return
                global_step = state["global_step"]
                local_step = state["local_step"]
                state["global_step"] += 1
                state["local_step"] += 1

                current_pose = state["current_pose"]
                if current_pose is None:
                    current_pose = waypoint_pose(idx)
                original_pose = state["original_pose"]

                o_row = OD.obs_to_row(
                    obs, global_step, waypoint=idx, expected_waypoint=idx,
                    waypoint_path_done=path_done, waypoint_pose=current_pose,
                    current_waypoint_pose=current_pose,
                    ttm_waypoint_pose=original_pose,
                    waypoint_started=True, started_waypoint=idx)
                state["orientation_logs"].append(o_row)
                OD.update_deltas(state["orientation_logs"])
                # Record the gripper-vs-CORRECT (original/pre-injection)
                # orientation angle -- ttm_waypoint_angle -- the SAME signal the
                # live detector's C1 arrival and C4 prop now judge. On clean runs
                # this equals the old gripper-vs-commanded angle at static
                # waypoints (no threshold change) but is larger at dynamic
                # ride-along waypoints, so their calibrated arrival/prop bounds
                # rise / abstain instead of the detector false-firing there.
                angle = float(o_row.get("ttm_waypoint_angle", float("nan")))
                if not np.isfinite(angle):
                    angle = float(o_row.get("live_waypoint_angle", float("nan")))
                if np.isfinite(angle):
                    prev = state["orientation_prev"]
                    positive = 0
                    if np.isfinite(prev):
                        inc = max(0.0, angle - prev)
                        if inc > 0.0:
                            state["orientation_runup"] += inc
                            positive = 1
                    state["orientation_prev"] = angle
                    god = o_row.get("gripper_orientation_delta", float("nan"))
                    orientation_rows.append([
                        task, idx, local_step, global_step, angle,
                        state["orientation_runup"], positive, god,
                        int(bool(path_done)),
                    ])

                t_row = TD.obs_to_row(
                    obs, global_step, waypoint=idx, expected_waypoint=idx,
                    waypoint_path_done=path_done, waypoint_pose=current_pose,
                    current_waypoint_pose=current_pose,
                    ttm_waypoint_pose=original_pose,
                    waypoint_started=True, started_waypoint=idx)
                state["transition_logs"].append(t_row)
                TD.update_deltas(state["transition_logs"])
                dist = float(t_row.get("live_waypoint_distance", float("nan")))
                if np.isfinite(dist):
                    prev = state["transition_prev"]
                    positive = 0
                    if np.isfinite(prev):
                        inc = max(0.0, dist - prev)
                        if inc > 0.0:
                            state["transition_runup"] += inc
                            positive = 1
                    state["transition_prev"] = dist
                    gm = t_row.get("gripper_motion", float("nan"))
                    transition_rows.append([
                        task, idx, local_step, global_step, dist,
                        state["transition_runup"], positive, gm,
                        int(bool(path_done)),
                    ])

            def patched_waypoint(point):
                idx = waypoint_index_from_point(point)
                if idx is None:
                    idx = state["next_waypoint"]
                state["next_waypoint"] = max(state["next_waypoint"], int(idx) + 1)
                reset_waypoint(idx)
                state["original_pose"] = waypoint_pose(idx)
                original_waypoint(point)
                state["current_pose"] = waypoint_pose(idx)

            def patched_step(obs):
                original_step(obs)
                step = state["global_step"]

                s_row = SD.obs_to_row(obs, step)
                s_row["task"] = task
                s_row["episode"] = episode
                grip_logs.append(s_row)
                SD.update_deltas(grip_logs)

                c_row = CD.obs_to_row(obs, step)
                c_row["task"] = task
                c_row["episode"] = episode
                torque_logs.append(c_row)
                CD.update_deltas(torque_logs)

                f_row = FD.obs_to_row(obs, step, camera_names=())
                f_row["task"] = task
                f_row["episode"] = episode
                freeze_logs.append(f_row)
                FD.update_deltas(freeze_logs)

                feed(obs, path_done=False)

            def patched_waypoint_end(point):
                idx = waypoint_index_from_point(point)
                if state["waypoint"] is None:
                    reset_waypoint(idx)
                original_waypoint_end(point)
                try:
                    obs = env_wrapper._task_env.get_observation()
                    feed(obs, path_done=True)
                except Exception:
                    pass
                state["waypoint"] = None
                state["original_pose"] = None
                state["current_pose"] = None

            env_wrapper.on_env_waypoint = patched_waypoint
            env_wrapper.on_env_step = patched_step
            env_wrapper.on_env_waypoint_end = patched_waypoint_end

            print(f"    {task} ep{episode + 1}/{episodes} running...", flush=True)
            try:
                env_wrapper.reset()
                # Wire the waypoint-START hook so per-frame in-motion callbacks
                # know which waypoint is active; without it feed() drops every
                # in-motion frame (state["waypoint"] stays None) and only the
                # path_done arrival frame survives -> empty per-waypoint baselines.
                demo = env_wrapper.get_success(
                    callable_each_waypoint=patched_waypoint)
            finally:
                env_wrapper.on_env_step = original_step
                env_wrapper.on_env_waypoint = original_waypoint
                env_wrapper.on_env_waypoint_end = original_waypoint_end

            if demo is None:
                print(f"    {task}: attempt {attempts} no successful demo "
                      f"(have {completed}/{target})", flush=True)
                continue

            write_drift_csv(
                ORIENT_DRIFT / f"{task}.clean.ep{episode}.csv",
                ["task", "waypoint", "local_step", "step", "angle_to_target",
                 "runup", "runup_positive", "gripper_orientation_delta", "path_done"],
                orientation_rows)
            write_drift_csv(
                TRANS_DRIFT / f"{task}.clean.ep{episode}.csv",
                ["task", "waypoint", "local_step", "step", "dist_to_target",
                 "runup", "runup_positive", "gripper_motion", "path_done"],
                transition_rows)
            # Unified raw per-frame dump (all four signals) for offline
            # threshold derivation by compute_thresholds.py.
            if CD.DEFAULT_COLLISION_METHOD == '3':
                residuals.write_episode(RAW_DIR / task / f"ep{episode}.csv",
                                        task, episode, torque_logs, CD)
            write_raw_csv(
                RAW_DIR / task / f"ep{episode}.csv",
                task, episode, transition_rows, orientation_rows,
                torque_logs, grip_logs)

            filtered_grip = [
                row for row in grip_logs
                if row.get("grip_force", 0.0) > MIN_GRIP_FORCE_FOR_STATS
            ]
            grip_episode_stats.append({
                metric: summarize([row[metric] for row in filtered_grip])
                for metric in GF.METRICS
            })
            torque_episode_stats.append({
                metric: summarize([row[metric] for row in torque_logs])
                for metric in TS.METRICS
            })
            all_grip_rows.extend(filtered_grip)
            all_torque_rows.extend(torque_logs)
            all_freeze_rows.extend(freeze_logs)
            completed += 1
            print(
                f"    {task} ep{completed}/{target} done (attempt {attempts}): "
                f"frames={len(torque_logs)} grip_filtered={len(filtered_grip)}",
                flush=True,
            )
    finally:
        env_wrapper.shutdown()

    if completed < target:
        print(f"    WARNING {task}: only {completed}/{target} clean episodes "
              f"after {attempts} attempts (cap {attempt_cap})", flush=True)
    if completed <= 0:
        raise RuntimeError(f"{task}: no clean episodes completed")

    orient_base = EO._compute_clean_baseline(task, completed)
    trans_base = ET._compute_clean_baseline(task, completed)
    write_json(ORIENT_STATS / f"{task}.json", orient_base)
    write_json(TRANS_STATS / f"{task}.json", trans_base)
    write_json(
        GRIP_STATS / f"{task}_success_grip_force_stats.json",
        grip_task_json(task, completed, grip_episode_stats, all_grip_rows))
    write_json(
        TORQUE_STATS / f"{task}_success_torque_stats.json",
        torque_task_json(task, completed, torque_episode_stats, all_torque_rows))
    write_json(FREEZE_STATS / f"{task}.json", freezing_task_json(task, completed, all_freeze_rows))

    return {
        "task": task,
        "episodes_completed": completed,
        "orientation_wps": len(orient_base.get("arrival_stats_by_waypoint", {})),
        "transition_wps": len(trans_base.get("arrival_stats_by_waypoint", {})),
        "grip_rows": len(all_grip_rows),
        "torque_rows": len(all_torque_rows),
        "freeze_rows": len(all_freeze_rows),
    }


def clean_episode_count(task):
    """Number of clean raw-dump episodes already on disk for a task
    (<out-root>/clean_data/<task>/ep*.csv)."""
    d = RAW_DIR / task
    if not d.exists():
        return 0
    return sum(1 for p in d.glob("ep*.csv") if p.stem[2:].isdigit())


def task_done(task, episodes):
    if CD.DEFAULT_COLLISION_METHOD == '3':
        try:
            _, count = residuals.load_envelopes(RAW_DIR, task, residuals.observer_parameters(CD))
            if count < episodes:
                return False
        except (ValueError, OSError, TypeError):
            return False
    paths = [
        ORIENT_STATS / f"{task}.json",
        TRANS_STATS / f"{task}.json",
        GRIP_STATS / f"{task}_success_grip_force_stats.json",
        TORQUE_STATS / f"{task}_success_torque_stats.json",
        FREEZE_STATS / f"{task}.json",
    ]
    if not all(path.exists() for path in paths):
        return False
    try:
        for path in paths[:2]:
            data = json.loads(path.read_text())
            if data.get("n_clean_episodes", 0) < episodes:
                return False
            # An arrival baseline with an empty per-waypoint map is a failed
            # calibration (no in-motion frames were parsed): treat it as NOT
            # done so --skip-completed re-runs it instead of keeping the stub.
            if not data.get("arrival_stats_by_waypoint"):
                return False
        for path in paths[2:4]:
            if json.loads(path.read_text()).get("episodes_completed", 0) < episodes:
                return False
        if json.loads(paths[4].read_text()).get("n_clean_episodes", 0) < episodes:
            return False
    except Exception:
        return False
    return True


def rebuild_all_task_summaries(tasks, episodes, failed):
    completed_grip = [
        task for task in tasks
        if (GRIP_STATS / f"{task}_success_grip_force_stats.json").exists()
    ]
    if completed_grip:
        write_json(
            GRIP_STATS / "ALL_TASKS_success_grip_force_stats.json",
            GF.all_tasks_stats_json(GRIP_STATS, tasks, episodes, failed))
    completed_torque = [
        task for task in tasks
        if (TORQUE_STATS / f"{task}_success_torque_stats.json").exists()
    ]
    if completed_torque:
        write_json(
            TORQUE_STATS / "ALL_TASKS_success_torque_stats.json",
            TS.all_tasks_stats_json(TORQUE_STATS, tasks, episodes, failed))


def run_one_subprocess(task, args):
    cmd = [
        sys.executable,
        "-u",
        str(paths.SOURCE_DIR / 'calibration/calibrate_all_clean.py'),
        "--task",
        task,
        "--episodes",
        str(args.episodes),
        "--max-attempts",
        str(args.max_attempts),
        "--out-root",
        str(args.out_root),
    ]
    if args.show_sim:
        cmd.append("--show-sim")
    if args.force:
        cmd.append("--force")
    env = os.environ.copy()
    start = time.monotonic()
    process = subprocess.run(cmd, cwd=str(paths.PROJECT_ROOT), env=env)
    return process.returncode, round(time.monotonic() - start, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", default=[],
                        help="Task to calibrate. Omit for all configured tasks.")
    parser.add_argument("--tasks", type=int, default=None,
                        help="Limit to first N configured tasks when --task is omitted.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel task subprocesses. Use 10 if your machine can handle it.")
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument(
        "--out-root",
        default=str(DEFAULT_CALIBRATION_ROOT),
        help=(
            "Root folder for all calibration outputs. Default: "
            "aha_output/aha_calibration"
        ),
    )
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--show-sim", action="store_true")
    args = parser.parse_args()

    configure_output_root(args.out_root)
    configure_env(headless=not args.show_sim)
    tasks = args.task or available_tasks()
    if args.tasks is not None:
        tasks = tasks[:args.tasks]

    if args.force:
        skipped = []
        tasks_to_run = tasks
    else:
        # Default: check how many clean episodes each task already has and only
        # rerun the ones short of --episodes (run_task then retries failed demos
        # until each reaches the target). --force redoes everything.
        counts = {task: clean_episode_count(task) for task in tasks}
        skipped = [task for task in tasks if counts[task] >= args.episodes]
        tasks_to_run = [task for task in tasks if task not in set(skipped)]
        for task in tasks_to_run:
            print(f"  {task:<30} has {counts[task]}/{args.episodes} — will rerun",
                  flush=True)

    print(
        f"all-clean calibration: requested={len(tasks)} skipped={len(skipped)} "
        f"to_run={len(tasks_to_run)} episodes={args.episodes} workers={args.workers} "
        f"out_root={CALIBRATION_ROOT}",
        flush=True,
    )
    if not tasks_to_run:
        rebuild_all_task_summaries(tasks, args.episodes, [])
        print("nothing to do")
        return

    failed = []
    if args.workers > 1 and len(tasks_to_run) > 1:
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_one_subprocess, task, args): task
                for task in tasks_to_run
            }
            for future in as_completed(futures):
                task = futures[future]
                rc, elapsed = future.result()
                done += 1
                if rc == 0:
                    print(f"  [{done}/{len(tasks_to_run)}] {task:<28} done {elapsed}s", flush=True)
                else:
                    failed.append((task, f"exit={rc}"))
                    print(f"  [{done}/{len(tasks_to_run)}] {task:<28} FAIL exit={rc}", flush=True)
    else:
        for i, task in enumerate(tasks_to_run, 1):
            if args.skip_completed and not args.force and task_done(task, args.episodes):
                print(f"  [{i}/{len(tasks_to_run)}] {task:<28} skip", flush=True)
                continue
            try:
                print(f"  [{i}/{len(tasks_to_run)}] {task} start", flush=True)
                result = run_task(task, args.episodes, args.max_attempts, args.show_sim)
                print(
                    f"  [{i}/{len(tasks_to_run)}] {task} done "
                    f"episodes={result['episodes_completed']} "
                    f"orient_wp={result['orientation_wps']} "
                    f"trans_wp={result['transition_wps']}",
                    flush=True,
                )
            except Exception as exc:
                failed.append((task, str(exc)))
                print(f"  [{i}/{len(tasks_to_run)}] {task} FAIL {exc}", flush=True)

    rebuild_all_task_summaries(tasks, args.episodes, failed)
    if failed:
        write_json(
            CALIBRATION_ROOT / "all_clean_calibration_failed_tasks.json",
            [{"task": task, "error": error} for task, error in failed])
    print(f"finished: failed={len(failed)}")


if __name__ == "__main__":
    main()
