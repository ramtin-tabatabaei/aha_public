"""Record and plot slip-detector telemetry for one task run.

This is intentionally separate from the slip detector implementation. It runs
either a clean success episode or a FailGen slip/grasp failure episode, feeds
the same live slip detector used by the BT runner, and saves:

  - aha_output/slip_plots/<task>.clean.csv/png
  - aha_output/slip_plots/<task>.<failtype>.wp<wp>.csv/png

Examples:
  python aha_scripts/main_bt_run/plot_slip_trace.py \
      --task change_clock --failure grasp --waypoint 1 --headless
  python aha_scripts/main_bt_run/plot_slip_trace.py \
      --task change_clock --failure none --headless
"""

from aha_publish import paths

import argparse
import csv
import logging
import os
from pathlib import Path
import sys

ROOT = (paths.PROJECT_ROOT)
FAILGEN_ROOT = (paths.FAILGEN_ROOT)
CONFIGS_PATH = FAILGEN_ROOT / "failgen" / "configs"
OUT_DIR = (paths.OUTPUT_DIR / 'slip_plots')
COPPELIA = paths.COPPELIASIM_ROOT
SLIP_FAILTYPES = ("slip", "grasp")

for path in (str(FAILGEN_ROOT), str(paths.PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


def configure_runtime_env():
    os.environ["COPPELIASIM_ROOT"] = str(COPPELIA)
    os.environ["LD_LIBRARY_PATH"] = (
        os.environ.get("LD_LIBRARY_PATH", "") + ":" + str(COPPELIA)
    )
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(COPPELIA)
    os.environ.pop("QT_QPA_PLATFORM", None)
    os.environ.setdefault("DISPLAY", ":1")
    os.environ.setdefault("AHA_FAIL_EXTREME", "1")
    os.environ.setdefault("AHA_FAIL_DEBUG", "1")
    os.environ.setdefault("AHA_DETECTOR_VLM_AUTO", "1")
    os.environ.setdefault("AHA_VLM_CONFIRM_DETECTORS", "")


def load_config(task_name):
    import yaml

    path = CONFIGS_PATH / f"{task_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f)


def failure_configs(config):
    return [f for f in config.get("failures", []) if f.get("type") in SLIP_FAILTYPES]


def resolve_failure(config, failtype, waypoint):
    if failtype == "none":
        return failtype, waypoint, {}

    failures = failure_configs(config)
    if failtype is None:
        # Most configs have useful grasp waypoints; slip waypoints are often [].
        failure_cfg = next((f for f in failures if f.get("type") == "grasp"), None)
        failure_cfg = failure_cfg or next((f for f in failures if f.get("type") == "slip"), None)
    else:
        failure_cfg = next((f for f in failures if f.get("type") == failtype), None)
    if failure_cfg is None:
        raise ValueError(f"No slip/grasp failure config found for {failtype or 'default'}")

    waypoints = [int(w) for w in failure_cfg.get("waypoints", [])]
    if waypoint is None:
        if waypoints:
            waypoint = min(waypoints)
        elif failure_cfg.get("type") == "slip":
            grasp_cfg = next((f for f in failures if f.get("type") == "grasp"), {})
            grasp_wps = [int(w) for w in grasp_cfg.get("waypoints", [])]
            if grasp_wps:
                waypoint = min(grasp_wps)
            else:
                data_wps = config.get("data", {}).get("waypoints", [])
                if data_wps:
                    waypoint = min(int(w) for w in data_wps)
        if waypoint is None:
            raise ValueError(
                f"Failure {failure_cfg.get('type')} has no waypoint. Pass --waypoint."
            )
    return failure_cfg.get("type"), int(waypoint), failure_cfg


def disable_all_failures(env_wrapper):
    for failure in env_wrapper.manager._failures:
        failure.set_enabled(False)


def configure_slip_failure(env_wrapper, failtype, waypoint, failure_cfg):
    target = None
    for failure in env_wrapper.manager._failures:
        if failure.failure_type == failtype:
            failure.set_enabled(True)
            target = failure
        else:
            failure.set_enabled(False)

    if target is not None:
        target.change_waypoint_fail_name(f"waypoint{waypoint}")
        return target

    if failtype == "slip":
        from failgen.fail_slip import SlipFailure

        target = SlipFailure(
            robot=env_wrapper.robot,
            name="slip_trace",
            waypoint_indices=[waypoint],
            num_steps_till_fail=int(failure_cfg.get("fail_after", 1)),
        )
    elif failtype == "grasp":
        from failgen.fail_grasp import GraspFailure

        target = GraspFailure(
            robot=env_wrapper.robot,
            name="grasp_trace",
            waypoint_indices=[waypoint],
        )
    else:
        raise ValueError(f"Unsupported slip failtype: {failtype}")

    target.set_enabled(True)
    target.change_waypoint_fail_name(f"waypoint{waypoint}")
    target.set_obj_base(env_wrapper._obj_base)
    env_wrapper.manager.add_failure(target)
    return target


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


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task",
        "waypoint",
        "local_step",
        "step",
        "grip_force",
        "left_grip_force",
        "right_grip_force",
        "grip_force_drop",
        "grip_force_delta",
        "gripper_open",
        "is_holding",
        "holding_required_phase",
        "prior_holding_streak",
        "steps_since_holding",
        "force_released",
        "force_drop_crossed",
        "slip_candidate_active",
        "low_force_streak",
        "slip_score",
        "slip_reason",
        "grip_force_threshold",
        "grip_force_drop_threshold",
        "path_done",
        "slip_fired",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_trace(path, rows, task_name, label, fired_steps):
    if not rows:
        raise RuntimeError("No slip telemetry rows were recorded.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import os
    method = os.getenv("AHA_SLIP_METHOD", "1").strip()
    single = (method != "2")   # method 1 / default = proportional, single panel

    def _val(r, key):
        try:
            x = float(r.get(key))
            return x if x == x else None   # drop NaN
        except (TypeError, ValueError):
            return None

    steps = [int(r["step"]) for r in rows]
    grip = [float(r["grip_force"]) for r in rows]

    if single:
        fig, ax0 = plt.subplots(1, 1, figsize=(12, 5))
        axes = [ax0]
    else:
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f"{task_name} {label}: slip telemetry (method {method})")

    axes[0].plot(steps, grip, label="grip_force", color="#2563eb")
    if single:
        # Method 1: show the held peak and the proportional slip level
        # ((1-drop_frac)*peak) -- grip crossing below it (while a grasp is
        # established) is the slip trigger.
        drop_frac = float(os.getenv("AHA_SLIP_DROP_FRAC", "0.8"))
        peak = [_val(r, "held_peak") for r in rows]
        px = [s for s, p in zip(steps, peak) if p and p > 0]
        py = [p for p in peak if p and p > 0]
        if px:
            axes[0].plot(px, py, label="held peak", color="#10b981",
                         linestyle=":", alpha=0.8)
            axes[0].plot(px, [(1 - drop_frac) * p for p in py],
                         label=f"slip level ({int(drop_frac*100)}% drop)",
                         color="#a855f7", linestyle="--", alpha=0.8)
    axes[0].set_ylabel("grip_force")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=8)

    if not single:
        drop = [float(r["grip_force_drop"]) for r in rows]
        drop_thr = [float(r["grip_force_drop_threshold"]) for r in rows]
        axes[1].plot(steps, drop, label="grip_force_drop", color="#f97316")
        axes[1].plot(steps, drop_thr, label="drop threshold",
                     color="#ef4444", linestyle="--")
        axes[1].set_ylabel("force_drop")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(loc="upper right")
    axes[-1].set_xlabel("frame / detector step")

    waypoint_starts = []
    last_waypoint = None
    for row in rows:
        waypoint = row["waypoint"]
        if waypoint != last_waypoint:
            waypoint_starts.append((int(row["step"]), waypoint))
            last_waypoint = waypoint

    # Gripper CLOSE command frames (grasp starts) from telemetry.
    gripper_close_frames = [int(row["step"]) for row in rows
                            if _val(row, "gripper_close_cmd") == 1]

    # Preferred: the exact frame the runner issued the open command, tagged in
    # the telemetry (gripper_open_cmd=1). This is earlier and more accurate than
    # any jaw-motion measurement. Fall back to the continuous jaw rise, then the
    # binarized gripper_open, for telemetry that predates those columns.
    gripper_open_frames = [int(row["step"]) for row in rows
                           if _val(row, "gripper_open_cmd") == 1]
    if not gripper_open_frames:
        cont = [a for a in (_val(r, "gripper_open_amount") for r in rows)
                if a is not None]
        open_key, open_thr = (("gripper_open_amount", min(cont) + 0.2) if cont
                              else ("gripper_open", 0.5))
        prev_open = None
        for row in rows:
            v = _val(row, open_key)
            if v is not None:
                if prev_open is not None and prev_open < open_thr <= v:
                    gripper_open_frames.append(int(row["step"]))
                prev_open = v

    for ax in axes:
        for step, waypoint in waypoint_starts:
            ax.axvline(step, color="#7c3aed", linestyle="--", alpha=0.35)
            ax.text(
                step,
                0.98,
                f"wp{waypoint}",
                transform=ax.get_xaxis_transform(),
                rotation=90,
                va="top",
                ha="right",
                fontsize=8,
                color="#7c3aed",
            )
        for step in gripper_close_frames:
            ax.axvline(step, color="#0891b2", linestyle="-", alpha=0.8)
            ax.text(
                step, 0.98, "close cmd",
                transform=ax.get_xaxis_transform(), rotation=90,
                va="top", ha="right", fontsize=8, color="#0891b2",
            )
        for step in gripper_open_frames:
            ax.axvline(step, color="#16a34a", linestyle="-", alpha=0.7)
            ax.text(
                step,
                0.98,
                "open cmd",
                transform=ax.get_xaxis_transform(),
                rotation=90,
                va="top",
                ha="left",
                fontsize=8,
                color="#16a34a",
            )
        for row in rows:
            if not bool(row["holding_required_phase"]):
                ax.axvspan(int(row["step"]) - 0.5, int(row["step"]) + 0.5,
                           color="#e5e7eb", alpha=0.18, linewidth=0)
            if int(row["path_done"]):
                ax.axvline(int(row["step"]), color="#64748b", linestyle=":", alpha=0.18)
        for i, step in enumerate(fired_steps):
            ax.axvline(step, color="#dc2626", linestyle="-", linewidth=1.8,
                       alpha=0.9,
                       label="slip fired" if (ax is axes[0] and i == 0) else None)
    if fired_steps:
        # re-collect the legend: the fired line is added after the first
        # legend() call, so it would otherwise be missing from it
        axes[0].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def record_trace(task_name, failtype, waypoint, failure_cfg, *, headless=True):
    import numpy as np
    from failgen.env_wrapper import FailGenEnvWrapper
    from aha_publish.running.live_detectors.slip import LiveDetector

    env_wrapper = FailGenEnvWrapper(
        task_name=task_name,
        headless=headless,
        record=False,
        save_data=True,
        no_failures=(failtype == "none"),
        save_path=str(paths.BACKEND_DATA_DIR / 'slip_trace'),
        save_keyframes_only=True,
    )

    if failtype == "none":
        disable_all_failures(env_wrapper)
    else:
        configure_slip_failure(env_wrapper, failtype, waypoint, failure_cfg)

    detector = LiveDetector(
        task_name,
        env_wrapper=env_wrapper,
        vlm_enabled=False,
        show_plot=False,
        failure_waypoint=waypoint,
    )

    state = {
        "waypoint": None,
        "pose": None,
        "next_waypoint": 0,
        "feed_count": 0,
    }
    trace_rows = []
    fired_global_steps = []
    original_step = env_wrapper.on_env_step
    original_waypoint = env_wrapper.on_env_waypoint
    original_waypoint_end = env_wrapper.on_env_waypoint_end

    def patched_waypoint(point):
        idx = waypoint_index_from_point(point)
        if idx is None:
            idx = state["next_waypoint"]
        state["next_waypoint"] = max(state["next_waypoint"], int(idx) + 1)
        state["waypoint"] = idx
        state["pose"] = waypoint_pose(idx)
        original_waypoint(point)

    def feed(obs, path_done=False):
        idx = state["waypoint"]
        if idx is None:
            return
        state["feed_count"] += 1
        pose = state["pose"]
        if pose is None:
            pose = waypoint_pose(idx)
        fire_count_before = len(detector.paused_steps)
        detector.step(
            obs,
            waypoint=idx,
            path_done=path_done,
            waypoint_pose=pose,
            original_pose=pose,
            report=True,
        )
        if not detector.logs:
            return
        row = detector.logs[-1]
        global_step = len(trace_rows)
        fired_now = len(detector.paused_steps) > fire_count_before
        if fired_now:
            fired_global_steps.append(global_step)
        thr = detector.frozen_thr or {}
        trace_rows.append({
            "task": task_name,
            "waypoint": idx,
            "local_step": int(row.get("step", 0)),
            "step": global_step,
            "grip_force": float(row.get("grip_force", np.nan)),
            "left_grip_force": float(row.get("left_grip_force", np.nan)),
            "right_grip_force": float(row.get("right_grip_force", np.nan)),
            "grip_force_drop": float(row.get("grip_force_drop", np.nan)),
            "grip_force_delta": float(row.get("grip_force_delta", np.nan)),
            "gripper_open": float(getattr(obs, "gripper_open", np.nan)),
            "is_holding": bool(row.get("is_holding", False)),
            "holding_required_phase": bool(row.get("holding_required_phase", True)),
            "prior_holding_streak": int(row.get("prior_holding_streak", 0)),
            "steps_since_holding": (
                "" if row.get("steps_since_holding") is None
                else int(row.get("steps_since_holding"))
            ),
            "force_released": bool(row.get("force_released", False)),
            "force_drop_crossed": bool(row.get("force_drop_crossed", False)),
            "slip_candidate_active": bool(detector._candidate_active),
            "low_force_streak": int(
                max(
                    detector._consec,
                    detector._grasp_lost_consec,
                    detector._never_held_consec,
                )
            ),
            "slip_score": float(row.get("slip_score", 0.0)),
            "slip_reason": row.get("slip_reason", ""),
            "grip_force_threshold": float(
                thr.get(
                    "grip_force_threshold",
                    detector.settings["threshold_overrides"].get(
                        "grip_force_threshold", 1.0
                    ),
                )
            ),
            "grip_force_drop_threshold": float(
                thr.get(
                    "grip_force_drop_threshold",
                    detector.settings["threshold_overrides"].get(
                        "grip_force_drop_threshold", 1.0
                    ),
                )
            ),
            "path_done": int(bool(path_done)),
            "slip_fired": fired_now,
        })

    def patched_step(obs):
        original_step(obs)
        feed(obs, path_done=False)

    def patched_waypoint_end(point):
        idx = waypoint_index_from_point(point)
        if state["waypoint"] is None:
            state["waypoint"] = idx
        original_waypoint_end(point)
        try:
            obs = env_wrapper._task_env.get_observation()
            feed(obs, path_done=True)
        except Exception:
            pass
        state["waypoint"] = None
        state["pose"] = None

    env_wrapper.on_env_waypoint = patched_waypoint
    env_wrapper.on_env_step = patched_step
    env_wrapper.on_env_waypoint_end = patched_waypoint_end

    try:
        if failtype == "none":
            attempts = env_wrapper._max_failure_attempts
            last_error = None
            demo = None
            ctr_loop = env_wrapper.robot.arm.joints[0].is_control_loop_enabled()
            env_wrapper.robot.arm.set_control_loop_enabled(True)
            try:
                while attempts > 0:
                    random_seed = np.random.get_state()
                    env_wrapper.reset()
                    try:
                        demo = env_wrapper._env._scene.get_demo(
                            callable_each_step=env_wrapper.on_env_step,
                            callable_each_waypoint=env_wrapper.on_env_waypoint,
                            callable_each_end_waypoint=env_wrapper.on_env_waypoint_end,
                        )
                        demo.random_seed = random_seed
                        break
                    except Exception as exc:
                        last_error = exc
                        attempts -= 1
                        logging.info("Bad demo. %s", exc)
            finally:
                env_wrapper.robot.arm.set_control_loop_enabled(ctr_loop)
            if demo is None:
                print(f"Warn >>> clean run ended without task success: {last_error}")
            success = demo is not None
        else:
            _demo, success = env_wrapper.get_failure()
    finally:
        detector.close()
        env_wrapper.shutdown()

    if not trace_rows:
        print(
            "Warn >>> no slip telemetry rows recorded "
            f"(fed={state['feed_count']}, detector_fed={detector.fed}, "
            f"disabled={detector.disabled_reason})"
        )
    return trace_rows, detector, bool(success), fired_global_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--failure", choices=("none",) + SLIP_FAILTYPES, default=None)
    parser.add_argument("--waypoint", type=int)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--show-gui", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    configure_runtime_env()
    config = load_config(args.task)
    failtype, waypoint, failure_cfg = resolve_failure(config, args.failure, args.waypoint)

    rows, detector, success, fired_steps = record_trace(
        args.task,
        failtype,
        waypoint,
        failure_cfg,
        headless=args.headless,
    )

    if failtype == "none":
        stem = f"{args.task}.clean"
        label = "clean"
    else:
        stem = f"{args.task}.{failtype}.wp{waypoint}"
        label = f"{failtype}@wp{waypoint}"

    csv_path = args.out_dir / f"{stem}.csv"
    png_path = args.out_dir / f"{stem}.png"
    write_csv(csv_path, rows)
    plot_trace(png_path, rows, args.task, label, fired_steps)

    fired = bool(detector.paused_steps)
    print(f"task={args.task} failure={failtype} waypoint={waypoint} success={success}")
    print(f"frames={len(rows)} slip_fired={fired} fired_steps={fired_steps[:10]}")
    print(f"fired_waypoints={sorted(detector.fired_waypoints)}")
    print(f"csv -> {csv_path}")
    print(f"plot -> {png_path}")


if __name__ == "__main__":
    main()
