"""Record and plot collision-detector torque telemetry for one task run.

This is intentionally separate from the collision detector implementation. It
runs either a FailGen collision episode or a clean success episode, records the
same per-frame torque features the detector uses, and saves:

  - aha_output/collision_torque_plots/<task>.collision.wp<wp>.csv
  - aha_output/collision_torque_plots/<task>.collision.wp<wp>.png
  - aha_output/collision_torque_plots/<task>.clean.csv
  - aha_output/collision_torque_plots/<task>.clean.png

Examples:
  python aha_scripts/main_bt_run/plot_collision_torque_trace.py \
      --task change_channel --waypoint 1 --headless
  python aha_scripts/main_bt_run/plot_collision_torque_trace.py \
      --task change_channel --failure none --headless
"""

from aha_publish import paths

import argparse
import csv
import os
from pathlib import Path
import sys

ROOT = (paths.PROJECT_ROOT)
FAILGEN_ROOT = (paths.FAILGEN_ROOT)
CONFIGS_PATH = FAILGEN_ROOT / "failgen" / "configs"
COLLISION_DIR = (paths.SOURCE_DIR / 'detectors' / 'collision')
OUT_DIR = (paths.OUTPUT_DIR / 'collision_torque_plots')
COPPELIA = paths.COPPELIASIM_ROOT

for path in (str(COLLISION_DIR), str(FAILGEN_ROOT), str(paths.PROJECT_ROOT)):
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


def load_config(task_name):
    import yaml

    path = CONFIGS_PATH / f"{task_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f)


def all_task_names():
    """Every task that has a config yaml, sorted (used by --tasks all)."""
    return sorted(p.stem for p in CONFIGS_PATH.glob("*.yaml"))


def collision_config(config):
    return next(
        (f for f in config.get("failures", []) if f.get("type") == "collision"),
        {},
    )


def default_collision_waypoint(config):
    cfg = collision_config(config)
    waypoints = [int(w) for w in cfg.get("waypoints", [])]
    if not waypoints:
        raise ValueError("Task config has no collision waypoints")
    return min(waypoints)


def configure_collision_failure(env_wrapper, config, waypoint):
    from failgen.fail_collision import CollisionFailure

    target = None
    for failure in env_wrapper.manager._failures:
        if failure.failure_type == CollisionFailure.FAILURE_TYPE:
            failure.set_enabled(True)
            target = failure
        else:
            failure.set_enabled(False)

    if target is None:
        cfg = collision_config(config)
        target = CollisionFailure(
            robot=env_wrapper.robot,
            name="collision_trace",
            waypoints_indices=[waypoint],
            mode=cfg.get("mode", "approach"),
            approach_distance_range=tuple(
                cfg.get("approach_distance_range", [0.10, 0.20])
            ),
            approach_axis=cfg.get("approach_axis", "z"),
            runtime_duration_steps=cfg.get("runtime_duration_steps", 60),
        )
        target.set_obj_base(env_wrapper._obj_base)
        env_wrapper.manager.add_failure(target)

    target.change_waypoint_fail_name(f"waypoint{waypoint}")
    return target


def disable_all_failures(env_wrapper):
    for failure in env_wrapper.manager._failures:
        failure.set_enabled(False)


def detector_settings(task_name, consecutive_frames=None):
    import detector

    settings = detector.default_detector_settings()
    settings = detector.apply_task_torque_stats_thresholds(
        settings,
        task_name,
        manual_threshold_keys=settings.get("manual_threshold_keys", ()),
    )
    if consecutive_frames is not None:
        settings["consecutive_collision_frames"] = int(consecutive_frames)
    return settings


# Custom side view: a VisionSensor we attach to the scene ourselves, matching the
# aha_side_camera used by aha_scripts/waypoints_screenshot.py and the collision
# detector's interactive.py, so the "side" photo is the same view as everywhere
# else in the project.
SIDE_CAMERA_NAME = "aha_side_camera"
SIDE_CAMERA_RESOLUTION = [256, 256]
SIDE_CAMERA_POSITION = (-0.547, -0.803, 1.209)  # nudged ~0.45 m forward along the look axis
SIDE_CAMERA_ORIENTATION_DEG = (-100, 38.3, -180.0)  # panned left


def create_aha_side_camera():
    """Create the standalone aha_side_camera VisionSensor in the running sim."""
    import numpy as np
    from pyrep.const import RenderMode
    from pyrep.objects.vision_sensor import VisionSensor

    cam = VisionSensor.create(SIDE_CAMERA_RESOLUTION)
    cam.set_name(SIDE_CAMERA_NAME)
    cam.set_render_mode(RenderMode.OPENGL3)
    cam.set_position(list(SIDE_CAMERA_POSITION))
    cam.set_orientation(np.radians(SIDE_CAMERA_ORIENTATION_DEG))
    return cam


def _save_detection_images(obs, out_dir, task_name, waypoint, failure, step,
                           side_camera=None):
    """Save front, side and wrist RGB snapshots at the collision-detection frame.

    front and wrist come from the high-dim observation; side is the custom
    aha_side_camera VisionSensor. Any view whose image is missing is skipped.
    """
    import numpy as np
    from PIL import Image
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "clean" if failure == "none" else f"collision.wp{waypoint}"

    views = {
        "front": getattr(obs, "front_rgb", None),
        "wrist": getattr(obs, "wrist_rgb", None),
    }
    if side_camera is not None:
        try:
            # capture_rgb() returns float RGB in [0, 1].
            views["side"] = np.clip(
                side_camera.capture_rgb() * 255.0, 0, 255).astype(np.uint8)
        except Exception as exc:
            print(f"  [capture] side camera capture failed: {exc}")

    saved = []
    for name, rgb in views.items():
        if rgb is None:
            continue
        path = out_dir / f"{task_name}.{tag}.detect_step{step}.{name}.png"
        try:
            Image.fromarray(np.asarray(rgb).astype(np.uint8)).save(path)
            saved.append(str(path))
        except Exception as exc:
            print(f"  [capture] failed to save {name}: {exc}")
    for p in saved:
        print(f"detection image -> {p}")
    return saved


def record_trace(
    task_name,
    waypoint,
    *,
    failure="collision",
    headless=True,
    consecutive_frames=None,
    capture_images_dir=None,
    save_residual=True,
):
    import detector
    from failgen.env_wrapper import FailGenEnvWrapper

    settings = detector_settings(task_name, consecutive_frames)
    config = load_config(task_name)

    # Camera RGB (front / side views) is only rendered when we intend to save a
    # snapshot at the detection frame; otherwise skip it for a big speedup.
    capture_images = capture_images_dir is not None

    env_wrapper = FailGenEnvWrapper(
        task_name=task_name,
        headless=headless,
        record=False,
        # The collision detector only reads low-dim obs (joint_forces, gripper
        # touch forces, gripper state). save_data=False triggers
        # set_all_high_dim(False), so CoppeliaSim skips rendering all 5 camera
        # views every step -- the dominant per-step cost. Huge speedup. Enable it
        # only when we need the front/side photo at the detection frame.
        save_data=capture_images,
        save_path=str(paths.BACKEND_DATA_DIR / 'collision_torque_trace'),
        save_keyframes_only=False,
    )

    if failure == "collision":
        configure_collision_failure(env_wrapper, config, waypoint)
    elif failure == "none":
        disable_all_failures(env_wrapper)
    else:
        raise ValueError(f"Unsupported failure mode: {failure}")

    # Per-waypoint rise_frac gates calibrated from clean runs; {} -> flat floor.
    rise_gates = detector.load_rise_frac_gates(task_name)
    # Method-3 momentum-observer per-joint residual thresholds, calibrated from a
    # clean run (None -> fall back to the warmup residual envelope).
    residual_stats = detector.load_residual_stats(task_name)

    logs = []
    rows = []
    paused_steps = []
    waypoint_frames = []  # (waypoint_index, frame) at each waypoint boundary
    step_counter = 0
    current_wp = 0        # ordinal of the waypoint currently being executed
    frozen_thr = None
    consecutive = 0
    captured = {"done": False}   # front/side/wrist snapshot saved at detection?
    side_cam = {"sensor": None}  # lazily-created aha_side_camera VisionSensor
    original_step = env_wrapper.on_env_step
    original_wp_end = env_wrapper.on_env_waypoint_end

    def patched_wp_end(point):
        # on_env_waypoint_end fires when the arm finishes a waypoint; the current
        # step counter is the frame it landed on. Record (ordinal, frame) so the
        # plot can mark waypoint boundaries, then advance the current-waypoint
        # ordinal used to pick this waypoint's rise_frac gate.
        nonlocal current_wp
        original_wp_end(point)
        waypoint_frames.append((current_wp, step_counter))
        current_wp += 1

    env_wrapper.on_env_waypoint_end = patched_wp_end

    def append_row(row, reason=""):
        holding = bool(row["is_holding"])
        norm_thr = (
            frozen_thr["tq_norm_hold"] if holding else frozen_thr["tq_norm_free"]
        ) if frozen_thr else None
        delta_thr = (
            frozen_thr["tq_delta_hold"] if holding else frozen_thr["tq_delta_free"]
        ) if frozen_thr else None
        rows.append({
            "step": row["step"],
            "torque_norm": row["torque_norm"],
            "torque_delta": row["torque_delta"],
            "torque_norm_threshold": norm_thr,
            "torque_delta_threshold": delta_thr,
            "torque_norm_crossed": row.get("torque_norm_crossed", False),
            "torque_delta_crossed": row.get("torque_delta_crossed", False),
            "torque_baseline": row.get("torque_baseline", ""),
            "torque_rise": row.get("torque_rise", ""),
            "torque_rise_frac": row.get("torque_rise_frac", ""),
            "torque_rise_gate": row.get("torque_rise_gate", ""),
            "torque_proportional_spike": row.get("torque_proportional_spike", False),
            "residual_norm": row.get("residual_norm", ""),
            "residual_score": row.get("residual_score", ""),
            "residual_threshold": row.get("residual_threshold", ""),
            "residual_crossed": row.get("residual_crossed", False),
            **{f"r{j}": (float(row["mo_residual"][j])
                        if row.get("mo_residual") is not None else "")
               for j in range(7)},
            "waypoint": row.get("waypoint", ""),
            "collision_raw": row.get("collision", False),
            "consecutive_collision_frames": consecutive,
            "collision_reason": reason,
            "is_holding": holding,
        })

    def patched_step(obs):
        nonlocal step_counter, frozen_thr, consecutive
        original_step(obs)
        step = step_counter
        step_counter += 1

        # Create the side camera once, early, so it renders through the episode
        # and yields a valid frame when we snapshot at the detection step.
        if capture_images and side_cam["sensor"] is None:
            try:
                side_cam["sensor"] = create_aha_side_camera()
            except Exception as exc:
                print(f"  [capture] could not create side camera: {exc}")
                side_cam["sensor"] = False  # don't retry every step

        row = detector.obs_to_row(obs, step)
        logs.append(row)
        detector.update_deltas(logs)

        if step < detector.WARMUP_STEPS:
            row["collision"] = False
            consecutive = 0
            append_row(row, "warmup")
            return

        if frozen_thr is None:
            frozen_thr = detector.freeze_thresholds(
                logs[:detector.WARMUP_STEPS],
                settings["threshold_overrides"],
                residual_stats=residual_stats,
            )

        # Per-waypoint gate for the waypoint currently executing (falls back to
        # the default floor when this task/waypoint was not calibrated).
        row["waypoint"] = current_wp
        row["torque_rise_gate"] = rise_gates.get(current_wp)

        raw, reason = detector.check_collision(
            row,
            frozen_thr,
            torque_rule=settings["torque_rule"],
            torque_norm_weight=settings["torque_norm_weight"],
            torque_delta_weight=settings["torque_delta_weight"],
            torque_score_threshold=settings["torque_score_threshold"],
            use_touch_force=settings.get("use_touch_force", False),
            torque_rise_gate=row["torque_rise_gate"],
        )
        row["collision"] = raw
        if raw:
            consecutive += 1
        else:
            consecutive = 0
        required = max(1, int(settings["consecutive_collision_frames"]))
        if raw and consecutive >= required:
            paused_steps.append(step)
            # Save front/side camera snapshots at the first confirmed detection.
            if capture_images and not captured["done"]:
                _save_detection_images(
                    obs, capture_images_dir, task_name, waypoint, failure, step,
                    side_camera=side_cam["sensor"] or None)
                captured["done"] = True
        append_row(row, reason)

    env_wrapper.on_env_step = patched_step

    try:
        if failure == "collision":
            _demo, success = env_wrapper.get_failure()
        else:
            try:
                _demo = env_wrapper.get_success()
                success = _demo is not None
            except RuntimeError as exc:
                if not rows:
                    raise
                print(f"Warn >>> clean run ended without task success: {exc}")
                success = False
    finally:
        env_wrapper.shutdown()

    # On a clean run, calibrate the method-3 per-joint residual envelope: the max
    # |r_i| over the settled, post-warmup clean frames. This is the De Luca
    # per-joint threshold basis (scaled by RESIDUAL_K in the detector).
    # `episode_residual_max` is returned so a multi-episode caller can take the
    # element-wise max over several clean runs before persisting.
    episode_residual_max = None
    if failure == "none" and logs:
        import numpy as np
        clean = [r["mo_residual"] for r in logs
                 if r.get("mo_residual") is not None
                 and not r.get("_mo_settling", False)
                 and int(r.get("step", 0)) >= detector.WARMUP_STEPS]
        if clean:
            episode_residual_max = np.max(np.abs(np.array(clean)), axis=0)
            print(f"clean per-joint max |r_i| (Nm): "
                  f"{np.round(episode_residual_max, 2)}")
            if save_residual:
                path = detector.save_residual_stats(
                    task_name, episode_residual_max)
                print(f"residual_stats -> {path}")

    return (rows, paused_steps, settings, bool(success), waypoint_frames,
            episode_residual_max)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "step",
        "torque_norm",
        "torque_delta",
        "torque_norm_threshold",
        "torque_delta_threshold",
        "torque_norm_crossed",
        "torque_delta_crossed",
        "torque_baseline",
        "torque_rise",
        "torque_rise_frac",
        "torque_rise_gate",
        "torque_proportional_spike",
        "residual_norm",
        "residual_score",
        "residual_threshold",
        "residual_crossed",
        "r0", "r1", "r2", "r3", "r4", "r5", "r6",
        "waypoint",
        "collision_raw",
        "consecutive_collision_frames",
        "collision_reason",
        "is_holding",
    ]
    # The detector keeps gaining residual columns (per-joint thresholds
    # rthr0..rthr6, residual_joints_over, ...). DictWriter raises on any key
    # missing from fieldnames, which used to abort the whole collision plot,
    # so append whatever extra columns the drift rows carry instead of a
    # hard-coded list going stale on the next detector change.
    seen = set(fields)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def confirmed_steps_from_rows(rows, required=None):
    """Frames the collision detector actually CONFIRMED, from a drift CSV.

    A raw crossing (`collision_raw == True`) only becomes a detection after
    `required` consecutive frames. Plotting every raw frame marks candidates
    the live detector never fired on, and pushes the "first detection" marker
    earlier -- often into the previous waypoint -- than the waypoint the run
    log and summary/vlm_events CSVs report. Use this so plot and CSV agree.
    """
    if required is None:
        try:
            import detector as _det
            required = int(getattr(
                _det, "DEFAULT_CONSECUTIVE_COLLISION_FRAMES", 2))
        except Exception:
            required = 2
    required = max(1, int(required))

    out, consecutive = [], 0
    for r in rows:
        raw = str(r.get("collision_raw")) == "True"
        # Prefer the counter the live detector logged; fall back to counting.
        try:
            consecutive = int(float(r.get("consecutive_collision_frames")))
        except (TypeError, ValueError):
            consecutive = consecutive + 1 if raw else 0
        if raw and consecutive >= required:
            try:
                out.append(int(float(r["step"])))
            except (TypeError, ValueError, KeyError):
                pass
    return out


def _waypoint_at_frame(waypoint_frames, frame):
    """Waypoint index whose segment contains `frame` (None if unknown)."""
    wp = None
    for idx, start in sorted(waypoint_frames or [], key=lambda t: t[1]):
        if start <= frame:
            wp = idx
        else:
            break
    return wp


def _floats(rows, key):
    out = []
    for r in rows:
        v = r.get(key, "")
        out.append(float(v) if v not in (None, "") else None)
    return out


def plot_trace(path, rows, paused_steps, task_name, label,
               waypoint_frames=None, injected_wp=None):
    """Visualize the proportional-spike collision method (AHA_COLLISION_METHOD=1).

    Panels:
      1. torque_norm (blue) vs its running baseline (orange). The method reacts
         to the GAP between them, not to an absolute level.
      2. torque_rise_frac (purple) vs the RISE_FRAC gate (red). This is the
         quantity the detector actually thresholds on.
      3. torque_delta (green), kept for reference.
    Vertical lines: grey = waypoint boundaries (the injected collision waypoint
    is red), violet dotted = frames the detector fired.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import detector

    default_gate = detector.DEFAULT_TORQUE_RISE_FRAC

    steps = [int(r["step"]) for r in rows]
    torque_norm = _floats(rows, "torque_norm")
    baseline = _floats(rows, "torque_baseline")
    rise_frac = _floats(rows, "torque_rise_frac")
    torque_delta = _floats(rows, "torque_delta")
    # Per-frame gate (per-waypoint, calibrated). Missing -> the default gate.
    gate = [g if g is not None else default_gate
            for g in _floats(rows, "torque_rise_gate")]
    spike = [str(r.get("torque_proportional_spike", "")) == "True" for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fig.suptitle(
        f"{task_name} {label}: proportional-spike collision "
        f"(per-waypoint rise_frac gate, floor={default_gate:g})"
    )

    axes[0].plot(steps, torque_norm, label="torque_norm", color="#2563eb")
    if any(v is not None for v in baseline):
        axes[0].plot(steps, baseline, label="baseline (trailing median)",
                     color="#f59e0b", linestyle="--")
    axes[0].set_ylabel("torque_norm")

    axes[1].plot(steps, rise_frac, label="torque_rise_frac", color="#7c3aed")
    axes[1].plot(steps, gate, label="per-waypoint gate", color="#ef4444",
                 linestyle="--", drawstyle="steps-mid")
    # Mark frames where the proportional spike condition (frac AND abs floor) held.
    for s, on in zip(steps, spike):
        if on:
            axes[1].axvspan(s - 0.5, s + 0.5, color="#7c3aed", alpha=0.12)
    axes[1].set_ylabel("torque_rise_frac")

    axes[2].plot(steps, torque_delta, label="torque_delta", color="#16a34a")
    axes[2].set_ylabel("torque_delta")
    axes[2].set_xlabel("frame / simulator step")

    # Waypoint boundaries.
    seen_labels = set()
    for idx, frame in (waypoint_frames or []):
        is_injected = injected_wp is not None and idx == injected_wp
        color = "#dc2626" if is_injected else "#9ca3af"
        lbl = "collision waypoint" if is_injected else "waypoint"
        for ax in axes:
            ax.axvline(frame, color=color, linestyle="-",
                       alpha=0.8 if is_injected else 0.45, linewidth=1.3)
        key = (lbl,)
        axes[0].text(frame, axes[0].get_ylim()[1], f"wp{idx}",
                     color=color, fontsize=8, va="bottom", ha="center")
        if key not in seen_labels:
            seen_labels.add(key)

    # Detector fires.
    for step in paused_steps:
        for ax in axes:
            ax.axvline(step, color="#000000", linestyle=":", alpha=0.6)

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_residual_trace(path, rows, paused_steps, task_name, label,
                        waypoint_frames=None, injected_wp=None):
    """Visualize the momentum-observer collision method (AHA_COLLISION_METHOD=3).

    Single panel: per-joint residual |r_i| (the De Luca external-torque estimate)
    with each joint's calibrated threshold as a dotted line of the same color. A
    collision is declared when at least DEFAULT_RESIDUAL_MIN_JOINTS joints are
    over their own line at once.
    Vertical lines: grey = waypoint boundaries (injected collision waypoint red),
    faint black dotted = frames the detector fired, bold red = first detection.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import detector

    stats = detector.load_residual_stats(task_name)
    thr_vec = (detector.residual_threshold_vector(stats)
               if stats is not None else None)
    min_joints = int(getattr(detector, "DEFAULT_RESIDUAL_MIN_JOINTS", 1))

    steps = [int(r["step"]) for r in rows]

    def _rv(r, j):
        v = r.get(f"r{j}", "")
        return float(v) if v not in (None, "") else 0.0

    resid = np.abs(np.array([[_rv(r, j) for j in range(7)] for r in rows],
                            dtype=float))

    fig, ax = plt.subplots(1, 1, figsize=(11, 6))
    fig.suptitle(f"{task_name} {label}: De Luca generalized-momentum observer "
                 f"(collision method 3, fire when >= {min_joints} joints over)")

    colors = plt.cm.tab10(np.linspace(0, 1, 7))
    for j in range(7):
        ax.plot(steps, resid[:, j], color=colors[j], linewidth=1.2,
                label=f"|r{j+1}|")
        if thr_vec is not None:
            ax.axhline(thr_vec[j], color=colors[j], linestyle=":",
                       linewidth=1.0, alpha=0.7)
    ax.set_ylabel("per-joint residual |r_i|  (Nm)")
    ax.set_xlabel("frame / simulator step")
    ax.legend(loc="upper left", fontsize=7, ncol=7,
              title="dotted = per-joint threshold")

    for idx, frame in (waypoint_frames or []):
        is_inj = injected_wp is not None and idx == injected_wp
        color = "#dc2626" if is_inj else "#9ca3af"
        ax.axvline(frame, color=color, linestyle="-",
                   alpha=0.8 if is_inj else 0.4, linewidth=1.3)
        ax.text(frame, ax.get_ylim()[1], f"wp{idx}", color=color,
                fontsize=8, va="bottom", ha="center")
    for step in paused_steps:
        ax.axvline(step, color="#000000", linestyle=":", alpha=0.3)
    # First detection: the frame the detector first confirms the failure.
    if paused_steps:
        first_detect = min(paused_steps)
        fire_wp = _waypoint_at_frame(waypoint_frames, first_detect)
        wp_txt = "" if fire_wp is None else f" (wp{fire_wp})"
        ax.axvline(first_detect, color="#ef4444", linestyle="-",
                   linewidth=2.4, alpha=0.95,
                   label=f"first detection @ frame {first_detect}{wp_txt}")
        ax.legend(loc="upper left", fontsize=7, ncol=7)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_task(args, task_name, detector, np):
    """Run calibration/plot for a single task (the body of the old main())."""
    config = load_config(task_name)
    waypoint = (
        args.waypoint
        if args.waypoint is not None
        else default_collision_waypoint(config)
    )

    # Multi-episode calibration: run --repeats clean episodes and keep the
    # element-wise max of the per-joint residual envelope across all of them,
    # then persist once. For --repeats 1 this is identical to a single run.
    n_repeats = max(1, args.repeats)
    calibrating = args.failure == "none" and n_repeats > 1
    n_workers = max(1, args.workers)

    if calibrating and n_workers > 1:
        # Parallel: each episode runs in its own subprocess/CoppeliaSim, mirroring
        # run_all_tasks' thread-pool-of-subprocesses pattern. Aggregate the
        # per-joint envelopes (element-wise max) and save once.
        import json
        import subprocess
        import tempfile
        from concurrent.futures import ThreadPoolExecutor, as_completed

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"resid_{task_name}_"))

        def _run_worker(ep):
            out_json = tmp_dir / f"ep{ep}.json"
            cmd = [sys.executable, str(paths.SOURCE_DIR / 'running/plot_collision_torque_trace.py'),
                   "--task", task_name, "--failure", "none", "--headless",
                   "--emit-residual", str(out_json)]
            if args.consecutive_collision_frames is not None:
                cmd += ["--consecutive-collision-frames",
                        str(args.consecutive_collision_frames)]
            subprocess.run(cmd, stdin=subprocess.DEVNULL, check=True,
                           cwd=str(paths.SOURCE_DIR / 'running'))
            with open(out_json) as f:
                vals = json.load(f).get("per_joint_max") or []
            return np.asarray(vals, dtype=float) if vals else None

        residual_env = None
        print(f"calibrating {task_name}: {n_repeats} clean episodes, "
              f"{n_workers} workers")
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_run_worker, ep) for ep in range(n_repeats)]
            for done, fut in enumerate(as_completed(futs), 1):
                em = fut.result()
                if em is not None:
                    residual_env = (em if residual_env is None
                                    else np.maximum(residual_env, em))
                print(f"  episode {done}/{n_repeats} done")

        if residual_env is not None:
            path = detector.save_residual_stats(task_name, residual_env)
            print(f"residual_stats ({n_repeats} episodes x {n_workers} workers, "
                  f"element-wise max) -> {path}")
            print(f"per-joint max |r_i| (Nm): {np.round(residual_env, 2)}")
        return

    residual_env = None
    for ep in range(n_repeats):
        if n_repeats > 1:
            print(f"=== episode {ep + 1}/{n_repeats} ===")
        (rows, paused_steps, settings, success, waypoint_frames,
         episode_residual_max) = record_trace(
            task_name,
            waypoint,
            failure=args.failure,
            headless=args.headless,
            consecutive_frames=args.consecutive_collision_frames,
            capture_images_dir=(args.out_dir / "detection_frames"
                                if args.capture_images else None),
            # Suppress per-episode saving while calibrating; save the aggregate
            # envelope once below.
            save_residual=not calibrating,
        )
        if episode_residual_max is not None:
            residual_env = (episode_residual_max if residual_env is None
                            else np.maximum(residual_env, episode_residual_max))

    if calibrating and residual_env is not None:
        path = detector.save_residual_stats(task_name, residual_env)
        print(f"residual_stats ({n_repeats} episodes, element-wise max) "
              f"-> {path}")
        print(f"per-joint max |r_i| (Nm): {np.round(residual_env, 2)}")

    if args.failure == "collision":
        stem = f"{task_name}.collision.wp{waypoint}"
        label = f"collision@wp{waypoint}"
    else:
        stem = f"{task_name}.clean"
        label = "clean"

    csv_path = args.out_dir / f"{stem}.csv"
    png_path = args.out_dir / f"{stem}.png"
    write_csv(csv_path, rows)
    injected_wp = waypoint if args.failure == "collision" else None
    method = os.getenv("AHA_COLLISION_METHOD", "3").strip()
    plot_fn = plot_residual_trace if method == "3" else plot_trace
    plot_fn(png_path, rows, paused_steps, task_name, label,
            waypoint_frames=waypoint_frames, injected_wp=injected_wp)

    fired = any(str(r["collision_raw"]) == "True" for r in rows)
    print(f"task={task_name} failure={args.failure} waypoint={waypoint} success={success}")
    print(f"frames={len(rows)} raw_collision_seen={fired} flagged_steps={paused_steps[:10]}")
    print(f"torque_rule={settings['torque_rule']} consecutive={settings['consecutive_collision_frames']}")
    print(f"csv -> {csv_path}")
    print(f"plot -> {png_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task")
    parser.add_argument(
        "--tasks",
        choices=("all",),
        help="Run for every task config in failgen/configs (ignores --task). "
             "Tasks run sequentially; each still uses --workers internally. "
             "A task that errors (e.g. no collision waypoints) is logged and "
             "skipped so the sweep continues.",
    )
    parser.add_argument("--waypoint", type=int)
    parser.add_argument(
        "--failure",
        choices=("collision", "none"),
        default="collision",
        help="Use 'none' for a clean/no-collision baseline run.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Shortcut for --failure none.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--show-gui", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    parser.add_argument("--consecutive-collision-frames", type=int)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--capture-images", action="store_true",
        help="Render and save front/side camera photos at the detection frame "
             "(enables high-dim obs; slower).")
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Number of clean episodes to run for method-3 residual "
             "calibration. The saved per-joint envelope is the element-wise "
             "max over all repeats (only meaningful with --failure none).")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Run the --repeats clean episodes in parallel, each in its own "
             "subprocess/CoppeliaSim (like run_all_tasks). >1 only applies to "
             "multi-repeat --failure none calibration.")
    parser.add_argument(
        "--emit-residual", type=Path, default=None,
        help=argparse.SUPPRESS)  # internal: one worker episode -> JSON per-joint max
    args = parser.parse_args()
    if args.clean:
        args.failure = "none"

    if not args.tasks and not args.task:
        parser.error("provide --task <name> or --tasks all")

    configure_runtime_env()
    import detector
    import numpy as np

    # Internal worker mode: run exactly one clean episode and dump its per-joint
    # residual max to --emit-residual (no shared-file save, no CSV/plot). The
    # parent process aggregates these across workers.
    if args.emit_residual is not None:
        config = load_config(args.task)
        waypoint = (
            args.waypoint
            if args.waypoint is not None
            else default_collision_waypoint(config)
        )
        (_rows, _paused, _settings, _success, _frames,
         episode_residual_max) = record_trace(
            args.task, waypoint, failure=args.failure, headless=args.headless,
            consecutive_frames=args.consecutive_collision_frames,
            save_residual=False,
        )
        import json
        vals = ([] if episode_residual_max is None
                else [float(v) for v in np.asarray(episode_residual_max).ravel()])
        args.emit_residual.parent.mkdir(parents=True, exist_ok=True)
        with open(args.emit_residual, "w") as f:
            json.dump({"per_joint_max": vals}, f)
        print(f"emit_residual -> {args.emit_residual} ({vals})")
        return

    tasks = all_task_names() if args.tasks == "all" else [args.task]
    multi = len(tasks) > 1

    failures = []
    for i, task_name in enumerate(tasks, 1):
        if multi:
            print(f"\n########## [{i}/{len(tasks)}] {task_name} ##########")
        try:
            run_task(args, task_name, detector, np)
        except Exception as exc:  # keep the sweep alive across per-task errors
            if not multi:
                raise
            print(f"FAIL {task_name}: {type(exc).__name__}: {exc}")
            failures.append(task_name)

    if multi:
        ok = len(tasks) - len(failures)
        print(f"\n########## sweep done: {ok}/{len(tasks)} ok ##########")
        if failures:
            print(f"failed ({len(failures)}): {', '.join(failures)}")


if __name__ == "__main__":
    main()
