"""Batch evaluation + tuning of the live ORIENTATION detector's DRIFT mode.

The orientation detector has two options (live_detectors/orientation.py):
  * ARRIVAL (option 2): judge the settled endpoint at waypoint completion.
  * DRIFT   (option 1): fire mid-motion when the angle to the waypoint's
            ORIGINAL target trends up by >= theta_drift for N consecutive
            frames. theta_drift is GLOBAL (one value for every task) and is
            derived here from data, not hand-picked.

Pipeline (headless BT runs, no VLM, orientation detector only):

  COLLECT  per task, two runs with drift FIRING DISABLED but drift LOGGING on:
             - clean run  (also calibrates the per-waypoint arrival baseline)
             - rotation-failure run at one BT-monitored orientation waypoint
           Every frame's drift signal is appended to a CSV.
  TUNE     read the clean + failure CSVs across all tasks and pick the global
           (theta_drift, N): the smallest theta with ZERO clean false-fires at
           some small N, that still catches the most injected rotations. The
           clean transients set the floor; the failures confirm separability.
  EVAL     re-run clean (negative) + rotation-failure (positive) with MODE=drift
           and the tuned (theta, N). Fire on clean = FP; fire at the injected
           waypoint = TP. Reports a confusion matrix + drift-vs-arrival leadtime.

Run:
  python aha_scripts/main_bt_run/eval_orientation_detector.py [--tasks N]
         [--task name ...] [--workers K] [--phases collect,tune,eval]
"""

from aha_publish import paths
import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = str(paths.PROJECT_ROOT)
BT_DIR = str(paths.BT_DIR)
CFG_DIR = str(paths.FAILGEN_ROOT / 'failgen/configs')
RUNNER = str(paths.SOURCE_DIR / 'running/waypoints_interactive_bt_conditions.py')
STATS_DIR = str(paths.OUTPUT_DIR / 'orientation_arrival_stats')
OUT_DIR = str(paths.OUTPUT_DIR / 'orientation_eval')
OUT_CSV = os.path.join(OUT_DIR, "results.csv")
DRIFT_DIR = os.path.join(OUT_DIR, "drift_logs")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
TUNED_JSON = os.path.join(OUT_DIR, "tuned_drift.json")
PY = sys.executable

ROTATION_TYPES = ("rotation_x", "rotation_y", "rotation_z")
RUN_TIMEOUT = 900
# A rotation injection whose resulting arrival deviation is below this (rad) is
# too small to count as a real failure (uniform sampling can land near zero).
MIN_REAL_ROTATION = 0.05
# Global theta_drift = worst clean transient * SCALE + MARGIN. Same formula
# family as the per-waypoint arrival baseline, so the threshold stays
# data-derived and robust to the large run-to-run variance of the natural
# reorientation transient (a single clean episode underestimates it).
DRIFT_SCALE = 1.5
DRIFT_MARGIN = 0.05
DRIFT_N = 3                 # consecutive frames, fixed (filters single-frame noise)
CALIB_EPISODES = 10        # clean episodes per task (>=~10 for a stable mean+3*std)
# Gradient (option 3): rate-of-change ceiling = mean + K*std of the clean
# per-frame orientation step, pooled over all clean motion frames (statistical
# calibration like slip/collision, NOT the drift ceiling*scale formula).
GRADIENT_SIGMA_K = 4
GRADIENT_N = 2             # consecutive frames for the gradient check
ARRIVAL_SIGMA_K = 3        # (back-compat) arrival = max(floor, mean + 3*std) per wp
DIVERGE_SIGMA_K = 4        # diverge error band = mean + 4*std (per task)
ARRIVAL_FLOOR = float(os.getenv("AHA_ORIENTATION_ARRIVAL_THRESHOLD", "0.3"))  # rad floor
# New per-waypoint arrival threshold = max(floor, SCALE * worst clean arrival angle).
ARRIVAL_SCALE = float(os.getenv("AHA_ORIENTATION_ARRIVAL_SCALE", "1.3"))
# New per-waypoint C4 PROP factor = max(PROP_K_MIN, PROP_RATIO_SCALE * worst clean
# prop ratio); clean ratio = angle / max(median-of-last-WINDOW, PROP_FLOOR),
# recomputed offline exactly like the live orientation detector.
PROP_K_MIN = float(os.getenv("AHA_ORIENTATION_PROP_K", "1.5"))
PROP_RATIO_SCALE = float(os.getenv("AHA_ORIENTATION_PROP_RATIO_SCALE", "1.1"))
PROP_WINDOW = int(os.getenv("AHA_ORIENTATION_PROP_WINDOW", "5"))
PROP_FLOOR = float(os.getenv("AHA_ORIENTATION_PROP_FLOOR", "0.1"))

COPPELIA = str(paths.COPPELIASIM_ROOT)


def base_env(extra=None):
    env = dict(os.environ)
    env["COPPELIASIM_ROOT"] = COPPELIA
    env["LD_LIBRARY_PATH"] = env.get("LD_LIBRARY_PATH", "") + ":" + COPPELIA
    env["QT_QPA_PLATFORM_PLUGIN_PATH"] = COPPELIA
    env.pop("QT_QPA_PLATFORM", None)            # need a real GL context
    env["DISPLAY"] = os.environ.get("DISPLAY", ":1")
    env["AHA_LIVE_DETECTORS"] = "orientation"   # isolate the orientation detector
    env["AHA_FAIL_DEBUG"] = "1"
    env["AHA_NO_SIDE_CAMERA"] = "1"             # no VLM/window here; skip the
                                                 # per-step side-camera render
    if extra:
        env.update(extra)
    return env


def prepared_tasks():
    return sorted(f[:-len(".bt_conditions.json")]
                  for f in os.listdir(BT_DIR) if f.endswith(".bt_conditions.json"))


def load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def bt_orientation_waypoints(task):
    """Waypoint indices where the prepared BT monitors the orientation detector."""
    d = json.load(open(os.path.join(BT_DIR, f"{task}.bt_conditions.json")))
    stages = (d.get("review") or {}).get("stages") or (d.get("generated") or {}).get("stages") or []
    out = []
    for idx, st in enumerate(stages):
        for hc in (st.get("hold_conditions") or []):
            if hc.get("selected") is False:
                continue
            det = str(hc.get("detector") or hc.get("failure") or "").lower()
            links = [str(l.get("failure")).lower()
                     for l in (hc.get("failure_links") or []) if isinstance(l, dict)]
            cond = str(hc.get("condition") or hc.get("original_condition") or "")
            if det == "orientation" or "orientation" in links or "orientation_maintained" in cond:
                out.append(idx)
                break
    return out


def pick_positive(task):
    """Choose (failtype, waypoint): a rotation failure at the first BT-monitored
    orientation waypoint that also has such a failure configured."""
    cfg = load_yaml(os.path.join(CFG_DIR, f"{task}.yaml"))
    by_wp = {}
    for f in cfg.get("failures", []):
        ty = f.get("type")
        if ty in ROTATION_TYPES:
            for wp in f.get("waypoints", []):
                by_wp.setdefault(int(wp), []).append(ty)
    monitored = bt_orientation_waypoints(task)
    for wp in monitored:
        if wp in by_wp:
            for ty in ROTATION_TYPES:
                if ty in by_wp[wp]:
                    return ty, wp
    # fall back to any waypoint with a rotation failure
    for wp in sorted(by_wp):
        return by_wp[wp][0], wp
    return None, None


def pick_all_positives(task):
    """Every (failtype, waypoint) combo: each rotation failure type configured
    at each BT-monitored orientation waypoint (not just the first/preferred one)."""
    cfg = load_yaml(os.path.join(CFG_DIR, f"{task}.yaml"))
    by_wp = {}   # wp -> list of rotation failtypes available
    for f in cfg.get("failures", []):
        ty = f.get("type")
        if ty in ROTATION_TYPES:
            for wp in f.get("waypoints", []):
                by_wp.setdefault(int(wp), []).append(ty)
    monitored = bt_orientation_waypoints(task)
    out = []
    for wp in monitored:
        for ty in by_wp.get(wp, []):
            out.append((ty, wp))
    return out


SUMMARY_FIRE = re.compile(r"orientation\s+DETECTED orientation x(\d+)")
SUMMARY_OK = re.compile(r"orientation\s+ok — no orientation")
SUMMARY_DISABLED = re.compile(r"orientation\s+DISABLED")
SUMMARY_SKIPPED = re.compile(r"orientation\s+skipped")
FIRE_LINE = re.compile(r"\[DETECTOR\] ORIENTATION failure at waypoint (\d+) \(step (\d+)\)")
FIRE_MODE_LINE = re.compile(r"\[DETECTOR\] ORIENTATION failure at waypoint \d+ \(step \d+\) mode=(\w+)")
FIRED_BY = re.compile(r"fired_by=(\w+)")
WP_VAL = re.compile(r"wp(\d+)=([0-9.]+)")
HEADER_LINE = re.compile(r"Waypoint (\d+) of \d+")


def _arrival_diag(stdout):
    """Per-waypoint arrival deviation (rad). The diag line contains both
    'per-waypoint:' and 'drift-peak per-waypoint:'; isolate the arrival one."""
    per_wp = {}
    for line in stdout.splitlines():
        if "per-waypoint:" not in line:
            continue
        arrival_part = line.split("drift-peak per-waypoint:")[0]
        seg = arrival_part.split("per-waypoint:", 1)
        if len(seg) == 2:
            for wp, val in WP_VAL.findall(seg[1]):
                per_wp[int(wp)] = float(val)
    return per_wp


def parse_run(stdout):
    fires = FIRE_LINE.findall(stdout)
    fired_wps = sorted(int(w) for w, _ in fires)
    fire_steps = {int(w): int(s) for w, s in fires}
    fired_steps = [int(s) for _, s in fires]
    fired_modes = FIRE_MODE_LINE.findall(stdout)
    status = "ran"
    fired = False
    if SUMMARY_FIRE.search(stdout):
        fired = True
    elif SUMMARY_OK.search(stdout):
        fired = False
    elif SUMMARY_DISABLED.search(stdout):
        status = "disabled"
    elif SUMMARY_SKIPPED.search(stdout):
        status = "skipped"
    else:
        status = "no_summary"
    fired_by = FIRED_BY.search(stdout)
    return {"fired": fired or bool(fired_wps), "fired_waypoints": fired_wps,
            "fire_steps": fire_steps, "fired_steps": fired_steps,
            "fired_modes": fired_modes, "status": status,
            "fired_by": fired_by.group(1) if fired_by else "",
            "per_wp": _arrival_diag(stdout)}


def plot_path_for(task, phase, failtype=None, wp=None):
    if phase == "negative":
        stem = f"{task}.clean"
    else:
        stem = f"{task}.{failtype}.wp{wp}"
    return os.path.join(PLOTS_DIR, f"{stem}.png")


def drift_log_path_for(task, phase, failtype=None, wp=None):
    if phase == "negative":
        stem = f"{task}.clean"
    else:
        stem = f"{task}.{failtype}.wp{wp}"
    return os.path.join(PLOTS_DIR, f"{stem}.csv")


def run_bt(task, *, failure="none", failure_waypoint=None, env_extra=None,
           plot_path=None, drift_log_path=None, plot_label=None,
           stop_after_wp=None):
    """Run one headless BT episode. If stop_after_wp is set, the subprocess is
    killed the instant it's no longer possible to change the verdict: either a
    live ORIENTATION fire is seen, or the BT's waypoint header advances past
    stop_after_wp with no fire (miss). An early-killed run never reaches the
    final "Live detector summary" line, so info["per_wp"] (the measured arrival
    deviation) comes back empty -- the fired/fired_waypoints verdict itself is
    unaffected since FIRE_LINE is matched live, streamed as it happens."""
    cmd = [PY, "-u", RUNNER, "--task", task, "--failure", failure,
           "--headless", "--mode", "auto", "--vlm-checks", "off",
           "--no-side-camera-window"]
    run_env_extra = dict(env_extra or {})
    if plot_path:
        run_env_extra["MPLBACKEND"] = "Agg"
    if drift_log_path:
        os.makedirs(os.path.dirname(drift_log_path), exist_ok=True)
        if os.path.exists(drift_log_path):
            os.remove(drift_log_path)
        run_env_extra["AHA_ORIENTATION_DRIFT_LOG"] = drift_log_path
        run_env_extra["AHA_ORIENTATION_DRIFT_LOG_STREAM"] = "1"
        run_env_extra["AHA_ORIENTATION_DRIFT_LOG_STREAM_EVERY"] = "10"
    cmd.append("--no-detector-plots")
    if failure != "none" and failure_waypoint is not None:
        cmd += ["--failure-waypoint", str(failure_waypoint)]
    t0 = time.time()
    p = subprocess.Popen(cmd, env=base_env(run_env_extra), stdin=subprocess.DEVNULL,
                         cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, start_new_session=True)

    def _kill():
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass

    wd = threading.Timer(RUN_TIMEOUT, _kill)
    wd.daemon = True
    wd.start()
    lines = []
    timed_out = False
    try:
        for line in p.stdout:
            lines.append(line)
            if time.time() - t0 > RUN_TIMEOUT:
                timed_out = True
                break
            if stop_after_wp is not None:
                if FIRE_LINE.search(line):
                    break   # live fire is enough to score the run
                h = HEADER_LINE.search(line)
                if h and int(h.group(1)) > stop_after_wp:
                    break   # advanced past the last waypoint we care about
    finally:
        wd.cancel()
        _kill()
        try:
            p.wait(timeout=10)
        except Exception:
            pass
    out = "".join(lines)
    info = parse_run(out)
    info["elapsed"] = round(time.time() - t0, 1)
    info["stdout"] = out
    info["plot"] = plot_path or ""
    if timed_out:
        info["status"] = "timeout"
    return info


# ----------------------------- COLLECT --------------------------------------

def collect(task, episodes=CALIB_EPISODES, with_fail=True):
    """K clean runs (write the rich baseline + log clean per-frame signals) and,
    when with_fail, one rotation-failure run (log failure drift for drift-theta
    tuning / eval). Drift firing disabled (theta huge) so the BT always runs
    every waypoint and we get full trajectories. Multiple clean episodes are
    essential: the natural transient varies run-to-run and a single episode
    underestimates the mean/std."""
    os.makedirs(DRIFT_DIR, exist_ok=True)
    ft, wp = pick_positive(task)
    no_fire = {"AHA_ORIENTATION_MODE": "arrival",
               "AHA_ORIENTATION_ARRIVAL_THRESHOLD": "100",
               "AHA_ORIENTATION_DRIFT_INCREASE": "100"}

    # clear any stale logs for this task
    for f in os.listdir(DRIFT_DIR):
        if f.startswith(f"{task}.clean.ep") or f == f"{task}.fail.csv":
            os.remove(os.path.join(DRIFT_DIR, f))

    last_status = None
    for i in range(max(1, episodes)):
        clean_log = os.path.join(DRIFT_DIR, f"{task}.clean.ep{i}.csv")
        clean = run_bt(task, failure="none",
                       env_extra={**no_fire, "AHA_ORIENTATION_DRIFT_LOG": clean_log})
        last_status = clean["status"]
    # Rich baseline (arrival mean/std per wp, gradient & diverge thresholds) from
    # the per-frame clean CSVs.
    base = _compute_clean_baseline(task, episodes)
    os.makedirs(STATS_DIR, exist_ok=True)
    with open(os.path.join(STATS_DIR, f"{task}.json"), "w") as f:
        json.dump(base, f, indent=2)

    fail = {"status": "skipped" if with_fail else "baseline_only"}
    if with_fail and ft is not None:
        fail_log = os.path.join(DRIFT_DIR, f"{task}.fail.csv")
        fail = run_bt(task, failure=ft, failure_waypoint=wp,
                      env_extra={**no_fire, "AHA_ORIENTATION_DRIFT_LOG": fail_log})
    return {"task": task, "failtype": ft, "inj_wp": wp,
            "clean_status": last_status, "fail_status": fail.get("status"),
            "n_waypoints": len(base.get("arrival_stats_by_waypoint", {})),
            "gradient_threshold": base.get("gradient_threshold"),
            "diverge_error_threshold": base.get("diverge_error_threshold")}


def calibrate(task, episodes=CALIB_EPISODES):
    """Run K clean episodes with a huge arrival threshold and per-frame logging
    on (never fires => full episode logged), then build the statistical per-task
    baseline (arrival mean/std per wp, gradient & diverge thresholds) from the
    clean per-frame CSVs. Clean-only analog of collect(); returns (last_info,
    base) exactly like the transition harness so the pipeline can drive it."""
    os.makedirs(DRIFT_DIR, exist_ok=True)
    no_fire = {"AHA_ORIENTATION_MODE": "arrival",
               "AHA_ORIENTATION_ARRIVAL_THRESHOLD": "100"}
    stop_wp = max(bt_orientation_waypoints(task), default=None)

    # clear any stale clean logs for this task
    for f in os.listdir(DRIFT_DIR):
        if f.startswith(f"{task}.clean.ep"):
            os.remove(os.path.join(DRIFT_DIR, f))

    last = None
    for i in range(max(1, episodes)):
        clean_log = os.path.join(DRIFT_DIR, f"{task}.clean.ep{i}.csv")
        last = run_bt(task, failure="none", stop_after_wp=stop_wp,
                      env_extra={**no_fire, "AHA_ORIENTATION_DRIFT_LOG": clean_log})
    base = _compute_clean_baseline(task, episodes)
    os.makedirs(STATS_DIR, exist_ok=True)
    with open(os.path.join(STATS_DIR, f"{task}.json"), "w") as f:
        json.dump(base, f, indent=2)
    return last, base


def _max_runup(series):
    """Cumulative positive increase of an error series.

    Adds only positive frame-to-frame increases; decreases do not subtract.
    """
    if not series:
        return 0.0
    total = 0.0
    prev = series[0]
    for value in series[1:]:
        total += max(0.0, value - prev)
        prev = value
    return total


def _runup_series(series):
    """Cumulative positive runup value for every frame."""
    if not series:
        return []
    total = 0.0
    prev = series[0]
    out = []
    for i, value in enumerate(series):
        if i > 0:
            total += max(0.0, value - prev)
        out.append(total)
        prev = value
    return out


def _angle_value(row):
    for key in ("angle_to_target", "angle_to_original"):
        try:
            value = float(row.get(key, "nan"))
        except ValueError:
            continue
        if value == value:
            return value
    return float("nan")


def _robust_std_around_median(values):
    import numpy as np

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return 1.4826 * mad


def _episode_index(path):
    match = re.search(r'\.clean\.ep(\d+)\.csv$', os.path.basename(path))
    return int(match.group(1)) if match else 10**9


def _max_prop_ratio(series, window=PROP_WINDOW, floor=PROP_FLOOR):
    """Worst clean C4-PROP ratio over an ordered list of per-frame angles.

    Mirrors the live orientation detector: at frame i (non-empty history) the
    ratio is angle_i / max(median(last `window` angles before i), floor). The max
    over all frames is the largest a clean run ever pushes prop, so a per-waypoint
    factor of PROP_RATIO_SCALE * this stays just above clean."""
    import numpy as np
    recent = []
    worst = 0.0
    for ang in series:
        if not (ang == ang):
            continue
        if recent:
            ref = max(float(np.median(recent)), floor)
            if ref > 0:
                worst = max(worst, ang / ref)
        recent.append(ang)
        if len(recent) > window:
            recent.pop(0)
    return worst


def _compute_clean_baseline(task, episodes):
    """Per-task baselines from the clean per-frame CSVs. Anchored on the two
    plots: PLOT 1 = error-to-target (angle_to_original); PLOT 2 = its change.
      * error_stats: PLOT-1 max/min/mean/std over clean motion frames.
      * segment_rise_by_waypoint: PLOT-2 analysis -- per waypoint segment, does
        the error ever rise on a clean run (any positive change) and the largest
        such run-up (how much PLOT 1 grows), worst over episodes.
      * arrival_stats_by_waypoint: per wp, mean/std of the arrival error.
      * gradient_threshold = step mean+K*std; diverge_error_threshold =
        error mean+K*std (pooled motion frames)."""
    import glob
    import numpy as np
    per_wp_arrivals = {}    # wp -> [per-episode pre-arrival error]
    per_wp_rise = {}        # wp -> [per-episode max run-up]
    per_wp_positive_runup_samples = {}  # wp -> all positive second-plot samples
    per_wp_pos = {}         # wp -> [per-episode any-positive bool]
    per_wp_prop_ratio = {}  # wp -> [per-episode worst clean C4-prop ratio]
    steps = []              # pooled per-frame gradient (motion frames)
    errors = []             # pooled angle_to_original (motion frames)
    clean_paths = sorted(
        glob.glob(os.path.join(DRIFT_DIR, f"{task}.clean.ep*.csv")),
        key=_episode_index,
    )
    clean_paths = clean_paths[:max(1, int(episodes))]
    for cpath in clean_paths:
        ep_arrival = {}     # wp -> last in-motion error before waypoint completion
        ep_series = {}      # wp -> ordered list of motion-frame errors
        with open(cpath) as f:
            for r in csv.DictReader(f):
                wp = int(r["waypoint"])
                done = r.get("path_done") == "1"
                ang = _angle_value(r)
                try:
                    god = float(r.get("gripper_orientation_delta", "nan"))
                except ValueError:
                    continue
                if done:
                    continue
                else:
                    if ang == ang:
                        errors.append(ang)
                        ep_series.setdefault(wp, []).append(ang)
                        ep_arrival[wp] = ang
                    if god == god:
                        steps.append(god)
        for wp, v in ep_arrival.items():
            per_wp_arrivals.setdefault(wp, []).append(v)
        for wp, s in ep_series.items():
            per_wp_rise.setdefault(wp, []).append(_max_runup(s))
            per_wp_positive_runup_samples.setdefault(wp, []).extend(
                value for value in _runup_series(s) if value > 0.0)
            per_wp_pos.setdefault(wp, []).append(
                any(s[j] > s[j - 1] for j in range(1, len(s))))
            per_wp_prop_ratio.setdefault(wp, []).append(_max_prop_ratio(s))

    arrival_stats = {}
    for wp, vals in sorted(per_wp_arrivals.items()):
        a = np.asarray(vals, dtype=float)
        median = float(np.median(a))
        std = float(a.std())
        robust_std = _robust_std_around_median(a)
        arrival_stats[str(wp)] = {
            "mean": round(float(a.mean()), 5),
            "median": round(median, 5),
            "std": round(std, 5),
            "robust_std": round(robust_std, 5),
            "max": round(float(a.max()), 5),
            "threshold": round(
                float(max(ARRIVAL_FLOOR, ARRIVAL_SCALE * float(a.max()))),
                5,
            ),
            "source": "last_motion_frame_before_waypoint_done",
            "n": int(a.size),
        }
    prop_stats = {}
    for wp in sorted(per_wp_prop_ratio):
        ratios = np.asarray(per_wp_prop_ratio[wp], dtype=float)
        max_ratio = float(ratios.max()) if ratios.size else 0.0
        prop_stats[str(wp)] = {
            "max_ratio": round(max_ratio, 5),
            "mean_ratio": round(float(ratios.mean()), 5) if ratios.size else 0.0,
            "prop_k": round(float(max(PROP_K_MIN, PROP_RATIO_SCALE * max_ratio)), 5),
            "window": PROP_WINDOW,
            "floor": PROP_FLOOR,
            "source": "worst_clean_angle_over_median_of_last_window",
            "n": int(ratios.size),
        }
    segment_rise = {}
    for wp in sorted(per_wp_rise):
        rises = np.asarray(per_wp_rise[wp], dtype=float)
        positive_samples = np.asarray(
            per_wp_positive_runup_samples.get(wp, []), dtype=float)
        median = float(np.median(rises))
        std = float(rises.std())
        positive_std = float(positive_samples.std()) if positive_samples.size else 0.0
        robust_std = _robust_std_around_median(rises)
        segment_rise[str(wp)] = {
            "any_positive": bool(any(per_wp_pos.get(wp, []))),
            "max_rise": round(float(rises.max()), 5),
            "mean_rise": round(float(rises.mean()), 5),
            "median_rise": round(median, 5),
            "std_rise": round(std, 5),
            "positive_std_rise": round(positive_std, 5),
            "robust_std_rise": round(robust_std, 5),
            "threshold": round(
                float(max(ARRIVAL_FLOOR, 1.5 * float(rises.max()))),
                5,
            ),
            "source": "max_runup_of_angle_to_target_between_waypoint_boundaries",
            "n": int(rises.size)}
    g = np.asarray(steps, dtype=float) if steps else np.zeros(1)
    e = np.asarray(errors, dtype=float) if errors else np.zeros(1)
    grad_thr = round(float(g.mean() + GRADIENT_SIGMA_K * g.std()), 5)
    div_thr = round(float(e.mean() + DIVERGE_SIGMA_K * e.std()), 5)
    return {
        "task": task,
        "n_clean_episodes": max(1, episodes),
        "signal": "orientation: PLOT1=angle to waypoint-defined orientation (rad)",
        "error_stats": {"max": round(float(e.max()), 5),
                        "min": round(float(e.min()), 5),
                        "mean": round(float(e.mean()), 5),
                        "std": round(float(e.std()), 5),
                        "n_frames": int(e.size)},
        "segment_rise_by_waypoint": segment_rise,
        "arrival_stats_by_waypoint": arrival_stats,
        "prop_stats_by_waypoint": prop_stats,
        "arrival_sigma_k": ARRIVAL_SIGMA_K,
        "arrival_scale": ARRIVAL_SCALE,
        "arrival_floor": ARRIVAL_FLOOR,
        "prop_k_min": PROP_K_MIN,
        "prop_ratio_scale": PROP_RATIO_SCALE,
        "gradient_threshold": grad_thr,
        "gradient_clean_mean": round(float(g.mean()), 5),
        "gradient_clean_std": round(float(g.std()), 5),
        "gradient_sigma_k": GRADIENT_SIGMA_K,
        "diverge_error_threshold": div_thr,
        "diverge_error_mean": round(float(e.mean()), 5),
        "diverge_error_std": round(float(e.std()), 5),
        "diverge_sigma_k": DIVERGE_SIGMA_K,
        "design": ("PLOT1 error_stats = max/min/mean/std of angle-to-target; "
                   "PLOT2 segment_rise = any positive change between waypoints "
                   "and the run-up magnitude; arrival[wp]=max(floor,median+%g*robust_std) "
                   "using the last in-motion frame before waypoint completion; "
                   "gradient=step mean+%g*std; diverge=error mean+%g*std"
                   % (ARRIVAL_SIGMA_K, GRADIENT_SIGMA_K, DIVERGE_SIGMA_K)),
    }


# ------------------------------- TUNE ---------------------------------------

def _read_drift_csv(path):
    """task -> waypoint -> list of (drift_increase, increasing_flag, path_done)
    over the motion (path_done == 0)."""
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                inc = float(r["drift_increase"])
            except (ValueError, KeyError):
                continue
            wp = int(r["waypoint"])
            flag = r.get("increasing_flag") == "1"
            done = r.get("path_done") == "1"
            out.setdefault(r["task"], {}).setdefault(wp, []).append((inc, flag, done))
    return out


def _max_consec_ge(series, theta):
    """Longest run of consecutive frames with finite increase >= theta (during
    motion, increasing_flag set)."""
    best = cur = 0
    for inc, flag, done in series:
        ok = flag and (inc == inc) and inc >= theta and not done
        cur = cur + 1 if ok else 0
        best = max(best, cur)
    return best


def _clean_gradient_stats(tasks):
    """Pool the per-frame gripper-orientation step (rad/frame) over all clean
    MOTION frames across tasks; return (mean, std, max, n). The global gradient
    threshold = mean + K*std (statistical calibration, like slip/collision)."""
    import glob
    vals = []
    for t in tasks:
        for cpath in glob.glob(os.path.join(DRIFT_DIR, f"{t}.clean.ep*.csv")):
            if not os.path.exists(cpath):
                continue
            with open(cpath) as f:
                for r in csv.DictReader(f):
                    if r.get("path_done") == "1":
                        continue
                    try:
                        god = float(r.get("gripper_orientation_delta", "nan"))
                    except ValueError:
                        continue
                    if god == god:  # not NaN
                        vals.append(god)
    if not vals:
        return 0.0, 0.0, 0.0, 0
    import numpy as np
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std()), float(a.max()), len(a)


def tune(tasks):
    """Global theta_drift = worst clean transient * SCALE + MARGIN, derived from
    ALL clean episodes across tasks (same formula as the arrival baseline). Then
    verify it catches the injected rotations at N consec frames. Data-derived and
    robust to run-to-run variance of the natural reorientation transient."""
    import glob
    clean_series = []   # one entry per (task, episode, waypoint)
    clean_peak = 0.0
    fail = {}           # task -> wp -> series
    inj = {}            # task -> injected wp
    for t in tasks:
        for cpath in sorted(glob.glob(os.path.join(DRIFT_DIR, f"{t}.clean.ep*.csv"))):
            for _, wps in _read_drift_csv(cpath).items():
                for wp, s in wps.items():
                    peak = max((i for i, fl, d in s if fl and not d), default=0.0)
                    clean_peak = max(clean_peak, peak)
                    clean_series.append(s)
        for tk, series in _read_drift_csv(os.path.join(DRIFT_DIR, f"{t}.fail.csv")).items():
            fail.setdefault(tk, {}).update(series)
        ft, wp = pick_positive(t)
        if wp is not None:
            inj[t] = wp

    theta = round(clean_peak * DRIFT_SCALE + DRIFT_MARGIN, 3)
    N = DRIFT_N

    # verify: held-in clean false-fires (should be 0) and injected-rotation recall
    fp = sum(1 for s in clean_series if _max_consec_ge(s, theta) >= N)
    fail_peaks = {t: max((i for i, fl, d in fail.get(t, {}).get(wp, []) if fl and not d),
                         default=0.0) for t, wp in inj.items()}
    tp = sum(1 for t, wp in inj.items()
             if _max_consec_ge(fail.get(t, {}).get(wp, []), theta) >= N)
    min_fail_peak = min(fail_peaks.values()) if fail_peaks else 0.0

    # Gradient (option 3): global rate threshold = mean + K*std of clean motion.
    g_mean, g_std, g_max, g_n = _clean_gradient_stats(tasks)
    grad_thr = round(g_mean + GRADIENT_SIGMA_K * g_std, 4)

    result = {"theta_drift": theta, "consecutive_frames": N,
              "clean_drift_ceiling": round(clean_peak, 4),
              "scale": DRIFT_SCALE, "margin": DRIFT_MARGIN,
              "n_clean_series": len(clean_series),
              "clean_false_fires": fp, "true_positives": tp,
              "n_positives": len(inj),
              "min_failure_peak": round(min_fail_peak, 4),
              "headroom": round(min_fail_peak - theta, 4),
              "gradient_threshold": grad_thr,
              "gradient_clean_mean": round(g_mean, 4),
              "gradient_clean_std": round(g_std, 4),
              "gradient_clean_max": round(g_max, 4),
              "gradient_sigma_k": GRADIENT_SIGMA_K,
              "gradient_n_clean_frames": g_n,
              "gradient_consecutive_frames": GRADIENT_N,
              "design": "global theta_drift = worst clean transient * %.2f + %.2f; "
                        "gradient_threshold = clean per-frame-step mean + %d*std "
                        "(pooled over all clean motion frames)"
                        % (DRIFT_SCALE, DRIFT_MARGIN, GRADIENT_SIGMA_K)}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(TUNED_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print("\n=== TUNE (global, data-derived) ===")
    print(f"  worst clean transient over {len(clean_series)} clean episode-waypoints: "
          f"{result['clean_drift_ceiling']} rad")
    print(f"  global theta_drift = {theta} rad  (= {result['clean_drift_ceiling']}"
          f"*{DRIFT_SCALE}+{DRIFT_MARGIN}),  N = {N} consec frames")
    print(f"  held-in check: clean false-fires={fp}/{len(clean_series)}, "
          f"injected-rotation TP={tp}/{len(inj)}")
    print(f"  min failure peak={result['min_failure_peak']} rad  "
          f"=> headroom above theta = {result['headroom']} rad")
    print(f"  GRADIENT: clean per-frame step mean={g_mean:.4f} std={g_std:.4f} "
          f"(max={g_max:.4f}, n={g_n})  =>  threshold = mean+{GRADIENT_SIGMA_K}*std "
          f"= {grad_thr} rad/frame, N = {GRADIENT_N} consec")
    print(f"  written to {TUNED_JSON}")
    return result


# ------------------------------- EVAL ---------------------------------------

FIELDS = ["task", "phase", "failtype", "inj_wp", "arrival_dev", "verdict",
          "fired", "fired_waypoints", "fired_by", "fire_step", "status", "elapsed"]


def write_rows(rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    exists = os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def eval_tasks(tasks, tuned, workers):
    theta = str(tuned["theta_drift"])
    N = str(tuned["consecutive_frames"])
    drift_env = {"AHA_ORIENTATION_MODE": "drift",
                 "AHA_ORIENTATION_DRIFT_INCREASE": theta,
                 "AHA_ORIENTATION_DRIFT_CONSECUTIVE": N}
    jobs = []
    for t in tasks:
        ft, wp = pick_positive(t)
        jobs.append(("negative", t, None, None))
        jobs.append(("positive", t, ft, wp))

    def do(job):
        phase, t, ft, wp = job
        if phase == "negative":
            info = run_bt(t, failure="none", env_extra=drift_env)
            return {"task": t, "phase": phase, "failtype": "none", "inj_wp": "",
                    "verdict": "FP" if info["fired"] else "TN",
                    "fired": info["fired"], "fired_waypoints": info["fired_waypoints"],
                    "fired_by": info["fired_by"], "status": info["status"],
                    "elapsed": info["elapsed"]}
        if ft is None:
            return {"task": t, "phase": phase, "verdict": "SKIP",
                    "failtype": "none", "inj_wp": "", "fired": "",
                    "fired_waypoints": "", "status": "no_rotation_case", "elapsed": 0}
        info = run_bt(t, failure=ft, failure_waypoint=wp, env_extra=drift_env)
        hit = wp in info["fired_waypoints"]
        verdict = "TP" if hit else ("FN_fired_other" if info["fired"] else "FN")
        return {"task": t, "phase": phase, "failtype": ft, "inj_wp": wp,
                "arrival_dev": info["per_wp"].get(wp, ""),
                "verdict": verdict, "fired": info["fired"],
                "fired_waypoints": info["fired_waypoints"],
                "fired_by": info["fired_by"],
                "fire_step": info["fire_steps"].get(wp, ""),
                "status": info["status"], "elapsed": info["elapsed"]}

    print(f"\n=== EVAL drift mode (theta={theta} rad, N={N}) — {len(jobs)} runs ===")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(do, j): j for j in jobs}
        for fut in as_completed(futs):
            row = fut.result()
            write_rows([row])
            print(f"  {row['phase']:<8} {row['task']:<26} {row['verdict']:<14} "
                  f"fired_wps={row.get('fired_waypoints')} inj_wp={row.get('inj_wp')} "
                  f"({row['status']}, {row['elapsed']}s)", flush=True)
    summarize()


def summarize():
    if not os.path.exists(OUT_CSV):
        print("no results yet")
        return
    rows = list(csv.DictReader(open(OUT_CSV)))
    latest = {}
    for r in rows:
        if r["phase"] in ("negative", "positive"):
            latest[(r["task"], r["phase"])] = r
    TP = sum(1 for r in latest.values() if r["verdict"] == "TP")
    FN = sum(1 for r in latest.values() if r["verdict"].startswith("FN"))
    FP = sum(1 for r in latest.values() if r["verdict"] == "FP")
    TN = sum(1 for r in latest.values() if r["verdict"] == "TN")
    SK = sum(1 for r in latest.values() if r["verdict"] == "SKIP")
    print("\n" + "=" * 60)
    print("  ORIENTATION DETECTOR (DRIFT mode) — confusion matrix")
    print("=" * 60)
    print(f"  Positives (injected rotation):  TP={TP}  FN={FN}")
    print(f"  Negatives (clean):              TN={TN}  FP={FP}")
    if SK:
        print(f"  Skipped (no rotation case):     {SK}")
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec = TP / (TP + FN) if (TP + FN) else 0.0
    print(f"  Precision={prec:.2f}  Recall={rec:.2f}")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=10, help="first N prepared tasks")
    ap.add_argument("--task", action="append", default=[], help="explicit task(s)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--calib-episodes", type=int, default=CALIB_EPISODES)
    ap.add_argument("--phases", default="collect,tune,eval")
    args = ap.parse_args()

    tasks = args.task or prepared_tasks()[:args.tasks]
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    print(f"tasks ({len(tasks)}): {', '.join(tasks)}")

    # "calibrate" = clean episodes + write baseline only (no failure run); used
    # for the all-tasks baseline rebuild. "collect" additionally runs the failure
    # for drift-theta tuning / eval.
    if "collect" in phases or "calibrate" in phases:
        with_fail = ("tune" in phases or "eval" in phases) and "collect" in phases
        label = "COLLECT" if with_fail else "CALIBRATE"
        print(f"\n=== {label} ({len(tasks)} tasks, {args.calib_episodes} clean "
              f"episodes each) ===", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(collect, t, args.calib_episodes, with_fail): t
                    for t in tasks}
            for fut in as_completed(futs):
                info = fut.result()
                print(f"  {label.lower()} {info['task']:<26} "
                      f"clean={info['clean_status']} wp={info['n_waypoints']} "
                      f"grad_thr={info['gradient_threshold']} "
                      f"diverge_thr={info['diverge_error_threshold']}", flush=True)

    tuned = None
    if "tune" in phases:
        tuned = tune(tasks)
    if tuned is None and os.path.exists(TUNED_JSON):
        tuned = json.load(open(TUNED_JSON))

    if "eval" in phases:
        if not tuned:
            print("no tuned threshold available; run the tune phase first")
            return
        eval_tasks(tasks, tuned, args.workers)


if __name__ == "__main__":
    main()
