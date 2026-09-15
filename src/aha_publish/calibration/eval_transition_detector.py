"""Batch evaluation of the live TRANSITION detector over the prepared-BT tasks.

Three phases per task (each phase = one headless BT run, no VLM, transition
detector only):

  1. CALIBRATE  K clean runs with a huge arrival threshold (MODE=arrival) so the
                detector never fires and the BT runs every waypoint, with
                AHA_TRANSITION_DRIFT_LOG on to log every frame. The statistical
                per-task baseline (arrival mean/std per waypoint, gradient &
                diverge thresholds) is derived from those CSVs and written to
                aha_output/transition_arrival_stats/<task>.json.
  2. NEGATIVE   clean run with the calibrated baseline loaded. A fire = FP, no
                fire = TN. (Held-out: a different random episode than calibrate.)
  3. POSITIVE   a translation failure injected at one BT-monitored waypoint, with
                the baseline loaded. Fire at the injected waypoint = TP, else FN.

Run:
  python aha_scripts/main_bt_run/eval_transition_detector.py [--tasks N]
         [--workers K] [--phases calibrate,negative,positive] [--task name ...]

Results stream to aha_output/transition_eval/results.csv and a confusion matrix
is printed at the end (also re-printable with --summarize).
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
STATS_DIR = str(paths.OUTPUT_DIR / 'transition_arrival_stats')
OUT_DIR = str(paths.OUTPUT_DIR / 'transition_eval')
OUT_CSV = os.path.join(OUT_DIR, "results.csv")
DRIFT_DIR = os.path.join(OUT_DIR, "drift_logs")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
PY = sys.executable

TRANSLATION_TYPES = ("translation_x", "translation_y", "translation_z", "translation")
RUN_TIMEOUT = 900

# Statistical per-task calibration (mirrors the orientation harness, in meters):
#   arrival[wp] threshold = max(floor, median + K*robust_std) of the clean arrival
#       distance (last in-motion frame before path_done) per waypoint.
#   gradient_threshold = mean + K*std of the per-frame gripper_motion over all
#       clean MOTION frames (pooled), like slip/collision.
#   diverge_error_threshold = mean + K*std of dist_to_original over all clean
#       MOTION frames (pooled).
GRADIENT_SIGMA_K = 4
DIVERGE_SIGMA_K = 4
ARRIVAL_SIGMA_K = 3        # (back-compat) arrival = max(floor, mean + 3*std) per wp
ARRIVAL_FLOOR = float(os.getenv("AHA_TRANSITION_ARRIVAL_THRESHOLD", "0.02"))  # m floor
# New per-waypoint arrival threshold = max(floor, SCALE * worst clean arrival dist).
ARRIVAL_SCALE = float(os.getenv("AHA_TRANSITION_ARRIVAL_SCALE", "1.3"))
# New per-waypoint C4 PROP factor = max(PROP_K_MIN, PROP_RATIO_SCALE * worst clean
# prop ratio). The clean prop ratio at a frame = dist / max(median-of-last-WINDOW,
# PROP_FLOOR), recomputed offline exactly like the live detector so a clean run
# never trips prop while a real "reached then diverged" still fires.
PROP_K_MIN = float(os.getenv("AHA_TRANSITION_PROP_K", "1.5"))
PROP_RATIO_SCALE = float(os.getenv("AHA_TRANSITION_PROP_RATIO_SCALE", "1.1"))
PROP_WINDOW = int(os.getenv("AHA_TRANSITION_PROP_WINDOW", "5"))
PROP_FLOOR = float(os.getenv("AHA_TRANSITION_PROP_FLOOR", "0.01"))
GRADIENT_N = 2             # consecutive frames for the gradient check
CALIB_EPISODES = 10        # clean episodes per task for stable thresholds

COPPELIA = str(paths.COPPELIASIM_ROOT)


def base_env(extra=None):
    # Mirrors the user's run_orient_detector_eval.py environment so behavior
    # matches the orientation eval: a real GL context for the vision sensors
    # (no offscreen), DISPLAY set, only the transition detector enabled, no VLM.
    env = dict(os.environ)
    env["COPPELIASIM_ROOT"] = COPPELIA
    env["LD_LIBRARY_PATH"] = env.get("LD_LIBRARY_PATH", "") + ":" + COPPELIA
    env["QT_QPA_PLATFORM_PLUGIN_PATH"] = COPPELIA
    env.pop("QT_QPA_PLATFORM", None)            # need a real GL context
    env["DISPLAY"] = os.environ.get("DISPLAY", ":1")
    env["AHA_LIVE_DETECTORS"] = "transition"    # isolate the transition detector
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


def bt_transition_waypoints(task):
    """Waypoint indices where the prepared BT monitors the transition detector."""
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
            if det == "transition" or "transition" in links or "reaches_waypoint" in cond:
                out.append(idx)
                break
    return out


def pick_positive(task):
    """Choose (failtype, waypoint): a translation failure at the first BT-monitored
    transition waypoint that also has such a failure configured."""
    cfg = load_yaml(os.path.join(CFG_DIR, f"{task}.yaml"))
    by_wp = {}   # wp -> list of translation failtypes available
    for f in cfg.get("failures", []):
        ty = f.get("type")
        if ty in TRANSLATION_TYPES:
            for wp in f.get("waypoints", []):
                by_wp.setdefault(int(wp), []).append(ty)
    monitored = bt_transition_waypoints(task)
    for wp in monitored:
        if wp in by_wp:
            # prefer translation_x, else first available
            for ty in TRANSLATION_TYPES:
                if ty in by_wp[wp]:
                    return ty, wp
    return None, None


def pick_all_positives(task):
    """Every (failtype, waypoint) combo: each translation failure type configured
    at each BT-monitored transition waypoint (not just the first/preferred one)."""
    cfg = load_yaml(os.path.join(CFG_DIR, f"{task}.yaml"))
    by_wp = {}   # wp -> list of translation failtypes available
    for f in cfg.get("failures", []):
        ty = f.get("type")
        if ty in TRANSLATION_TYPES:
            for wp in f.get("waypoints", []):
                by_wp.setdefault(int(wp), []).append(ty)
    monitored = bt_transition_waypoints(task)
    out = []
    for wp in monitored:
        for ty in by_wp.get(wp, []):
            out.append((ty, wp))
    return out


SUMMARY_FIRE = re.compile(r"transition\s+DETECTED transition x(\d+).*waypoint (\d+)")
SUMMARY_OK = re.compile(r"transition\s+ok — no transition")
SUMMARY_DISABLED = re.compile(r"transition\s+DISABLED")
SUMMARY_SKIPPED = re.compile(r"transition\s+skipped")
FIRE_LINE = re.compile(r"\[DETECTOR\] TRANSITION failure at waypoint (\d+)")
FIRE_STEP_LINE = re.compile(r"\[DETECTOR\] TRANSITION failure at waypoint \d+ \(step (\d+)\)")
FIRE_MODE_LINE = re.compile(r"\[DETECTOR\] TRANSITION failure at waypoint \d+ \(step \d+\) mode=(\w+)")
DIAG_LINE = re.compile(r"arrival distance per-waypoint: (.+)")
WP_DIST = re.compile(r"wp(\d+)=([0-9.]+)")
HEADER_LINE = re.compile(r"Waypoint (\d+) of \d+")


def parse_run(stdout):
    """Return dict: fired(bool), fired_waypoints(set), status, diag(str),
    per_wp(dict wp->arrival distance m)."""
    fired_wps = set(int(m) for m in FIRE_LINE.findall(stdout))
    fired_steps = [int(m) for m in FIRE_STEP_LINE.findall(stdout)]
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
    diag = ""
    per_wp = {}
    m = DIAG_LINE.search(stdout)
    if m:
        diag = m.group(1).strip()
        for wp, val in WP_DIST.findall(diag):
            per_wp[int(wp)] = float(val)
    return {"fired": fired or bool(fired_wps),
            "fired_waypoints": sorted(fired_wps), "status": status, "diag": diag,
            "per_wp": per_wp, "fired_steps": fired_steps, "fired_modes": fired_modes}


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


def _float_or_nan(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def render_plot_from_drift(csv_path, png_path, task, label, fired_steps):
    if not csv_path or not os.path.exists(csv_path):
        return ""
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return ""

    steps = [int(float(r.get("step") or 0)) for r in rows]
    distance = [_float_or_nan(r.get("dist_to_target")) for r in rows]
    runup = [_float_or_nan(r.get("runup")) for r in rows]
    waypoint_starts = []
    last_wp = object()
    for row in rows:
        wp = row.get("waypoint", "")
        if wp != last_wp:
            waypoint_starts.append((int(float(row.get("step") or 0)), wp))
            last_wp = wp

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return render_plot_with_pillow(
            png_path, task, label, steps, distance, runup, rows,
            waypoint_starts, fired_steps)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f"{task} {label}: transition detector telemetry")
    axes[0].plot(steps, distance, color="#2563eb", label="distance_to_target")
    axes[0].set_ylabel("distance (m)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].plot(steps, runup, color="#16a34a", label="cumulative positive runup")
    axes[1].set_xlabel("frame / detector step")
    axes[1].set_ylabel("runup (m)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right")

    for ax in axes:
        for step, wp in waypoint_starts:
            ax.axvline(step, color="#7c3aed", linestyle="--", alpha=0.35)
            ax.text(step, 0.98, f"wp{wp}", transform=ax.get_xaxis_transform(),
                    rotation=90, va="top", ha="right", fontsize=8, color="#7c3aed")
        for row in rows:
            if int(float(row.get("path_done") or 0)):
                ax.axvline(int(float(row.get("step") or 0)), color="#64748b",
                           linestyle=":", alpha=0.18)
        for step in fired_steps:
            ax.axvline(step, color="#ef4444", linestyle="-", alpha=0.75)

    fig.tight_layout()
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return png_path


def render_plot_with_pillow(png_path, task, label, steps, distance, runup, rows,
                            waypoint_starts, fired_steps):
    from PIL import Image, ImageDraw, ImageFont
    import math

    width, height = 1400, 820
    margin_l, margin_r = 90, 30
    top, gap, panel_h = 80, 70, 280
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    draw.text((margin_l, 25), f"{task} {label}: transition detector telemetry",
              fill=(20, 20, 20), font=font)

    x_min = min(steps) if steps else 0
    x_max = max(steps) if steps else 1
    if x_max <= x_min:
        x_max = x_min + 1

    def finite_range(values):
        vals = [v for v in values if math.isfinite(v)]
        if not vals:
            return 0.0, 1.0
        lo, hi = min(vals), max(vals)
        if hi <= lo:
            hi = lo + 1.0
        pad = (hi - lo) * 0.08
        return lo - pad, hi + pad

    def xcoord(step):
        return margin_l + (step - x_min) / (x_max - x_min) * (width - margin_l - margin_r)

    def draw_panel(y0, values, y_label, color):
        y_min, y_max = finite_range(values)
        x0, x1 = margin_l, width - margin_r
        y1 = y0 + panel_h
        draw.rectangle((x0, y0, x1, y1), outline=(180, 180, 180), width=1)
        draw.text((12, y0 + 8), y_label, fill=(40, 40, 40), font=font)
        for i in range(5):
            y = y0 + i * panel_h / 4
            draw.line((x0, y, x1, y), fill=(235, 235, 235))
        for step, wp in waypoint_starts:
            x = xcoord(step)
            draw.line((x, y0, x, y1), fill=(180, 150, 220), width=1)
            draw.text((x + 3, y0 + 3), f"wp{wp}", fill=(100, 50, 160), font=font)
        for row in rows:
            if int(float(row.get("path_done") or 0)):
                x = xcoord(int(float(row.get("step") or 0)))
                draw.line((x, y0, x, y1), fill=(210, 210, 210), width=1)
        for step in fired_steps:
            x = xcoord(step)
            draw.line((x, y0, x, y1), fill=(230, 60, 60), width=2)

        def ycoord(value):
            if not math.isfinite(value):
                return None
            return y1 - (value - y_min) / (y_max - y_min) * panel_h

        points = []
        for step, value in zip(steps, values):
            y = ycoord(value)
            if y is None:
                if len(points) > 1:
                    draw.line(points, fill=color, width=2)
                points = []
                continue
            points.append((xcoord(step), y))
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
        draw.text((x0, y1 + 8), f"step {x_min}", fill=(80, 80, 80), font=font)
        draw.text((x1 - 90, y1 + 8), f"step {x_max}", fill=(80, 80, 80), font=font)

    draw_panel(top, distance, "distance_to_target (m)", (37, 99, 235))
    draw_panel(top + panel_h + gap, runup, "cumulative positive runup (m)", (22, 163, 74))

    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    img.save(png_path)
    return png_path


def run_bt(task, *, failure="none", failure_waypoint=None, env_extra=None,
           plot_path=None, drift_log_path=None, plot_label=None,
           stop_after_wp=None):
    """Run one headless BT episode. If stop_after_wp is set, the subprocess is
    killed the instant it's no longer possible to change the verdict: either a
    live TRANSITION fire is seen, or the BT's waypoint header advances past
    stop_after_wp with no fire (miss). This is the same early-exit trick
    fast_collision_sweep.py uses. Tradeoff: an early-killed run never reaches
    the final "Live detector summary" line, so info["diag"]/info["per_wp"]
    (the measured offset at the injected waypoint) come back empty — the
    fired/fired_waypoints verdict itself is unaffected since FIRE_LINE is
    matched live, streamed as it happens."""
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
        run_env_extra["AHA_TRANSITION_DRIFT_LOG"] = drift_log_path
        run_env_extra["AHA_TRANSITION_DRIFT_LOG_STREAM"] = "1"
        run_env_extra["AHA_TRANSITION_DRIFT_LOG_STREAM_EVERY"] = "10"
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
    if plot_path and drift_log_path:
        try:
            info["plot"] = render_plot_from_drift(
                drift_log_path, plot_path, task, plot_label or failure,
                info.get("fired_steps", []))
        except Exception as exc:
            info["plot"] = f"plot_err:{exc}"
    if timed_out:
        info["status"] = "timeout"
    return info


def calibrate(task, episodes=CALIB_EPISODES):
    """Run K clean episodes with a huge arrival threshold and per-frame logging
    on (never fires => full episode logged), then build the statistical per-task
    baseline (arrival mean/std per wp, gradient & diverge thresholds) from the
    clean per-frame CSVs. Multiple episodes are essential: clean arrival distance
    on dynamic waypoints varies run-to-run and a single episode underestimates
    the mean/std."""
    os.makedirs(DRIFT_DIR, exist_ok=True)
    no_fire = {"AHA_TRANSITION_MODE": "arrival",
               "AHA_TRANSITION_ARRIVAL_THRESHOLD": "100"}
    stop_wp = max(bt_transition_waypoints(task), default=None)

    # clear any stale clean logs for this task
    for f in os.listdir(DRIFT_DIR):
        if f.startswith(f"{task}.clean.ep"):
            os.remove(os.path.join(DRIFT_DIR, f))

    last = None
    for i in range(max(1, episodes)):
        clean_log = os.path.join(DRIFT_DIR, f"{task}.clean.ep{i}.csv")
        last = run_bt(task, failure="none", stop_after_wp=stop_wp,
                      env_extra={**no_fire, "AHA_TRANSITION_DRIFT_LOG": clean_log})
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


def _distance_value(row):
    for key in ("dist_to_target", "dist_to_original"):
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


def _max_prop_ratio(series, window=PROP_WINDOW, floor=PROP_FLOOR):
    """Worst clean C4-PROP ratio over an ordered list of per-frame distances.

    Mirrors the live detector exactly: at frame i (with a non-empty history) the
    ratio is dist_i / max(median(last `window` distances before i), floor). The
    max of that over all frames is the largest a clean run ever pushes prop, so a
    per-waypoint factor of PROP_RATIO_SCALE * this stays just above clean.
    """
    import numpy as np
    recent = []
    worst = 0.0
    for dist in series:
        if not (dist == dist):          # skip NaN, matches live finite() gate
            continue
        if recent:
            ref = max(float(np.median(recent)), floor)
            if ref > 0:
                worst = max(worst, dist / ref)
        recent.append(dist)
        if len(recent) > window:
            recent.pop(0)
    return worst


def _episode_index(path):
    match = re.search(r'\.clean\.ep(\d+)\.csv$', os.path.basename(path))
    return int(match.group(1)) if match else 10**9


def _compute_clean_baseline(task, episodes):
    """Per-task baselines from the clean per-frame CSVs. Anchored on the two
    plots: PLOT 1 = distance-to-target (m); PLOT 2 = cumulative positive runup.
      * error_stats: PLOT-1 max/min/mean/std over clean motion frames.
      * segment_rise_by_waypoint: PLOT-2 -- per waypoint segment, does the error
        ever rise on a clean run and the cumulative positive run-up.
      * arrival_stats_by_waypoint: per wp, median/robust-std of the pre-arrival distance.
      * gradient_threshold = gripper_motion mean+K*std; diverge_error_threshold =
        distance mean+K*std (pooled motion frames)."""
    import glob
    import numpy as np
    per_wp_arrivals = {}    # wp -> [per-episode pre-arrival distance]
    per_wp_rise = {}        # wp -> [per-episode max run-up]
    per_wp_positive_runup_samples = {}  # wp -> all positive second-plot samples
    per_wp_pos = {}         # wp -> [per-episode any-positive bool]
    per_wp_prop_ratio = {}  # wp -> [per-episode worst clean C4-prop ratio]
    steps = []              # pooled per-frame gripper_motion (motion frames)
    errors = []             # pooled distance-to-target (motion frames)
    clean_paths = sorted(
        glob.glob(os.path.join(DRIFT_DIR, f"{task}.clean.ep*.csv")),
        key=_episode_index,
    )
    clean_paths = clean_paths[:max(1, int(episodes))]
    for cpath in clean_paths:
        ep_arrival = {}     # wp -> last in-motion distance before waypoint completion
        ep_series = {}      # wp -> ordered list of motion-frame distances
        with open(cpath) as f:
            for r in csv.DictReader(f):
                try:
                    wp = int(r["waypoint"])
                except (ValueError, KeyError, TypeError):
                    continue
                done = r.get("path_done") == "1"
                dist = _distance_value(r)
                try:
                    gm = float(r.get("gripper_motion", "nan"))
                except ValueError:
                    continue
                if done:
                    continue
                else:
                    if dist == dist:
                        errors.append(dist)
                        ep_series.setdefault(wp, []).append(dist)
                        ep_arrival[wp] = dist
                    if gm == gm:
                        steps.append(gm)
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
            "source": "worst_clean_dist_over_median_of_last_window",
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
            "source": "max_runup_of_distance_to_target_between_waypoint_boundaries",
            "n": int(rises.size)}
    g = np.asarray(steps, dtype=float) if steps else np.zeros(1)
    e = np.asarray(errors, dtype=float) if errors else np.zeros(1)
    grad_thr = round(float(g.mean() + GRADIENT_SIGMA_K * g.std()), 5)
    div_thr = round(float(e.mean() + DIVERGE_SIGMA_K * e.std()), 5)
    return {
        "task": task,
        "n_clean_episodes": max(1, episodes),
        "signal": "transition: PLOT1=distance to waypoint-defined position (m)",
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
        "design": ("PLOT1 error_stats = max/min/mean/std of distance-to-target; "
                   "PLOT2 segment_rise = any positive change between waypoints "
                   "and the cumulative positive run-up magnitude; "
                   "arrival[wp]=max(floor,median+%g*robust_std) using the last "
                   "in-motion frame before waypoint completion; "
                   "runup_threshold[wp]=max(floor,1.5*max_clean_cumulative_runup); "
                   "gradient=gripper_motion mean+%g*std; diverge=distance "
                   "mean+%g*std" % (ARRIVAL_SIGMA_K, GRADIENT_SIGMA_K,
                                    DIVERGE_SIGMA_K)),
    }


def negative(task, env_extra=None, plot_path=None):
    stop_wp = max(bt_transition_waypoints(task), default=None)
    drift_path = drift_log_path_for(task, "negative") if plot_path else None
    # Drift rows stream during the run, so early-killing still leaves plot data.
    return run_bt(task, failure="none", env_extra=env_extra, plot_path=plot_path,
                  drift_log_path=drift_path, plot_label="clean",
                  stop_after_wp=stop_wp)


def positive(task, failtype, wp, env_extra=None, plot_path=None):
    # Stop once the injected waypoint has fired or passed. This keeps eval fast;
    # the streamed drift CSV still gives enough telemetry for the plot.
    drift_path = drift_log_path_for(task, "positive", failtype, wp) if plot_path else None
    return run_bt(task, failure=failtype, failure_waypoint=wp, env_extra=env_extra,
                  plot_path=plot_path, drift_log_path=drift_path,
                  plot_label=f"{failtype}@wp{wp}", stop_after_wp=wp)


FIELDS = ["task", "phase", "failtype", "inj_wp", "offset", "label", "verdict",
          "fired", "fired_waypoints", "status", "elapsed", "plot", "diag"]


def write_rows(rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    exists = os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def reset_outputs():
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(OUT_CSV):
        os.remove(OUT_CSV)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    for name in os.listdir(PLOTS_DIR):
        path = os.path.join(PLOTS_DIR, name)
        if os.path.isfile(path):
            os.remove(path)


def evaluate(tasks, workers, phases, threshold_extra, calib_episodes=CALIB_EPISODES,
             all_positives=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    reset_outputs()
    print(f"results -> {OUT_CSV}", flush=True)
    print(f"plots   -> {PLOTS_DIR}", flush=True)
    # positive_cases: (task, failtype, wp) for every combo to test. With
    # all_positives, that's every translation failtype configured at every
    # BT-monitored waypoint (not just one pick per task).
    positive_cases = []
    for t in tasks:
        cases = pick_all_positives(t) if all_positives else [pick_positive(t)]
        cases = [(ft, wp) for ft, wp in cases if ft is not None]
        if not cases:
            positive_cases.append((t, None, None))   # SKIP row, no translation case
        else:
            positive_cases += [(t, ft, wp) for ft, wp in cases]

    # Phase 1: calibrate (parallel) — must finish before neg/pos load baselines.
    if "calibrate" in phases:
        print(f"\n=== CALIBRATE ({len(tasks)} tasks, {calib_episodes} episodes each) ===", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(calibrate, t, calib_episodes): t for t in tasks}
            for fut in as_completed(futs):
                t = futs[fut]
                info, base = fut.result()
                arrival_wps = sorted(int(k) for k in
                                     base.get("arrival_stats_by_waypoint", {}))
                row = {"task": t, "phase": "calibrate", "label": "clean",
                       "fired": info["fired"], "fired_waypoints": info["fired_waypoints"],
                       "status": info["status"], "elapsed": info["elapsed"],
                       "diag": info["diag"],
                       "verdict": (f"grad_thr={base.get('gradient_threshold')} "
                                   f"diverge_thr={base.get('diverge_error_threshold')}")}
                write_rows([row])
                print(f"  calib {t:<26} {info['status']:<10} {info['elapsed']:>5}s  "
                      f"baseline_wps={arrival_wps} "
                      f"grad_thr={base.get('gradient_threshold')} "
                      f"diverge_thr={base.get('diverge_error_threshold')}", flush=True)

    # Phase 2 + 3: negative and positive (parallel), baselines on disk.
    jobs = []
    if "negative" in phases:
        jobs += [("negative", t, None, None) for t in tasks]
    if "positive" in phases:
        jobs += [("positive", t, ft, wp) for t, ft, wp in positive_cases]

    def do(job):
        phase, t, ft, wp = job
        if phase == "negative":
            ppath = plot_path_for(t, phase)
            info = negative(t, env_extra=threshold_extra, plot_path=ppath)
            label, verdict = "clean", ("FP" if info["fired"] else "TN")
            return {"task": t, "phase": phase, "label": label, "verdict": verdict,
                    "failtype": "none", "inj_wp": "",
                    "fired": info["fired"], "fired_waypoints": info["fired_waypoints"],
                    "status": info["status"], "elapsed": info["elapsed"],
                    "plot": info.get("plot", ""), "diag": info["diag"]}
        else:
            if ft is None:
                return {"task": t, "phase": phase, "label": "fail", "verdict": "SKIP",
                        "failtype": "none", "inj_wp": "", "fired": "", "fired_waypoints": "",
                        "status": "no_translation_case", "elapsed": 0, "plot": "", "diag": ""}
            ppath = plot_path_for(t, phase, ft, wp)
            info = positive(t, ft, wp, env_extra=threshold_extra, plot_path=ppath)
            hit = wp in info["fired_waypoints"]
            verdict = "TP" if hit else ("FN_fired_other" if info["fired"] else "FN")
            offset = info["per_wp"].get(wp)   # arrival dist at injected wp ~= |offset|
            return {"task": t, "phase": phase, "label": "fail", "verdict": verdict,
                    "failtype": ft, "inj_wp": wp, "offset": "" if offset is None else round(offset, 3),
                    "fired": info["fired"], "fired_waypoints": info["fired_waypoints"],
                    "status": info["status"], "elapsed": info["elapsed"],
                    "plot": info.get("plot", ""), "diag": info["diag"]}

    if jobs:
        print(f"\n=== EVAL negative+positive ({len(jobs)} runs) ===", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(do, j): j for j in jobs}
            for fut in as_completed(futs):
                row = fut.result()
                write_rows([row])
                tag = f"{row.get('failtype', 'none')}@wp{row['inj_wp']}" if row["phase"] == "positive" else "clean"
                print(f"  {row['phase']:<8} {row['task']:<26} {tag:<20} {row['verdict']:<14} "
                      f"fired_wps={row['fired_waypoints']} "
                      f"({row['status']}, {row['elapsed']}s)", flush=True)

    summarize()


def summarize():
    if not os.path.exists(OUT_CSV):
        print("no results yet")
        return
    rows = list(csv.DictReader(open(OUT_CSV)))
    # keep only the latest eval rows (negative/positive), dedupe by
    # (task, phase, failtype, inj_wp) last -- a task can have multiple positive
    # rows (one per failtype/waypoint combo when swept with --all-positives), so
    # deduping by (task, phase) alone would silently drop all but one of them.
    latest = {}
    for r in rows:
        if r["phase"] in ("negative", "positive"):
            latest[(r["task"], r["phase"], r.get("failtype", ""), r.get("inj_wp", ""))] = r
    TP = sum(1 for r in latest.values() if r["verdict"] == "TP")
    FN = sum(1 for r in latest.values() if r["verdict"].startswith("FN"))
    FP = sum(1 for r in latest.values() if r["verdict"] == "FP")
    TN = sum(1 for r in latest.values() if r["verdict"] == "TN")
    SK = sum(1 for r in latest.values() if r["verdict"] == "SKIP")
    print("\n" + "=" * 60)
    print("  TRANSITION DETECTOR — confusion matrix (per task/run)")
    print("=" * 60)
    print(f"  Positives (injected translation):  TP={TP}  FN={FN}")
    print(f"  Negatives (clean):                 TN={TN}  FP={FP}")
    if SK:
        print(f"  Skipped (no translation case):     {SK}")
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec = TP / (TP + FN) if (TP + FN) else 0.0
    print(f"  Precision={prec:.2f}  Recall={rec:.2f}")
    # Recall conditioned on a meaningful injected offset (>= 0.2 m). Tiny random
    # offsets (uniform sampling can land near 0) are not real failures to catch.
    pos = [r for r in latest.values() if r["phase"] == "positive"]
    def offv(r):
        try:
            return float(r.get("offset") or "nan")
        except ValueError:
            return float("nan")
    big = [r for r in pos if offv(r) >= 0.2]
    big_tp = sum(1 for r in big if r["verdict"] == "TP")
    if big:
        print(f"  Recall on offsets>=0.2m: {big_tp}/{len(big)}={big_tp/len(big):.2f} "
              f"(excludes {len(pos)-len(big)} sub-0.2m / unknown-offset positives)")
    print("=" * 60)
    # offenders
    fps = [(r["task"], r.get("diag", "")) for r in latest.values() if r["verdict"] == "FP"]
    fns = [(r["task"], r.get("failtype", ""), r.get("inj_wp", ""), r.get("offset", ""))
           for r in latest.values() if r["verdict"].startswith("FN")]
    if fps:
        print("  FP tasks:")
        for t, d in sorted(fps):
            print(f"     {t:<26} {d}")
    if fns:
        print("  FN tasks (failtype@wp  offset m):")
        for t, ft, wp, o in sorted(fns):
            print(f"     {t:<26} {ft}@wp{wp}  offset={o}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=None, help="limit to first N prepared tasks")
    ap.add_argument("--task", action="append", help="run only these named tasks")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--calib-episodes", type=int, default=CALIB_EPISODES)
    ap.add_argument("--phases", default="calibrate,negative,positive")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--arrival-threshold", default=None,
                    help="override AHA_TRANSITION_ARRIVAL_THRESHOLD for neg/pos eval")
    ap.add_argument("--baseline-scale", default=None)
    ap.add_argument("--baseline-margin", default=None)
    ap.add_argument("--all-positives", action="store_true",
                    help="sweep every translation failtype at every BT-monitored "
                         "waypoint (instead of one pick per task)")
    args = ap.parse_args()

    if args.summarize:
        summarize()
        return

    tasks = args.task or prepared_tasks()
    if args.tasks:
        tasks = tasks[:args.tasks]
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]

    threshold_extra = {}
    if args.arrival_threshold is not None:
        threshold_extra["AHA_TRANSITION_ARRIVAL_THRESHOLD"] = str(args.arrival_threshold)
    if args.baseline_scale is not None:
        threshold_extra["AHA_TRANSITION_BASELINE_SCALE"] = str(args.baseline_scale)
    if args.baseline_margin is not None:
        threshold_extra["AHA_TRANSITION_BASELINE_MARGIN"] = str(args.baseline_margin)

    print(f"tasks={len(tasks)} workers={args.workers} phases={phases} "
          f"calib_episodes={args.calib_episodes} all_positives={args.all_positives} "
          f"threshold_extra={threshold_extra}")
    evaluate(tasks, args.workers, phases, threshold_extra or None, args.calib_episodes,
             all_positives=args.all_positives)


if __name__ == "__main__":
    main()
