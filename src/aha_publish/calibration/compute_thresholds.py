"""STEP 2 of the two-step calibration: derive detector thresholds OFFLINE from
the raw per-frame clean data saved by calibrate_all_clean.py (STEP 1), with
every formula parameter exposed as a CLI flag so you can sweep them (e.g. is
transition prop_k=1.5 right? is the 0.02 arrival floor right?) WITHOUT touching
the simulator.

Input : <out-root>/clean_data/<task>/ep*.csv   (unified raw, all four signals)
Output:
  * <out-root>/transition_arrival_stats/<task>.json   (per-waypoint thresholds,
        live-read)
  * <out-root>/orientation_arrival_stats/<task>.json  (per-waypoint thresholds,
        live-read)
  * <out-root>/torque_stats/<task>_success_torque_stats.json
        (collision torque thresholds,
        live-read) + torque_stats/<task>_rise_frac_gates.json (per-wp rise gates,
        live-read) -- both now clean_data-derived, like the pose detectors above.
  * <out-root>/threshold_report/<detector>.csv        (all computed values, for
        comparison)

Formulas reproduced exactly from the live detectors:
  transition/orientation, per waypoint over the clean episodes:
    arrival_thr = max(arrival_floor, arrival_scale * stat(ep_arrival))
    runup_thr   = max(arrival_floor, rise_scale   * max(ep_runup))
    prop_k      = max(prop_k_min,    prop_ratio_scale * stat(ep_prop_ratio))
      where stat defaults to max, but transition uses median by default.
      ep_arrival    = worst last in-motion (path_done==0) signal value among
                      that episode's contiguous occurrences of the waypoint
      ep_runup      = worst cumulative positive rise among those occurrences
      ep_prop_ratio = worst per-occurrence max_i signal_i /
                      max(median(last W previous values), prop_floor)
  collision (per task, torque_norm & torque_delta):
    thr = mean + k*std   (and, with --torque-max-floor, >= max + std)
          where mean/std/max = median across episodes of per-episode mean/std/max
  slip (per task, grip_force over holding frames grip>min_force):
    thr = min(ceiling, max(grip_floor, mean - grip_k*std))
          where mean/std = average across episodes of per-episode mean/std

Usage:
  python aha_scripts/main_bt_run/compute_thresholds.py                # defaults
  python aha_scripts/main_bt_run/compute_thresholds.py \
         --trans-prop-k-min 2.0 --trans-arrival-floor 0.03            # sweep
  python aha_scripts/main_bt_run/compute_thresholds.py --dry-run --task change_clock
"""

from aha_publish import paths
import argparse
import csv
import glob
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = (paths.PROJECT_ROOT)
if str(paths.PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(paths.PROJECT_ROOT))

DEFAULT_CALIBRATION_ROOT = (paths.CALIBRATION_DIR)
CALIBRATION_ROOT = DEFAULT_CALIBRATION_ROOT
RAW_DIR = CALIBRATION_ROOT / "clean_data"
TRANS_STATS = CALIBRATION_ROOT / "transition_arrival_stats"
ORIENT_STATS = CALIBRATION_ROOT / "orientation_arrival_stats"
REPORT_DIR = CALIBRATION_ROOT / "threshold_report"
TORQUE_STATS_DIR = CALIBRATION_ROOT / "torque_stats"

# Collision method-1 rise-frac gate params, inlined from the collision detector
# so this script derives every collision threshold from clean_data with no import
# dependency on aha_scripts/detectors/collision. Keep the env var names and
# defaults identical to the detector so the live-read gates JSON stays unchanged.
TORQUE_BASELINE_WINDOW = int(os.getenv("AHA_COLLISION_BASELINE_WINDOW", "10"))
TORQUE_RISE_FRAC_FLOOR = float(os.getenv("AHA_COLLISION_RISE_FRAC", "1.0"))
RISE_FRAC_GATE_MARGIN = float(os.getenv("AHA_COLLISION_GATE_MARGIN", "0.05"))
RISE_FRAC_GATE_STD_K = float(os.getenv("AHA_COLLISION_GATE_STD_K", "0.0"))


def configure_output_root(out_root):
    """Point all read/write calibration paths at one output tree."""
    global CALIBRATION_ROOT, RAW_DIR, TRANS_STATS, ORIENT_STATS
    global REPORT_DIR, TORQUE_STATS_DIR
    CALIBRATION_ROOT = Path(out_root).expanduser().resolve()
    RAW_DIR = CALIBRATION_ROOT / "clean_data"
    TRANS_STATS = CALIBRATION_ROOT / "transition_arrival_stats"
    ORIENT_STATS = CALIBRATION_ROOT / "orientation_arrival_stats"
    REPORT_DIR = CALIBRATION_ROOT / "threshold_report"
    TORQUE_STATS_DIR = CALIBRATION_ROOT / "torque_stats"


# --------------------------------------------------------------------------- #
# raw-data loading
# --------------------------------------------------------------------------- #
def _f(x):
    """CSV cell -> float or None (blank cells are absent, never 0)."""
    if x is None or x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def load_episodes(task, raw_dir):
    """Return [ep0_rows, ep1_rows, ...]; each rows list is ordered by frame."""
    episodes = []
    for path in sorted(glob.glob(str(raw_dir / task / "ep*.csv")),
                       key=lambda p: int(Path(p).stem[2:])):
        with open(path) as f:
            episodes.append(list(csv.DictReader(f)))
    return episodes


def tasks_with_raw(raw_dir):
    if not raw_dir.exists():
        return []
    return sorted(p.name for p in raw_dir.iterdir()
                  if p.is_dir() and any(p.glob("ep*.csv")))


# --------------------------------------------------------------------------- #
# shared per-segment primitives (identical math to the live detectors)
# --------------------------------------------------------------------------- #
def max_runup(series):
    total, prev = 0.0, (series[0] if series else 0.0)
    for v in series[1:]:
        total += max(0.0, v - prev)
        prev = v
    return total


def max_prop_ratio(series, window, floor):
    recent, worst = [], 0.0
    for v in series:
        if recent:
            ref = max(float(np.median(recent[-window:])), floor)
            if ref > 0:
                worst = max(worst, v / ref)
        recent.append(v)
    return worst


def _episode_waypoint_segments(rows, signal_col):
    """Yield (wp, ordered signal values) for each contiguous waypoint occurrence.

    Some RLBench tasks repeat all waypoints until the success condition is met.
    The live prop detector resets its recent-median history at every waypoint
    start, so calibration must not stitch two occurrences of the same waypoint
    together.
    """
    current_wp, current, last_local = None, [], None
    for r in rows:
        wp = r.get("waypoint")
        if wp in (None, ""):
            continue
        if r.get("path_done") == "1":
            continue
        val = _f(r.get(signal_col))
        if val is None:
            continue
        wp = int(wp)
        local_step = int(float(r.get("local_step") or 0))
        starts_new_segment = (
            current
            and (wp != current_wp
                 or (last_local is not None and local_step <= last_local))
        )
        if starts_new_segment:
            yield current_wp, current
            current = []
        current_wp = wp
        current.append(val)
        last_local = local_step
    if current:
        yield current_wp, current


def per_waypoint_arrays(episodes, signal_col, prop_window, prop_floor):
    """wp -> {'arrival': [...], 'runup': [...], 'prop': [...]} across episodes.

    Each episode contributes one value per waypoint: the worst value among all
    contiguous occurrences of that waypoint in the episode. That preserves the
    "10 runs -> 10 values" calibration shape while matching the live detector's
    per-waypoint-start reset behavior.
    """
    wp_arrival, wp_runup, wp_prop = {}, {}, {}
    for rows in episodes:
        episode_values = {}
        for wp, s in _episode_waypoint_segments(rows, signal_col):
            if not s:
                continue
            values = episode_values.setdefault(
                wp, {"arrival": [], "runup": [], "prop": []})
            values["arrival"].append(s[-1])
            values["runup"].append(max_runup(s))
            values["prop"].append(max_prop_ratio(s, prop_window, prop_floor))
        for wp, values in episode_values.items():
            wp_arrival.setdefault(wp, []).append(max(values["arrival"]))
            wp_runup.setdefault(wp, []).append(max(values["runup"]))
            wp_prop.setdefault(wp, []).append(max(values["prop"]))
    return wp_arrival, wp_runup, wp_prop


def waypoint_thresholds(episodes, signal_col, p):
    """Compute the per-waypoint stats/thresholds dict for a pose detector."""
    wp_arrival, wp_runup, wp_prop = per_waypoint_arrays(
        episodes, signal_col, p["prop_window"], p["prop_floor"])
    arrival_stats, prop_stats, segment_rise = {}, {}, {}
    for wp in sorted(wp_arrival):
        a = np.asarray(wp_arrival[wp], dtype=float)
        arrival_base = (
            float(np.median(a)) if p.get("arrival_stat") == "median"
            else float(a.max())
        )
        arrival_stats[str(wp)] = {
            "mean": round(float(a.mean()), 5),
            "median": round(float(np.median(a)), 5),
            "std": round(float(a.std()), 5),
            "max": round(float(a.max()), 5),
            "threshold": round(float(max(p["arrival_floor"],
                                         p["arrival_scale"] * arrival_base)), 5),
            "n": int(a.size),
        }
        rises = np.asarray(wp_runup.get(wp, [0.0]), dtype=float)
        segment_rise[str(wp)] = {
            "max_rise": round(float(rises.max()), 5),
            "mean_rise": round(float(rises.mean()), 5),
            "threshold": round(float(max(p["arrival_floor"],
                                         p["rise_scale"] * float(rises.max()))), 5),
            "n": int(rises.size),
        }
        ratios = wp_prop.get(wp, [])
        max_ratio = float(max(ratios)) if ratios else 0.0
        median_ratio = float(np.median(ratios)) if ratios else 0.0
        prop_base = median_ratio if p.get("prop_stat") == "median" else max_ratio
        prop_stats[str(wp)] = {
            "max_ratio": round(max_ratio, 5),
            "median_ratio": round(median_ratio, 5),
            "prop_k": round(float(max(p["prop_k_min"],
                                      p["prop_ratio_scale"] * prop_base)), 5),
            "window": p["prop_window"],
            "floor": p["prop_floor"],
            "n": len(ratios),
        }
    return arrival_stats, prop_stats, segment_rise


# --------------------------------------------------------------------------- #
# per-episode scalar stats (collision / slip)
# --------------------------------------------------------------------------- #
def episode_metric_stats(episodes, col, frame_filter=None):
    """[{'mean','std','max'} per episode] for a scalar column."""
    out = []
    for rows in episodes:
        vals = []
        for r in rows:
            v = _f(r.get(col))
            if v is None:
                continue
            if frame_filter and not frame_filter(v):
                continue
            vals.append(v)
        if vals:
            out.append({
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "max": float(np.max(vals)),
                "min": float(np.min(vals)),
            })
    return out


def _agg(ep_stats, key, how):
    vals = [s[key] for s in ep_stats]
    if not vals:
        return float("nan")
    return float(np.median(vals) if how == "median" else np.mean(vals))


def collision_threshold(episodes, col, k, use_max_floor):
    st = episode_metric_stats(episodes, col)
    if not st:
        return None
    mean = _agg(st, "mean", "median")
    std = _agg(st, "std", "median")
    mx = _agg(st, "max", "median")
    thr = mean + k * std
    if use_max_floor:
        thr = max(thr, mx + std)
    return {"mean": round(mean, 5), "std": round(std, 5), "max": round(mx, 5),
            "threshold": round(float(thr), 5), "n_episodes": len(st)}


def collision_threshold_rows(task, episodes, k, use_max_floor):
    """Absolute torque_norm/torque_delta thresholds derived from clean_data.

    Same formula the live detector applies (mean + k*std, floored by max+std),
    but computed directly from the raw clean episodes here -- median across
    episodes of each per-episode mean/std/max -- instead of reading a frozen
    torque_stats JSON. This is the clean_data source of truth for collision.
    """
    rows = []
    for metric in ("torque_norm", "torque_delta"):
        res = collision_threshold(episodes, metric, k, use_max_floor)
        if not res:
            continue
        rows.append({
            "task": task,
            "metric": metric,
            "k": float(k),
            "use_max_floor": bool(use_max_floor),
            "mean": res["mean"],
            "std": res["std"],
            "max": res["max"],
            "threshold": res["threshold"],
            "n_episodes": res["n_episodes"],
        })
    return rows


def collision_torque_stats_json(task, episodes):
    """Per-task torque_stats JSON in the exact schema the live collision detector
    reads (``apply_task_torque_stats_thresholds`` -> median-of-episode mean/std/
    max), but built from clean_data so the runtime torque_norm/torque_delta
    thresholds are clean_data-derived -- exactly like the transition/orientation
    arrival_stats JSONs. Same path/format as detectors/torque_stats/run.py so the
    live detector reads it unchanged.
    """
    metrics = {}
    for metric in ("torque_norm", "torque_delta"):
        ep_rows = []
        all_vals = []
        for i, rows in enumerate(episodes):
            vals = [v for v in (_f(r.get(metric)) for r in rows) if v is not None]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            ep_rows.append({
                "episode": i,
                "stats": {
                    "count": int(arr.size),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                },
            })
            all_vals.extend(vals)
        agg = {}
        if all_vals:
            a = np.asarray(all_vals, dtype=float)
            agg = {"count": int(a.size), "mean": float(np.mean(a)),
                   "std": float(np.std(a)), "min": float(np.min(a)),
                   "max": float(np.max(a))}
        avg = {}
        if ep_rows:
            for key in ("count", "mean", "std", "min", "max"):
                avg[key] = float(np.mean([e["stats"][key] for e in ep_rows]))
        metrics[metric] = {
            "aggregate_all_frames": agg,
            "average_of_episode_stats": avg,
            "episodes": ep_rows,
        }
    return {
        "task": task,
        "episodes_completed": len(episodes),
        "episodes_requested": len(episodes),
        "metrics": metrics,
    }


def save_collision_torque_stats(task, episodes):
    """Write <task>_success_torque_stats.json from clean_data (live-read)."""
    path = TORQUE_STATS_DIR / f"{task}_success_torque_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(collision_torque_stats_json(task, episodes), f, indent=2)
    return path


def rise_frac_gate(max_clean_rise_frac, std_clean_rise_frac=0.0):
    """Per-waypoint proportional-spike gate from clean-run rise_frac stats.

    gate = max(FLOOR, clean_max * (1 + margin) + std_k * clean_std)
    Inlined from the collision detector; keep in sync with its defaults.
    """
    mx = float(max_clean_rise_frac) if np.isfinite(max_clean_rise_frac) else 0.0
    sd = float(std_clean_rise_frac) if np.isfinite(std_clean_rise_frac) else 0.0
    return max(TORQUE_RISE_FRAC_FLOOR,
               mx * (1.0 + RISE_FRAC_GATE_MARGIN) + RISE_FRAC_GATE_STD_K * sd)


def save_rise_frac_gates(task, gates):
    """Write the per-(task, waypoint) rise_frac gates JSON the detector reads.

    Same format as the collision detector's writer so the live-read file is
    unchanged: <out-root>/torque_stats/<task>_rise_frac_gates.json.
    """
    path = TORQUE_STATS_DIR / f"{task}_rise_frac_gates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": task,
        "floor": TORQUE_RISE_FRAC_FLOOR,
        "gates": {str(int(k)): float(v) for k, v in gates.items()},
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def collision_rise_gate_rows(task, episodes):
    """Method-1 collision gates from clean torque_rise_frac, per waypoint.

    The live method-1 detector fires when torque_rise_frac clears the current
    waypoint gate and the absolute rise is large enough. Compute each episode's
    max clean rise_frac per waypoint, take the median across episodes, then pass
    it through the detector's rise_frac_gate() helper.
    """
    per_wp = {}
    window = TORQUE_BASELINE_WINDOW
    for rows in episodes:
        # Every frame with a valid torque feeds the baseline -- including frames
        # whose waypoint is blank -- so the trailing window matches the
        # contiguous stream the live detector sees. Only the per-waypoint max
        # attribution skips frames without a waypoint.
        seq = []
        for r in rows or []:
            torque_norm = _f(r.get("torque_norm"))
            if torque_norm is None or not np.isfinite(torque_norm):
                continue
            wp = r.get("waypoint")
            try:
                wp = int(float(wp)) if wp not in (None, "") else None
            except (TypeError, ValueError):
                wp = None
            seq.append((wp, torque_norm))

        norms = [tn for _, tn in seq]
        ep_max = {}
        for i, (wp, torque_norm) in enumerate(seq):
            prior = norms[max(0, i - window):i]
            baseline = float(np.median(prior)) if prior else torque_norm
            rise = torque_norm - baseline
            rise_frac = float(rise / baseline) if baseline > 1e-9 else 0.0
            if wp is None or not np.isfinite(rise_frac):
                continue
            ep_max[wp] = max(ep_max.get(wp, float("-inf")), rise_frac)
        for wp, value in ep_max.items():
            per_wp.setdefault(wp, []).append(value)

    rows = []
    gates = {}
    for wp in sorted(per_wp):
        values = per_wp[wp]
        median_value = float(np.median(values))
        max_value = float(max(values))
        gate = float(rise_frac_gate(median_value))
        gates[wp] = gate
        rows.append({
            "task": task,
            "waypoint": wp,
            "rise_frac_gate": round(gate, 5),
            "median_clean_rise_frac": round(median_value, 5),
            "max_clean_rise_frac": round(max_value, 5),
            "n_episodes": len(values),
        })
    return gates, rows


def slip_threshold(episodes, col, k, floor, ceiling, min_force):
    st = episode_metric_stats(episodes, col, frame_filter=lambda v: v > min_force)
    if not st:
        return None
    mean = _agg(st, "mean", "mean")
    std = _agg(st, "std", "mean")
    thr = min(ceiling, max(floor, mean - k * std))
    return {"mean": round(mean, 5), "std": round(std, 5),
            "threshold": round(float(thr), 5), "n_episodes": len(st)}


# --------------------------------------------------------------------------- #
# JSON assembly for the live pose-detector loaders
# --------------------------------------------------------------------------- #
def pose_stats_json(task, signal_name, arrival_stats, prop_stats, segment_rise,
                    p, n_episodes):
    return {
        "task": task,
        "signal": signal_name,
        "n_clean_episodes": n_episodes,
        "arrival_stats_by_waypoint": arrival_stats,
        "prop_stats_by_waypoint": prop_stats,
        "segment_rise_by_waypoint": segment_rise,
        "arrival_floor": p["arrival_floor"],
        "arrival_scale": p["arrival_scale"],
        "prop_k_min": p["prop_k_min"],
        "prop_ratio_scale": p["prop_ratio_scale"],
        "rise_scale": p["rise_scale"],
        "source": "compute_thresholds.py (offline, from calibration clean_data)",
    }


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- #
def build_params(args):
    trans = dict(arrival_floor=args.trans_arrival_floor,
                 arrival_scale=args.trans_arrival_scale,
                 prop_k_min=args.trans_prop_k_min,
                 prop_ratio_scale=args.trans_prop_ratio_scale,
                 prop_window=args.trans_prop_window,
                 prop_floor=args.trans_prop_floor,
                 rise_scale=args.trans_rise_scale,
                 arrival_stat="median",
                 prop_stat="median")
    ori = dict(arrival_floor=args.ori_arrival_floor,
               arrival_scale=args.ori_arrival_scale,
               prop_k_min=args.ori_prop_k_min,
               prop_ratio_scale=args.ori_prop_ratio_scale,
               prop_window=args.ori_prop_window,
               prop_floor=args.ori_prop_floor,
               rise_scale=args.ori_rise_scale,
               arrival_stat="median",
               prop_stat="median")
    return trans, ori


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default=str(DEFAULT_CALIBRATION_ROOT),
                    help="calibration output root; defaults to "
                         "aha_output/aha_calibration")
    ap.add_argument("--raw-dir", default=None,
                    help="override raw clean-data directory; defaults to "
                         "<out-root>/clean_data")
    ap.add_argument("--task", action="append", default=[],
                    help="limit to these tasks (repeatable)")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute + print + report, but do not overwrite the "
                         "live per-waypoint JSONs")
    ap.add_argument("--no-report", action="store_true")
    # transition knobs
    ap.add_argument("--trans-arrival-floor", type=float, default=0.015)
    ap.add_argument("--trans-arrival-scale", type=float, default=1.3)
    ap.add_argument("--trans-prop-k-min", type=float, default=1.5)
    ap.add_argument("--trans-prop-ratio-scale", type=float, default=1.3)
    ap.add_argument("--trans-prop-window", type=int, default=5)
    ap.add_argument("--trans-prop-floor", type=float, default=0.01)
    ap.add_argument("--trans-rise-scale", type=float, default=1.5)
    # orientation knobs
    ap.add_argument("--ori-arrival-floor", type=float, default=0.1745)
    ap.add_argument("--ori-arrival-scale", type=float, default=1.3)
    ap.add_argument("--ori-prop-k-min", type=float, default=1.5)
    ap.add_argument("--ori-prop-ratio-scale", type=float, default=1.3)
    ap.add_argument("--ori-prop-window", type=int, default=5)
    ap.add_argument("--ori-prop-floor", type=float, default=0.1)
    ap.add_argument("--ori-rise-scale", type=float, default=1.5)
    # collision knobs
    ap.add_argument("--torque-k", type=float, default=3.0)
    ap.add_argument("--torque-max-floor", action="store_true", default=True)
    ap.add_argument("--no-torque-max-floor", dest="torque_max_floor",
                    action="store_false")
    # slip knobs
    ap.add_argument("--grip-k", type=float, default=2.0)
    ap.add_argument("--grip-floor", type=float, default=0.3)
    ap.add_argument("--grip-ceiling", type=float, default=float("inf"))
    ap.add_argument("--grip-min-force", type=float, default=0.2)
    args = ap.parse_args()

    configure_output_root(args.out_root)
    raw_dir = Path(args.raw_dir).expanduser().resolve() if args.raw_dir else RAW_DIR
    trans_p, ori_p = build_params(args)
    tasks = args.task or tasks_with_raw(raw_dir)
    if not tasks:
        print(f"No raw data under {raw_dir} — run STEP 1 (calibrate_all_clean.py) first.")
        return

    rows_report = {"transition": [], "orientation": [],
                   "collision": [], "collision_rise": [], "slip": []}
    print(f"compute_thresholds: {len(tasks)} task(s) from {raw_dir}"
          f" -> {CALIBRATION_ROOT}"
          f"{'  [DRY-RUN]' if args.dry_run else ''}")

    for task in tasks:
        episodes = load_episodes(task, raw_dir)
        n = len(episodes)
        if n == 0:
            print(f"  {task:<30} no episodes — skipped")
            continue

        # --- transition + orientation (write live JSONs) ---
        for name, col, sig, params, stats_dir in (
            ("transition", "distance_m", "distance (m)", trans_p, TRANS_STATS),
            ("orientation", "angle_rad", "angle (rad)", ori_p, ORIENT_STATS),
        ):
            a_stats, p_stats, rise = waypoint_thresholds(episodes, col, params)
            if not args.dry_run and a_stats:
                write_json(stats_dir / f"{task}.json",
                           pose_stats_json(task, sig, a_stats, p_stats, rise, params, n))
            for wp in sorted(a_stats, key=int):
                rows_report[name].append({
                    "task": task, "waypoint": wp,
                    "arrival_threshold": a_stats[wp]["threshold"],
                    "arrival_median_clean": a_stats[wp]["median"],
                    "arrival_max_clean": a_stats[wp]["max"],
                    "runup_threshold": rise[wp]["threshold"],
                    "runup_max_clean": rise[wp]["max_rise"],
                    "prop_k": p_stats[wp]["prop_k"],
                    "prop_median_ratio": p_stats[wp]["median_ratio"],
                    "prop_max_ratio": p_stats[wp]["max_ratio"],
                    "n_episodes": n,
                })

        # --- collision (torque) --- derived from clean_data, not torque_stats JSON
        rows_report["collision"].extend(
            collision_threshold_rows(
                task, episodes, args.torque_k, args.torque_max_floor))
        rise_gates, rise_rows = collision_rise_gate_rows(task, episodes)
        rows_report["collision_rise"].extend(rise_rows)
        if not args.dry_run:
            # live-read by the collision detector at runtime, now clean_data-derived
            save_collision_torque_stats(task, episodes)
            if rise_gates:
                save_rise_frac_gates(task, rise_gates)

        # --- slip (grip) ---
        res = slip_threshold(episodes, "grip_force", args.grip_k, args.grip_floor,
                             args.grip_ceiling, args.grip_min_force)
        if res:
            rows_report["slip"].append(
                {"task": task, "metric": "grip_force", "k": args.grip_k, **res})

        tw = rows_report["transition"]
        n_wp = len([r for r in tw if r["task"] == task])
        print(f"  {task:<30} eps={n} trans_wp={n_wp} "
              f"{'(not written)' if args.dry_run else '(written)'}")

    if not args.no_report:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        for det, rows in rows_report.items():
            if not rows:
                continue
            path = REPORT_DIR / f"{det}.csv"
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        print(f"\nReport CSVs -> {REPORT_DIR}")
        # quick sweep-friendly summary
        for det in ("transition", "orientation"):
            rows = rows_report[det]
            if rows:
                thr = [r["arrival_threshold"] for r in rows]
                pk = [r["prop_k"] for r in rows]
                print(f"  {det:<12} arrival_thr[min/median/max]="
                      f"{min(thr):.4f}/{statistics.median(thr):.4f}/{max(thr):.4f}"
                      f"  prop_k[median/max]={statistics.median(pk):.3f}/{max(pk):.3f}")
        for det in ("collision", "slip"):
            rows = rows_report[det]
            if rows:
                thr = [r["threshold"] for r in rows]
                print(f"  {det:<12} threshold[min/median/max]="
                      f"{min(thr):.4f}/{statistics.median(thr):.4f}/{max(thr):.4f}")

    if not args.dry_run:
        print("\nThreshold files written. The public calibration entry point saves "
              "the matching runtime parameters in each task's manifest.")


if __name__ == "__main__":
    main()
