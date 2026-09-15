"""Batch evaluation of the live detectors AND their VLM confirmation layer.

For each prepared-BT task this harness exercises five live detectors
(orientation, transition, slip, collision, freezing) in HOLD-ONLY mode — the
pre/post-condition VLM checks are switched off (``--vlm-predicates __none__``)
so only the detector → VLM-confirm path is measured. Each run isolates one
detector (``AHA_LIVE_DETECTORS=<name>``) and turns on the auto VLM confirmation
(``AHA_DETECTOR_VLM_AUTO=1``); a clear VLM "no failure" verdict retracts the
detection.

Two things are measured:

  (1) the CHEAP detector — does it fire on a real injected failure, and stay
      quiet on a clean run?
  (2) the VLM confirmation — when the detector fires correctly (real failure),
      does the VLM confirm it (say YES)?  And when the detector is deliberately
      made to fire on a clean scene (AHA_FORCE_FIRE_WAYPOINT), does the VLM
      catch that it is wrong (say NO / retract)?

Phases per (task, detector):

  POSITIVE   inject the failure type this detector should catch at a
             BT-monitored waypoint.  detector fires  -> VLM runs.
               VLM YES/UNKNOWN  -> TP (detector right, VLM confirmed/kept)
               VLM NO (retract) -> FN (detector right, VLM wrongly retracted)
               detector silent  -> DET_MISS (cheap detector missed it; no VLM)
  NEGATIVE   clean run, no forcing.  Measures the cheap detector's false-alarm
             rate.  detector silent -> TN_det ; detector fires -> DET_FP.
  FORCED     clean run with AHA_FORCE_FIRE_WAYPOINT set so the detector fires on
             a scene with NO real failure.  (VLM-capable detectors only.)
               VLM NO (retract)  -> TN (VLM correctly rejected a wrong detection)
               VLM YES/UNKNOWN   -> FP (VLM failed to reject a wrong detection)

The freezing detector has NO VLM verifier, so it only runs POSITIVE/NEGATIVE
(cheap-detector accuracy) and is excluded from the VLM confusion matrix.

The TP/FN/TN/FP counts in the printed summary are for the VLM confirmation
layer (what the user asked for); the cheap-detector recall/specificity are
reported alongside.

Run (from the `aha` conda env):
  python aha_scripts/main_bt_run/eval_detectors_vlm.py [--tasks 10]
         [--detectors orientation,transition,slip,collision,freezing]
         [--workers 2] [--cameras side_rgb,wrist_rgb,front_rgb]
         [--task name ...] [--phases positive,negative,forced]

Results stream to aha_output/detector_vlm_eval/results.csv; a summary is
printed at the end (also re-printable with --summarize).
"""

from aha_publish import paths
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import aha_publish.running.detector_vlm_config as DVC

PROJECT_ROOT = str(paths.PROJECT_ROOT)
BT_DIR = str(paths.BT_DIR)
CFG_DIR = str(paths.FAILGEN_ROOT / 'failgen/configs')
RUNNER = str(paths.SOURCE_DIR / 'running/waypoints_interactive_bt_conditions.py')
OUT_DIR = str(paths.OUTPUT_DIR / 'detector_vlm_eval')
OUT_CSV = os.path.join(OUT_DIR, "results.csv")
# Slip-trace plots (same format/method as aha_output/slip_method1/plots).
SLIP_PLOTS_DIR = os.path.join(OUT_DIR, "slip_plots")
# Collision torque-trace plots (same method as collision_method1 / plot_collision_torque_trace).
COLLISION_PLOTS_DIR = os.path.join(OUT_DIR, "collision_plots")
PY = sys.executable

COPPELIA = str(paths.COPPELIASIM_ROOT)
RUN_TIMEOUT = 1200

ALL_DETECTORS = ("orientation", "transition", "slip", "collision", "freezing")
VLM_DETECTORS = ("orientation", "transition", "slip", "collision")  # have a verifier

# Detector -> the failgen failure types it is supposed to catch, best first.
DETECTOR_FAILTYPES = {
    "transition": ("translation_x", "translation_y", "translation_z", "translation",
                   "combined_transition"),
    "orientation": ("rotation_x", "rotation_y", "rotation_z", "no_rotation",
                    "combined_orientation"),
    "slip": ("slip", "grasp"),
    "collision": ("collision",),
    "freezing": ("freezing",),
}

# Detector -> the BT hold-condition predicate that turns it on at a waypoint.
DETECTOR_PREDICATE = {
    "collision": "no_collision",
    "freezing": "not_frozen",
    "slip": "maintains_grasp",
    "orientation": "orientation_maintained",
    "transition": "reaches_waypoint",
}

# Per-detector summary line emitted by LiveDetectorMonitor.print_summary().
SUMMARY_DETECTED = {n: re.compile(rf"{n}\s+DETECTED {n} x(\d+)") for n in ALL_DETECTORS}
SUMMARY_OK = {n: re.compile(rf"{n}\s+ok — no {n}") for n in ALL_DETECTORS}
SUMMARY_DISABLED = {n: re.compile(rf"{n}\s+DISABLED") for n in ALL_DETECTORS}
SUMMARY_SKIPPED = {n: re.compile(rf"{n}\s+skipped") for n in ALL_DETECTORS}

# Fire / verdict / retract lines printed during the run.
FIRE_LINE = re.compile(
    r"\[DETECTOR\] (ORIENTATION|TRANSITION|SLIP|COLLISION|FREEZING) "
    r"(?:failure|detected) at waypoint (\d+)")
VLM_VERDICT = re.compile(
    r"\[vlm:[^\]]*\]\s+(orientation|transition|slip|collision)=(YES|NO|UNKNOWN)")
VLM_MODEL = re.compile(
    r"\[vlm:([^\]]+)\]\s+(orientation|transition|slip|collision)=")
# The one-line VLM verdict also carries the free-text reason before "cameras=".
VLM_REASON = re.compile(
    r"\[vlm:[^\]]*\]\s+(orientation|transition|slip|collision)="
    r"(?:YES|NO|UNKNOWN)\s+(.*?)\s+cameras=")
VLM_TOKENS = re.compile(
    r"\[vlm:(orientation|transition|slip|collision)\]\s+tokens:\s+"
    r"(\d+)\s+in\s+\+\s+(\d+)\s+out"
    r"(?:\s+\(reasoning:\s+(\d+)\))?")
RETRACT_LINE = re.compile(r"\[detector:(\w+)\] detection RETRACTED")
GRID_SAVED = re.compile(r"\[vlm:(?:slip|collision)\] camera grid saved -> (\S+)")
FORCED_FIRE_MARK = re.compile(r"FORCED eval false-positive")

# Prices are USD per 1M tokens for OpenAI standard processing.
# Override with AHA_VLM_INPUT_PRICE_PER_1M / AHA_VLM_OUTPUT_PRICE_PER_1M if needed.
OPENAI_PRICE_PER_1M = {
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.5": (5.00, 30.00),
}


# --------------------------------------------------------------------------- #
# Task / failure planning
# --------------------------------------------------------------------------- #
def prepared_tasks():
    return sorted(f[:-len(".bt_conditions.json")]
                  for f in os.listdir(BT_DIR) if f.endswith(".bt_conditions.json"))


def load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def bt_monitored_waypoints(task, detector):
    """Waypoint indices where the prepared BT runs `detector` as a hold check."""
    path = os.path.join(BT_DIR, f"{task}.bt_conditions.json")
    if not os.path.exists(path):
        return []
    d = json.load(open(path))
    stages = (d.get("review") or {}).get("stages") or (d.get("generated") or {}).get("stages") or []
    predicate = DETECTOR_PREDICATE[detector]
    out = []
    for idx, st in enumerate(stages):
        for hc in (st.get("hold_conditions") or []):
            if hc.get("selected") is False:
                continue
            det = str(hc.get("detector") or "").strip().lower()
            fail = str(hc.get("failure") or "").strip().lower()
            links = [str(l.get("failure")).lower()
                     for l in (hc.get("failure_links") or []) if isinstance(l, dict)]
            cond = str(hc.get("condition") or hc.get("original_condition") or "")
            if (det == detector or fail == detector or detector in links
                    or predicate in cond):
                out.append(idx)
                break
    return out


def config_failtypes_by_wp(task):
    """{failure_type: set(waypoints)} from the task's failgen config."""
    path = os.path.join(CFG_DIR, f"{task}.yaml")
    if not os.path.exists(path):
        return {}
    cfg = load_yaml(path)
    by_type = {}
    for f in cfg.get("failures", []) or []:
        ty = f.get("type")
        if not ty:
            continue
        for wp in f.get("waypoints", []) or []:
            by_type.setdefault(ty, set()).add(int(wp))
    return by_type


# Abstain ceilings, mirrored from the live detectors (kept in sync via import
# below; these are the fallbacks if the import is unavailable).
_ABSTAIN_CEIL_DEFAULT = {"transition": 0.30, "orientation": 0.80}


def _abstain_ceil(detector):
    try:
        import importlib
        mod = importlib.import_module(f"aha_publish.running.live_detectors.{detector}")
        return float(mod.ARRIVAL_ABSTAIN_CEIL)
    except Exception:
        return _ABSTAIN_CEIL_DEFAULT.get(detector, float("inf"))


def abstained_waypoints(task, detector):
    """Waypoints the orientation/transition detector ABSTAINS from -- its clean
    arrival max exceeds the abstain ceiling, so the episode-start (original) pose
    is not a valid reference there (dynamic target / unconstrained orientation).
    The eval must not inject at these (no clean-derived threshold can separate
    clean from a failure), so plan_case skips them. {} for other detectors."""
    if detector not in ("transition", "orientation"):
        return set()
    stats_dir = f"{detector}_arrival_stats"
    path = os.path.join(str(paths.CALIBRATION_DIR), stats_dir, f"{task}.json")
    if not os.path.exists(path):
        return set()
    ceil = _abstain_ceil(detector)
    try:
        data = json.load(open(path))
    except Exception:
        return set()
    out = set()
    for k, v in (data.get("arrival_stats_by_waypoint") or {}).items():
        mx = float(v.get("max", v.get("mean", 0.0)))
        if mx > ceil:
            out.add(int(k))
    return out


def plan_case(task, detector):
    """Return (failtype, pos_wp, force_wp).

    pos_wp:  a BT-monitored waypoint that also has a matching failure configured
             (falls back to the failure's own configured waypoint if the BT does
             not monitor this detector anywhere — still a valid detector test).
    force_wp: a BT-monitored waypoint to force a clean-run firing at (so the
             detector is actually enabled there); falls back to pos_wp.
    """
    monitored = bt_monitored_waypoints(task, detector)
    by_type = config_failtypes_by_wp(task)
    failtype, pos_wp = None, None
    for ty in DETECTOR_FAILTYPES[detector]:
        wps = by_type.get(ty)
        if not wps:
            continue
        # prefer a waypoint the BT monitors with this detector (no abstain filter
        # -- every task is evaluated; degenerate waypoints may FN, accepted)
        common = sorted(set(wps) & set(monitored))
        if common:
            failtype, pos_wp = ty, common[0]
            break
        if failtype is None:  # fallback: the failure's own configured waypoint
            failtype, pos_wp = ty, sorted(wps)[0]
    # Force the clean-run false-positive at the SAME waypoint the real slip is
    # injected at (where the object is actually held), so the FP is a fair
    # counterpart to the positive run — not the approach pose (wp0).
    force_wp = (pos_wp if pos_wp is not None
                else (monitored[0] if monitored else None))
    return failtype, pos_wp, force_wp


def injection_candidates(task, detector):
    """Ordered (failtype, waypoint) candidates to inject a `detector` failure at:
    BT-monitored, NON-abstained waypoints that have a matching failure configured,
    ascending by waypoint. A caller probes these in order and uses the first where
    the injection actually MANIFESTS (some waypoints are pass-throughs where the
    failure has no effect -- e.g. a rotation at a waypoint that doesn't set the
    gripper's settled orientation). One failtype per waypoint (first by priority)."""
    # No abstain filter: every BT-monitored waypoint with a configured failure is
    # a candidate, so no task is excluded as N/A. Degenerate waypoints (clean
    # arrival ~= the failure size) may FN -- accepted. The probe still skips
    # super-small no-op injections and prefers a manifesting waypoint.
    monitored = set(bt_monitored_waypoints(task, detector))
    by_type = config_failtypes_by_wp(task)
    wp_to_ft = {}
    for ty in DETECTOR_FAILTYPES[detector]:          # failtype priority order
        for wp in by_type.get(ty, set()):
            if wp in monitored:
                wp_to_ft.setdefault(wp, ty)
    return [(wp_to_ft[wp], wp) for wp in sorted(wp_to_ft)]


# --------------------------------------------------------------------------- #
# Slip-trace plotting (identical method to slip_method1 / eval_slip_all.save_plot)
# --------------------------------------------------------------------------- #
def save_slip_plot(drift_log, task, stem, label):
    """Render <stem>.csv + <stem>.png into SLIP_PLOTS_DIR from the slip detector's
    drift log, using the very same helpers eval_slip_all uses for slip_method1
    (plot_slip_trace_bt.write_csv/_read_drift_csv + plot_slip_trace.plot_trace).
    Returns the PNG path, or "" / "plot_err:..." on nothing-to-plot / failure."""
    from pathlib import Path
    try:
        from aha_publish.running.plot_slip_trace_bt import write_csv, _read_drift_csv
        from aha_publish.running.plot_slip_trace import plot_trace
    except Exception as exc:  # matplotlib / import issues shouldn't kill the eval
        return f"plot_err:{exc}"
    rows = _read_drift_csv(drift_log)
    if not rows:
        return ""
    os.makedirs(SLIP_PLOTS_DIR, exist_ok=True)
    all_fired = sorted({int(r["step"]) for r in rows if r.get("slip_fired")})
    png_path = Path(SLIP_PLOTS_DIR) / f"{stem}.png"
    try:
        # write_csv and plot_trace both expect pathlib.Path (they call .parent).
        write_csv(Path(SLIP_PLOTS_DIR) / f"{stem}.csv", rows)
        plot_trace(png_path, rows, task, label, all_fired)
        return str(png_path)
    except Exception as exc:
        return f"plot_err:{exc}"


# --------------------------------------------------------------------------- #
# Collision torque-trace plotting (same method as collision_method1 /
# plot_collision_torque_trace.plot_trace, driven by AHA_COLLISION_DRIFT_LOG)
# --------------------------------------------------------------------------- #
def save_collision_plot(drift_log, task, stem, label, injected_wp=None):
    """Render <stem>.csv + <stem>.png into COLLISION_PLOTS_DIR from the collision
    detector's per-step torque drift log, using plot_collision_torque_trace
    (the same proportional-spike method as collision_method1). Returns the PNG
    path, or "" / "plot_err:..." on nothing-to-plot / failure."""
    import csv as _csv
    from pathlib import Path
    try:
        from aha_publish.running.plot_collision_torque_trace import plot_trace, write_csv, confirmed_steps_from_rows
    except Exception as exc:
        return f"plot_err:{exc}"
    if not drift_log or not os.path.exists(drift_log):
        return ""
    rows = list(_csv.DictReader(open(drift_log, newline="")))
    if not rows:
        return ""
    os.makedirs(COLLISION_PLOTS_DIR, exist_ok=True)

    def _wp(r):
        try:
            return int(float(r.get("waypoint", "")))
        except (TypeError, ValueError):
            return None
    # Waypoint boundaries: (ordinal, last frame of that waypoint run).
    waypoint_frames, cur, last_step = [], None, None
    for r in rows:
        w = _wp(r)
        if w is not None and w != cur:
            if cur is not None and last_step is not None:
                waypoint_frames.append((cur, last_step))
            cur = w
        try:
            last_step = int(float(r.get("step", 0)))
        except (TypeError, ValueError):
            pass
    if cur is not None and last_step is not None:
        waypoint_frames.append((cur, last_step))
    # Frames the detector CONFIRMED a collision (violet fire lines) -- raw
    # crossings that never reached N consecutive frames are not fires.
    paused = confirmed_steps_from_rows(rows)
    png_path = Path(COLLISION_PLOTS_DIR) / f"{stem}.png"
    try:
        write_csv(Path(COLLISION_PLOTS_DIR) / f"{stem}.csv", rows)
        plot_trace(png_path, rows, paused, task, label,
                   waypoint_frames=waypoint_frames, injected_wp=injected_wp)
        return str(png_path)
    except Exception as exc:
        return f"plot_err:{exc}"


# --------------------------------------------------------------------------- #
# Running the BT
# --------------------------------------------------------------------------- #
def base_env(detector, cameras, extra=None):
    env = dict(os.environ)
    env["COPPELIASIM_ROOT"] = COPPELIA
    env["LD_LIBRARY_PATH"] = env.get("LD_LIBRARY_PATH", "") + ":" + COPPELIA
    env["QT_QPA_PLATFORM_PLUGIN_PATH"] = COPPELIA
    env.pop("QT_QPA_PLATFORM", None)            # need a real GL context
    env["DISPLAY"] = os.environ.get("DISPLAY", ":1")
    env["AHA_LIVE_DETECTORS"] = detector        # isolate one detector
    env["AHA_FAIL_DEBUG"] = "1"
    DVC.apply_detector_vlm_env(
        env, cameras=cameras, confirm_detectors=None,
        immediate=False, collision_method=False)
    if extra:
        env.update(extra)
    return env


def run_bt(task, detector, cameras, *, vlm_model=None, failure="none", failure_waypoint=None,
           force_fire_wp=None, extreme=False, plot_stem=None, plot_label=None):
    cmd = [PY, RUNNER, "--task", task, "--failure", failure,
           "--headless", "--mode", "auto",
           "--vlm-checks", "both", "--vlm-predicates", "__none__",
           "--no-side-camera-window", "--no-detector-plots"]
    if vlm_model:
        cmd += ["--vlm-model", vlm_model]
    if failure != "none" and failure_waypoint is not None:
        cmd += ["--failure-waypoint", str(failure_waypoint)]
    extra = {}
    if force_fire_wp is not None:
        extra["AHA_FORCE_FIRE_WAYPOINT"] = str(force_fire_wp)
    if extreme:
        # make the injected offset deterministic at the largest configured
        # magnitude so the cheap detector reliably has something to catch.
        extra["AHA_FAIL_EXTREME"] = "1"
    drift_log = None            # slip torque/grip drift log
    collision_drift_log = None  # collision torque drift log
    if detector == "slip":
        # Confirm synchronously the instant the detector fires: the deferred
        # confirm (default 4 frames, flushed only at waypoint transitions) never
        # resolves when the fire lands on the last waypoint, so the VLM would
        # never run. 0 = run it now, guaranteeing a verdict for every fire.
        extra["AHA_VLM_CONFIRM_DELAY_FRAMES"] = "0"
        if plot_stem:
            # Capture per-step telemetry so we can render the slip_method1 trace plot.
            import tempfile
            drift_log = tempfile.mktemp(prefix="aha_slip_drift_", suffix=".csv")
            extra["AHA_SLIP_DRIFT_LOG"] = drift_log
            # Save the camera montage that goes into the VLM into a per-run folder.
            extra["AHA_SLIP_VLM_SAVE_GRID"] = os.path.join(SLIP_PLOTS_DIR,
                                                           f"{plot_stem}_vlm_grids")
    elif detector == "collision":
        extra["AHA_VLM_CONFIRM_DELAY_FRAMES"] = "0"   # run the VLM on every fire
        extra["AHA_COLLISION_METHOD"] = os.getenv("AHA_COLLISION_METHOD", "3")           # configured collision method
        if plot_stem:
            # Capture per-step torque telemetry -> collision_method1 trace plot.
            import tempfile
            collision_drift_log = tempfile.mktemp(prefix="aha_collision_drift_",
                                                  suffix=".csv")
            extra["AHA_COLLISION_DRIFT_LOG"] = collision_drift_log
            # Save the camera montage that goes into the VLM into a per-run folder.
            extra["AHA_COLLISION_VLM_SAVE_GRID"] = os.path.join(
                COLLISION_PLOTS_DIR, f"{plot_stem}_vlm_grids")
    t0 = time.time()
    try:
        p = subprocess.run(cmd, env=base_env(detector, cameras, extra),
                           stdin=subprocess.DEVNULL, cwd=PROJECT_ROOT,
                           capture_output=True, text=True, timeout=RUN_TIMEOUT)
        out = (p.stdout or "") + "\n" + (p.stderr or "")
        info = parse_run(out, detector)
        # The runner ends on a "Press Space..." input() that raises EOFError under
        # a /dev/null stdin (non-zero exit) AFTER the episode + summary finished.
        # That is cosmetic: trust the parsed detector summary, not the exit code.
        info["returncode"] = p.returncode
        if info["status"] == "no_output" and p.returncode != 0:
            info["status"] = f"exit{p.returncode}"
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        info = parse_run(out, detector)
        info["status"] = "timeout"
        info["returncode"] = None
    # Render the per-run detector trace plot (slip_method1 / collision_method1).
    info["slip_plot"] = ""
    if drift_log is not None:
        info["slip_plot"] = save_slip_plot(drift_log, task, plot_stem, plot_label)
        try:
            os.remove(drift_log)
        except OSError:
            pass
    elif collision_drift_log is not None:
        inj_wp = (failure_waypoint if failure != "none" and failure_waypoint is not None
                  else None)
        info["slip_plot"] = save_collision_plot(collision_drift_log, task, plot_stem,
                                                plot_label, inj_wp)
        try:
            os.remove(collision_drift_log)
        except OSError:
            pass
    info["elapsed"] = round(time.time() - t0, 1)
    return info


def _price_per_1m(model):
    input_override = os.getenv("AHA_VLM_INPUT_PRICE_PER_1M", "").strip()
    output_override = os.getenv("AHA_VLM_OUTPUT_PRICE_PER_1M", "").strip()
    if input_override and output_override:
        try:
            return float(input_override), float(output_override)
        except ValueError:
            pass

    key = (model or "").strip().lower()
    return OPENAI_PRICE_PER_1M.get(key, (0.0, 0.0))


def _format_usd(value):
    return f"{float(value):.6f}" if value else ""


def parse_run(stdout, detector):
    """Parse fire waypoints, the VLM verdict, retraction and the summary status."""
    fired_wps = sorted(int(wp) for name, wp in FIRE_LINE.findall(stdout)
                       if name.lower() == detector)
    verdicts = [v for n, v in VLM_VERDICT.findall(stdout) if n == detector]
    vlm_verdict = verdicts[-1] if verdicts else ""    # last verdict in the run
    models = [m.strip() for m, n in VLM_MODEL.findall(stdout) if n == detector]
    vlm_model = models[-1] if models else ""
    reasons = [r.strip() for n, r in VLM_REASON.findall(stdout) if n == detector]
    vlm_reason = reasons[-1] if reasons else ""       # reason for the last verdict
    token_rows = [
        (int(input_tokens), int(output_tokens), int(reasoning_tokens or 0))
        for n, input_tokens, output_tokens, reasoning_tokens
        in VLM_TOKENS.findall(stdout)
        if n == detector
    ]
    vlm_calls = len(token_rows)
    vlm_input_tokens = sum(row[0] for row in token_rows)
    vlm_output_tokens = sum(row[1] for row in token_rows)
    vlm_reasoning_tokens = sum(row[2] for row in token_rows)
    input_price, output_price = _price_per_1m(vlm_model)
    vlm_cost_usd = (
        (vlm_input_tokens * input_price + vlm_output_tokens * output_price)
        / 1_000_000.0
        if vlm_model and token_rows and (input_price or output_price)
        else 0.0
    )
    retracted = any(n == detector for n in RETRACT_LINE.findall(stdout))
    # VLM-confirmed fires: a "detection RETRACTED" line overturns the most recent
    # unretracted fire of the same detector (retraction is printed right after the
    # fire+verdict it refers to). Walking fire/retract events in document order
    # yields the fires that SURVIVED VLM confirmation, so downstream scoring can
    # judge the end-to-end pipeline (detector + VLM) rather than the raw detector.
    events = [(m.start(), "fire", int(m.group(2)))
              for m in FIRE_LINE.finditer(stdout) if m.group(1).lower() == detector]
    events += [(m.start(), "retract", None)
               for m in RETRACT_LINE.finditer(stdout) if m.group(1) == detector]
    events.sort()
    surviving = []
    for _, kind, wp in events:
        if kind == "fire":
            surviving.append(wp)
        elif surviving:                       # retract kills the latest live fire
            surviving.pop()
    fired_wps_confirmed = sorted(set(surviving))
    forced = bool(FORCED_FIRE_MARK.search(stdout))
    grids = GRID_SAVED.findall(stdout) if detector in ("slip", "collision") else []

    status = "ran"
    if SUMMARY_DETECTED[detector].search(stdout):
        status = "ran"
    elif SUMMARY_OK[detector].search(stdout):
        status = "ran"
    elif SUMMARY_DISABLED[detector].search(stdout):
        status = "disabled"
    elif SUMMARY_SKIPPED[detector].search(stdout):
        status = "skipped"
    elif not stdout.strip():
        status = "no_output"
    return {"fired_waypoints": fired_wps, "fired": bool(fired_wps),
            "fired_waypoints_confirmed": fired_wps_confirmed,
            "fired_confirmed": bool(fired_wps_confirmed),
            "vlm_verdict": vlm_verdict, "vlm_ran": bool(verdicts),
            "vlm_model": vlm_model, "vlm_calls": vlm_calls,
            "vlm_input_tokens": vlm_input_tokens,
            "vlm_output_tokens": vlm_output_tokens,
            "vlm_reasoning_tokens": vlm_reasoning_tokens,
            "vlm_cost_usd": vlm_cost_usd,
            "vlm_reason": vlm_reason, "vlm_grids": grids,
            "retracted": retracted, "forced_seen": forced, "status": status,
            "stdout_tail": stdout[-1500:]}


# --------------------------------------------------------------------------- #
# Outcome classification
# --------------------------------------------------------------------------- #
def classify_positive(detector, info, inj_wp):
    fired_here = (inj_wp in info["fired_waypoints"]) if inj_wp is not None else info["fired"]
    if not fired_here:
        if info["fired"]:
            return "DET_MISS_other", "detector fired at other waypoint(s), not the injected one"
        return "DET_MISS", "cheap detector did not fire on the injected failure"
    # detector fired at the right place -> judge the VLM
    if detector not in VLM_DETECTORS:
        return "TP_det", "detector fired (no VLM verifier for this detector)"
    if not info["vlm_ran"]:
        return "TP_novlm", "detector fired but VLM did not run"
    if info["vlm_verdict"] == "NO" or info["retracted"]:
        return "FN", "detector right, but VLM wrongly retracted the real failure"
    return "TP", f"detector right and VLM confirmed (verdict={info['vlm_verdict']})"


def classify_negative(detector, info):
    if not info["fired"]:
        return "TN_det", "cheap detector correctly silent on clean run"
    # detector false-alarmed on a clean run
    if detector in VLM_DETECTORS and (info["retracted"] or info["vlm_verdict"] == "NO"):
        return "DET_FP_vlm_saved", "cheap detector false-alarmed; VLM retracted it"
    return "DET_FP", "cheap detector false-alarmed on a clean run"


def classify_forced(detector, info):
    if not info["fired"] and not info["forced_seen"] and not info["retracted"]:
        return "FORCE_FAIL", "forced firing never triggered (detector not active at wp?)"
    if not info["vlm_ran"]:
        return "FORCE_NOVLM", "forced firing but VLM did not run"
    if info["vlm_verdict"] == "NO" or info["retracted"]:
        return "TN", "VLM correctly rejected the deliberately-wrong detection"
    return "FP", f"VLM failed to reject a wrong detection (verdict={info['vlm_verdict']})"


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
FIELDS = ["task", "detector", "phase", "ground_truth", "failtype", "inj_wp",
          "force_wp", "detector_fired", "fired_waypoints", "vlm_ran",
          "vlm_verdict", "vlm_reason", "vlm_model", "vlm_calls",
          "vlm_input_tokens", "vlm_output_tokens", "vlm_reasoning_tokens",
          "vlm_cost_usd", "retracted", "outcome", "note", "status", "elapsed",
          "slip_plot", "vlm_grid"]


def write_rows(rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    exists = os.path.exists(OUT_CSV)
    if exists:
        with open(OUT_CSV, newline="") as f:
            reader = csv.DictReader(f)
            old_fields = reader.fieldnames or []
            old_rows = list(reader)
        if any(field not in old_fields for field in FIELDS):
            with open(OUT_CSV, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                for r in old_rows:
                    clean = {field: r.get(field, "") for field in FIELDS}
                    w.writerow(clean)
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


# --------------------------------------------------------------------------- #
# Evaluation driver
# --------------------------------------------------------------------------- #
def build_jobs(tasks, detectors, phases, cameras, vlm_model):
    jobs = []
    for task in tasks:
        for det in detectors:
            failtype, pos_wp, force_wp = plan_case(task, det)
            if "positive" in phases:
                jobs.append(dict(task=task, detector=det, phase="positive",
                                 cameras=cameras, failtype=failtype, pos_wp=pos_wp,
                                 force_wp=force_wp, vlm_model=vlm_model))
            if "negative" in phases:
                jobs.append(dict(task=task, detector=det, phase="negative",
                                 cameras=cameras, failtype=failtype, pos_wp=pos_wp,
                                 force_wp=force_wp, vlm_model=vlm_model))
            if "forced" in phases and det in VLM_DETECTORS:
                jobs.append(dict(task=task, detector=det, phase="forced",
                                 cameras=cameras, failtype=failtype, pos_wp=pos_wp,
                                 force_wp=force_wp, vlm_model=vlm_model))
    return jobs


def run_job(job):
    task, det, phase = job["task"], job["detector"], job["phase"]
    row = {"task": task, "detector": det, "phase": phase,
           "failtype": job["failtype"] or "", "inj_wp": "", "force_wp": ""}
    if phase == "positive":
        row["ground_truth"] = "fail"
        if not job["failtype"] or job["pos_wp"] is None:
            row.update(outcome="SKIP", note="no matching failure type in config",
                       status="skipped", detector_fired="", fired_waypoints="",
                       vlm_ran="", vlm_verdict="", retracted="", elapsed=0)
            return row
        row["inj_wp"] = job["pos_wp"]
        # slip_method1 naming: <task>.<failtype>.wp<wp>
        stem = f"{task}.{job['failtype']}.wp{job['pos_wp']}"
        info = run_bt(task, det, job["cameras"], vlm_model=job.get("vlm_model"),
                      failure=job["failtype"],
                      failure_waypoint=job["pos_wp"], extreme=True,
                      plot_stem=stem, plot_label=f"{job['failtype']}@wp{job['pos_wp']}")
        outcome, note = classify_positive(det, info, job["pos_wp"])
    elif phase == "negative":
        row["ground_truth"] = "clean"
        info = run_bt(task, det, job["cameras"], vlm_model=job.get("vlm_model"),
                      failure="none",
                      plot_stem=f"{task}.clean", plot_label="clean")
        outcome, note = classify_negative(det, info)
    else:  # forced
        row["ground_truth"] = "clean"
        if job["force_wp"] is None:
            row.update(outcome="SKIP", note="no BT-monitored waypoint to force",
                       status="skipped", detector_fired="", fired_waypoints="",
                       vlm_ran="", vlm_verdict="", retracted="", elapsed=0)
            return row
        row["force_wp"] = job["force_wp"]
        # forced FP is a clean scene; keep it distinct from the negative clean plot
        stem = f"{task}.clean.forced_wp{job['force_wp']}"
        info = run_bt(task, det, job["cameras"], vlm_model=job.get("vlm_model"),
                      failure="none",
                      force_fire_wp=job["force_wp"],
                      plot_stem=stem, plot_label=f"clean(forced@wp{job['force_wp']})")
        outcome, note = classify_forced(det, info)

    row.update(outcome=outcome, note=note, status=info["status"],
               detector_fired=info["fired"], fired_waypoints=info["fired_waypoints"],
               vlm_ran=info["vlm_ran"], vlm_verdict=info["vlm_verdict"],
               vlm_reason=info.get("vlm_reason", ""),
               vlm_model=info.get("vlm_model", ""),
               vlm_calls=info.get("vlm_calls", 0),
               vlm_input_tokens=info.get("vlm_input_tokens", 0),
               vlm_output_tokens=info.get("vlm_output_tokens", 0),
               vlm_reasoning_tokens=info.get("vlm_reasoning_tokens", 0),
               vlm_cost_usd=_format_usd(info.get("vlm_cost_usd", 0.0)),
               retracted=info["retracted"], elapsed=info["elapsed"],
               slip_plot=info.get("slip_plot", ""),
               vlm_grid=";".join(info.get("vlm_grids", [])))
    return row


def print_run_report(row):
    """Human-readable per-run report: what the detector said, what the VLM said,
    and the VLM's reason."""
    phase = row["phase"]
    scene = ("REAL failure injected" if phase == "positive"
             else "CLEAN run, forced false-positive" if phase == "forced"
             else "CLEAN run")
    wp = row.get("inj_wp") or row.get("force_wp") or "?"
    fired = row.get("fired_waypoints", "")
    det = f"FIRED at waypoint(s) {fired}" if row.get("detector_fired") in (True, "True") else "did NOT fire"
    if str(row.get("vlm_ran")) in ("True", "1"):
        vlm = row.get("vlm_verdict") or "UNKNOWN"
    else:
        vlm = "did not run"
    reason = row.get("vlm_reason") or "-"
    plot = row.get("slip_plot", "")
    plot_note = (os.path.basename(plot) if plot and not str(plot).startswith("plot_err")
                 else (plot or "-"))
    print(f"\n  ── {phase.upper()} — {row['detector']} @ {row['task']} (wp{wp}) — {scene}")
    print(f"       detector : {det}")
    print(f"       VLM      : {vlm}")
    print(f"       reason   : {reason}")
    if row.get("vlm_cost_usd"):
        token_bits = (
            f"{row.get('vlm_input_tokens', 0)} in + "
            f"{row.get('vlm_output_tokens', 0)} out"
        )
        calls = row.get("vlm_calls", 0)
        model = row.get("vlm_model") or "unknown-model"
        print(
            f"       cost     : ${float(row['vlm_cost_usd']):.4f} "
            f"({calls} call(s), {token_bits}, {model})"
        )
    else:
        print("       cost     : -")
    print(f"       outcome  : {row['outcome']}   ({row.get('status')}, {row.get('elapsed', 0)}s)")
    print(f"       plot     : {plot_note}")
    grids = [g for g in (row.get("vlm_grid", "") or "").split(";") if g]
    if grids:
        shown = ", ".join(os.path.basename(g) for g in grids)
        print(f"       vlm imgs : {shown}  (in {os.path.dirname(grids[0])})", flush=True)
    else:
        print("       vlm imgs : -", flush=True)


def evaluate(tasks, detectors, phases, cameras, workers, vlm_model):
    os.makedirs(OUT_DIR, exist_ok=True)
    jobs = build_jobs(tasks, detectors, phases, cameras, vlm_model)
    print(f"tasks={len(tasks)} detectors={detectors} phases={phases} "
          f"cameras={cameras or '(default)'} vlm_model={vlm_model or '(runner default)'} "
          f"jobs={len(jobs)} workers={workers}",
          flush=True)
    if "slip" in detectors:
        print(f"slip plots -> {SLIP_PLOTS_DIR}", flush=True)
    if "collision" in detectors:
        print(f"collision plots -> {COLLISION_PLOTS_DIR}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_job, j): j for j in jobs}
        for fut in as_completed(futs):
            row = fut.result()
            write_rows([row])
            print_run_report(row)
    # Only summarize what this invocation actually ran, not stale rows from
    # earlier runs of other tasks/detectors sitting in results.csv.
    summarize(only={(t, d) for t in tasks for d in detectors})


def summarize(only=None):
    if not os.path.exists(OUT_CSV):
        print("no results yet")
        return
    rows = list(csv.DictReader(open(OUT_CSV)))
    if only is not None:
        rows = [r for r in rows if (r["task"], r["detector"]) in only]
    # keep the latest row per (task, detector, phase)
    latest = {}
    for r in rows:
        latest[(r["task"], r["detector"], r["phase"])] = r
    vals = list(latest.values())

    def count(pred):
        return sum(1 for r in vals if pred(r))

    print("\n" + "=" * 72)
    print("  VLM CONFIRMATION LAYER — confusion matrix (per task/detector)")
    print("=" * 72)
    total_cost = sum(float(r.get("vlm_cost_usd") or 0.0) for r in vals)
    total_input_tokens = sum(int(float(r.get("vlm_input_tokens") or 0)) for r in vals)
    total_output_tokens = sum(int(float(r.get("vlm_output_tokens") or 0)) for r in vals)
    total_vlm_calls = sum(int(float(r.get("vlm_calls") or 0)) for r in vals)
    if total_vlm_calls:
        print(
            f"  VLM API usage: calls={total_vlm_calls}  "
            f"input_tokens={total_input_tokens}  "
            f"output_tokens={total_output_tokens}  "
            f"estimated_cost=${total_cost:.4f}"
        )
    TP = count(lambda r: r["outcome"] == "TP")
    FN = count(lambda r: r["outcome"] == "FN")
    TN = count(lambda r: r["outcome"] == "TN")
    FP = count(lambda r: r["outcome"] == "FP")
    print(f"  Real failure, detector fired:   VLM confirm TP={TP}   VLM retract FN={FN}")
    print(f"  Forced wrong detection (clean): VLM retract TN={TN}   VLM keep   FP={FP}")
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec = TP / (TP + FN) if (TP + FN) else 0.0
    print(f"  VLM precision={prec:.2f}  recall={rec:.2f}")

    print("\n" + "-" * 72)
    print("  CHEAP DETECTOR — accuracy (per task/detector)")
    print("-" * 72)
    det_hit = count(lambda r: r["phase"] == "positive" and r["outcome"] in
                    ("TP", "FN", "TP_det", "TP_novlm"))
    det_miss = count(lambda r: r["phase"] == "positive" and r["outcome"].startswith("DET_MISS"))
    det_pos_total = det_hit + det_miss
    det_tn = count(lambda r: r["phase"] == "negative" and r["outcome"] == "TN_det")
    det_fp = count(lambda r: r["phase"] == "negative" and r["outcome"].startswith("DET_FP"))
    det_neg_total = det_tn + det_fp
    print(f"  Recall (fires on injected failure):    {det_hit}/{det_pos_total}"
          f"{'  (%.2f)' % (det_hit/det_pos_total) if det_pos_total else ''}")
    print(f"  Specificity (silent on clean run):     {det_tn}/{det_neg_total}"
          f"{'  (%.2f)' % (det_tn/det_neg_total) if det_neg_total else ''}")

    # per-detector breakdown
    print("\n  Per-detector outcomes:")
    for det in ALL_DETECTORS:
        dr = [r for r in vals if r["detector"] == det]
        if not dr:
            continue
        tally = {}
        for r in dr:
            tally[r["outcome"]] = tally.get(r["outcome"], 0) + 1
        line = "  ".join(f"{k}={v}" for k, v in sorted(tally.items()))
        print(f"    {det:<12} {line}")

    skipped = count(lambda r: r["outcome"] == "SKIP")
    errs = count(lambda r: r["status"] not in ("ran",) and r["outcome"] != "SKIP")
    if skipped or errs:
        print(f"\n  ({skipped} skipped — no matching failure/waypoint; "
              f"{errs} runs with non-clean status)")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", type=int, default=10, help="first N prepared tasks")
    ap.add_argument("--task", action="append", help="run only these named tasks")
    # Default focuses on the VLM-confirmation eval the user asked for: exactly
    # 4 VLM checks per task = orientation {positive, forced} + transition
    # {positive, forced}.  Add --detectors / --phases to widen (e.g. add the
    # 'negative' phase for cheap-detector specificity, or slip/collision/freezing
    # for cheap-detector recall).
    ap.add_argument("--detectors", default="orientation,transition",
                    help="comma-separated subset of detectors "
                         f"(any of {','.join(ALL_DETECTORS)})")
    ap.add_argument("--phases", default="positive,forced",
                    help="positive (inject real failure -> VLM confirm), "
                         "forced (clean + forced fire -> VLM retract), "
                         "negative (clean, cheap-detector specificity; no VLM)")
    ap.add_argument("--cameras", default=DVC.DEFAULT_DETECTOR_VLM_CAMERAS,
                    help="AHA_VLM_CONFIRM_CAMERAS for the VLM (use 'default' to keep "
                         "the full production 5-camera set)")
    ap.add_argument(
        "--vlm-model",
        default=DVC.DEFAULT_DETECTOR_VLM_MODEL,
        help=(
            "OpenAI model for detector VLM confirmations. Defaults to "
            "OPENAI_DETECTOR_VLM_MODEL, then OPENAI_MODEL, then gpt-5.4."
        ),
    )
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()

    if args.summarize:
        summarize()
        return

    tasks = args.task or prepared_tasks()[:args.tasks]
    detectors = [d.strip() for d in args.detectors.split(",")
                 if d.strip() in ALL_DETECTORS]
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    cameras = "" if args.cameras.strip().lower() == "default" else args.cameras.strip()
    evaluate(tasks, detectors, phases, cameras, args.workers, args.vlm_model)


if __name__ == "__main__":
    main()
