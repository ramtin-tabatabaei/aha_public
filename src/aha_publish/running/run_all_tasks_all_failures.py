"""Run EVERY prepared task with EVERY failure defined in its config, all detectors
live at once, optionally with detector VLM confirmation, and save per-task /
per-failure plots + CSVs in one folder.

For each task (task 1 .. end, alphabetical over prepared BTs):
  * one CLEAN episode (no failure injected)
  * one episode per (failure type, waypoint) defined in the task's failgen yaml
    -- e.g. rotation_z@wp0, rotation_z@wp2, slip@wp2, collision@wp1, freezing@wp3.

Every episode runs the BT with ALL five cheap detectors live simultaneously
(orientation, transition, slip, collision, freezing). By default there is NO VLM
checking (--vlm-checks off, AHA_DETECTOR_VLM_AUTO=0). Pass
--vlm-confirm slip,collision to let those detector fires call the VLM verifier
and retract clear false positives. Pass --vlm-prepost to ALSO run the BT
pre/post-condition VLM checks at every waypoint boundary (--vlm-checks both);
optionally restrict them with --vlm-prepost-predicates /
--vlm-prepost-arrival-predicates. Their checkpoint verdicts are recorded per
episode in the prepost_* columns of results.csv (a FAIL verdict stops the BT,
so later waypoints -- and any failure injected there -- never execute).
Each detector's per-step drift log is captured
and rendered into a trace plot (freezing has no trace plot -- only its
fire/status is recorded).

Output layout (--out, default aha_output/all_tasks_all_failures):

  <out>/
    <task>/
      clean/
        orientation.png orientation_drift.csv  transition.png transition_drift.csv
        slip.png slip_drift.csv  collision.png collision_drift.csv  run.log
      rotation_z_wp0/ ...
      slip_wp2/ ...
    results.csv        <- MAIN: one row per (task, case, detector)
    summary.csv        <- per-detector TP/TN/FP/FN + precision/recall/specificity
    misclassified.csv  <- only the FP/FN rows
    run_all.log

Scoring (all detectors independent, per episode):
  * clean episode: a detector that fires -> FP, silent -> TN.
  * failure episode, responsible detector (owns the failure type):
      - episode never reached the injection waypoint (the BT aborts the whole
        episode when ANY detector's hold check is violated)  -> NA_unreached
      - slip injected while the robot held nothing (no slip can manifest,
        judged from is_holding in the slip drift telemetry)   -> NA_no_grasp
      - freezing injected at the final waypoint (the freezing detector skips
        it by design; also skipped at enumeration when the task's
        gripper-sequence n_waypoints is known)                -> NA_final_wp
      - otherwise fired at the injection waypoint (or wp+1) -> TP, else FN.
  * failure episode, any OTHER detector: fired -> CF (cross-fire; often a
    genuine physical consequence of the injection, NOT counted in precision),
    silent -> TN.

Usage:
  python aha_scripts/main_bt_run/run_all_tasks_all_failures.py --workers 3
  python aha_scripts/main_bt_run/run_all_tasks_all_failures.py --redo --workers 4 --vlm-confirm slip,collision
  python aha_scripts/main_bt_run/run_all_tasks_all_failures.py --redo --workers 4 --failures slip,collision --vlm-confirm slip,collision
  python aha_scripts/main_bt_run/run_all_tasks_all_failures.py --redo --vlm-prepost --vlm-prepost-predicates end_effector_aligned_with
  python aha_scripts/main_bt_run/run_all_tasks_all_failures.py --start 1 --limit 5
  python aha_scripts/main_bt_run/run_all_tasks_all_failures.py --task close_box --task open_box
  python aha_scripts/main_bt_run/run_all_tasks_all_failures.py --rescore   # no sim
"""

from aha_publish import paths
import argparse
import csv
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import aha_publish.running.detector_vlm_config as DVC

HERE = str(paths.SOURCE_DIR / 'running')
ROOT = str(paths.PROJECT_ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)                 # plot_* modules + EV internal imports
PY = sys.executable

# Reuse the existing harness for task/config discovery, env setup and run parsing.
_spec = importlib.util.spec_from_file_location(
    "ev", str(paths.SOURCE_DIR / 'running/eval_detectors_vlm.py'))
EV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EV)

DETECTORS = ("orientation", "transition", "slip", "collision", "freezing")
LIVE_ALL = ",".join(DETECTORS)
# failgen failure `type` -> the detector responsible for catching it.
FAILTYPE_DETECTOR = {ft: det for det, fts in EV.DETECTOR_FAILTYPES.items() for ft in fts}
# Failure types no detector is expected to catch (a failed grasp means the
# object was never held, so there is nothing for the cheap detectors to see).
# Not enumerated, and dropped entirely by --rescore.
EXCLUDED_FAILTYPES = {"grasp"}
RUN_TIMEOUT = 1800
# Max attempts for a case whose injection waypoint gets skipped (NA_unreached):
# the skip is stochastic, so re-run and only finalize NA if all attempts skip.
NA_RETRIES = 5

_write_lock = threading.Lock()

RESULT_FIELDS = ["task", "case", "failure", "waypoint", "ground_truth",
                 "responsible_detector", "detector", "fired", "fired_waypoints",
                 "vlm_ran", "vlm_verdict", "vlm_retracted", "vlm_model",
                 "vlm_calls", "vlm_input_tokens", "vlm_output_tokens",
                 "vlm_reasoning_tokens", "vlm_cost_usd", "vlm_reason", "vlm_error",
                 "vlm_grids", "prepost_checks", "prepost_fails",
                 "prepost_fail_wps", "outcome", "status", "plot", "elapsed"]

VLM_AUTO_CONFIRM = re.compile(
    r"\[vlm\] auto-confirming (orientation|transition|slip|collision) detection")
VLM_CONFIRM_FAILED = re.compile(r"\[vlm\] confirmation failed:\s*(.*)")

# Pre/post-condition VLM checkpoint lines (only with --vlm-prepost):
#   [vlm] checking preconditions for waypoint 2...            (single kind)
#   [vlm] checking preconditions for waypoint 3 and postconditions for waypoint 2...
#   VLM PRE CHECK: PASS | VLM PRE+POST CHECK: FAIL | ...
# "VLM <kind> SIMULATOR CHECK" lines are sim-backed condition shortcuts printed
# for the same checkpoint: on PASS the real VLM verdict follows (that line is
# the check), on FAIL/UNKNOWN the checkpoint ends there (the simulator line IS
# the check).
PREPOST_CHECKING = re.compile(
    r"\[vlm\] checking (pre|post)conditions for waypoint (\d+)"
    r"(?: and postconditions for waypoint (\d+))?")
PREPOST_VERDICT = re.compile(
    r"VLM ([A-Z+]+(?: SIMULATOR)?) CHECK: (PASS|FAIL|UNKNOWN)")


def _prepost_fields(stdout):
    """Per-episode pre/post-condition VLM stats parsed from run.log: number of
    checkpoints whose verdict came back, number of FAILs, and (for each FAIL)
    the checkpoint's waypoint label from the preceding '[vlm] checking' line."""
    checks, fails, fail_wps = 0, 0, []
    where = ""
    for line in stdout.splitlines():
        m = PREPOST_CHECKING.search(line)
        if m:
            kind, wp, post_wp = m.groups()
            where = (f"pre{wp}+post{post_wp}" if post_wp is not None
                     else f"{kind}{wp}")
            continue
        m = PREPOST_VERDICT.search(line)
        if not m:
            continue
        kind, verdict = m.groups()
        if "SIMULATOR" not in kind or verdict != "PASS":
            checks += 1
        if verdict == "FAIL":
            fails += 1
            fail_wps.append(where or "?")
    return {"prepost_checks": checks, "prepost_fails": fails,
            "prepost_fail_wps": " ".join(fail_wps)}


# --------------------------------------------------------------------------- #
# Case enumeration
# --------------------------------------------------------------------------- #
GRIPPER_SEQ_DIR = str(paths.GRIPPER_DIR)


def _n_waypoints(task):
    try:
        with open(os.path.join(GRIPPER_SEQ_DIR, f"{task}.json")) as f:
            return int(json.load(f)["n_waypoints"])
    except (OSError, KeyError, TypeError, ValueError):
        return None


def enumerate_cases(task):
    """[(case_name, failure_type, waypoint, responsible_detector)] for a task:
    one clean case + one per (failure type, waypoint) in the task's yaml.

    Freezing at the task's FINAL waypoint is skipped: the freezing detector
    ignores the final waypoint by design, so the case is unwinnable (was 85
    guaranteed-FN episodes across the 92 tasks)."""
    cases = [("clean", "none", None, "")]
    by_type = EV.config_failtypes_by_wp(task)          # {type: set(wp)} from yaml
    n_wp = _n_waypoints(task)
    for ftype in sorted(by_type):
        if ftype in EXCLUDED_FAILTYPES:
            continue
        det = FAILTYPE_DETECTOR.get(ftype, "")
        for wp in sorted(by_type[ftype]):
            if det == "freezing" and n_wp is not None and wp == n_wp - 1:
                continue
            cases.append((f"{ftype}_wp{wp}", ftype, wp, det))
    return cases


# --------------------------------------------------------------------------- #
# One episode: all detectors live, optional detector VLM
# --------------------------------------------------------------------------- #
def run_episode(task, failure, waypoint, case_dir, *,
                vlm_confirm="", vlm_model=None, vlm_cameras=None,
                vlm_trace=False, prepost=False, prepost_predicates="",
                prepost_arrival_predicates=""):
    """Run one BT episode with all detectors live. Returns
    (stdout, elapsed, drift_paths)."""
    os.makedirs(case_dir, exist_ok=True)
    drift = {d: os.path.join(case_dir, f"{d}_drift.csv")
             for d in ("orientation", "transition", "slip", "collision")}
    capture = os.path.join(case_dir, "capture.jsonl")   # replayable input stream
    for p in list(drift.values()) + [capture]:          # start each run clean
        if os.path.exists(p):
            os.remove(p)
    extra = {
        "AHA_LIVE_DETECTORS": LIVE_ALL,                 # every detector live
        "AHA_COLLISION_METHOD": os.getenv("AHA_COLLISION_METHOD", "3"),                    # configured collision method
        # orientation/transition: fire ONLY via C1 arrival. C4 prop is gone from
        # the detectors; this also drops the diverge mode and the instant
        # target-mutation shortcut (mode=target),
        # which otherwise fires at local_step 0 before any approach telemetry
        # and leaves the injection waypoint absent from the trace plot.
        "AHA_ORIENTATION_MODE": "arrival",
        "AHA_ORIENTATION_TARGET_CHANGE_ENABLED": "0",
        # arrival-threshold floor: 10 deg (0.1745 rad) instead of 0.3 rad, so
        # smaller orientation errors (e.g. rotation injections that only nudge
        # the live angle) can trip C1 arrival.
        "AHA_ORIENTATION_ARRIVAL_THRESHOLD": os.getenv("AHA_ORIENTATION_ARRIVAL_THRESHOLD", "0.1745"),
        "AHA_TRANSITION_MODE": "arrival",
        "AHA_TRANSITION_TARGET_CHANGE_ENABLED": "0",
        # arrival-threshold floor: 0.015 m (was 0.02).
        "AHA_TRANSITION_ARRIVAL_THRESHOLD": os.getenv("AHA_TRANSITION_ARRIVAL_THRESHOLD", "0.015"),
        "AHA_ORIENTATION_DRIFT_LOG": drift["orientation"],
        "AHA_TRANSITION_DRIFT_LOG": drift["transition"],
        "AHA_SLIP_DRIFT_LOG": drift["slip"],
        "AHA_COLLISION_DRIFT_LOG": drift["collision"],
        "AHA_DETECTOR_CAPTURE": capture,                # per-step inputs for replay
    }
    if vlm_confirm:
        DVC.apply_detector_vlm_env(
            extra, cameras=vlm_cameras, confirm_detectors=vlm_confirm)
    else:
        extra["AHA_DETECTOR_VLM_AUTO"] = "0"             # NO VLM
    env = EV.base_env(LIVE_ALL, "", extra)              # COPPELIA/DISPLAY setup + overrides

    # --vlm-cameras default (resolves to ""): use EACH detector's OWN built-in
    # camera set (slip = 5 incl. wrist_depth, collision = its own), matching the
    # interactive runner, which never sets this var. base_env unconditionally
    # pins the 4-cam eval default, so clear it here to actually hand control back
    # to the detectors.
    if vlm_confirm and not DVC.detector_vlm_cameras(vlm_cameras):
        env.pop("AHA_VLM_CONFIRM_CAMERAS", None)

    # --vlm-checks both is needed by BOTH VLM channels; which one actually runs
    # is steered by --vlm-predicates (pre/post condition checks) and
    # AHA_VLM_CONFIRM_DETECTORS (detector fire confirmation). Without
    # --vlm-prepost the predicates are pinned to __none__ so no pre/post
    # condition call ever happens.
    if prepost:
        vlm_args = ["--vlm-checks", "both"]
        if prepost_predicates:
            vlm_args += ["--vlm-predicates", prepost_predicates]
        if prepost_arrival_predicates:
            vlm_args += ["--vlm-arrival-predicates", prepost_arrival_predicates]
    elif vlm_confirm:
        vlm_args = ["--vlm-checks", "both", "--vlm-predicates", "__none__"]
    else:
        vlm_args = ["--vlm-checks", "off"]
    cmd = [PY, EV.RUNNER, "--task", task, "--failure", failure,
           "--headless", "--mode", "auto", *vlm_args,
           "--no-side-camera-window", "--no-detector-plots"]
    if vlm_model:
        cmd += ["--vlm-model", vlm_model]
    if vlm_trace:
        cmd += ["--vlm-trace"]
    if failure != "none" and waypoint is not None:
        cmd += ["--failure-waypoint", str(waypoint)]

    t0 = time.time()
    try:
        p = subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL, cwd=EV.PROJECT_ROOT,
                           capture_output=True, text=True, timeout=RUN_TIMEOUT)
        out = (p.stdout or "") + "\n" + (p.stderr or "")
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        out += "\n[TIMEOUT]"
    elapsed = round(time.time() - t0, 1)
    with open(os.path.join(case_dir, "run.log"), "w") as f:
        f.write("$ " + " ".join(cmd) + "\n\n" + out)
    return out, elapsed, drift


# --------------------------------------------------------------------------- #
# Plot rendering from the live drift logs
# --------------------------------------------------------------------------- #
def _read_dicts(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _renumber_steps(rows):
    """The orientation/transition live drift logs write `step` PER WAYPOINT
    (it equals local_step, resetting to 0 at every waypoint), but the trace
    plotters use `step` as the global x axis -- every waypoint segment then
    starts at x=0 and the line loops back on itself. Renumber to row order,
    the same global frame index the *_pipeline plots use."""
    for i, r in enumerate(rows):
        r["step"] = i
    return rows


def _norm_orientation_rows(rows):
    """plot_orientation_trace.plot_trace/write_csv expect the enriched rows its
    own read_drift_rows() builds (typed values + per-waypoint threshold keys
    that need a live detector). Offline we only have the raw drift CSV, so
    type-convert and fill the threshold keys with NaN (the plot then simply
    skips the threshold markers). Without this every orientation plot died
    with KeyError('arrival_threshold')."""
    out = []
    for r in rows:
        try:
            out.append({
                "task": r["task"],
                "waypoint": (int(float(r["waypoint"]))
                             if r.get("waypoint") not in ("", None) else None),
                "local_step": int(float(r["local_step"])),
                "step": int(float(r["step"])),
                "angle_to_target": float(r["angle_to_target"]),
                "runup": float(r["runup"]),
                "runup_positive": int(float(r["runup_positive"])),
                "gripper_orientation_delta": float(r["gripper_orientation_delta"]),
                "path_done": int(float(r["path_done"])),
                "arrival_threshold": float("nan"),
                "rise_threshold": float("nan"),
                "envelope_threshold": float("nan"),
                "orientation_fired": False,
            })
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _fired_steps(rows, fired_wps):
    """First global step of each fired waypoint (for the plot's fire markers)."""
    steps = []
    for wp in fired_wps:
        for r in rows:
            try:
                if int(float(r.get("waypoint", "nan"))) == wp:
                    steps.append(int(float(r["step"])))
                    break
            except (TypeError, ValueError):
                continue
    return sorted(steps)


def _waypoint_boundaries_from_sibling(drift):
    """[(waypoint, first_global_step)] for the collision plot's waypoint lines.

    The collision drift log records only a GLOBAL step (its `waypoint` column is
    empty), so borrow the waypoint timeline from a sibling detector whose drift
    log carries both a global step and a waypoint. Slip is the natural source:
    its `step` is global (unlike orientation/transition, which reset per
    waypoint) and it is live in every episode."""
    for det in ("slip",):
        rows = _read_dicts(drift.get(det))
        if not rows or "waypoint" not in rows[0] or "step" not in rows[0]:
            continue
        bounds, cur = [], None
        for r in rows:
            try:
                w = int(float(r["waypoint"]))
                s = int(float(r["step"]))
            except (TypeError, ValueError):
                continue
            if w != cur:
                bounds.append((w, s))
                cur = w
        if bounds:
            return bounds
    return []


def render_plots(case_dir, task, label, drift, info, stdout="",
                 failure="none", waypoint=None):
    """Render a trace plot per detector into case_dir. Returns {det: png_path}.

    orientation/transition use the *_pipeline plotters (same output as the
    transition_c1c2 folder): distance/angle to target, the C1 arrival threshold
    and the C1 arrival threshold, waypoint-reached lines, and a fire
    marker (labeled with the firing condition) at the detection frame."""
    from pathlib import Path
    out = {}

    # The pipeline plotters recompute the C1 arrival threshold from env + the
    # baked per-task baseline at render time. This renderer runs in the OUTER
    # process (not the episode subprocess), so mirror run_episode's env here --
    # otherwise the plotted threshold silently falls back to the module
    # defaults (flat 0.3 rad / 0.02 m, no baseline), which is NOT what the live
    # detector used to fire.
    os.environ["AHA_ORIENTATION_ARRIVAL_USE_BASELINE"] = "1"
    os.environ["AHA_ORIENTATION_ARRIVAL_THRESHOLD"] = "0.1745"
    os.environ["AHA_TRANSITION_ARRIVAL_USE_BASELINE"] = "1"
    os.environ["AHA_TRANSITION_ARRIVAL_THRESHOLD"] = "0.015"

    # orientation / transition: c1c2-style pipeline plot with both thresholds
    # and the detection-frame fire marker parsed from the run stdout.
    for det, pipe_name, parser_name in (
            ("orientation", "orientation_pipeline", "eval_orientation_detector"),
            ("transition", "transition_pipeline", "eval_transition_detector")):
        if not drift.get(det) or not os.path.exists(drift[det]):
            continue
        try:
            pipe = __import__(pipe_name)
            parser = __import__(parser_name)
            fi = parser.parse_run(stdout) if stdout else {}
            png = os.path.join(case_dir, f"{det}.png")
            res = pipe.render_plot_from_drift(
                drift[det], png, task, label,
                fi.get("fired_steps", []),
                fired_waypoints=fi.get("fired_waypoints"),
                fired_modes=fi.get("fired_modes"))
            out[det] = res or png
        except Exception as exc:                        # a plot failure must not kill the run
            out[det] = f"plot_err:{exc}"

    # slip: same helpers slip_method1 uses (unchanged).
    rows = _read_dicts(drift["slip"])
    if rows:
        try:
            from aha_publish.running.plot_slip_trace_bt import write_csv as slip_write_csv, _read_drift_csv
            from aha_publish.running.plot_slip_trace import plot_trace as slip_plot
            srows = _read_drift_csv(drift["slip"])
            if srows:
                png = os.path.join(case_dir, "slip.png")
                fired = sorted({int(r["step"]) for r in srows if r.get("slip_fired")})
                slip_write_csv(Path(os.path.join(case_dir, "slip_trace.csv")), srows)
                slip_plot(Path(png), srows, task, label, fired)
                out["slip"] = png
        except Exception as exc:
            out["slip"] = f"plot_err:{exc}"

    # collision: method-aware trace. Method 3 (momentum observer, the default)
    # draws the per-joint De Luca residual plot; method 1 draws the torque-spike
    # trace. Mirror plot_collision_torque_trace.main()'s dispatch so the plot
    # matches the detector that actually ran. Both plotters share the signature
    # (path, rows, paused_steps, task, label, waypoint_frames=, injected_wp=).
    if os.path.exists(drift["collision"]):
        try:
            from aha_publish.running.plot_collision_torque_trace import plot_trace, plot_residual_trace, write_csv as coll_write, confirmed_steps_from_rows
            method = os.getenv("AHA_COLLISION_METHOD", "3").strip()
            coll_plot = plot_residual_trace if method == "3" else plot_trace
            crows = _read_dicts(drift["collision"])
            if crows:
                png = os.path.join(case_dir, "collision.png")
                # Confirmed fires only (raw crossings need N consecutive
                # frames): plotting raw candidates puts "first detection" in an
                # earlier waypoint than the run log / summary CSV report.
                paused = confirmed_steps_from_rows(crows)
                wp_frames = _waypoint_boundaries_from_sibling(drift)
                inj = waypoint if failure == "collision" else None
                coll_write(Path(os.path.join(case_dir, "collision_trace.csv")), crows)
                coll_plot(Path(png), crows, paused, task, label,
                          waypoint_frames=wp_frames, injected_wp=inj)
                out["collision"] = png
        except Exception as exc:
            out["collision"] = f"plot_err:{exc}"

    return out


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score(det, info, is_clean, responsible, inj_wp, *,
          reached=True, slip_holding=None, freezing_skips_wp=False):
    """Outcome for one (detector, episode):

      TP/TN/FP/FN     as usual, except FP is now CLEAN-episode-only.
      CF              cross-fire: a non-responsible detector fired during
                      another failure's episode. Often a genuine physical
                      consequence of the injection (e.g. a translation offset
                      ramming the table IS a collision), so it must not
                      pollute FP counts.
      NA_unreached    the episode ended before the injection waypoint started,
                      so the failure was never injected -- nothing to detect.
      NA_no_grasp     slip only: the injection hit while the robot was not
                      holding anything, so no slip could manifest.
      NA_final_wp     freezing only: injected at the final waypoint, which the
                      freezing detector skips by design.
    """
    # Score the END-TO-END pipeline (cheap detector + VLM confirmation): a fire
    # the VLM retracted is not a pipeline detection, so use the VLM-confirmed fire
    # set. This turns a retracted fire on a clean episode into TN (not FP) and a
    # wrongly-retracted real failure into FN. Falls back to the raw fires for
    # detectors/episodes with no VLM (retracted fires there don't exist).
    fired = info.get("fired_confirmed", info["fired"])
    fwps = info.get("fired_waypoints_confirmed", info["fired_waypoints"])
    if is_clean:
        return "FP" if fired else "TN"
    if det != responsible:
        return "CF" if fired else "TN"
    # "Can't-manifest" NAs take precedence: if the failure physically could not
    # occur, a fire is spurious, not a true catch, so these stay NA even if the
    # detector fired.
    if freezing_skips_wp:
        return "NA_final_wp"
    if det == "slip" and slip_holding is False:
        return "NA_no_grasp"
    # A detector that fired at (or one waypoint after) the injection point
    # DETECTED the failure -> TP, even if the arm then skipped the waypoint's
    # motion (orientation's target-mode catches the corrupted goal at wp start).
    # Checked before the reached/skip branch so a genuine catch is never
    # downgraded to NA_unreached.
    hit = (inj_wp in fwps or (inj_wp + 1) in fwps) if inj_wp is not None else fired
    if hit:
        return "TP"
    if not reached:
        return "NA_unreached"
    return "FN"


_FREEZE_SKIP_RE = re.compile(r"skipping final waypoint (\d+)")


def _drift_max_wp(drift):
    """Highest waypoint with any telemetry across the drift logs. The failure
    hook runs when its waypoint STARTS, so telemetry at wp >= inj_wp means the
    injection actually happened."""
    best = -1
    for p in drift.values():
        if not p or not os.path.exists(p):
            continue
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    best = max(best, int(float(r["waypoint"])))
                except (KeyError, TypeError, ValueError):
                    continue
    return best


def _finished_injection_wp(out, wp):
    """True if the arm actually executed the injected waypoint's MOTION to
    completion, i.e. stdout contains '[move] finished waypoint N path ...'.

    When the motion planner skips the injected waypoint (it prints '[move]
    planning path for waypoint N' then jumps straight to the next waypoint with
    no 'executing'/'finished path' line), the responsible orientation/transition
    detector never receives an arrival observation for that waypoint -- it never
    had a chance to judge the failure. Such an episode is scored NA_unreached
    rather than a false miss (FN). This uses the LOGICAL waypoint index (matching
    inj_wp / fired_waypoints), unlike the raw scene-object index in the drift
    logs, which RLBench renumbers per episode."""
    if wp is None:
        return True
    return re.search(rf"finished waypoint {int(wp)} path", out) is not None


def _slip_holding_at_injection(slip_csv, inj_wp):
    """True/False: was the robot holding an object when the slip injection hit
    (start of inj_wp)? Judged from slip telemetry in a window spanning the last
    frames before the waypoint starts and its first frames. None if unknown."""
    if not slip_csv or not os.path.exists(slip_csv):
        return None
    with open(slip_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    first = next((i for i, r in enumerate(rows)
                  if r.get("waypoint") not in ("", None)
                  and int(float(r["waypoint"])) == inj_wp), None)
    if first is None:
        return None
    window = rows[max(0, first - 10):first + 3]

    def _i(r, k):
        try:
            return int(float(r.get(k) or 0))
        except (TypeError, ValueError):
            return 0
    return any(_i(r, "is_holding") or _i(r, "prior_holding_streak")
               for r in window)


def episode_extras(is_clean, responsible, waypoint, out, drift):
    """(reached, slip_holding, freezing_skips) for score(), from the episode's
    stdout + drift logs -- everything on disk, so --rescore can reuse it."""
    reached, slip_holding, freezing_skips = True, None, False
    if not is_clean and waypoint is not None:
        if responsible in ("orientation", "transition"):
            # Pose-corruption failures: only a MISS on a waypoint the arm truly
            # executed to arrival is a real FN; a planner-skipped waypoint is NA.
            reached = _finished_injection_wp(out, waypoint)
        else:
            # slip/collision/freezing: keep the telemetry-based reach test (a
            # frozen arm legitimately never "finishes" its path, so the stdout
            # signal would wrongly mark freezing catches unreached).
            reached = _drift_max_wp(drift) >= waypoint
        if responsible == "slip":
            slip_holding = _slip_holding_at_injection(drift["slip"], waypoint)
        if responsible == "freezing":
            m = _FREEZE_SKIP_RE.search(out)
            freezing_skips = m is not None and int(m.group(1)) == waypoint
    return reached, slip_holding, freezing_skips


# --------------------------------------------------------------------------- #
# One job = one (task, case) episode -> per-detector result rows
# --------------------------------------------------------------------------- #
def run_job(task, case, out_dir, na_retries=NA_RETRIES, *,
            vlm_confirm="", vlm_model=None, vlm_cameras=None,
            vlm_trace=False, prepost=False, prepost_predicates="",
            prepost_arrival_predicates=""):
    case_name, failure, waypoint, responsible = case
    case_dir = os.path.join(out_dir, task, case_name)
    is_clean = failure == "none"
    label = "clean" if is_clean else f"{failure}@wp{waypoint}"

    # A skipped injection waypoint (NA_unreached) is often stochastic -- the
    # motion planner sometimes skips the corrupted waypoint's path. Re-run up to
    # na_retries times to give the failure a chance to actually execute; only
    # finalize NA if every attempt skips it. A genuine catch (TP, incl.
    # orientation target-mode) or a real miss on an executed waypoint (FN) stops
    # retrying immediately -- those are not NA. Each attempt overwrites the case
    # dir, so the kept run.log/drift/raw-capture belong to the final attempt.
    attempt = 0
    while True:
        attempt += 1
        out, elapsed, drift = run_episode(
            task, failure, waypoint, case_dir,
            vlm_confirm=vlm_confirm, vlm_model=vlm_model,
            vlm_cameras=vlm_cameras, vlm_trace=vlm_trace,
            prepost=prepost, prepost_predicates=prepost_predicates,
            prepost_arrival_predicates=prepost_arrival_predicates)
        info = _add_vlm_errors({d: EV.parse_run(out, d) for d in DETECTORS}, out)
        reached, slip_holding, freezing_skips = episode_extras(
            is_clean, responsible, waypoint, out, drift)
        if is_clean or attempt >= na_retries:
            break
        resp_outcome = score(responsible, info[responsible], is_clean,
                             responsible, waypoint, reached=reached,
                             slip_holding=slip_holding,
                             freezing_skips_wp=freezing_skips)
        if resp_outcome != "NA_unreached":
            break
        print(f"[retry] {task}/{case_name}: injection wp {waypoint} skipped "
              f"(NA_unreached), attempt {attempt}/{na_retries} -> re-running",
              flush=True)

    plots = render_plots(case_dir, task, label, drift, info, stdout=out,
                         failure=failure, waypoint=waypoint)

    prepost_fields = _prepost_fields(out)               # per-episode, same on every row
    rows = []
    for det in DETECTORS:
        info_det = info[det]
        rows.append({
            "task": task, "case": case_name, "failure": failure,
            "waypoint": "" if waypoint is None else waypoint,
            "ground_truth": "clean" if is_clean else "fail",
            "responsible_detector": responsible, "detector": det,
            "fired": info_det["fired"],
            "fired_waypoints": " ".join(map(str, info_det["fired_waypoints"])),
            **_vlm_row_fields(info_det),
            **prepost_fields,
            "outcome": score(det, info_det, is_clean, responsible, waypoint,
                             reached=reached, slip_holding=slip_holding,
                             freezing_skips_wp=freezing_skips),
            "status": info_det["status"], "plot": plots.get(det, ""),
            "elapsed": elapsed,
        })
    return rows


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _r(x):
    return "nan" if (x is None or (isinstance(x, float) and math.isnan(x))) else round(x, 4)


def _vlm_row_fields(info):
    return {
        "vlm_ran": info.get("vlm_ran", ""),
        "vlm_verdict": info.get("vlm_verdict", ""),
        "vlm_retracted": info.get("retracted", ""),
        "vlm_model": info.get("vlm_model", ""),
        "vlm_calls": info.get("vlm_calls", ""),
        "vlm_input_tokens": info.get("vlm_input_tokens", ""),
        "vlm_output_tokens": info.get("vlm_output_tokens", ""),
        "vlm_reasoning_tokens": info.get("vlm_reasoning_tokens", ""),
        "vlm_cost_usd": info.get("vlm_cost_usd", ""),
        "vlm_reason": info.get("vlm_reason", ""),
        "vlm_error": info.get("vlm_error", ""),
        "vlm_grids": " ".join(info.get("vlm_grids") or []),
    }


def _add_vlm_errors(info, stdout):
    """Attach detector-specific VLM errors parsed from run.log.

    A failed OpenAI call prints no final "[vlm:model] detector=YES/NO" verdict,
    so EV.parse_run correctly reports no verdict but loses the reason. Track the
    detector named by the preceding auto-confirm line and save the failure text.
    """
    pending_detector = None
    errors = {}
    for line in stdout.splitlines():
        m = VLM_AUTO_CONFIRM.search(line)
        if m:
            pending_detector = m.group(1)
            continue
        m = VLM_CONFIRM_FAILED.search(line)
        if m and pending_detector:
            errors[pending_detector] = m.group(1).strip()
            pending_detector = None
    for det, message in errors.items():
        if det in info:
            info[det]["vlm_error"] = message
            info[det]["vlm_ran"] = True
    return info


def rewrite_summaries(out_dir):
    results = os.path.join(out_dir, "results.csv")
    if not os.path.exists(results):
        return
    with open(results, newline="") as f:
        rows = list(csv.DictReader(f))
    per = {d: {k: 0 for k in ("TP", "TN", "FP", "FN", "CF", "NA")} for d in DETECTORS}
    mis = []
    for r in rows:
        d, o = r["detector"], r["outcome"]
        if d in per:
            if o.startswith("NA"):
                per[d]["NA"] += 1
            elif o in per[d]:
                per[d][o] += 1
        if o in ("FP", "FN"):
            mis.append(r)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["detector", "TP", "TN", "FP", "FN", "cross_fire", "NA",
                    "precision", "recall", "specificity"])
        for d in DETECTORS:
            c = per[d]
            tp, tn, fp, fn = c["TP"], c["TN"], c["FP"], c["FN"]
            prec = tp / (tp + fp) if (tp + fp) else float("nan")
            rec = tp / (tp + fn) if (tp + fn) else float("nan")
            spec = tn / (tn + fp) if (tn + fp) else float("nan")
            w.writerow([d, tp, tn, fp, fn, c["CF"], c["NA"],
                        _r(prec), _r(rec), _r(spec)])
    with open(os.path.join(out_dir, "misclassified.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        w.writeheader()
        for r in mis:
            w.writerow({k: r.get(k, "") for k in RESULT_FIELDS})


def append_results(out_dir, rows):
    results = os.path.join(out_dir, "results.csv")
    with _write_lock:
        exists = os.path.exists(results)
        if exists:
            with open(results, newline="") as f:
                header = next(csv.reader(f), [])
            if header != RESULT_FIELDS:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                backup = f"{results}.schema_mismatch.{stamp}.bak"
                os.replace(results, backup)
                print(f"[run_all] results.csv schema changed; previous file -> "
                      f"{backup}", flush=True)
                exists = False
        with open(results, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
            if not exists:
                w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in RESULT_FIELDS})
        rewrite_summaries(out_dir)


# --------------------------------------------------------------------------- #
# Offline re-scoring: rebuild results.csv from run.log + drift csvs on disk
# with the CURRENT scoring rules. No simulation.
# --------------------------------------------------------------------------- #
def _parse_case_name(case):
    """'clean' -> ('none', None); '<ftype>_wp<N>' -> (ftype, N); else None."""
    if case == "clean":
        return "none", None
    head, sep, tail = case.rpartition("_wp")
    if not sep or not tail.isdigit():
        return None
    return head, int(tail)


def rescore(out_dir):
    old_elapsed = {}
    results = os.path.join(out_dir, "results.csv")
    if os.path.exists(results):
        with open(results, newline="") as f:
            for r in csv.DictReader(f):
                old_elapsed[(r["task"], r["case"], r["detector"])] = r.get("elapsed", "")
        os.replace(results, results + ".bak")

    n_cases = 0
    all_rows = []
    for task in sorted(os.listdir(out_dir)):
        tdir = os.path.join(out_dir, task)
        if not os.path.isdir(tdir):
            continue
        for case in sorted(os.listdir(tdir)):
            cdir = os.path.join(tdir, case)
            runlog = os.path.join(cdir, "run.log")
            parsed = _parse_case_name(case)
            if not os.path.isfile(runlog) or parsed is None:
                continue
            failure, waypoint = parsed
            if failure in EXCLUDED_FAILTYPES:
                continue
            is_clean = failure == "none"
            responsible = "" if is_clean else FAILTYPE_DETECTOR.get(failure, "")
            with open(runlog, errors="ignore") as f:
                out = f.read()
            drift = {d: os.path.join(cdir, f"{d}_drift.csv")
                     for d in ("orientation", "transition", "slip", "collision")}
            info = _add_vlm_errors({d: EV.parse_run(out, d) for d in DETECTORS}, out)
            reached, slip_holding, freezing_skips = episode_extras(
                is_clean, responsible, waypoint, out, drift)
            prepost_fields = _prepost_fields(out)
            n_cases += 1
            for det in DETECTORS:
                png = os.path.join(cdir, f"{det}.png")
                info_det = info[det]
                all_rows.append({
                    "task": task, "case": case, "failure": failure,
                    "waypoint": "" if waypoint is None else waypoint,
                    "ground_truth": "clean" if is_clean else "fail",
                    "responsible_detector": responsible, "detector": det,
                    "fired": info_det["fired"],
                    "fired_waypoints": " ".join(map(str, info_det["fired_waypoints"])),
                    **_vlm_row_fields(info_det),
                    **prepost_fields,
                    "outcome": score(det, info_det, is_clean, responsible, waypoint,
                                     reached=reached, slip_holding=slip_holding,
                                     freezing_skips_wp=freezing_skips),
                    "status": info_det["status"],
                    "plot": png if os.path.exists(png) else "",
                    "elapsed": old_elapsed.get((task, case, det), ""),
                })

    with open(results, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    rewrite_summaries(out_dir)
    print(f"[rescore] {n_cases} episodes re-scored -> {results} "
          f"(previous file kept as results.csv.bak)")


def replot(out_dir):
    """No simulation: regenerate every detector plot from the run.log + drift
    CSVs already on disk, using the current render_plots. Reuses whatever python
    is running this (must have matplotlib -- use the `aha` conda env)."""
    n = 0
    for task in sorted(os.listdir(out_dir)):
        tdir = os.path.join(out_dir, task)
        if not os.path.isdir(tdir):
            continue
        for case in sorted(os.listdir(tdir)):
            cdir = os.path.join(tdir, case)
            runlog = os.path.join(cdir, "run.log")
            parsed = _parse_case_name(case)
            if not os.path.isfile(runlog) or parsed is None:
                continue
            failure, waypoint = parsed
            if failure in EXCLUDED_FAILTYPES:
                continue
            with open(runlog, errors="ignore") as f:
                out = f.read()
            drift = {d: os.path.join(cdir, f"{d}_drift.csv")
                     for d in ("orientation", "transition", "slip", "collision")}
            info = {d: EV.parse_run(out, d) for d in DETECTORS}
            label = "clean" if failure == "none" else f"{failure}@wp{waypoint}"
            render_plots(cdir, task, label, drift, info, stdout=out,
                         failure=failure, waypoint=waypoint)
            n += 1
            if n % 200 == 0:
                print(f"[replot] {n} episodes done", flush=True)
    print(f"[replot] regenerated plots for {n} episodes")


# --------------------------------------------------------------------------- #
def main():
    global RUN_TIMEOUT
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(paths.RUNS_DIR),
                    help="parent output folder (default aha_output/all_tasks_all_failures)")
    ap.add_argument("--workers", type=int, default=3,
                    help="parallel episodes (default 3; each episode is a full "
                         "CoppeliaSim instance at ~3-5 GB RAM -- with 31 GB and "
                         "no swap, >3 risks an OOM kill)")
    ap.add_argument("--redo", action="store_true",
                    help="rerun every case even if it already has results "
                         "(default: resume -- skip (task, case) pairs already "
                         "in results.csv)")
    ap.add_argument("--rescore", action="store_true",
                    help="no simulation: rebuild results.csv + summaries from "
                         "the run.log / drift csvs already on disk, applying "
                         "the current scoring rules")
    ap.add_argument("--replot", action="store_true",
                    help="no simulation: regenerate every detector plot from "
                         "the drift csvs on disk (needs matplotlib -- run with "
                         "the `aha` conda env python)")
    ap.add_argument("--task", action="append", default=None,
                    help="run only these named task(s); repeatable")
    ap.add_argument("--failures", default="",
                    help="comma-separated failure types to run, e.g. "
                         "slip,collision. Clean baseline episodes are still "
                         "included for specificity scoring. Default: all.")
    ap.add_argument("--waypoints", default="",
                    help="comma-separated injection waypoint indices to run, "
                         "e.g. 2 or 2,3. Filters failure cases only; the clean "
                         "baseline episode is still included. Default: all.")
    ap.add_argument("--no-clean", action="store_true",
                    help="skip the clean baseline episode and run only the "
                         "selected failure cases (no TN/FP specificity rows)")
    ap.add_argument("--start", type=int, default=1,
                    help="1-based index into the task list to start from (default 1)")
    ap.add_argument("--limit", type=int, default=None,
                    help="run at most this many tasks from --start")
    ap.add_argument("--timeout", type=int, default=RUN_TIMEOUT,
                    help=f"per-episode timeout seconds (default {RUN_TIMEOUT})")
    ap.add_argument("--na-retries", type=int, default=NA_RETRIES,
                    help="max attempts for a case whose injection waypoint is "
                         f"skipped before finalizing NA (default {NA_RETRIES})")
    ap.add_argument("--vlm-confirm", default="",
                    help="comma-separated live detectors whose fires should be "
                         "confirmed by the detector VLM, e.g. slip,collision. "
                         "Default blank keeps the historical no-VLM run.")
    ap.add_argument("--vlm-model",
                    default=DVC.DEFAULT_DETECTOR_VLM_MODEL,
                    help="optional OpenAI model name passed through to the BT "
                         "runner for detector VLM confirmation")
    ap.add_argument("--vlm-cameras",
                    default=DVC.DEFAULT_DETECTOR_VLM_CAMERAS,
                    help="AHA_VLM_CONFIRM_CAMERAS for detector VLM confirmation "
                         "(default matches detector_vlm_eval; use 'default' for "
                         "the detector modules' production defaults)")
    ap.add_argument("--vlm-trace", action="store_true",
                    help="save detector VLM prompts/responses under "
                         "aha_output/vlm_traces")
    ap.add_argument("--vlm-prepost", action="store_true",
                    help="ALSO run the BT pre/post-condition VLM checks at "
                         "every waypoint boundary (--vlm-checks both). Verdicts "
                         "land in the prepost_* columns of results.csv. NOTE: a "
                         "FAIL verdict stops the BT, so waypoints after it "
                         "never run.")
    ap.add_argument("--vlm-prepost-predicates", default="",
                    help="comma-separated predicate names the pre/post VLM "
                         "checks are restricted to (passed through as "
                         "--vlm-predicates), e.g. end_effector_aligned_with. "
                         "Default: check all conditions.")
    ap.add_argument("--vlm-prepost-arrival-predicates", default="",
                    help="comma-separated predicate names verified at WAYPOINT "
                         "ARRIVAL instead of the pre/post boundary (passed "
                         "through as --vlm-arrival-predicates), e.g. "
                         "gripper_oriented_for.")
    args = ap.parse_args()

    RUN_TIMEOUT = args.timeout
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    if args.rescore:
        rescore(out_dir)
        return
    if args.replot:
        replot(out_dir)
        return

    valid_vlm_confirm = {d.strip() for d in args.vlm_confirm.split(",") if d.strip()}
    unknown_vlm_confirm = valid_vlm_confirm - set(DETECTORS)
    if unknown_vlm_confirm:
        ap.error("--vlm-confirm contains unknown detector(s): "
                 + ",".join(sorted(unknown_vlm_confirm)))
    vlm_confirm = ",".join(d for d in DETECTORS if d in valid_vlm_confirm)

    if not args.vlm_prepost and (args.vlm_prepost_predicates
                                 or args.vlm_prepost_arrival_predicates):
        ap.error("--vlm-prepost-predicates/--vlm-prepost-arrival-predicates "
                 "require --vlm-prepost")

    tasks = args.task or EV.prepared_tasks()
    if not args.task:
        tasks = tasks[args.start - 1:]
        if args.limit is not None:
            tasks = tasks[:args.limit]

    allowed_failures = {f.strip() for f in args.failures.split(",") if f.strip()}
    if allowed_failures:
        configured_failures = set()
        for task in tasks:
            configured_failures.update(EV.config_failtypes_by_wp(task))
        unknown_failures = allowed_failures - configured_failures - {"none"}
        if unknown_failures:
            ap.error("--failures contains unknown/unconfigured failure type(s): "
                     + ",".join(sorted(unknown_failures)))

    try:
        allowed_waypoints = {int(w) for w in args.waypoints.split(",") if w.strip()}
    except ValueError:
        ap.error("--waypoints must be comma-separated integers")

    def _selected_cases(task):
        cases = enumerate_cases(task)
        if allowed_failures:
            cases = [case for case in cases
                     if case[1] == "none" or case[1] in allowed_failures]
        if allowed_waypoints:
            cases = [case for case in cases
                     if case[1] == "none" or case[2] in allowed_waypoints]
        if args.no_clean:
            cases = [case for case in cases if case[1] != "none"]
        return cases

    # Resume: a (task, case) pair reaches results.csv only after its episode
    # fully finished and parsed, so pairs already there are safe to skip and
    # half-run episodes (killed mid-flight) get rerun automatically. For --redo,
    # start a fresh result CSV rather than appending duplicate rows to the old
    # one; per-case files are still overwritten by run_episode().
    done_pairs = set()
    results_path = os.path.join(out_dir, "results.csv")
    if args.redo and os.path.exists(results_path):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup = f"{results_path}.redo.{stamp}.bak"
        os.replace(results_path, backup)
        print(f"[run_all] --redo: previous results.csv -> {backup}", flush=True)
    if not args.redo and os.path.exists(results_path):
        with open(results_path, newline="") as f:
            for r in csv.DictReader(f):
                done_pairs.add((r["task"], r["case"]))

    cases_by_task = {t: _selected_cases(t) for t in tasks}
    jobs = [(t, c) for t in tasks for c in cases_by_task[t]
            if (t, c[0]) not in done_pairs]
    skipped = sum(len(cases_by_task[t]) for t in tasks) - len(jobs)
    print(f"[run_all] out={out_dir} tasks={len(tasks)} episodes={len(jobs)} "
          f"(skipped {skipped} already-done) workers={args.workers} "
          f"failures={','.join(sorted(allowed_failures)) if allowed_failures else 'all'} "
          f"(all detectors live, "
          f"{'VLM confirm=' + vlm_confirm + ', cameras=' + (DVC.detector_vlm_cameras(args.vlm_cameras) or 'per-detector default') + ', model=' + args.vlm_model if vlm_confirm else 'no detector VLM'}, "
          f"{'prepost VLM=' + (args.vlm_prepost_predicates or 'all predicates') if args.vlm_prepost else 'no prepost VLM'})\n",
          flush=True)

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(
                    run_job, t, c, out_dir, args.na_retries,
                    vlm_confirm=vlm_confirm, vlm_model=args.vlm_model,
                    vlm_cameras=args.vlm_cameras, vlm_trace=args.vlm_trace,
                    prepost=args.vlm_prepost,
                    prepost_predicates=args.vlm_prepost_predicates,
                    prepost_arrival_predicates=args.vlm_prepost_arrival_predicates): (t, c)
                for t, c in jobs}
        for fut in as_completed(futs):
            t, c = futs[fut]
            done += 1
            try:
                rows = fut.result()
            except Exception as exc:                    # keep the batch alive
                print(f"[{done}/{len(jobs)}] {t}/{c[0]} ERROR: {exc}", flush=True)
                continue
            append_results(out_dir, rows)
            fired = [r["detector"] for r in rows if str(r["fired"]) == "True"]
            print(f"[{done}/{len(jobs)}] {t}/{c[0]} "
                  f"fired={','.join(fired) or 'none'} ({rows[0]['elapsed']}s)", flush=True)

    rewrite_summaries(out_dir)
    print(f"\n[run_all] done. results -> {os.path.join(out_dir, 'results.csv')}")
    print(f"[run_all] summary -> {os.path.join(out_dir, 'summary.csv')}")
    print(f"[run_all] misclassified -> {os.path.join(out_dir, 'misclassified.csv')}")


if __name__ == "__main__":
    main()
