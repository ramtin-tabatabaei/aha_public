"""bt_gui-embedded ORIENTATION detector.

A copy of aha_scripts/detectors/orientation/interactive.py's real-time detection
(``patched_step`` + ``prompt_vlm_confirmation`` + live telemetry plot), adapted
to be fed by the bt_gui BT run.

Detection rule: the arrival check compares the gripper with the original
waypoint orientation so injected rotation failures are caught from the executed
motion/arrival state, not from an instant target-mutation shortcut.
"""

from aha_publish import paths

import os
from pathlib import Path

import numpy as np

from ._bundle import (
    VLM_AUTO, FORCE_FIRE_WAYPOINT, FORCE_FIRE_REASON, load_detector_bundle)

NAME = "orientation"
FAILURE = "orientation"
CONDITION = "orientation_maintained() == True"
# No live waypoint pose needed: arrival angle is measured against the composed reference
# (ttm_waypoint_distance), never the live target.
NEEDS_WAYPOINT_POSE = False

# Flat floor for the arrival threshold (rad): the smallest arrival angle worth
# firing on -- above clean servo noise (~0.07 rad) and below a real mis-
# orientation. DECOUPLED from the abstain ceiling (low floor catches real
# failures whose settled deviation can be well under the injected rotation; the
# high ceiling drops unconstrained waypoints).
ARRIVAL_ANGLE_THRESHOLD = float(
    os.getenv("AHA_ORIENTATION_ARRIVAL_THRESHOLD", "0.3"))
TARGET_ORIENTATION_GAP_THRESHOLD = float(
    os.getenv("AHA_ORIENTATION_TARGET_GAP_THRESHOLD", "0.30"))
# Target-orientation-change shortcut. Defaults OFF: the detector should fire only
# from the selected motion/arrival checks unless explicitly requested.
TARGET_CHANGE_ENABLED = os.getenv(
    "AHA_ORIENTATION_TARGET_CHANGE_ENABLED", "0"
).strip().lower() in ("1", "true", "yes", "on")
TARGET_CHANGE_START_WINDOW = int(
    os.getenv("AHA_ORIENTATION_TARGET_CHANGE_START_WINDOW", "3"))
POSITION_TARGET_CHANGE_SUPPRESS_THRESHOLD = float(
    os.getenv(
        "AHA_ORIENTATION_POSITION_GAP_SUPPRESS_THRESHOLD",
        "0.03",
    ))
_REPO_ROOT = (paths.PROJECT_ROOT)
_CALIBRATION_ROOT = Path(os.getenv(
    "AHA_CALIBRATION_ROOT",
    str(paths.CALIBRATION_DIR),
)).expanduser()
ARRIVAL_STATS_DIR = Path(os.getenv(
    "AHA_ORIENTATION_STATS_DIR",
    str(_CALIBRATION_ROOT / "orientation_arrival_stats"),
)).expanduser()

# Three conditions (all driven by the per-task clean-run baselines):
#
# C1 ARRIVAL  - at waypoint completion, the angle to the waypoint-defined
#               orientation > max(floor, mean + K1*std) of the clean arrival
#               error for that waypoint.  K1 default 2.
# C2 ENVELOPE - during motion, the live angle to the target leaves the clean
#               envelope: err > max(error_max + Kmax*std, error_mean + Kmean*std)
#               from the per-task PLOT-1 error_stats.  Kmax=1, Kmean=3.
# C3 DIVERGE  - during motion, the error keeps rising (delta > 0) for MORE than
#               N consecutive frames AND the accumulated rise (PLOT-2) exceeds
#               mean + K3*std of the clean per-waypoint segment rises.  N=3, K3=3.
# C1 arrival threshold is a ROBUST per-waypoint band keyed off the WORST clean
# arrival angle ever seen at that waypoint (not mean+K*std): thr =
# max(floor, max_clean*SCALE + MARGIN). Robust to tiny-std waypoints (where
# mean+K*std nips a held-out clean run) and to a single anomalous clean episode,
# while staying well below the injected-failure rotation (~1.57 rad).
ARRIVAL_SCALE = float(os.getenv("AHA_ORIENTATION_ARRIVAL_SCALE", "1.3"))
ARRIVAL_MARGIN = float(os.getenv("AHA_ORIENTATION_ARRIVAL_MARGIN", "0.05"))
# ABSTAIN: a waypoint whose WORST clean arrival angle exceeds this is not
# tracking its episode-start (original) orientation -- the gripper roll is
# unconstrained there (e.g. a button press, where any roll succeeds, reads a
# large and run-to-run-variable angle even when clean), so no clean-derived
# threshold can separate clean from a failure. The detector ABSTAINS
# (threshold -> inf) at such waypoints; the eval injects elsewhere. DECOUPLED from
# the arrival floor: the floor is low (catch real failures), this ceiling is high
# (only drop waypoints whose clean arrival is so large -- approaching the ~1.57
# rad injected rotation -- that it can't be separated from a failure, e.g. a
# free-roll button press reading ~2.4 rad even when clean).
ARRIVAL_ABSTAIN_CEIL = float(os.getenv("AHA_ORIENTATION_ARRIVAL_ABSTAIN_CEIL", "0.80"))
ARRIVAL_SIGMA_K = float(os.getenv("AHA_ORIENTATION_ARRIVAL_SIGMA_K", "3"))  # back-compat
ENVELOPE_MAX_K = float(os.getenv("AHA_ORIENTATION_ENVELOPE_MAX_K", "1"))
ENVELOPE_MEAN_K = float(os.getenv("AHA_ORIENTATION_ENVELOPE_MEAN_K", "3"))
ENVELOPE_CONSEC = int(os.getenv("AHA_ORIENTATION_ENVELOPE_CONSEC", "2"))
DIVERGE_CONSEC = int(os.getenv("AHA_ORIENTATION_DIVERGE_CONSEC", "3"))  # ">N"
# C3 diverge run-up threshold = max_clean_runup*SCALE + floor2 (rad), with a
# run-up abstain: where the worst clean approach run-up exceeds floor2 the clean
# reorientation is too large/variable to separate from a failure, so C3 abstains
# (C1 backstops -> no recall lost). At active waypoints (clean run-up <= floor2)
# the threshold is always >= floor2 >= any clean run-up, so no clean run fires.
# floor2 = 0.3 rad (vs 0.1 m for transition): clean orientation run-ups are larger.
DIVERGE_RISE_SCALE = float(os.getenv("AHA_ORIENTATION_DIVERGE_RISE_SCALE", "1.5"))
DIVERGE_RISE_FLOOR2 = float(os.getenv("AHA_ORIENTATION_DIVERGE_RISE_FLOOR2", "0.3"))
DIVERGE_RISE_K = float(os.getenv("AHA_ORIENTATION_DIVERGE_RISE_K", "4"))  # back-compat
# Optional per-task env override for the envelope threshold (else from baseline).
_ENV_THR_ENV = os.getenv("AHA_ORIENTATION_ENVELOPE_THRESHOLD")

# AHA_ORIENTATION_MODE selects which condition(s) are live. Comma-list of any of
# {arrival, envelope, diverge}; "all" == every condition.
# Default = arrival ONLY: the detector fires at waypoint completion when the
# arrival angle exceeds the calibrated per-waypoint band. C4 prop (the early
# "got close, then diverged" check) has been REMOVED -- a "prop" entry in the
# mode string is ignored.
ORIENTATION_MODE = os.getenv(
    "AHA_ORIENTATION_MODE", "arrival").strip().lower()

# When set, every frame's signals are appended to this CSV so the eval harness
# can derive the thresholds offline from clean + failure trajectories.
DRIFT_LOG_PATH = os.getenv("AHA_ORIENTATION_DRIFT_LOG", "").strip()
DRIFT_LOG_STREAM = os.getenv("AHA_ORIENTATION_DRIFT_LOG_STREAM", "").strip().lower() in (
    "1", "true", "yes", "on")
DRIFT_LOG_STREAM_EVERY = max(1, int(os.getenv("AHA_ORIENTATION_DRIFT_LOG_STREAM_EVERY", "20")))

def _parse_modes(spec):
    """Set of enabled conditions from the AHA_ORIENTATION_MODE string.

    "prop" is dropped: the C4 proportional-divergence check no longer exists, so
    older callers passing AHA_ORIENTATION_MODE=arrival,prop still get arrival only."""
    spec = (spec or "").strip().lower()
    if spec in ("", "all"):
        return {"arrival", "envelope", "diverge"}
    if spec == "both":          # back-compat
        return {"arrival", "envelope"}
    return {p.strip() for p in spec.split(",") if p.strip() and p.strip() != "prop"}


def load_clean_stats(task_name):
    """Per-task clean-run scalars (gradient_threshold, diverge_error_threshold)
    from the arrival baseline JSON, or {} if absent."""
    import json
    try:
        path = ARRIVAL_STATS_DIR / f"{task_name}.json"
        if path.exists():
            return json.loads(path.read_text())
    except Exception as exc:
        print(f"  [orientation] could not load clean stats: {exc}")
    return {}


def load_arrival_baseline(task_name):
    """Per-waypoint clean arrival error stats in rad for a task, or {} if none.

    New baselines use the last in-motion frame before waypoint completion and
    provide threshold=max(floor, median + K*robust_std). Older baselines are still
    accepted by deriving the threshold from mean/std.
    """
    import json
    try:
        path = ARRIVAL_STATS_DIR / f"{task_name}.json"
        if path.exists():
            data = json.loads(path.read_text())
            stats = data.get("arrival_stats_by_waypoint")
            if stats:
                out = {}
                for k, v in stats.items():
                    mean = float(v.get("mean", 0.0))
                    std = float(v.get("std", 0.0))
                    mx = float(v.get("max", mean))
                    median = float(v.get("median", mean))
                    threshold = float(
                        v.get(
                            "threshold",
                            max(ARRIVAL_ANGLE_THRESHOLD,
                                median + ARRIVAL_SIGMA_K * std),
                        )
                    )
                    out[int(k)] = {
                        "mean": mean,
                        "median": median,
                        "std": std,
                        "max": mx,
                        "threshold": threshold,
                    }
                return out
            raw = data.get("clean_arrival_dev_by_waypoint", {})
            return {
                int(k): {
                    "mean": float(v),
                    "median": float(v),
                    "std": 0.0,
                    "max": float(v),
                    "threshold": max(ARRIVAL_ANGLE_THRESHOLD, float(v)),
                }
                for k, v in raw.items()
            }
    except Exception as exc:
        print(f"  [orientation] could not load arrival baseline: {exc}")
    return {}


def load_rise_baseline(task_name):
    """Per-waypoint clean runup stats in rad from segment_rise_by_waypoint."""
    import json
    try:
        path = ARRIVAL_STATS_DIR / f"{task_name}.json"
        if path.exists():
            data = json.loads(path.read_text())
            rise = data.get("segment_rise_by_waypoint", {})
            out = {}
            for k, v in rise.items():
                mean = float(v.get("mean_rise", 0.0))
                median = float(v.get("median_rise", mean))
                mx = float(v.get("max_rise", mean))
                std = (float(v["std_rise"]) if "std_rise" in v
                       else max(0.0, mx - mean) / 3.0)
                positive_std = float(v.get("positive_std_rise", std))
                threshold = float(
                    v.get(
                        "threshold",
                        max(ARRIVAL_ANGLE_THRESHOLD, 1.5 * mx),
                    )
                )
                out[int(k)] = {
                    "mean": mean,
                    "median": median,
                    "std": std,
                    "positive_std": positive_std,
                    "max": mx,
                    "threshold": threshold,
                }
            return out
    except Exception as exc:
        print(f"  [orientation] could not load rise baseline: {exc}")
    return {}


def _envelope_threshold(stats):
    """C2 envelope threshold (rad) from per-task PLOT-1 error_stats:
    max(error_max + Kmax*std, error_mean + Kmean*std). Inf if no stats."""
    es = (stats or {}).get("error_stats")
    if not es:
        return float("inf")
    mx, mean, std = (float(es.get("max", 0.0)), float(es.get("mean", 0.0)),
                     float(es.get("std", 0.0)))
    return max(mx + ENVELOPE_MAX_K * std, mean + ENVELOPE_MEAN_K * std)


def _position_gap(pose, reference_pose):
    try:
        live = np.asarray(pose, dtype=float).reshape(-1)
        reference = np.asarray(reference_pose, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return float("nan")
    if live.size < 3 or reference.size < 3:
        return float("nan")
    return float(np.linalg.norm(live[:3] - reference[:3]))


class LiveDetector:
    NEEDS_ORIGINAL_POSE = True

    def __init__(self, task_name, *, env_wrapper=None, vlm_enabled=True,
                 vlm_model=None, vlm_trace=False, show_plot=True,
                 n_waypoints=0, failure_waypoint=None):
        self.task_name = task_name
        self.env_wrapper = env_wrapper
        self.vlm_enabled = vlm_enabled
        self.vlm_model = vlm_model
        self.vlm_trace = vlm_trace
        self.failure_waypoint = failure_waypoint

        b = load_detector_bundle(NAME)
        self.det = b.detector
        self.plot_mod = b.plot
        self.vlm_mod = b.vlm
        self.WARMUP = int(self.det.WARMUP_STEPS)
        self.settings = self.det.default_detector_settings()
        # Description-driven thresholds: locked once, independent of warmup.
        self.frozen_thr = self.det.freeze_thresholds(
            [], self.settings.get("threshold_overrides"))
        self.arrival_thr = ARRIVAL_ANGLE_THRESHOLD
        self.arrival_baseline = load_arrival_baseline(task_name)
        self.rise_baseline = load_rise_baseline(task_name)
        # Which condition(s) are live this run.
        modes = _parse_modes(ORIENTATION_MODE)
        self.arrival_enabled = "arrival" in modes
        self.envelope_enabled = "envelope" in modes
        self.diverge_enabled = "diverge" in modes
        # C2 envelope threshold (per-task): env override > baseline > inf.
        stats = load_clean_stats(task_name)
        self.env_thr = (float(_ENV_THR_ENV) if _ENV_THR_ENV not in (None, "")
                        else _envelope_threshold(stats))
        self.env_consec_n = max(1, ENVELOPE_CONSEC)
        self.diverge_consec_n = max(1, DIVERGE_CONSEC)
        print(f"  [orientation] modes={sorted(modes)}")
        if self.arrival_enabled:
            print(
                "  [orientation] C1 arrival: fire at waypoint end when the angle "
                f"to the waypoint orientation > "
                f"max({self.arrival_thr:g}, median+{ARRIVAL_SIGMA_K:g}*robust_std) per waypoint"
            )
        if self.envelope_enabled:
            print(
                "  [orientation] C2 envelope: fire mid-motion when the angle to "
                f"target > {self.env_thr:.3f} rad "
                f"(max(max+{ENVELOPE_MAX_K:g}std, mean+{ENVELOPE_MEAN_K:g}std)) "
                f"for {self.env_consec_n} consecutive frames"
            )
        if self.diverge_enabled:
            print(
                "  [orientation] C3 diverge: fire when the angle to target rises "
                f"for >{self.diverge_consec_n} consecutive frames AND the run-up > "
                f"max_clean_runup*{DIVERGE_RISE_SCALE:g}+{DIVERGE_RISE_FLOOR2:g} "
                f"(abstains where clean run-up > {DIVERGE_RISE_FLOOR2:g})"
            )
        if self.arrival_baseline and self.arrival_enabled:
            print(
                f"  [orientation] arrival baseline ({len(self.arrival_baseline)} wp): "
                + ", ".join(f"wp{w}>={self._waypoint_threshold(w):.3f}"
                            for w in sorted(self.arrival_baseline))
            )
        print()

        self.logs = []
        self.step_counter = 0
        self.paused_steps = []
        self._cur_wp = None
        self._local_step = 0
        self._consec = 0
        self._env_consec = 0
        self._diverge_consec = 0
        self._prev_err = float("nan")
        self._cumulative_runup = 0.0
        self.waypoint_obs = []  # observations during the active waypoint (for VLM)

        self.fed = 0
        self.checked = 0
        self.fire_count = 0
        self.first_fire = None
        self.fire_mode = None  # "arrival" | "drift" — which option fired first
        self.fired_waypoints = set()
        self._fired_this_execution = set()
        self.disabled_reason = None
        self.diag = {}
        self._wp_arrival_dev = {}  # waypoint -> worst arrival deviation (rad)
        self._wp_drift_peak = {}   # waypoint -> worst accumulated drift (rad)
        self._wp_grad_peak = {}    # waypoint -> worst per-frame step (rad/frame)
        self._vlm_rejected = set()  # waypoints whose detection the VLM refuted
        self.last_waypoint = None

        # Per-frame drift-signal CSV (for offline (theta,N) sweeping).
        self._drift_log_rows = []
        self._drift_log_path = DRIFT_LOG_PATH or None

        self.live_plot = None
        if show_plot and self.plot_mod is not None:
            try:
                self.live_plot = self.plot_mod.LiveTelemetryPlot(
                    task_name, FAILURE, list(range(n_waypoints or 0)))
            except Exception as exc:
                print(f"  [detector:orientation] live plot disabled ({exc})")

    def _waypoint_threshold(self, waypoint):
        """Fire threshold (rad) for this waypoint: max(flat floor, median+K*robust_std)
        of the clean last in-motion error before waypoint completion."""
        base = self.arrival_baseline.get(waypoint)
        if base is None:
            return self.arrival_thr
        if isinstance(base, dict):
            return max(self.arrival_thr, float(base.get("threshold", self.arrival_thr)))
        _mean, _std, _mx = base
        return max(self.arrival_thr, _mean + ARRIVAL_SIGMA_K * _std)

    def _abstain(self, waypoint):
        """True if this waypoint's worst clean arrival angle shows the gripper
        orientation is not set by this waypoint (unconstrained roll / dynamic):
        no clean-derived threshold can separate clean from a failure, so every
        condition abstains here. Data-driven (keys off the waypoint's clean max)."""
        base = self.arrival_baseline.get(waypoint)
        if base is None:
            return False
        mx = float(base.get("max", 0.0)) if isinstance(base, dict) else base[2]
        return mx > ARRIVAL_ABSTAIN_CEIL

    def _rise_threshold(self, waypoint):
        """C3 threshold (rad): max(floor, 1.5 * max cumulative clean runup)."""
        base = self.rise_baseline.get(waypoint)
        if base is None:
            return float("inf")
        if isinstance(base, dict):
            return max(ARRIVAL_ANGLE_THRESHOLD, float(base.get("threshold", 0.0)))
        _mean, _std, _mx = base
        return max(ARRIVAL_ANGLE_THRESHOLD, 1.5 * _mx)

    def _reset_waypoint(self, waypoint):
        self._fired_this_execution.clear()
        self._vlm_rejected.discard(waypoint)
        self._cur_wp = waypoint
        self.logs = []
        self._local_step = 0
        self._consec = 0
        self._env_consec = 0
        self._diverge_consec = 0
        self._prev_err = float("nan")
        self._cumulative_runup = 0.0
        self.waypoint_obs = []

    def step(self, obs, waypoint=None, path_done=False, waypoint_pose=None,
             original_pose=None, report=True, phase="move"):
        if self.disabled_reason is not None:
            return False
        self.fed += 1
        if report:
            self.checked += 1
        self.last_waypoint = waypoint
        det = self.det
        try:
            if waypoint != self._cur_wp:
                self._reset_waypoint(waypoint)
            self.waypoint_obs.append(obs)
            local_step = self._local_step
            self._local_step += 1
            self.step_counter += 1

            reference_pose = (
                original_pose if original_pose is not None else waypoint_pose)
            row = det.obs_to_row(
                obs, local_step, waypoint=waypoint, expected_waypoint=waypoint,
                waypoint_path_done=path_done, waypoint_pose=waypoint_pose,
                ttm_waypoint_pose=reference_pose, waypoint_started=True,
                started_waypoint=waypoint)
            self.logs.append(row)
            if len(self.logs) > 300:
                del self.logs[0]
            det.update_deltas(self.logs)

            # Failgen rotation/no_rotation failures mutate the waypoint target
            # before motion starts. Detect that orientation-only target change
            # directly; relying only on the gripper's settled pose can mislabel
            # it as translation when IK preserves orientation but shifts XYZ.
            target_gap = row.get("waypoint_orientation_gap", float("nan"))
            try:
                target_gap = float(target_gap)
            except (TypeError, ValueError):
                target_gap = float("nan")
            target_gap_thr = max(
                TARGET_ORIENTATION_GAP_THRESHOLD,
                self._waypoint_threshold(waypoint),
            )
            target_position_gap = _position_gap(waypoint_pose, original_pose)
            position_target_changed = (
                np.isfinite(target_position_gap)
                and target_position_gap >=
                POSITION_TARGET_CHANGE_SUPPRESS_THRESHOLD)
            orientation_target_changed = (
                np.isfinite(target_gap) and target_gap >= target_gap_thr)
            if (TARGET_CHANGE_ENABLED
                    and report and local_step <= TARGET_CHANGE_START_WINDOW
                    and np.isfinite(target_gap)
                    and target_gap >= target_gap_thr
                    and waypoint not in self._fired_this_execution
                    and waypoint not in self._vlm_rejected):
                reason = (
                    f"waypoint target orientation changed {target_gap:.3f} rad "
                    f"from its original orientation (>= {target_gap_thr:.3f})")
                return self._do_fire(
                    row, row["step"], waypoint, reason, "target")

            if phase == "gripper":
                self._update_plot()
                return False

            # NOTE: previously, once the commanded target's orientation had
            # diverged from the original (i.e. a rotation/no_rotation mutation was
            # present) past the start window, the tracking checks were suppressed
            # and detection was left to the mode=target shortcut. Now arrival
            # judges the gripper against the ORIGINAL orientation, so a mutated
            # target is exactly what it must catch -- suppressing here would drop
            # every rotation failure. A translation-only change leaves the
            # orientation error ~0, so it still won't false-fire. Hence: never
            # suppress the tracking checks on a target-orientation change.
            late_orientation_target_change = (
                local_step > TARGET_CHANGE_START_WINDOW
                and orientation_target_changed)
            suppress_tracking_for_position_change = False

            # Per-frame orientation step (rate of change) for the gradient check.
            god = row.get("gripper_orientation_delta", float("nan"))
            try:
                god = float(god)
            except (TypeError, ValueError):
                god = float("nan")
            if np.isfinite(god):
                self._wp_grad_peak[waypoint] = max(
                    self._wp_grad_peak.get(waypoint, 0.0), god)

            # Orientation error = angle between the gripper and the CORRECT
            # (pre-injection / original) waypoint orientation -- ttm_waypoint_angle
            # (gripper vs ttm_waypoint_pose == original_pose). EVERY condition
            # (C1 arrival, and C2/C3 when enabled) uses this signal so a
            # target-orientation mutation that the arm faithfully tracks
            # (rotation / no_rotation failures) is exposed, instead of reading ~0
            # against the mutated command. A translation-only mutation leaves the
            # orientation unchanged, so this stays ~0 there and is not mislabeled.
            # The clean per-waypoint baselines are calibrated on THIS same signal
            # (calibrate_all_clean.py records ttm_waypoint_angle), so dynamic
            # ride-along waypoints -- where clean gripper-vs-original is large --
            # get a high threshold / abstain rather than false-firing. Falls back
            # to the live/commanded angle only when no original reference exists.
            err = row.get("ttm_waypoint_angle", float("nan"))
            err_d = row.get("ttm_waypoint_angle_delta", float("nan"))
            try:
                err = float(err)
                err_d = float(err_d)
            except (TypeError, ValueError):
                err = err_d = float("nan")
            if not np.isfinite(err):
                err = row.get("live_waypoint_angle", float("nan"))
                try:
                    err = float(err)
                except (TypeError, ValueError):
                    err = float("nan")
            if not np.isfinite(err):
                err = row.get("expected_live_waypoint_angle", float("nan"))
                try:
                    err = float(err)
                except (TypeError, ValueError):
                    err = float("nan")
            if not np.isfinite(err_d):
                err_d = row.get("live_waypoint_angle_delta", float("nan"))
                try:
                    err_d = float(err_d)
                except (TypeError, ValueError):
                    err_d = float("nan")
            # Live run-up = cumulative positive increase of the error during
            # this waypoint segment. It only adds the amount PLOT-1 increased;
            # decreases do not subtract from the cumulative value.
            runup_positive = False
            if np.isfinite(err):
                if np.isfinite(self._prev_err):
                    increase = max(0.0, err - self._prev_err)
                    if increase > 0.0:
                        self._cumulative_runup += increase
                        runup_positive = True
                self._prev_err = err
            runup = self._cumulative_runup if np.isfinite(err) else float("nan")
            if self._drift_log_path is not None:
                self._drift_log_rows.append(
                    (self.task_name, waypoint, local_step, row["step"],
                     err, runup, int(runup_positive),
                     god, int(bool(path_done))))
                if DRIFT_LOG_STREAM and len(self._drift_log_rows) >= DRIFT_LOG_STREAM_EVERY:
                    self._flush_drift_log()
            if np.isfinite(runup):
                self._wp_drift_peak[waypoint] = max(
                    self._wp_drift_peak.get(waypoint, 0.0), runup)

            if suppress_tracking_for_position_change:
                self._update_plot()
                return False

            # --- C2 ENVELOPE check (during the motion) -----------------------
            # Fire when the live error leaves the clean envelope
            # max(max+Kmax*std, mean+Kmean*std) for N consecutive frames.
            if (self.envelope_enabled and report
                    and waypoint not in self._fired_this_execution
                    and waypoint not in self._vlm_rejected):
                if np.isfinite(err) and err >= self.env_thr:
                    self._env_consec += 1
                else:
                    self._env_consec = 0
                if self._env_consec >= self.env_consec_n:
                    reason = (
                        f"orientation error {err:.3f} rad left the clean envelope "
                        f"(>= {self.env_thr:.3f}, {self._env_consec} consec)")
                    return self._do_fire(row, row["step"], waypoint, reason,
                                         "envelope")

            # --- C3 DIVERGE check (during the motion) ------------------------
            # Fire when the error keeps RISING (delta > 0) for MORE than N
            # consecutive frames AND the accumulated rise (PLOT-2) exceeds the
            # per-waypoint clean rise band (mean + K*std).
            if (self.diverge_enabled and report
                    and waypoint not in self._fired_this_execution
                    and waypoint not in self._vlm_rejected):
                if np.isfinite(err_d) and err_d > 0.0:
                    self._diverge_consec += 1
                else:
                    self._diverge_consec = 0
                rise_thr = self._rise_threshold(waypoint)
                if (self._diverge_consec > self.diverge_consec_n
                        and np.isfinite(runup) and runup >= rise_thr):
                    reason = (
                        f"orientation diverging: rose {runup:.3f} rad over "
                        f"{self._diverge_consec} consec frames (>= {rise_thr:.3f} "
                        f"= max_clean_runup*{DIVERGE_RISE_SCALE:g}+{DIVERGE_RISE_FLOOR2:g})")
                    return self._do_fire(row, row["step"], waypoint, reason,
                                         "diverge")

            # --- C1 ARRIVAL check -------------------------------------------
            # Only frames where the waypoint path has finished (and the arm
            # therefore sits at its commanded target) are judged.
            if not path_done:
                self._update_plot()
                return False

            # C1 arrival uses the gripper-vs-correct error (err is
            # ttm_waypoint_angle = gripper vs the original/pre-injection
            # orientation), so a mutated-target rotation the arm tracked cleanly
            # still shows a large arrival deviation and fires.
            dev = err
            if not np.isfinite(dev):
                self._update_plot()
                return False

            self.diag["max_arrival_dev"] = max(
                self.diag.get("max_arrival_dev", 0.0), dev)
            self._wp_arrival_dev[waypoint] = max(
                self._wp_arrival_dev.get(waypoint, 0.0), dev)

            if not self.arrival_enabled:
                self._update_plot()
                return False

            forced = self._maybe_force_fire(row, row["step"], waypoint, path_done, report)
            if forced is not None:
                return forced

            wp_thr = self._waypoint_threshold(waypoint)
            raw = dev >= wp_thr
            reason = (
                f"arrived {dev:.3f} rad off the waypoint's correct "
                f"orientation (>= {wp_thr:.3f})")
            if (raw and report and waypoint not in self._fired_this_execution
                    and waypoint not in self._vlm_rejected):
                return self._do_fire(row, row["step"], waypoint, reason,
                                     "arrival")
            self._update_plot()
            return False
        except Exception as exc:
            self.disabled_reason = f"runtime error: {exc}"
            print(f"  [detector:orientation] disabled mid-run — {exc}")
            return False

    def _do_fire(self, row, step, waypoint, reason, mode):
        """Common fire path for arrival, drift, and force-fire: bookkeeping,
        VLM confirmation, and VLM-retraction. Returns True if the detection
        stands, False if the VLM verifier refuted it."""
        self.fire_count += 1
        previously_fired = waypoint in self.fired_waypoints
        self.fired_waypoints.add(waypoint)
        self._fired_this_execution.add(waypoint)
        self._flush_drift_log()
        if self.first_fire is None:
            self.first_fire = (step, reason, waypoint)
            self.fire_mode = mode
        verdict = self._on_detection(step, reason, row, waypoint, mode)
        if VLM_AUTO and verdict is False:
            print("  [detector:orientation] detection RETRACTED — "
                  "VLM verifier says no orientation failure.")
            if not previously_fired:
                self.fired_waypoints.discard(waypoint)
            self._fired_this_execution.discard(waypoint)
            self._vlm_rejected.add(waypoint)
            self.fire_count -= 1
            self._update_plot()
            return False
        self._update_plot()
        return True

    def _maybe_force_fire(self, row, step, waypoint, path_done, report):
        """Eval hook: when AHA_FORCE_FIRE_WAYPOINT matches, fire on a clean scene
        so the VLM verifier's retraction can be measured. Returns True/False if it
        handled the step (mirrors the natural fire/retract path), else None."""
        if (FORCE_FIRE_WAYPOINT is None or waypoint != FORCE_FIRE_WAYPOINT
                or not path_done or not report
                or waypoint in self._fired_this_execution
                or waypoint in self._vlm_rejected):
            return None
        return self._do_fire(row, step, waypoint, FORCE_FIRE_REASON, "arrival")

    def _on_detection(self, step, reason, row, waypoint, mode="?"):
        print(
            f"\n  [DETECTOR] ORIENTATION failure at waypoint {waypoint} "
            f"(step {step}) mode={mode}  reason={reason}\n",
            flush=True,
        )
        self.paused_steps.append(step)
        if self.live_plot is not None:
            try:
                self.live_plot.update(
                    self.logs, self.frozen_thr, self.paused_steps, force=True)
            except Exception:
                pass
        verdict = None
        if self.vlm_enabled and self.vlm_mod is not None:
            verdict = self._prompt_vlm(row, waypoint)
        if not VLM_AUTO:
            try:
                input("  [orientation] press Enter to continue the run...")
            except EOFError:
                pass
        return verdict

    def _prompt_vlm(self, row, waypoint):
        vlm = self.vlm_mod
        try:
            sampled_obs, step_offsets = vlm.sample_waypoint_observations(
                list(self.waypoint_obs))
            if not VLM_AUTO:
                vlm.show_camera_sequence(
                    sampled_obs, step_offsets=step_offsets,
                    env_wrapper=self.env_wrapper,
                    title="Images that will be sent to orientation VLM")
        except Exception as exc:
            print(f"  [vlm] image preview failed: {exc}")
            sampled_obs, step_offsets = list(self.waypoint_obs), None

        if VLM_AUTO:
            print("  [vlm] auto-confirming orientation detection with the "
                  "VLM verifier...")
        else:
            raw = input("  Confirm this orientation failure with OpenAI VLM? [y/N]: ").strip().lower()
            if raw not in ("y", "yes"):
                return None
        try:
            result = vlm.confirm_orientation_with_openai(
                sampled_obs, step_offsets=step_offsets, row=row,
                telemetry_history=list(self.logs), env_wrapper=self.env_wrapper,
                task_name=self.task_name, waypoint_index=waypoint,
                model=self.vlm_model, preview_images=False, trace=self.vlm_trace)
        except Exception as exc:
            print(f"  [vlm] confirmation failed: {exc}")
            return None
        verdict = result.get("orientation_failure_happened")
        vtext = "YES" if verdict is True else "NO" if verdict is False else "UNKNOWN"
        print(f"  [vlm:{result.get('model')}] orientation={vtext}  "
              f"{result.get('explanation', '')}  "
              f"cameras={', '.join(result.get('camera_names', []))}")
        return verdict

    def _update_plot(self):
        if self.live_plot is None:
            return
        try:
            self.live_plot.update(self.logs, self.frozen_thr, self.paused_steps)
        except Exception as exc:
            print(f"  [detector:orientation] live plot update failed ({exc})")
            self.live_plot = None

    def diagnostics_line(self):
        per_wp = "; ".join(
            f"wp{wp}={val:.3f}" for wp, val in sorted(self._wp_arrival_dev.items())
        )
        drift_wp = "; ".join(
            f"wp{wp}={val:.3f}" for wp, val in sorted(self._wp_drift_peak.items())
        )
        grad_wp = "; ".join(
            f"wp{wp}={val:.3f}" for wp, val in sorted(self._wp_grad_peak.items())
        )
        fired_by = f"  fired_by={self.fire_mode}" if self.fire_mode else ""
        return (f"      diag: fed {self.fed} obs; max arrival deviation="
                f"{self.diag.get('max_arrival_dev', 0):.3f} rad "
                f"(fires >= {self.arrival_thr:.3f}){fired_by}"
                f"{('  per-waypoint: ' + per_wp) if per_wp else ''}"
                f"{('  drift-peak per-waypoint: ' + drift_wp) if drift_wp else ''}"
                f"{('  gradient-peak per-waypoint: ' + grad_wp) if grad_wp else ''}")

    def _flush_drift_log(self):
        if not self._drift_log_path or not self._drift_log_rows:
            return
        import csv
        path = self._drift_log_path
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            exists = os.path.exists(path)
            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                if not exists:
                    w.writerow(["task", "waypoint", "local_step", "step",
                                "angle_to_target", "runup",
                                "runup_positive", "gripper_orientation_delta",
                                "path_done"])
                w.writerows(self._drift_log_rows)
            self._drift_log_rows = []
        except Exception as exc:
            print(f"  [orientation] drift-log flush failed: {exc}")

    def close(self):
        self._flush_drift_log()
        if self.live_plot is not None:
            try:
                self.live_plot.plt.close(self.live_plot.fig)
            except Exception:
                pass
