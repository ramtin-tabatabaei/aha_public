"""bt_gui-embedded TRANSITION detector.

A copy of aha_scripts/detectors/transition/interactive.py's real-time detection
(``patched_step`` + ``prompt_vlm_confirmation`` + live telemetry plot), adapted
to be fed by the bt_gui BT run. Transition's distance check requires a non-None
waypoint spec, so an empty dict enables it (the gripper-state sub-check stays
off by default).

Target pose mutations are detected directly by comparing the current waypoint
position with the pose captured at episode start. Tracking/arrival checks compare
the gripper with the current commanded waypoint position, so orientation-only
target changes do not get mislabeled as transition failures.

Mirrors the orientation detector's multi-condition design exactly (with rad ->
m): AHA_TRANSITION_MODE selects any of {arrival, envelope, diverge}; all
thresholds are per-task, data-derived, loaded from the per-task baseline JSON
(env override > baseline file > default).
"""

from aha_publish import paths

import os
from pathlib import Path

import numpy as np

from ._bundle import (
    VLM_AUTO, FORCE_FIRE_WAYPOINT, FORCE_FIRE_REASON, load_detector_bundle)

NAME = "transition"
FAILURE = "transition"
CONDITION = "reaches_waypoint() == True"
# No live waypoint pose needed: arrival distance is measured against the composed reference
# (ttm_waypoint_distance), never the live target.
NEEDS_WAYPOINT_POSE = False

# Flat floor for the arrival threshold (m): the smallest arrival deviation worth
# firing on. It must sit ABOVE clean servo noise (~0.002 m) and BELOW the real
# failure arrival deviation. This is deliberately DECOUPLED from the abstain
# ceiling: the floor defines the smallest arrival error we care to flag, while
# the high ceiling drops dynamic waypoints.
ARRIVAL_DISTANCE_THRESHOLD = float(
    os.getenv("AHA_TRANSITION_ARRIVAL_THRESHOLD", "0.02"))
# By default every waypoint uses the flat ARRIVAL_DISTANCE_THRESHOLD (0.01 m):
# the per-task/per-waypoint calibrated baseline is IGNORED. Set
# AHA_TRANSITION_ARRIVAL_USE_BASELINE=1 to restore the calibrated per-waypoint
# thresholds (and the dynamic-waypoint abstain that keys off the same baseline).
ARRIVAL_USE_BASELINE = os.getenv(
    "AHA_TRANSITION_ARRIVAL_USE_BASELINE", "1").strip().lower() in (
        "1", "true", "yes", "on")
TARGET_POSITION_GAP_THRESHOLD = float(
    os.getenv("AHA_TRANSITION_TARGET_GAP_THRESHOLD", "0.05"))
TARGET_CHANGE_ENABLED = os.getenv(
    "AHA_TRANSITION_TARGET_CHANGE_ENABLED", "0"
).strip().lower() in ("1", "true", "yes", "on")
TARGET_CHANGE_START_WINDOW = int(
    os.getenv("AHA_TRANSITION_TARGET_CHANGE_START_WINDOW", "3"))
# Optional target-change shortcut warmup. Normal transition detection is
# arrival-gated; this only applies when AHA_TRANSITION_TARGET_CHANGE_ENABLED=1.
TARGET_CHANGE_MIN_STEP = int(
    os.getenv("AHA_TRANSITION_TARGET_CHANGE_MIN_STEP", "0"))
ORIENTATION_TARGET_CHANGE_SUPPRESS_THRESHOLD = float(
    os.getenv(
        "AHA_TRANSITION_ORIENTATION_GAP_SUPPRESS_THRESHOLD",
        os.getenv("AHA_ORIENTATION_TARGET_GAP_THRESHOLD", "0.30"),
    ))
_REPO_ROOT = (paths.PROJECT_ROOT)
_CALIBRATION_ROOT = Path(os.getenv(
    "AHA_CALIBRATION_ROOT",
    str(paths.CALIBRATION_DIR),
)).expanduser()
ARRIVAL_STATS_DIR = Path(os.getenv(
    "AHA_TRANSITION_STATS_DIR",
    str(_CALIBRATION_ROOT / "transition_arrival_stats"),
)).expanduser()
# Opt-in: when set, a clean run writes its per-waypoint arrival distances to
# ARRIVAL_STATS_DIR/<task>.json on close (legacy path; superseded by the eval
# harness, which writes the richer statistical baseline).
WRITE_BASELINE = os.getenv("AHA_TRANSITION_WRITE_BASELINE", "").strip() not in (
    "", "0", "false", "False")

# Three conditions (all driven by the per-task clean-run baselines):
#
# C1 ARRIVAL  - at waypoint completion, the distance to the waypoint-defined
#               position > max(floor, mean + K1*std) of the clean arrival
#               distance for that waypoint.  K1 default 2.
# C2 ENVELOPE - during motion, the live distance to the target leaves the clean
#               envelope: err > max(error_max + Kmax*std, error_mean + Kmean*std)
#               from the per-task PLOT-1 error_stats.  Kmax=1, Kmean=3.
# C3 DIVERGE  - during motion, the error keeps rising (delta > 0) for MORE than
#               N consecutive frames AND the accumulated run-up (live: err above
#               the running minimum) exceeds mean + K3*std of the clean
#               per-waypoint segment rises.  N=3, K3=3.
# C1 arrival threshold is a ROBUST per-waypoint band keyed off the WORST clean
# arrival distance ever seen at that waypoint (not mean+K*std): thr =
# max(floor, max_clean*SCALE + MARGIN). Mean+K*std nips held-out clean runs by a
# hair when std is tiny, and a single anomalous clean episode skews the mean;
# keying off the clean MAX with a multiplicative+additive margin is robust to
# both, while staying far below the injected-failure offset (~0.5 m).
ARRIVAL_SCALE = float(os.getenv("AHA_TRANSITION_ARRIVAL_SCALE", "1.3"))
ARRIVAL_MARGIN = float(os.getenv("AHA_TRANSITION_ARRIVAL_MARGIN", "0.2"))
# ABSTAIN: a waypoint whose WORST clean arrival distance exceeds this is not
# tracking its episode-start (original) pose -- the target is dynamic (an object
# that moves after capture) or otherwise unconstrained, so distance-to-original
# is meaningless there and no clean-derived threshold can separate clean from a
# failure. The detector ABSTAINS (threshold -> inf, never fires) at such
# waypoints; the eval injects elsewhere. DECOUPLED from the arrival floor: the
# floor is low (catch small real failures), this ceiling is high (only drop
# waypoints whose clean arrival is so large the original-pose reference is plainly
# invalid -- dynamic target / unconstrained). A consistently-large-but-stable
# waypoint (e.g. close_box wp2 ~1.15 m every run) is handled by the per-waypoint
# max-based threshold without abstaining; the ceiling is for waypoints that can't
# be separated from a failure at all.
ARRIVAL_ABSTAIN_CEIL = float(os.getenv("AHA_TRANSITION_ARRIVAL_ABSTAIN_CEIL", "0.30"))
# Back-compat: kept so older callers/plots that read the sigma knob still work.
ARRIVAL_SIGMA_K = float(os.getenv("AHA_TRANSITION_ARRIVAL_SIGMA_K", "3"))
ENVELOPE_MAX_K = float(os.getenv("AHA_TRANSITION_ENVELOPE_MAX_K", "1"))
ENVELOPE_MEAN_K = float(os.getenv("AHA_TRANSITION_ENVELOPE_MEAN_K", "3"))
ENVELOPE_CONSEC = int(os.getenv("AHA_TRANSITION_ENVELOPE_CONSEC", "2"))
DIVERGE_CONSEC = int(os.getenv("AHA_TRANSITION_DIVERGE_CONSEC", "3"))  # ">N"
# C3 diverge run-up threshold = max_clean_runup*SCALE + floor2. Scales with the
# waypoint's own worst clean approach run-up (curvier clean approaches -> higher
# threshold, so they don't false-fire), with floor2 the additive separation
# margin above clean. C1 arrival is the ground-truth backstop, so abstained /
# high-run-up waypoints simply don't get faster-than-arrival detection -- no
# recall is lost.
DIVERGE_RISE_SCALE = float(os.getenv("AHA_TRANSITION_DIVERGE_RISE_SCALE", "1.5"))
DIVERGE_RISE_FLOOR2 = float(os.getenv("AHA_TRANSITION_DIVERGE_RISE_FLOOR2", "0.1"))
DIVERGE_RISE_K = float(os.getenv("AHA_TRANSITION_DIVERGE_RISE_K", "3"))  # back-compat
# --- C2 SETTLE (baseline-free EARLY detection) ---------------------------------
# Fire BEFORE waypoint completion when the end-effector has SETTLED -- its
# per-frame motion collapsed for a few frames after the arm had been moving --
# yet the distance to the true (original/TTM) waypoint position is still >= the
# flat arrival threshold. The arm has stopped at the WRONG place. Gating on
# motion STILLNESS rather than on distance direction is what makes this robust
# to non-linear approaches: a clean detour swings the arm away at HIGH velocity,
# so it never looks settled until it genuinely arrives (where the distance is
# ~0). No per-task baseline is used; C1 arrival at path_done is the backstop.
#   SETTLE_MOTION  - EE per-frame motion (m) below which the arm counts as still.
#   SETTLE_CONSEC  - consecutive still frames required to declare "settled".
#   SETTLE_MOVE_ARM- the arm must first exceed this motion once, so the low
#                    motion at segment start (before the arm accelerates) does
#                    not look like an early settle.
SETTLE_MOTION = float(os.getenv("AHA_TRANSITION_SETTLE_MOTION", "0.005"))
SETTLE_CONSEC = int(os.getenv("AHA_TRANSITION_SETTLE_CONSEC", "4"))
SETTLE_MOVE_ARM = float(os.getenv("AHA_TRANSITION_SETTLE_MOVE_ARM", "0.01"))

# Optional per-task env override for the envelope threshold (else from baseline).
_ENV_THR_ENV = os.getenv("AHA_TRANSITION_ENVELOPE_THRESHOLD")

# AHA_TRANSITION_MODE selects which condition(s) are live. Comma-list of any of
# {arrival, envelope, diverge, settle}; "all" == every condition.
# Default = arrival ONLY: at waypoint completion the last in-motion distance to
# the original/TTM waypoint pose is compared with the calibrated per-waypoint
# threshold. C4 PROP (the early "reached, then diverged" check) has been
# REMOVED -- a "prop" entry in the mode string is ignored.
# C2 SETTLE (motion-stillness-while-far), C2 ENVELOPE and C3 DIVERGE are opt-in.
# The direct target-position change shortcut is opt-in via
# AHA_TRANSITION_TARGET_CHANGE_ENABLED=1. Re-enable extras with e.g.
# AHA_TRANSITION_MODE=arrival,settle,envelope,diverge.
TRANSITION_MODE = os.getenv(
    "AHA_TRANSITION_MODE", "arrival").strip().lower()

# When set, every frame's signals are appended to this CSV so the eval harness
# can derive the per-task thresholds offline from clean trajectories.
DRIFT_LOG_PATH = os.getenv("AHA_TRANSITION_DRIFT_LOG", "").strip()
DRIFT_LOG_STREAM = os.getenv("AHA_TRANSITION_DRIFT_LOG_STREAM", "").strip().lower() in (
    "1", "true", "yes", "on")
DRIFT_LOG_STREAM_EVERY = max(1, int(os.getenv("AHA_TRANSITION_DRIFT_LOG_STREAM_EVERY", "20")))


def _parse_modes(spec):
    """Set of enabled conditions from the AHA_TRANSITION_MODE string.

    "prop" is dropped: the C4 proportional-divergence check no longer exists, so
    older callers passing AHA_TRANSITION_MODE=arrival,prop still get arrival only."""
    spec = (spec or "").strip().lower()
    if spec in ("", "all"):
        return {"arrival", "envelope", "diverge", "settle"}
    if spec == "both":          # back-compat
        return {"arrival", "envelope"}
    return {p.strip() for p in spec.split(",") if p.strip() and p.strip() != "prop"}


def load_clean_stats(task_name):
    """Per-task clean-run scalars (error_stats, etc.) from the arrival baseline
    JSON, or {} if absent."""
    import json
    try:
        path = ARRIVAL_STATS_DIR / f"{task_name}.json"
        if path.exists():
            return json.loads(path.read_text())
    except Exception as exc:
        print(f"  [transition] could not load clean stats: {exc}")
    return {}


def load_arrival_baseline(task_name):
    """Per-waypoint clean arrival distance stats in m for a task, or {} if none.

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
                            max(ARRIVAL_DISTANCE_THRESHOLD,
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
            raw = data.get("clean_arrival_dist_by_waypoint", {})
            return {
                int(k): {
                    "mean": float(v),
                    "median": float(v),
                    "std": 0.0,
                    "max": float(v),
                    "threshold": max(ARRIVAL_DISTANCE_THRESHOLD, float(v)),
                }
                for k, v in raw.items()
            }
    except Exception as exc:
        print(f"  [transition] could not load arrival baseline: {exc}")
    return {}


def load_rise_baseline(task_name):
    """Per-waypoint clean cumulative runup stats in m from segment_rise_by_waypoint."""
    import json
    try:
        path = ARRIVAL_STATS_DIR / f"{task_name}.json"
        if path.exists():
            data = json.loads(path.read_text())
            rise = data.get("segment_rise_by_waypoint", {})
            out = {}
            for k, v in rise.items():
                mean = float(v.get("mean_rise", 0.0))
                mx = float(v.get("max_rise", mean))
                std = (float(v["std_rise"]) if "std_rise" in v
                       else max(0.0, mx - mean) / 3.0)
                threshold = float(
                    v.get(
                        "threshold",
                        max(ARRIVAL_DISTANCE_THRESHOLD, 1.5 * mx),
                    )
                )
                out[int(k)] = {
                    "mean": mean,
                    "std": std,
                    "max": mx,
                    "threshold": threshold,
                }
            return out
    except Exception as exc:
        print(f"  [transition] could not load rise baseline: {exc}")
    return {}


def _envelope_threshold(stats):
    """C2 envelope threshold (m) from per-task PLOT-1 error_stats:
    max(error_max + Kmax*std, error_mean + Kmean*std). Inf if no stats."""
    es = (stats or {}).get("error_stats")
    if not es:
        return float("inf")
    mx, mean, std = (float(es.get("max", 0.0)), float(es.get("mean", 0.0)),
                     float(es.get("std", 0.0)))
    return max(mx + ENVELOPE_MAX_K * std, mean + ENVELOPE_MEAN_K * std)


def _pose_quat(pose):
    try:
        arr = np.asarray(pose, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < 7:
        return None
    q = arr[3:7].astype(float)
    n = np.linalg.norm(q)
    if n <= 1e-12:
        return None
    return q / n


def _quat_angle(q1, q2):
    if q1 is None or q2 is None:
        return float("nan")
    dot = abs(float(np.dot(q1, q2)))
    dot = min(1.0, max(-1.0, dot))
    return float(2.0 * np.arccos(dot))


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

        b = load_detector_bundle(NAME)
        self.det = b.detector
        self.plot_mod = b.plot
        self.vlm_mod = b.vlm
        self.WARMUP = int(self.det.WARMUP_STEPS)
        self.settings = self.det.default_detector_settings()
        self.frozen_thr = self.det.freeze_thresholds(
            [], self.settings.get("threshold_overrides"))
        self.frozen_thr["waypoint_distance"] = ARRIVAL_DISTANCE_THRESHOLD
        self.arrival_thr = ARRIVAL_DISTANCE_THRESHOLD
        self.failure_waypoint = failure_waypoint
        self.arrival_baseline = load_arrival_baseline(task_name)
        self.rise_baseline = load_rise_baseline(task_name)
        # Which condition(s) are live this run.
        modes = _parse_modes(TRANSITION_MODE)
        self.arrival_enabled = "arrival" in modes
        self.envelope_enabled = "envelope" in modes
        self.diverge_enabled = "diverge" in modes
        self.settle_enabled = "settle" in modes
        # C2 envelope threshold (per-task): env override > baseline > inf.
        stats = load_clean_stats(task_name)
        self.env_thr = (float(_ENV_THR_ENV) if _ENV_THR_ENV not in (None, "")
                        else _envelope_threshold(stats))
        self.env_consec_n = max(1, ENVELOPE_CONSEC)
        self.diverge_consec_n = max(1, DIVERGE_CONSEC)
        print(f"  [transition] modes={sorted(modes)}")
        if self.arrival_enabled:
            print(
                "  [transition] C1 arrival: fire at waypoint end when the "
                f"distance to the waypoint position > "
                f"max({self.arrival_thr:g}, median+{ARRIVAL_SIGMA_K:g}*robust_std) per waypoint"
            )
        if self.envelope_enabled:
            print(
                "  [transition] C2 envelope: fire mid-motion when the distance "
                f"to target > {self.env_thr:.3f} m "
                f"(max(max+{ENVELOPE_MAX_K:g}std, mean+{ENVELOPE_MEAN_K:g}std)) "
                f"for {self.env_consec_n} consecutive frames"
            )
        if self.diverge_enabled:
            print(
                "  [transition] C3 diverge: fire when the distance to target "
                f"rises for >{self.diverge_consec_n} consecutive frames AND the "
                f"run-up > max({self.arrival_thr:g}, {DIVERGE_RISE_SCALE:g}*max_clean_runup) "
                "per waypoint"
            )
        if self.settle_enabled:
            print(
                "  [transition] C2 settle: fire before waypoint end when the "
                f"end-effector motion stays < {SETTLE_MOTION:g} for "
                f"{SETTLE_CONSEC} frames (after moving) while still "
                f">= {self.arrival_thr:g} m from the waypoint position"
            )
        if self.arrival_baseline and self.arrival_enabled:
            print(
                f"  [transition] arrival baseline ({len(self.arrival_baseline)} wp): "
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
        self._settle_consec = 0
        self._has_moved_arm = False
        self._prev_err = float("nan")
        self._last_motion_err = float("nan")
        self._last_motion_err_d = float("nan")
        self._cumulative_runup = 0.0
        self.recent_obs = []

        self.fed = 0
        self.checked = 0
        self.fire_count = 0
        self.first_fire = None
        self.fire_mode = None  # "arrival" | "envelope" | "diverge"
        self.fired_waypoints = set()
        self._fired_this_execution = set()
        self.disabled_reason = None
        self.diag = {}
        self._wp_arrival_dist = {}  # waypoint -> worst arrival distance (m)
        self._wp_rise_peak = {}     # waypoint -> worst accumulated run-up (m)
        self._vlm_rejected = set()  # waypoints whose detection the VLM refuted
        self.last_waypoint = None

        # Per-frame signal CSV (for offline per-task threshold calibration).
        self._drift_log_rows = []
        self._drift_log_path = DRIFT_LOG_PATH or None

        self.live_plot = None
        if show_plot and self.plot_mod is not None:
            try:
                _plot_save = os.getenv("AHA_TRANSITION_PLOT_SAVE", "").strip() or None
                self.live_plot = self.plot_mod.LiveTelemetryPlot(
                    task_name, FAILURE, list(range(n_waypoints or 0)),
                    save_path=_plot_save)
            except Exception as exc:
                print(f"  [detector:transition] live plot disabled ({exc})")

    def _waypoint_threshold(self, waypoint):
        """Fire threshold (m) for this waypoint. Default: the flat floor
        (self.arrival_thr, 0.01 m) for every waypoint. With
        AHA_TRANSITION_ARRIVAL_USE_BASELINE=1: max(flat floor,
        median+K*robust_std) of the clean last in-motion distance before
        waypoint completion."""
        if not ARRIVAL_USE_BASELINE:
            return self.arrival_thr
        base = self.arrival_baseline.get(waypoint)
        if base is None:
            return self.arrival_thr
        if isinstance(base, dict):
            return max(self.arrival_thr, float(base.get("threshold", self.arrival_thr)))
        _mean, _std, _mx = base
        return max(self.arrival_thr, _mean + ARRIVAL_SIGMA_K * _std)

    def _abstain(self, waypoint):
        """True if this waypoint's worst clean arrival distance shows it is not
        tracking its original pose (dynamic target / unconstrained): no clean-
        derived threshold can separate clean from a failure, so every condition
        abstains here. Data-driven (keys off the waypoint's own clean max)."""
        if not ARRIVAL_USE_BASELINE:            # flat-threshold mode: judge every wp
            return False
        base = self.arrival_baseline.get(waypoint)
        if base is None:
            return False
        mx = float(base.get("max", 0.0)) if isinstance(base, dict) else base[2]
        return mx > ARRIVAL_ABSTAIN_CEIL

    def _rise_threshold(self, waypoint):
        """C3 threshold (m): max(floor, 1.5 * max cumulative clean runup)."""
        base = self.rise_baseline.get(waypoint)
        if base is None:
            return float("inf")
        if isinstance(base, dict):
            return max(self.arrival_thr, float(base.get("threshold", 0.0)))
        _mean, _std, _mx = base
        return max(self.arrival_thr, 1.5 * _mx)

    def _reset_waypoint(self, waypoint):
        self._fired_this_execution.clear()
        self._vlm_rejected.discard(waypoint)
        self._cur_wp = waypoint
        self.logs = []
        self._local_step = 0
        self._consec = 0
        self._env_consec = 0
        self._diverge_consec = 0
        self._settle_consec = 0
        self._has_moved_arm = False
        self._prev_err = float("nan")
        self._last_motion_err = float("nan")
        self._last_motion_err_d = float("nan")
        self._cumulative_runup = 0.0

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
            self.recent_obs.append(obs)
            if len(self.recent_obs) > 9:
                self.recent_obs.pop(0)
            local_step = self._local_step
            self._local_step += 1
            self.step_counter += 1

            row = det.obs_to_row(
                obs, local_step, waypoint=waypoint, expected_waypoint=waypoint,
                waypoint_path_done=path_done, waypoint_pose=waypoint_pose,
                ttm_waypoint_pose=original_pose,
                waypoint_started=True, started_waypoint=waypoint)
            self.logs.append(row)
            if len(self.logs) > 300:
                del self.logs[0]
            det.update_deltas(self.logs)

            # Failgen translation failures mutate the waypoint target before
            # motion starts. The normal transition verdict is intentionally
            # arrival-gated below: when the robot reaches this waypoint, compare
            # the last in-motion distance to the original/TTM waypoint pose with
            # the calibrated threshold. The direct target-change shortcut is
            # kept as an opt-in diagnostic path only.
            target_position_gap = float("nan")
            try:
                live = np.asarray(waypoint_pose, dtype=float).reshape(-1)
                original = np.asarray(original_pose, dtype=float).reshape(-1)
                if live.size >= 3 and original.size >= 3:
                    target_position_gap = float(
                        np.linalg.norm(live[:3] - original[:3]))
            except (TypeError, ValueError):
                pass
            target_orientation_gap = _quat_angle(
                _pose_quat(waypoint_pose), _pose_quat(original_pose))
            orientation_target_changed = (
                np.isfinite(target_orientation_gap)
                and target_orientation_gap >=
                ORIENTATION_TARGET_CHANGE_SUPPRESS_THRESHOLD)
            target_gap_thr = TARGET_POSITION_GAP_THRESHOLD
            position_target_changed = (
                np.isfinite(target_position_gap)
                and target_position_gap >= target_gap_thr)
            if (TARGET_CHANGE_ENABLED
                    and report
                    and TARGET_CHANGE_MIN_STEP <= local_step
                    <= TARGET_CHANGE_MIN_STEP + TARGET_CHANGE_START_WINDOW
                    and position_target_changed
                    and waypoint not in self._fired_this_execution
                    and waypoint not in self._vlm_rejected):
                reason = (
                    f"waypoint target position changed {target_position_gap:.3f} m "
                    f"from its original position (>= {target_gap_thr:.3f})")
                return self._do_fire(
                    row, row["step"], waypoint, reason, "target")

            if phase == "gripper":
                self._update_plot()
                return False

            late_position_target_change = (
                TARGET_CHANGE_ENABLED
                and local_step > TARGET_CHANGE_START_WINDOW
                and position_target_changed)
            suppress_tracking_for_orientation_change = late_position_target_change

            # Per-frame signals: end-effector step (rate), distance to the
            # original/TTM waypoint pose, and its per-frame change. This is the
            # value judged at waypoint arrival, so a shifted waypoint target is
            # detected only when the robot reaches the shifted pose, not at the
            # instant the waypoint command changes.
            gm = row.get("gripper_motion", float("nan"))
            try:
                gm = float(gm)
            except (TypeError, ValueError):
                gm = float("nan")
            err = row.get("waypoint_distance", float("nan"))
            try:
                err = float(err)
            except (TypeError, ValueError):
                err = float("nan")
            if not np.isfinite(err):
                err = row.get("ttm_waypoint_distance", float("nan"))
                try:
                    err = float(err)
                except (TypeError, ValueError):
                    err = float("nan")
            if not np.isfinite(err):
                err = row.get("live_waypoint_distance", float("nan"))
                try:
                    err = float(err)
                except (TypeError, ValueError):
                    err = float("nan")
            if not np.isfinite(err):
                err = row.get("expected_live_waypoint_distance", float("nan"))
                try:
                    err = float(err)
                except (TypeError, ValueError):
                    err = float("nan")
            err_d = row.get("waypoint_distance_delta", float("nan"))
            try:
                err_d = float(err_d)
            except (TypeError, ValueError):
                err_d = float("nan")
            if not np.isfinite(err_d):
                err_d = row.get("ttm_waypoint_distance_delta", float("nan"))
                try:
                    err_d = float(err_d)
                except (TypeError, ValueError):
                    err_d = float("nan")
            if not np.isfinite(err_d):
                err_d = row.get("live_waypoint_distance_delta", float("nan"))
                try:
                    err_d = float(err_d)
                except (TypeError, ValueError):
                    err_d = float("nan")
            if not np.isfinite(err_d):
                err_d = row.get("expected_live_waypoint_distance_delta", float("nan"))
                try:
                    err_d = float(err_d)
                except (TypeError, ValueError):
                    err_d = float("nan")

            # Live run-up = cumulative positive increase of the distance during
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
                     err, runup, int(runup_positive), gm, int(bool(path_done))))
                if DRIFT_LOG_STREAM and len(self._drift_log_rows) >= DRIFT_LOG_STREAM_EVERY:
                    self._flush_drift_log()
            if np.isfinite(runup):
                self._wp_rise_peak[waypoint] = max(
                    self._wp_rise_peak.get(waypoint, 0.0), runup)

            if not path_done and np.isfinite(err):
                self._last_motion_err = err
                self._last_motion_err_d = err_d

            if suppress_tracking_for_orientation_change:
                self._update_plot()
                return False

            # --- C2 ENVELOPE check (during the motion) -----------------------
            # Fire when the live distance leaves the clean envelope
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
                        f"transition distance {err:.3f} m left the clean envelope "
                        f"(>= {self.env_thr:.3f}, {self._env_consec} consec)")
                    return self._do_fire(row, row["step"], waypoint, reason,
                                         "envelope")

            # --- C3 DIVERGE check (during the motion) ------------------------
            # Fire when the error keeps RISING (delta > 0) for MORE than N
            # consecutive frames AND the accumulated run-up exceeds the
            # per-waypoint clean band max(floor, 1.5*max cumulative clean runup).
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
                        f"transition diverging: rose {runup:.3f} m over "
                        f"{self._diverge_consec} consec frames (>= {rise_thr:.3f} "
                        f"= max({self.arrival_thr:g}, {DIVERGE_RISE_SCALE:g}*max_clean_runup))")
                    return self._do_fire(row, row["step"], waypoint, reason,
                                         "diverge")

            # --- C2 SETTLE check (baseline-free EARLY detection) ------------
            # The end-effector stopped moving (motion collapsed for N frames
            # after the arm had been moving) but is still >= threshold from the
            # true (original/TTM) waypoint position -> it settled on the wrong
            # place. Robust to non-linear approaches: a clean detour is still
            # moving fast here, so it never looks settled until it truly arrives
            # (distance ~0). Fires before path_done; C1 arrival is the backstop.
            if (self.settle_enabled and report and not path_done
                    and waypoint not in self._fired_this_execution
                    and waypoint not in self._vlm_rejected):
                if np.isfinite(gm) and gm >= SETTLE_MOVE_ARM:
                    self._has_moved_arm = True
                if self._has_moved_arm and np.isfinite(gm) and gm < SETTLE_MOTION:
                    self._settle_consec += 1
                else:
                    self._settle_consec = 0
                wp_thr = self._waypoint_threshold(waypoint)
                if (self._settle_consec >= SETTLE_CONSEC and np.isfinite(err)
                        and err >= wp_thr):
                    reason = (
                        f"end-effector settled (motion < {SETTLE_MOTION:g} for "
                        f"{self._settle_consec} frames) {err:.3f} m from the "
                        f"waypoint position (>= {wp_thr:.3f})")
                    return self._do_fire(row, row["step"], waypoint, reason,
                                         "settle")

            # --- C1 ARRIVAL check -------------------------------------------
            # Only the waypoint-completion frame is judged, using the previous
            # in-motion distance. This matches calibration, which derives the
            # per-waypoint threshold from the last motion frame before
            # path_done, and avoids firing at the instant the next waypoint
            # command becomes active.
            if not path_done:
                self._update_plot()
                return False

            arrival_err = (
                self._last_motion_err
                if np.isfinite(self._last_motion_err)
                else err
            )
            arrival_err_d = (
                self._last_motion_err_d
                if np.isfinite(self._last_motion_err_d)
                else err_d
            )
            if not np.isfinite(arrival_err):
                self._update_plot()
                return False

            self.diag["max_distance"] = max(
                self.diag.get("max_distance", 0.0), arrival_err)
            self._wp_arrival_dist[waypoint] = max(
                self._wp_arrival_dist.get(waypoint, 0.0), arrival_err)

            if not self.arrival_enabled:
                self._update_plot()
                return False

            check_row = dict(row)
            check_row["waypoint_distance"] = arrival_err
            check_row["waypoint_distance_delta"] = arrival_err_d
            check_row["waypoint_distance_source"] = row.get(
                "waypoint_distance_source", "ttm_recalculated")

            forced = self._maybe_force_fire(row, row["step"], waypoint, path_done, report)
            if forced is not None:
                return forced

            wp_thr = self._waypoint_threshold(waypoint)
            raw = arrival_err >= wp_thr
            reason = (
                f"previous in-motion distance to waypoint "
                f"({check_row['waypoint_distance_source']}) was "
                f"{arrival_err:.3f} m before arrival (>= {wp_thr:.3f})")
            if (raw and report and waypoint not in self._fired_this_execution
                    and waypoint not in self._vlm_rejected):
                return self._do_fire(check_row, row["step"], waypoint, reason,
                                     "arrival")
            self._update_plot()
            return False
        except Exception as exc:
            self.disabled_reason = f"runtime error: {exc}"
            print(f"  [detector:transition] disabled mid-run — {exc}")
            return False

    def _do_fire(self, row, step, waypoint, reason, mode):
        """Common fire path for arrival, envelope, diverge, and force-fire:
        bookkeeping, VLM confirmation, and VLM-retraction. Returns True if the
        detection stands, False if the VLM verifier refuted it."""
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
            print("  [detector:transition] detection RETRACTED — "
                  "VLM verifier says no transition failure.")
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
            f"\n  [DETECTOR] TRANSITION failure at waypoint {waypoint} "
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
                input("  [transition] press Enter to continue the run...")
            except EOFError:
                pass
        return verdict

    def _prompt_vlm(self, row, waypoint):
        vlm = self.vlm_mod
        try:
            sampled_obs, step_offsets = vlm.sample_recent_observations(
                list(self.recent_obs))
            if not VLM_AUTO:
                vlm.show_camera_sequence(
                    sampled_obs, step_offsets=step_offsets,
                    env_wrapper=self.env_wrapper,
                    title="Images that will be sent to transition VLM")
        except Exception as exc:
            print(f"  [vlm] image preview failed: {exc}")
            sampled_obs, step_offsets = list(self.recent_obs), None

        if VLM_AUTO:
            print("  [vlm] auto-confirming transition detection with the "
                  "VLM verifier...")
        else:
            raw = input("  Confirm this transition failure with OpenAI VLM? [y/N]: ").strip().lower()
            if raw not in ("y", "yes"):
                return None
        try:
            result = vlm.confirm_transition_with_openai(
                sampled_obs, step_offsets=step_offsets, row=row,
                telemetry_history=list(self.logs), env_wrapper=self.env_wrapper,
                task_name=self.task_name, waypoint_index=waypoint,
                model=self.vlm_model, preview_images=False, trace=self.vlm_trace)
        except Exception as exc:
            print(f"  [vlm] confirmation failed: {exc}")
            return None
        verdict = result.get("transition_failure_happened")
        vtext = "YES" if verdict is True else "NO" if verdict is False else "UNKNOWN"
        print(f"  [vlm:{result.get('model')}] transition={vtext}  "
              f"{result.get('explanation', '')}  "
              f"cameras={', '.join(result.get('camera_names', []))}")
        return verdict

    def _update_plot(self):
        if self.live_plot is None:
            return
        try:
            self.live_plot.update(self.logs, self.frozen_thr, self.paused_steps)
        except Exception as exc:
            print(f"  [detector:transition] live plot update failed ({exc})")
            self.live_plot = None

    def diagnostics_line(self):
        per_wp = "; ".join(
            f"wp{wp}={val:.3f}(>={self._waypoint_threshold(wp):.3f})"
            for wp, val in sorted(self._wp_arrival_dist.items())
        )
        rise_wp = "; ".join(
            f"wp{wp}={val:.3f}" for wp, val in sorted(self._wp_rise_peak.items())
        )
        fired_by = f"  fired_by={self.fire_mode}" if self.fire_mode else ""
        return (f"      diag: fed {self.fed} obs; max waypoint-distance="
                f"{self.diag.get('max_distance', 0):.3f} m "
                f"(fires >= {self.arrival_thr:.3f}){fired_by}"
                f"{('  arrival distance per-waypoint: ' + per_wp) if per_wp else ''}"
                f"{('  rise-peak per-waypoint: ' + rise_wp) if rise_wp else ''}")

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
                                "dist_to_target", "runup", "runup_positive",
                                "gripper_motion", "path_done"])
                w.writerows(self._drift_log_rows)
            self._drift_log_rows = []
        except Exception as exc:
            print(f"  [transition] drift-log flush failed: {exc}")

    def _write_baseline(self):
        """Persist this (clean) run's per-waypoint arrival distances so future
        runs load them as the calibrated baseline. The injected failure
        waypoint, if any, is excluded so it can't poison the baseline.

        Legacy single-value path: superseded by the eval harness, which writes
        the richer statistical baseline (arrival_stats_by_waypoint with mean/std,
        error_stats, segment_rise_by_waypoint)."""
        import json
        stats = {
            int(wp): round(float(val), 4)
            for wp, val in self._wp_arrival_dist.items()
            if wp is not None and wp != self.failure_waypoint
        }
        if not stats:
            return
        try:
            ARRIVAL_STATS_DIR.mkdir(parents=True, exist_ok=True)
            path = ARRIVAL_STATS_DIR / f"{self.task_name}.json"
            path.write_text(json.dumps({
                "task": self.task_name,
                "clean_arrival_dist_by_waypoint": {
                    str(wp): val for wp, val in sorted(stats.items())},
                "design": (
                    "per-waypoint clean arrival distance (m); detector "
                    "threshold = max(%.3f, mean+%g*std)" % (
                        self.arrival_thr, ARRIVAL_SIGMA_K)),
            }, indent=2))
            print(f"  [transition] wrote clean arrival baseline -> {path}")
        except Exception as exc:
            print(f"  [transition] could not write arrival baseline: {exc}")

    def close(self):
        self._flush_drift_log()
        if WRITE_BASELINE:
            self._write_baseline()
        if self.live_plot is not None:
            try:
                if self.live_plot.save_path:
                    import os as _os
                    _os.makedirs(_os.path.dirname(self.live_plot.save_path), exist_ok=True)
                    self.live_plot.fig.savefig(self.live_plot.save_path, dpi=150, bbox_inches="tight")
                    print(f"  [detector:transition] plot saved -> {self.live_plot.save_path}")
                self.live_plot.plt.close(self.live_plot.fig)
            except Exception:
                pass
