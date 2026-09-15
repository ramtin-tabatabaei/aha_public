"""bt_gui-embedded SLIP detector.

A copy of aha_scripts/detectors/slip/interactive.py's real-time detection (``patched_step``
+ ``prompt_vlm_confirmation`` + live telemetry plot), adapted to be fed by the
bt_gui BT run. Detection logic, holding-phase gate, thresholds and VLM flow are
unchanged.

The failgen slip injection opens the gripper SLOWLY (actuate velocity 0.1), so
grip force decays gently and the low-force / proportional-drop slip rules handle
it directly.

(A former BT-specific "gripper_opened_mid_path" grasp-lost rule was removed: it
fired on commanded/legitimate releases where residual finger-contact noise had
polluted held_peak, producing systematic false positives.)
"""

from aha_publish import paths

import os

import numpy as np

from ._bundle import (
    VLM_AUTO, FORCE_FIRE_WAYPOINT, FORCE_FIRE_REASON, load_detector_bundle)

NAME = "slip"
FAILURE = "slip"
CONDITION = "maintains_grasp() == True"
NEEDS_WAYPOINT_POSE = False
NEEDS_ORIGINAL_POSE = True

# Grip force (N) at/below which the fingers hold NOTHING (grasp failed or object
# dropped). Matches the grip-force calibration's own not-holding cutoff
# (min_grip_force_for_stats = 0.2); a real grasp -- even a weak one -- reads well
# above this, while a failed grasp reads ~0. Used by the grasp-not-established
# rule, which must NOT use the is_holding flag (its threshold sits far higher).
NOTHING_HELD_GRIP = float(os.getenv("AHA_SLIP_NOTHING_HELD_GRIP", "0.2"))
TARGET_POSITION_GAP_THRESHOLD = float(
    os.getenv("AHA_SLIP_POSITION_GAP_SUPPRESS_THRESHOLD", "0.03"))
TARGET_ORIENTATION_GAP_THRESHOLD = float(
    os.getenv("AHA_SLIP_ORIENTATION_GAP_SUPPRESS_THRESHOLD", "0.30"))

# When set, every frame's signals are appended to this CSV so the eval harness
# can derive signals offline from BT-main runs (clean + failure trajectories).
SLIP_DRIFT_LOG_PATH = os.getenv("AHA_SLIP_DRIFT_LOG", "").strip()


def _truthy(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


# --- slip detection v2 (tangential-shear / Wong & Zhu 2026) -----------------
# v2 detects INCIPIENT slip from a surge in the per-finger tangential (shear)
# force, before the grip magnitude collapses (which is what v1 / method 1
# waits for). It is calibration-free: each finger's press ("normal") direction
# and its stable tangential force are estimated at runtime from the grasp's own
# baseline, so no friction identification or object/pose calibration is needed.
#
# AHA_SLIP_V2=1 is a convenience alias for AHA_SLIP_METHOD=3; setting it here (if
# method is not already pinned) makes both the live gate below and the core
# detector's check_slip() agree on method 3.
if _truthy(os.getenv("AHA_SLIP_V2", "")) and os.getenv("AHA_SLIP_METHOD") is None:
    os.environ["AHA_SLIP_METHOD"] = "3"

# Per-finger contact force floor (N): a finger only contributes a shear signal
# while it is actually touching the object.
V2_CONTACT_N = float(os.getenv("AHA_SLIP_V2_CONTACT_N", "0.1"))
# Frames of stable contact needed before a finger's tangential baseline is
# trusted enough to flag a surge.
V2_BASELINE_FRAMES = int(os.getenv("AHA_SLIP_V2_BASELINE_FRAMES", "5"))
# Adaptive surge threshold = K * (std of the finger's stable tangential force).
V2_K = float(os.getenv("AHA_SLIP_V2_K", "4.0"))
# Absolute floor (N) on the surge threshold so a near-zero baseline std cannot
# trip on sensor noise. This is the calibration-free analogue of the paper's
# fixed Df_xy_TH; raise it for noisier grips.
V2_JUMP_FLOOR = float(os.getenv("AHA_SLIP_V2_JUMP_FLOOR", "0.05"))
# Optional fixed surge threshold (N). When > 0 it REPLACES the adaptive
# K*std rule with the paper's literal fixed Df_xy_TH behaviour.
V2_JUMP_ABS = float(os.getenv("AHA_SLIP_V2_JUMP_ABS", "0"))
# EMA rate for tracking each finger's press direction. Small = a stable,
# slowly-adapting normal so the slip event itself does not rotate the baseline.
V2_NORMAL_ALPHA = float(os.getenv("AHA_SLIP_V2_NORMAL_ALPHA", "0.05"))
# Maintain-phase gate (ON by default): a shear surge only fires on a HOLD
# waypoint that comes AFTER the grasp-acquisition waypoint (acq = the lowest
# hold-required waypoint, where the gripper closes). The grasp-close itself
# produces a shear transient as the fingers load up the object -- that is a
# grasp artefact, not a slip -- and it lands on the acquisition waypoint. Slips
# are injected on the later held/transport waypoints, so gating to the maintain
# phase removes the acquisition false positive structurally without any force
# heuristics. Disable with AHA_SLIP_V2_MAINTAIN_GATE=0. Tasks with no waypoint
# phase info fall through to "allowed" (no acquisition waypoint known).
V2_MAINTAIN_GATE = _truthy(os.getenv("AHA_SLIP_V2_MAINTAIN_GATE", "1"))
# Normal-force-trend gate (OFF by default): only fire a shear surge when the
# finger's normal (press) force is steady-or-falling, not rising. Sound in
# principle, but on lift-until-it-slips tasks the object is under increasing
# load right up to release, so the normal force is rising at the slip onset too
# and this gate suppresses the real detection. Kept as an optional extra filter
# (AHA_SLIP_V2_NORMAL_GATE=1). "Rising" = current normal force exceeds its slow
# EMA by more than max(ABS, FRAC*EMA); the tolerances absorb sensor noise.
V2_NORMAL_GATE = _truthy(os.getenv("AHA_SLIP_V2_NORMAL_GATE", "0"))
V2_NORMAL_RISE_FRAC = float(os.getenv("AHA_SLIP_V2_NORMAL_RISE_FRAC", "0.15"))
V2_NORMAL_RISE_ABS = float(os.getenv("AHA_SLIP_V2_NORMAL_RISE_ABS", "0.2"))


def _pose_array(pose):
    try:
        return np.asarray(pose, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None


def _position_gap(pose, reference_pose):
    live = _pose_array(pose)
    reference = _pose_array(reference_pose)
    if live is None or reference is None or live.size < 3 or reference.size < 3:
        return float("nan")
    return float(np.linalg.norm(live[:3] - reference[:3]))


def _pose_quat(pose):
    arr = _pose_array(pose)
    if arr is None or arr.size < 7:
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

        settings = self.det.default_detector_settings()
        try:
            apply_stats = getattr(
                self.det, "apply_task_grip_force_stats_thresholds", None)
            if apply_stats is not None:
                settings = apply_stats(settings, task_name)
        except Exception as exc:
            print(f"  [detector:slip] task grip-force stats unavailable ({exc}); "
                  f"using default thresholds.")
        self.settings = settings
        try:
            self.phase_info = self.det.load_holding_requirement_from_waypoints(
                task_name)
        except Exception:
            self.phase_info = {"enabled": False}

        # The grasp-not-established rule fires when grip force is ~0 at a maintain
        # waypoint. That only separates a real grasp failure from a clean run when
        # the task's CLEAN grasp reliably holds well above the not-held floor.
        # On push / barely-gripping tasks (e.g. close_box) the per-task grip
        # calibration collapses grip_force_threshold to ~0 (mean-k*std <= 0), so a
        # clean run ALSO reads ~0 grip at the maintain waypoint and the rule would
        # false-fire on every scenario. Gate it on the calibrated holding force
        # clearing the not-held floor. Uncalibrated tasks now fall back to the
        # 0.1 N floor (not 1.0 N), which sits below the not-held floor, so the
        # grasp-not-established check is treated as unreliable and disabled there
        # -- correct, since a 0.1 N holding threshold can't separate a clean grasp
        # from a grasp failure. (The check is off by default anyway.)
        _hold_thr = float(self.settings.get("threshold_overrides", {}).get(
            "grip_force_threshold", 0.1))
        self._grasp_check_reliable = _hold_thr > NOTHING_HELD_GRIP
        if not self._grasp_check_reliable:
            print(f"  [slip] grasp-not-established check DISABLED for {task_name}: "
                  f"calibrated holding force {_hold_thr:.3f} <= not-held floor "
                  f"{NOTHING_HELD_GRIP:g} (task does not reliably grip; the check "
                  f"could not separate a clean run from a grasp failure).")

        self.logs = []
        self.paused_steps = []
        self.step_counter = 0
        self.frozen_thr = None
        self.recent_obs = []
        self._candidate_active = False
        self._consec = 0
        self._latched = False
        self._ever_held = False      # held since the last commanded release
        self._grasp_lost_consec = 0  # consecutive gripper-open-mid-path frames
        self._never_held_consec = 0  # consecutive not-held frames in a maintain phase
        self._vlm_rejected = set()   # waypoints whose detection the VLM refuted
        # Verification mode only: delay the VLM look by this many fed frames after
        # a detection triggers, so the verifier sees the failure once it has
        # developed in the scene (a slip is barely visible at the trigger frame).
        # 0 -> confirm synchronously at the trigger frame (legacy behaviour). The
        # delay has no effect when VLM confirmation is disabled for this detector.
        self._vlm_confirm_delay = max(
            0, int(os.getenv("AHA_VLM_CONFIRM_DELAY_FRAMES", "4")))
        self._pending_vlm = None     # deferred VLM confirmation awaiting its frames

        self.fed = 0
        self.checked = 0
        self.fire_count = 0
        self.first_fire = None
        self.fired_waypoints = set()
        self.disabled_reason = None
        self.diag = {}
        self.last_waypoint = None

        self.live_plot = None
        if show_plot and self.plot_mod is not None:
            try:
                wp = failure_waypoint if failure_waypoint is not None else 0
                self.live_plot = self.plot_mod.LiveTelemetryPlot(
                    task_name, wp, self.WARMUP)
            except Exception as exc:
                print(f"  [detector:slip] live plot disabled ({exc})")

        self._drift_log_path = SLIP_DRIFT_LOG_PATH or None
        self._drift_log_rows = []
        self._last_obs_for_drift = None
        # Set by the runner the instant it issues an open_gripper command; the
        # next logged telemetry frame records it (gripper_open_cmd=1). This is
        # the exact command frame -- earlier and more accurate than the jaw
        # finishing its open motion.
        self._open_cmd_pending = False
        self._close_cmd_pending = False
        # Method-1 proportional state: peak grip during the current grasp, and
        # whether we are inside a grasp window (between a close and the next open
        # command). A grip drop outside this window is a commanded release, not a
        # slip -- so slip detection is gated on _grasp_active.
        self._held_peak = 0.0
        self._grasp_active = False

        # v2 (tangential-shear) per-finger state, indexed 0=left, 1=right. The
        # normal (press) direction and the running mean/variance of the stable
        # tangential force are the calibration-free baseline; they are reset at
        # the start of every grasp (mark_close_command) so each grasp calibrates
        # against its own object/pose.
        self._reset_shear_state()

    def _reset_shear_state(self):
        # Welford accumulators for the per-finger tangential-force baseline plus
        # the current press direction, indexed 0=left, 1=right.
        self._shear_normal_dir = [None, None]  # unit press direction / finger
        self._shear_tang_mean = [0.0, 0.0]
        self._shear_tang_m2 = [0.0, 0.0]
        self._shear_tang_n = [0, 0]
        self._shear_normal_ema = [None, None]  # slow EMA of normal-force magnitude

    def _in_maintain_phase(self, waypoint):
        """True when ``waypoint`` is a hold waypoint AFTER the grasp-acquisition
        waypoint (the lowest hold-required waypoint, where the gripper closes).
        Used to gate the v2 shear surge out of the grasp-formation window. When
        no waypoint phase info is available, returns True (nothing to gate on)."""
        hold_wps = self.phase_info.get("hold_required_waypoints") or []
        if not hold_wps or waypoint is None:
            return True
        try:
            wp = int(waypoint)
        except (TypeError, ValueError):
            return True
        return wp > min(hold_wps)

    def _update_shear_signal(self, row, allow_fire=True):
        """v2 (Wong & Zhu 2026): per-finger tangential-shear surge.

        Estimate each finger's press ("normal") direction and its stable
        tangential force from the grasp's own runtime baseline, then flag an
        INCIPIENT slip when the tangential force jumps above that finger's
        baseline -- before the grip magnitude collapses (which is what method 1
        waits for). Calibration-free: no friction identification, no object- or
        pose-specific tuning. Writes left/right_normal, left/right_tangential,
        tangential_jump and shear_surge onto ``row``. The caller's existing
        grasp-window / phase / geometry gates still decide whether the surge is
        allowed to fire.
        """
        forces = row.get("_grip_forces")
        if forces is None:
            return
        forces = np.asarray(forces, dtype=float).reshape(-1)
        if forces.size < 6:
            return
        surge_any = False
        max_jump = 0.0
        rising_any = False
        for i in range(2):
            side = "left" if i == 0 else "right"
            f3 = forces[3 * i:3 * i + 3]
            mag = float(np.linalg.norm(f3))
            if mag <= V2_CONTACT_N:
                # Finger not touching: no shear signal, but keep the learned
                # baseline so a brief loss of contact does not wipe calibration.
                row[side + "_normal"] = mag
                row[side + "_tangential"] = 0.0
                continue
            nd = self._shear_normal_dir[i]
            if nd is None:
                nd = f3 / mag
                self._shear_normal_dir[i] = nd
            f_normal, f_tang = self.det.decompose_finger_force(f3, nd)
            # Normal-force-trend gate: is the press force RISING (grasp closing)
            # versus its slow EMA? Decided against the CURRENT EMA, before it is
            # updated this frame, so a fast grasp-close ramp (EMA still lagging
            # far below) reads as rising while a settled grasp does not.
            ref = self._shear_normal_ema[i]
            rising = (
                V2_NORMAL_GATE and ref is not None
                and (f_normal - ref) > max(V2_NORMAL_RISE_ABS,
                                           V2_NORMAL_RISE_FRAC * ref))
            # Decide the surge against the CURRENT baseline (before this frame
            # updates it), so the slip event itself cannot mask its own jump.
            n = self._shear_tang_n[i]
            mean = self._shear_tang_mean[i]
            std = (self._shear_tang_m2[i] / n) ** 0.5 if n > 1 else 0.0
            established = n >= V2_BASELINE_FRAMES
            thr = V2_JUMP_ABS if V2_JUMP_ABS > 0.0 else max(
                V2_JUMP_FLOOR, V2_K * std)
            jump = f_tang - mean
            surge_i = established and jump >= thr and not rising and allow_fire
            # Track the normal-force EMA every contact frame (slow, so it lags a
            # closing ramp) -- kept outside the not-surging guard so it keeps
            # following even across a surge.
            self._shear_normal_ema[i] = (
                f_normal if ref is None
                else (1.0 - V2_NORMAL_ALPHA) * ref + V2_NORMAL_ALPHA * f_normal)
            # Update the tangential baseline only while NOT surging (Welford
            # online mean/variance); likewise only track the press direction
            # when stable, so a slip cannot rotate the reference into itself.
            if not surge_i:
                n += 1
                delta = f_tang - mean
                mean += delta / n
                self._shear_tang_m2[i] += delta * (f_tang - mean)
                self._shear_tang_mean[i] = mean
                self._shear_tang_n[i] = n
                new_nd = (1.0 - V2_NORMAL_ALPHA) * nd + V2_NORMAL_ALPHA * (f3 / mag)
                nn = float(np.linalg.norm(new_nd))
                if nn > 1e-9:
                    self._shear_normal_dir[i] = new_nd / nn
            row[side + "_normal"] = f_normal
            row[side + "_tangential"] = f_tang
            surge_any = surge_any or surge_i
            rising_any = rising_any or rising
            max_jump = max(max_jump, jump)
        row["shear_surge"] = bool(surge_any)
        row["tangential_jump"] = float(max_jump)
        row["normal_rising"] = bool(rising_any)

    def mark_open_command(self):
        """Called by the runner when it issues an open_gripper command: the grasp
        is intentionally released, so the grasp window closes (no slip after)."""
        self._open_cmd_pending = True
        self._grasp_active = False

    def mark_close_command(self):
        """Called by the runner when it issues a close_gripper command; opens a
        fresh grasp window and resets the held-peak baseline."""
        self._close_cmd_pending = True
        self._held_peak = 0.0
        self._grasp_active = True
        # Fresh grasp -> re-calibrate the per-finger shear baseline from scratch.
        self._reset_shear_state()

    def step(self, obs, waypoint=None, path_done=False, waypoint_pose=None,
             original_pose=None, report=True):
        logs_before = len(self.logs)
        self._last_obs_for_drift = obs
        result = self._step_impl(obs, waypoint, path_done, waypoint_pose,
                                 original_pose, report)
        if self._drift_log_path and len(self.logs) > logs_before:
            self._append_drift_row(waypoint, path_done, result)
        return result

    def _step_impl(self, obs, waypoint=None, path_done=False, waypoint_pose=None,
                   original_pose=None, report=True):
        if self.disabled_reason is not None:
            return False
        self.fed += 1
        if report:
            self.checked += 1
        self.last_waypoint = waypoint
        det = self.det
        try:
            step = self.step_counter
            self.step_counter += 1
            self.recent_obs.append(obs)
            if len(self.recent_obs) > 30:
                self.recent_obs.pop(0)
            # Advance any deferred VLM confirmation on every fed frame so it sees
            # the scene a few frames after the trigger (see _fire).
            self._tick_pending_vlm()

            row = det.obs_to_row(obs, step)
            row["waypoint"] = waypoint
            row["waypoint_path_done"] = path_done
            phase_gate = (
                self.settings.get("use_waypoint_phase_gate", True)
                and bool(self.phase_info.get("enabled", False)))
            row["holding_required_phase"] = (
                det.waypoint_requires_holding(
                    waypoint, self.phase_info, waypoint_path_done=path_done)
                if phase_gate else True)

            self.logs.append(row)
            # No cap: keep the full per-step history for the whole run. A rolling
            # cap here also silently froze the drift/trace CSV once reached, because
            # step()'s `len(self.logs) > logs_before` write-gate can never be true
            # when append+trim keeps the length constant. Unbounded keeps every
            # waypoint's rows (the recompute below is idempotent and cheap at run
            # lengths of a few hundred steps).
            det.update_deltas(self.logs)

            if step < self.WARMUP:
                self._candidate_active = False
                self._consec = 0
                self._latched = False
                self._update_plot()
                return False

            if self.frozen_thr is None:
                self.frozen_thr = det.freeze_thresholds(
                    self.logs[:self.WARMUP], self.settings["threshold_overrides"])
            det.update_holding_states(self.logs, self.frozen_thr)

            forced = self._maybe_force_fire(row, step, waypoint, path_done, report)
            if forced is not None:
                return forced

            self.diag["max_grip"] = max(
                self.diag.get("max_grip", 0.0), row.get("grip_force", 0.0))
            self.diag["max_drop"] = max(
                self.diag.get("max_drop", 0.0), row.get("grip_force_drop", 0.0))
            if row.get("is_holding"):
                self.diag["holding_steps"] = self.diag.get("holding_steps", 0) + 1

            position_gap = _position_gap(waypoint_pose, original_pose)
            orientation_gap = _quat_angle(
                _pose_quat(waypoint_pose), _pose_quat(original_pose))
            geometry_target_changed = (
                (np.isfinite(position_gap)
                 and position_gap >= TARGET_POSITION_GAP_THRESHOLD)
                or (np.isfinite(orientation_gap)
                    and orientation_gap >= TARGET_ORIENTATION_GAP_THRESHOLD)
            )
            if geometry_target_changed:
                self._candidate_active = False
                self._consec = 0
                self._latched = False
                self._grasp_lost_consec = 0
                self._never_held_consec = 0
                self._update_plot()
                return False

            if not row["holding_required_phase"]:
                # An open gripper while holding is not required is the commanded
                # end-of-grasp release, not a slip. Clear the "was held" latch so
                # a later hold-required waypoint does not fire grasp-lost on the
                # already-open gripper (the change_clock-style clean false
                # positive: the commanded open lands in a non-holding phase, so
                # without this the latch survives into the next waypoint).
                try:
                    _gopen = float(getattr(obs, "gripper_open", None))
                except (TypeError, ValueError):
                    _gopen = None
                if _gopen is not None and _gopen > 0.5:
                    self._ever_held = False
                # Suppress only once the commanded release has actually happened.
                # This used to return unconditionally, which switched slip
                # detection off for the whole of a release waypoint: an object
                # dropped early in the stage, many steps BEFORE the gripper was
                # told to open, was structurally undetectable (held_peak was never
                # even recorded, so the proportional rule below could not run).
                # A stage whose postconditions permit a release at its END does not
                # license losing the object at its start -- the object is still
                # meant to be in the gripper until the open command. _grasp_active
                # closes in mark_open_command(), so it is the correct boundary:
                # after it, a drop is the intended release; before it, a collapse
                # is a genuine slip and stays detectable.
                if not self._grasp_active:
                    self._candidate_active = False
                    self._consec = 0
                    self._latched = False
                    self._grasp_lost_consec = 0
                    self._never_held_consec = 0
                    self._update_plot()
                    return False

            try:
                gripper_open_now = float(getattr(obs, "gripper_open", None))
            except (TypeError, ValueError):
                gripper_open_now = None
            if row.get("is_holding"):
                self._ever_held = True
            elif (path_done and gripper_open_now is not None
                    and gripper_open_now > 0.5):
                # Gripper actions only run on path-done frames, so an open
                # gripper there is a commanded release, not a slip.
                self._ever_held = False

            # Grasp-lost rule (gripper_opened_mid_path) REMOVED: it fired on
            # commanded/legitimate releases where residual finger-contact noise
            # had polluted held_peak, producing systematic false positives (e.g.
            # weighing_scales, take_shoes_out_of_box). Slip held-then-lost events
            # are still covered by the low-force / proportional-drop rules above.
            # _grasp_lost_consec stays 0 (kept only for the slip_score max()).

            # Grasp-never-established rule (catches the failgen 'grasp' failure,
            # which disables the gripper-close so the object is never held).
            # Distinct from slip (held -> then lost): here the object is NEVER
            # held. Fire only in a MAINTAIN-hold phase -- a hold-required waypoint
            # AFTER the grasp-acquisition waypoint (the lowest hold-required wp),
            # where on a clean run the object IS held. The signal is "essentially
            # no grip force" (nothing between the fingers): a grasp failure reads
            # ~0 N (gripper closed on nothing / never closed), while even a weak
            # clean grasp reads well above NOTHING_HELD_GRIP. We do NOT use the
            # is_holding flag here: its threshold (mean-3*std of the *holding*
            # force, e.g. 1.79 N) sits far above the noise floor, so a light/weak
            # clean grasp can read below it and would false-fire. NOTHING_HELD_GRIP
            # matches the calibration's own not-holding cutoff (min_grip_force_for_stats).
            #
            # Crucially, also require that the object was NEVER held this grasp
            # (not self._ever_held). The instantaneous-grip signal alone is not
            # enough: on some tasks a clean, established grasp momentarily reads
            # below NOTHING_HELD_GRIP while transporting the object (e.g.
            # beat_the_buzz dips to ~0.16 N at the maintain waypoint), which used
            # to false-fire on the clean run. A grasp that WAS held and then reads
            # ~0 is a held-then-lost event -- that is a slip, caught by the
            # low-force / gripper-opened-mid-path rules above, NOT a never-
            # established grasp. _ever_held is set whenever grip clears the
            # calibrated holding threshold and only reset on a commanded release,
            # so on a real 'grasp' failure (gripper-close disabled, object never
            # held) it stays False and this rule still fires.
            if self.settings.get("enable_grasp_not_established_check", False):
                hold_wps = self.phase_info.get("hold_required_waypoints") or []
                acq_wp = min(hold_wps) if hold_wps else None
                is_maintain = (acq_wp is not None and waypoint in hold_wps
                               and waypoint > acq_wp)
                not_grasped_now = (
                    self._grasp_check_reliable
                    and is_maintain
                    and not self._ever_held
                    and row.get("grip_force", 0.0) <= NOTHING_HELD_GRIP)
                self._never_held_consec = (
                    self._never_held_consec + 1 if not_grasped_now else 0)
                required_ng = max(1, int(self.settings["consecutive_slip_frames"]))
                if (self._never_held_consec >= required_ng and report
                        and waypoint not in self.fired_waypoints
                        and waypoint not in self._vlm_rejected):
                    reason = (
                        f"grasp_not_established: ~no grip force in the maintain "
                        f"phase (wp{waypoint}), grip_force="
                        f"{row.get('grip_force', 0.0):.3f} <= "
                        f"{NOTHING_HELD_GRIP:g}")
                    if self._fire(step, reason, row, waypoint):
                        return True
            else:
                self._never_held_consec = 0

            # Method 1 (proportional): track the peak grip while a real grasp is
            # held -- "real" = grip cleared the baseline hold threshold at some
            # point -- and flag a slip when grip collapses by AHA_SLIP_DROP_FRAC
            # of that peak. Self-scaling per object; a load change that never
            # collapses (open_window) does not fire.
            _grip_now = float(row.get("grip_force", 0.0))
            _hold_thr = float((self.frozen_thr or {}).get("grip_force_threshold", 1.0))
            # Only track/detect inside the grasp window (close -> next open). A
            # drop after a commanded open (close_box) is a release, not a slip.
            # Only the grasp window gates detection. holding_required_phase used
            # to reset the peak too, which switched slip detection off entirely
            # at release waypoints (open_jar wp5) -- a slip injected there was
            # structurally undetectable even when it happened well before the
            # commanded open. _grasp_active alone is the correct guard: it closes
            # on mark_open_command(), so a drop after the commanded open is still
            # treated as a release, while a drop before it can fire.
            if not self._grasp_active:
                self._held_peak = 0.0
            elif _grip_now > _hold_thr:
                self._held_peak = max(self._held_peak, _grip_now)
            _grasp_est = (self._grasp_active
                          and self._held_peak > _hold_thr)   # a real grasp happened
            _drop_frac = float(os.getenv("AHA_SLIP_DROP_FRAC", "0.8"))
            _prop_collapse = bool(
                _grasp_est and self._held_peak > 0.0
                and _grip_now <= (1.0 - _drop_frac) * self._held_peak)
            row["held_peak"] = self._held_peak
            row["grasp_established"] = _grasp_est
            row["proportional_collapse"] = _prop_collapse
            row["prop_drop"] = (
                1.0 - _grip_now / self._held_peak if self._held_peak > 0.0 else 0.0)

            # v2 (tangential-shear): compute the per-finger shear surge before
            # check_slip reads row['shear_surge']. Only track inside a grasp --
            # outside it the baseline is meaningless (mirrors held_peak).
            if self._grasp_active:
                allow_fire = (not V2_MAINTAIN_GATE
                              or self._in_maintain_phase(waypoint))
                self._update_shear_signal(row, allow_fire=allow_fire)
            else:
                row["shear_surge"] = False
                row["tangential_jump"] = 0.0
                row["normal_rising"] = False

            # check_slip() still populates force_released / force_drop_crossed /
            # slip_score / slip_reason on the row (logged to the trace CSV); its
            # boolean is no longer the gate -- see the counter below.
            _slip_started, reason = det.check_slip(
                row, self.frozen_thr,
                required_prior_holding_steps=self.settings[
                    "required_prior_holding_steps"],
                hold_lost_grace_steps=self.settings["hold_lost_grace_steps"])
            # The per-frame slip signal used for the consecutive gate must match
            # the detection method: grip-level (force_released, method 1) or
            # force-drop (force_drop_crossed, method 2).
            _m = os.getenv("AHA_SLIP_METHOD", "1").strip()
            if _m == "2":
                _active = row.get("force_drop_crossed")          # force-drop
            elif _m == "3":
                _active = row.get("shear_surge")                 # v2 tangential shear
            else:
                _active = row.get("proportional_collapse",       # proportional
                                  row.get("force_released"))
            # Count consecutive frames of the slip signal itself. Previously the
            # counter was reset to 1 on every frame where check_slip() returned
            # True, so under method 1 (where slip_started IS _active) it could
            # only advance once the hold was declared lost by the grip-force
            # threshold -- i.e. it counted "frames below the acquisition
            # threshold", not frames of proportional collapse. A slip that
            # leaves residual load on the gripper (a handle that stays in
            # contact) then pinned the counter at 1 forever and never fired.
            if _active:
                self._candidate_active = True
                self._consec += 1
            elif self._candidate_active:
                self._candidate_active = False
                self._consec = 0
                self._latched = False
            else:
                self._consec = 0
                if row.get("prior_holding_streak", 0) > 0:
                    self._latched = False

            required = max(1, int(self.settings["consecutive_slip_frames"]))
            raw_slip = self._candidate_active and self._consec >= required
            confirmed_now = raw_slip and not self._latched
            if raw_slip:
                self._latched = True

            fired = (confirmed_now and report
                     and waypoint not in self.fired_waypoints
                     and waypoint not in self._vlm_rejected)
            if fired:
                return self._fire(step, reason, row, waypoint)
            self._update_plot()
            return False
        except Exception as exc:
            self.disabled_reason = f"runtime error: {exc}"
            print(f"  [detector:slip] disabled mid-run — {exc}")
            return False

    def _append_drift_row(self, waypoint, path_done, fired):
        row = self.logs[-1]
        thr = self.frozen_thr or {}
        try:
            gripper_open = float(getattr(self._last_obs_for_drift, "gripper_open",
                                         float("nan")))
        except (TypeError, ValueError):
            gripper_open = float("nan")
        # Continuous jaw-open fraction (0=closed, 1=open). obs.gripper_open is
        # binarized at >0.95, so it only flips when the jaws are nearly fully
        # open -- well after they start opening (and after the grip drops). The
        # raw get_open_amount() captures the start of the opening motion.
        gripper_open_amount = float("nan")
        try:
            gripper_open_amount = float(
                self.env_wrapper.robot.gripper.get_open_amount()[0])
        except Exception:
            pass
        open_cmd = 1 if self._open_cmd_pending else 0
        self._open_cmd_pending = False
        close_cmd = 1 if self._close_cmd_pending else 0
        self._close_cmd_pending = False
        self._drift_log_rows.append((
            self.task_name,
            "" if waypoint is None else int(waypoint),
            int(row.get("step", 0)),
            self.step_counter - 1,
            float(row.get("grip_force", float("nan"))),
            float(row.get("left_grip_force", float("nan"))),
            float(row.get("right_grip_force", float("nan"))),
            float(row.get("grip_force_drop", float("nan"))),
            float(row.get("grip_force_delta", float("nan"))),
            gripper_open,
            gripper_open_amount,
            open_cmd,
            close_cmd,
            float(row.get("held_peak", float("nan"))),
            float(row.get("prop_drop", float("nan"))),
            float(row.get("left_normal", float("nan"))),
            float(row.get("right_normal", float("nan"))),
            float(row.get("left_tangential", float("nan"))),
            float(row.get("right_tangential", float("nan"))),
            float(row.get("tangential_jump", float("nan"))),
            int(bool(row.get("shear_surge", False))),
            int(bool(row.get("normal_rising", False))),
            int(bool(row.get("is_holding", False))),
            int(bool(row.get("holding_required_phase", True))),
            int(row.get("prior_holding_streak", 0)),
            ("" if row.get("steps_since_holding") is None
             else int(row["steps_since_holding"])),
            int(bool(row.get("force_released", False))),
            int(bool(row.get("force_drop_crossed", False))),
            int(self._candidate_active),
            max(self._consec, self._grasp_lost_consec, self._never_held_consec),
            float(row.get("slip_score", 0.0)),
            row.get("slip_reason", ""),
            float(thr.get("grip_force_threshold",
                          self.settings["threshold_overrides"].get(
                              "grip_force_threshold", 1.0))),
            float(thr.get("grip_force_drop_threshold",
                          self.settings["threshold_overrides"].get(
                              "grip_force_drop_threshold", 1.0))),
            int(bool(path_done)),
            int(bool(fired)),
        ))
        if len(self._drift_log_rows) >= 100:
            self._flush_drift_log()

    def _flush_drift_log(self):
        if not self._drift_log_path or not self._drift_log_rows:
            return
        import csv as _csv
        path = self._drift_log_path
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            exists = os.path.exists(path)
            with open(path, "a", newline="") as f:
                w = _csv.writer(f)
                if not exists:
                    w.writerow([
                        "task", "waypoint", "local_step", "step",
                        "grip_force", "left_grip_force", "right_grip_force",
                        "grip_force_drop", "grip_force_delta", "gripper_open",
                        "gripper_open_amount", "gripper_open_cmd",
                        "gripper_close_cmd", "held_peak", "prop_drop",
                        "left_normal", "right_normal",
                        "left_tangential", "right_tangential",
                        "tangential_jump", "shear_surge", "normal_rising",
                        "is_holding", "holding_required_phase",
                        "prior_holding_streak", "steps_since_holding",
                        "force_released", "force_drop_crossed",
                        "slip_candidate_active", "low_force_streak",
                        "slip_score", "slip_reason",
                        "grip_force_threshold", "grip_force_drop_threshold",
                        "path_done", "slip_fired",
                    ])
                w.writerows(self._drift_log_rows)
            self._drift_log_rows = []
        except Exception as exc:
            print(f"  [slip] drift-log flush failed: {exc}")

    def _maybe_force_fire(self, row, step, waypoint, path_done, report):
        """Eval hook: when AHA_FORCE_FIRE_WAYPOINT matches, fire on a clean scene
        so the VLM verifier's retraction can be measured. Returns True/False if it
        handled the step (mirrors the natural fire/retract path), else None."""
        if (FORCE_FIRE_WAYPOINT is None or waypoint != FORCE_FIRE_WAYPOINT
                or not path_done or not report
                or waypoint in self.fired_waypoints
                or waypoint in self._vlm_rejected):
            return None
        return self._fire(step, FORCE_FIRE_REASON, row, waypoint)

    def _fire(self, step, reason, row, waypoint):
        """Register a detection: bookkeeping + announce now, then verify.

        The VLM look is run either synchronously (legacy) or, in verification
        mode with self._vlm_confirm_delay > 0, deferred by that many fed frames so
        the verifier sees the failure after it develops in the scene. Returns True
        while the detection stands (confirmed or still pending), False if a
        synchronous VLM verdict retracted it."""
        self.fire_count += 1
        self.fired_waypoints.add(waypoint)
        if self.first_fire is None:
            self.first_fire = (step, reason, waypoint)
        self._announce_detection(step, reason, row, waypoint)
        vlm_active = self.vlm_enabled and self.vlm_mod is not None
        if vlm_active and self._vlm_confirm_delay > 0:
            self._pending_vlm = {
                "countdown": self._vlm_confirm_delay,
                "step": step, "reason": reason, "row": row, "waypoint": waypoint,
                "obs_snapshot": list(self.recent_obs)}
            self._update_plot()
            return True
        verdict = self._run_vlm_and_pause(row, waypoint)
        return self._apply_vlm_verdict(verdict, waypoint)

    def _announce_detection(self, step, reason, row, waypoint):
        print(
            f"\n  [DETECTOR] SLIP detected at waypoint {waypoint} (step {step})  "
            f"reason={reason}  grip={row['grip_force']:.3f}  "
            f"drop={row['grip_force_drop']:.3f}  "
            f"prior_hold={row.get('prior_holding_streak')}\n",
            flush=True,
        )
        self.paused_steps.append(step)
        if self.live_plot is not None:
            try:
                self.live_plot.update(
                    self.logs, self.frozen_thr, self.paused_steps, force=True)
            except Exception:
                pass

    def _run_vlm_and_pause(self, row, waypoint):
        verdict = None
        if self.vlm_enabled and self.vlm_mod is not None:
            verdict = self._prompt_vlm(row, waypoint)
        if not VLM_AUTO:
            try:
                input("  [slip] press Enter to continue the run...")
            except EOFError:
                pass
        return verdict

    def _apply_vlm_verdict(self, verdict, waypoint):
        if VLM_AUTO and verdict is False:
            print("  [detector:slip] detection RETRACTED — "
                  "VLM verifier says no slip.")
            self.fired_waypoints.discard(waypoint)
            self._vlm_rejected.add(waypoint)
            self.fire_count -= 1
            self._update_plot()
            return False
        self._update_plot()
        return True

    def _resolve_pending_vlm(self):
        pending = self._pending_vlm
        # keep self._pending_vlm set so _prompt_vlm can read obs_snapshot from it
        verdict = self._run_vlm_and_pause(pending["row"], pending["waypoint"])
        self._pending_vlm = None
        self._apply_vlm_verdict(verdict, pending["waypoint"])

    def _tick_pending_vlm(self):
        if self._pending_vlm is None:
            return
        self._pending_vlm["countdown"] -= 1
        if self._pending_vlm["countdown"] <= 0:
            self._resolve_pending_vlm()

    def flush_pending_vlm(self):
        """Run any deferred VLM confirmation NOW. Called before the BT consumes
        fired_waypoints so a retraction lands in time even when fewer than
        self._vlm_confirm_delay frames remained in the waypoint."""
        if self._pending_vlm is not None:
            self._resolve_pending_vlm()

    def _prompt_vlm(self, row, waypoint):
        vlm = self.vlm_mod
        obs_source = (self._pending_vlm or {}).get("obs_snapshot") or list(self.recent_obs)
        try:
            sampled_obs, step_offsets = vlm.sample_recent_observations(
                obs_source)
            if not VLM_AUTO:
                vlm.show_camera_sequence(
                    sampled_obs, step_offsets=step_offsets,
                    env_wrapper=self.env_wrapper,
                    title="Images that will be sent to slip VLM")
        except Exception as exc:
            print(f"  [vlm] image preview failed: {exc}")
            sampled_obs, step_offsets = list(self.recent_obs), None

        if VLM_AUTO:
            print("  [vlm] auto-confirming slip detection with the VLM "
                  "verifier...")
        else:
            raw = input("  Confirm this slip with OpenAI VLM? [y/N]: ").strip().lower()
            if raw not in ("y", "yes"):
                return None
        try:
            result = vlm.confirm_slip_with_openai(
                sampled_obs, step_offsets=step_offsets, row=row,
                telemetry_history=list(self.logs), env_wrapper=self.env_wrapper,
                task_name=self.task_name, waypoint_index=waypoint,
                waypoints_description_path=self.settings.get(
                    "waypoints_description_path"),
                model=self.vlm_model,
                preview_images=os.getenv("AHA_SLIP_VLM_PREVIEW_IMAGES", "0") in ("1", "true"),
                trace=self.vlm_trace)
        except Exception as exc:
            print(f"  [vlm] confirmation failed: {exc}")
            return None
        verdict = result.get("slip_happened")
        vtext = "YES" if verdict is True else "NO" if verdict is False else "UNKNOWN"
        print(f"  [vlm:{result.get('model')}] slip={vtext}  "
              f"{result.get('explanation', '')}  "
              f"cameras={', '.join(result.get('camera_names', []))}")
        from ._bundle import log_detector_vlm_event
        log_detector_vlm_event("slip", waypoint, verdict, result,
                               getattr(self, "step_counter", None))
        return verdict

    def _update_plot(self):
        if self.live_plot is None:
            return
        try:
            self.live_plot.update(self.logs, self.frozen_thr, self.paused_steps)
        except Exception as exc:
            print(f"  [detector:slip] live plot update failed ({exc})")
            self.live_plot = None

    def diagnostics_line(self):
        thr = self.frozen_thr or {}
        return (f"      diag: fed {self.fed} obs; max grip="
                f"{self.diag.get('max_grip', 0):.3f} (hold>"
                f"{thr.get('grip_force_threshold', 'n/a')}); holding obs "
                f"{self.diag.get('holding_steps', 0)}; max drop="
                f"{self.diag.get('max_drop', 0):.3f}")

    def close(self):
        self._flush_drift_log()
        if self.live_plot is not None:
            try:
                self._update_plot()
                print("  [slip] showing final plot — close the window to continue...")
                self.live_plot.plt.show(block=True)
            except Exception as exc:
                print(f"  [slip] final plot failed: {exc}")
            try:
                self.live_plot.plt.close(self.live_plot.fig)
            except Exception:
                pass
