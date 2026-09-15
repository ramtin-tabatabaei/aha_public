"""bt_gui-embedded COLLISION detector.

A copy of aha_scripts/detectors/collision/interactive.py's real-time detection (the
``patched_step`` body + ``prompt_vlm_confirmation`` + live telemetry plot),
adapted so the bt_gui BT run feeds it observations instead of the detector
driving its own demo. Detection logic, thresholds and VLM flow are unchanged.
"""

from aha_publish import paths

import os

import numpy as np

from ._bundle import (
    VLM_AUTO, FORCE_FIRE_WAYPOINT, FORCE_FIRE_REASON, load_detector_bundle)

NAME = "collision"
FAILURE = "collision"
CONDITION = "no_collision() == True"
# Collision detection is purely torque-based: detectors/collision/detector.py
# never references a waypoint pose (obs_to_row takes only obs and step). The live
# pose was used solely for the geometry-change suppression below, which is dead
# now that the live target channel is gone -- so requiring it would disable the
# detector for nothing.
NEEDS_WAYPOINT_POSE = False
NEEDS_ORIGINAL_POSE = True

TARGET_POSITION_GAP_THRESHOLD = float(
    os.getenv("AHA_COLLISION_POSITION_GAP_SUPPRESS_THRESHOLD", "0.03"))
TARGET_ORIENTATION_GAP_THRESHOLD = float(
    os.getenv("AHA_COLLISION_ORIENTATION_GAP_SUPPRESS_THRESHOLD", "0.30"))
TARGET_CHANGE_START_WINDOW = int(
    os.getenv("AHA_COLLISION_TARGET_CHANGE_START_WINDOW", "3"))

# Optional per-step torque-telemetry log (same columns as
# plot_collision_torque_trace.write_csv). Set AHA_COLLISION_DRIFT_LOG to a path
# and the detector streams every fed frame there so a caller can render the
# collision_method1 torque plot for the actual run. Unset -> no logging.
COLLISION_DRIFT_LOG_PATH = os.getenv("AHA_COLLISION_DRIFT_LOG", "").strip()

# Column order the plot/CSV expect (kept in sync with plot_collision_torque_trace).
_DRIFT_FIELDS = (
    "step", "torque_norm", "torque_delta", "torque_norm_threshold",
    "torque_delta_threshold", "torque_norm_crossed", "torque_delta_crossed",
    "torque_baseline", "torque_rise", "torque_rise_frac", "torque_rise_gate",
    "torque_proportional_spike", "waypoint", "collision_raw",
    "consecutive_collision_frames", "collision_reason", "is_holding",
    # Method-3 momentum-observer per-joint residual |r_i| (empty under methods
    # 1/2). plot_residual_trace reads these columns as r0..r6.
    "r0", "r1", "r2", "r3", "r4", "r5", "r6", "residual_score", "residual_norm",
    # The per-joint threshold each |r_i| was compared against, plus the gate's
    # verdict, so a trace is self-describing when re-scored offline.
    "rthr0", "rthr1", "rthr2", "rthr3", "rthr4", "rthr5", "rthr6",
    "residual_threshold", "residual_crossed", "residual_joints_over",
)


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
    # This class-level copy is what LiveDetectorMonitor's skip gate actually
    # reads (__init__.py:325); the module-level constant above is not consulted.
    NEEDS_WAYPOINT_POSE = False
    NEEDS_ORIGINAL_POSE = True

    def __init__(self, task_name, *, env_wrapper=None, vlm_enabled=True,
                 vlm_model=None, vlm_trace=False, show_plot=True,
                 n_waypoints=0, failure_waypoint=None):
        self.task_name = task_name
        self.env_wrapper = env_wrapper
        self.vlm_enabled = vlm_enabled
        self.vlm_model = vlm_model
        self.vlm_trace = vlm_trace
        # waypoint index -> object the BT stage manipulates at that waypoint.
        # Filled in by LiveDetectorMonitor from the BT stages; the VLM verifier
        # uses it to tell intended contact (with this object) from unintended.
        self.bt_target_objects = {}
        # waypoint index -> that BT stage's contract: {primitive,
        # preconditions, postconditions}. Also filled in by
        # LiveDetectorMonitor; it replaces the description JSON's prose action
        # in the VLM prompt.
        self.bt_stage_contexts = {}

        b = load_detector_bundle(NAME)
        self.det = b.detector
        self.plot_mod = b.plot
        self.vlm_mod = b.vlm
        self.WARMUP = int(self.det.WARMUP_STEPS)

        # --- detector settings (copied from interactive.run setup) ---
        settings = self.det.default_detector_settings()
        try:
            settings = self.det.apply_task_torque_stats_thresholds(
                settings, task_name,
                manual_threshold_keys=settings.get("manual_threshold_keys", ()),
            )
        except Exception as exc:
            print(f"  [detector:collision] task torque stats unavailable ({exc}); "
                  f"using warmup-baseline thresholds.")
        self.settings = settings
        # Per-waypoint proportional-spike gates for collision method 1. Missing
        # entries fall back to the detector's global rise-fraction floor.
        self.rise_gates = self.det.load_rise_frac_gates(task_name)
        # Method-3 momentum-observer per-joint residual thresholds from a clean
        # run (None -> warmup residual-envelope fallback in freeze_thresholds).
        try:
            self.residual_stats = self.det.load_residual_stats(task_name)
        except Exception:
            self.residual_stats = None
        # Use the detector's configured persistence (default: one frame).
        # An optional env cap remains for deliberate tuning.
        _consec = int(self.settings.get(
            "consecutive_collision_frames",
            self.det.DEFAULT_CONSECUTIVE_COLLISION_FRAMES))
        _cap = os.getenv("AHA_COLLISION_MAX_CONSECUTIVE", "").strip()
        if _cap:
            _consec = min(_consec, int(_cap))
        self.settings["consecutive_collision_frames"] = max(1, _consec)

        # --- optional per-step torque drift log (for collision_method1 plots) ---
        self._drift_log_path = COLLISION_DRIFT_LOG_PATH or None
        self._drift_log_rows = []

        # --- per-run state (copied from interactive.run) ---
        self.logs = []
        self.paused_steps = []
        self.step_counter = 0
        self.frozen_thr = None
        self.consecutive = 0
        self.recent_obs = []

        # --- bookkeeping for the bt_gui coordinator ---
        self.fed = 0
        self.checked = 0
        self.fire_count = 0
        self.first_fire = None
        self.fired_waypoints = set()
        self._vlm_rejected = set()  # waypoints whose detection the VLM refuted
        # Verification mode only: delay the VLM look by this many fed frames after
        # a detection triggers, so the verifier sees the collision once it has
        # developed in the scene. 0 -> confirm synchronously at the trigger frame
        # (legacy). No effect when VLM confirmation is disabled for this detector.
        self._vlm_confirm_delay = max(
            0, int(os.getenv("AHA_VLM_CONFIRM_DELAY_FRAMES", "4")))
        self._pending_vlm = None    # deferred VLM confirmation awaiting its frames
        self.disabled_reason = None
        self.diag = {}
        self.last_waypoint = None
        self._cur_wp = None
        self._local_step = 0
        self._geometry_changed_from_start = False

        # --- live telemetry plot (same as interactive) ---
        self.live_plot = None
        if show_plot and self.plot_mod is not None:
            try:
                wp = failure_waypoint if failure_waypoint is not None else 0
                self.live_plot = self.plot_mod.LiveTelemetryPlot(
                    task_name, wp, self.WARMUP
                )
            except Exception as exc:
                print(f"  [detector:collision] live plot disabled ({exc})")

    # ---- per-step driver (copied patched_step body) -----------------------
    def step(self, obs, waypoint=None, path_done=False, waypoint_pose=None,
             original_pose=None, report=True):
        if self.disabled_reason is not None:
            return False
        self.fed += 1
        if report:
            self.checked += 1
        self.last_waypoint = waypoint
        det = self.det
        try:
            if waypoint != self._cur_wp:
                self._cur_wp = waypoint
                self._local_step = 0
                self._geometry_changed_from_start = False
            local_step = self._local_step
            self._local_step += 1
            step = self.step_counter
            self.step_counter += 1

            self.recent_obs.append(obs)
            if len(self.recent_obs) > 9:
                self.recent_obs.pop(0)
            # Advance any deferred VLM confirmation on every fed frame so it sees
            # the scene a few frames after the trigger (see _fire).
            self._tick_pending_vlm()

            row = det.obs_to_row(obs, step)
            self.logs.append(row)
            if len(self.logs) > 300:
                del self.logs[0]
            det.update_deltas(self.logs)

            self.diag["max_tq_norm"] = max(
                self.diag.get("max_tq_norm", 0.0), row.get("torque_norm", 0.0))
            self.diag["max_tq_delta"] = max(
                self.diag.get("max_tq_delta", 0.0), row.get("torque_delta", 0.0))

            position_gap = _position_gap(waypoint_pose, original_pose)
            orientation_gap = _quat_angle(
                _pose_quat(waypoint_pose), _pose_quat(original_pose))
            position_changed = (
                np.isfinite(position_gap)
                and position_gap >= TARGET_POSITION_GAP_THRESHOLD)
            orientation_changed = (
                np.isfinite(orientation_gap)
                and orientation_gap >= TARGET_ORIENTATION_GAP_THRESHOLD)
            geometry_changed = position_changed or orientation_changed
            if local_step <= TARGET_CHANGE_START_WINDOW and geometry_changed:
                self._geometry_changed_from_start = True
            late_geometry_disturbance = (
                local_step > TARGET_CHANGE_START_WINDOW
                and geometry_changed
                and not self._geometry_changed_from_start)
            row["waypoint_position_gap"] = position_gap
            row["waypoint_orientation_gap"] = orientation_gap

            if step < self.WARMUP:
                self.consecutive = 0
                row["suppression_reason"] = "warmup"
                self._append_drift_row(row, "warmup")
                self._update_plot()
                return False

            if self.frozen_thr is None:
                self.frozen_thr = det.freeze_thresholds(
                    self.logs[:self.WARMUP], self.settings["threshold_overrides"],
                    residual_stats=self.residual_stats)

            if self._geometry_changed_from_start:
                raw, reason = False, "initial_waypoint_pose_change_suppressed"
            else:
                raw, reason = det.check_collision(
                    row, self.frozen_thr,
                    torque_rule=self.settings["torque_rule"],
                    torque_norm_weight=self.settings["torque_norm_weight"],
                    torque_delta_weight=self.settings["torque_delta_weight"],
                    torque_score_threshold=self.settings["torque_score_threshold"],
                    use_touch_force=self.settings.get("use_touch_force", False),
                    torque_rise_gate=self.rise_gates.get(waypoint),
                )
                if late_geometry_disturbance and raw:
                    reason = (
                        f"{reason}+late_waypoint_pose_disturbance"
                        f"(position_gap={position_gap:.3f} m, "
                        f"orientation_gap={orientation_gap:.3f} rad)")
            row["threshold_crossed"] = raw
            row["collision_reason"] = reason
            row["collision"] = raw
            if raw:
                self.consecutive += 1
            else:
                self.consecutive = 0
            self._append_drift_row(row, reason)

            forced = self._maybe_force_fire(row, step, waypoint, path_done, report)
            if forced is not None:
                return forced

            required = max(1, int(self.settings["consecutive_collision_frames"]))
            confirmed = raw and self.consecutive >= required and report
            # Report once per waypoint (the BT run carries many steps per waypoint).
            if (confirmed and waypoint not in self.fired_waypoints
                    and waypoint not in self._vlm_rejected):
                return self._fire(step, reason, row, waypoint)
            self._update_plot()
            return False
        except Exception as exc:
            self.disabled_reason = f"runtime error: {exc}"
            print(f"  [detector:collision] disabled mid-run — {exc}")
            return False

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

    # ---- fire: bookkeeping + announce -> (deferred) VLM verify -------------
    def _fire(self, step, reason, row, waypoint):
        """Register a detection: bookkeeping + announce now, then verify.

        The VLM look is run either synchronously (legacy) or, in verification
        mode with self._vlm_confirm_delay > 0, deferred by that many fed frames so
        the verifier sees the collision after it develops in the scene. Returns
        True while the detection stands (confirmed or still pending), False if a
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
                "step": step, "reason": reason, "row": row, "waypoint": waypoint}
            self._update_plot()
            return True
        verdict = self._run_vlm_and_pause(row, waypoint)
        return self._apply_vlm_verdict(verdict, waypoint)

    # ---- on-detection: print -> plot -> (optional) VLM -> pause -----------
    def _announce_detection(self, step, reason, row, waypoint):
        print(
            f"\n  [DETECTOR] COLLISION detected at waypoint {waypoint} "
            f"(step {step})  reason={reason}  "
            f"torque_norm={row['torque_norm']:.3f}  "
            f"torque_delta={row['torque_delta']:.3f}\n",
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
                input("  [collision] press Enter to continue the run...")
            except EOFError:
                pass
        return verdict

    def _apply_vlm_verdict(self, verdict, waypoint):
        if VLM_AUTO and verdict is False:
            print("  [detector:collision] detection RETRACTED — "
                  "VLM verifier says no collision.")
            self.fired_waypoints.discard(waypoint)
            self._vlm_rejected.add(waypoint)
            self.fire_count -= 1
            self._update_plot()
            return False
        self._update_plot()
        return True

    def _peak_row_since(self, fire_step, fallback_row):
        """Return the peak-signal telemetry row seen since the trigger frame.

        The row captured at _fire is the collision ONSET (rising edge, e.g. the
        first sustained frame), whose signal is still low; it develops over the
        next few frames. Showing the VLM this onset row makes its numeric
        reasoning contradict the (freshly sampled) images. Pick the peak frame
        in [fire_step, now] so the event block matches the images.

        Ranked by the signal the detector actually fired on: the momentum
        observer's max|r_i| under method 3, torque_norm otherwise."""
        candidates = [r for r in self.logs
                      if int(r.get("step", -1)) >= int(fire_step)]
        if not candidates:
            return fallback_row
        method = os.getenv("AHA_COLLISION_METHOD", "3").strip()
        key = "residual_score" if method == "3" else "torque_norm"
        return max(candidates,
                   key=lambda r: float(r.get(key, 0.0) or 0.0))

    def _resolve_pending_vlm(self):
        pending = self._pending_vlm
        self._pending_vlm = None
        event_row = self._peak_row_since(pending["step"], pending["row"])
        verdict = self._run_vlm_and_pause(event_row, pending["waypoint"])
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

    # ---- VLM confirmation (copied prompt_vlm_confirmation) ----------------
    def _prompt_vlm(self, row, waypoint):
        vlm = self.vlm_mod
        try:
            sampled_obs, step_offsets = vlm.sample_recent_observations(
                list(self.recent_obs))
            if not VLM_AUTO:
                vlm.show_camera_sequence(
                    sampled_obs, step_offsets=step_offsets,
                    env_wrapper=self.env_wrapper,
                    title="Images that will be sent to collision VLM")
        except Exception as exc:
            print(f"  [vlm] image preview failed: {exc}")
            sampled_obs, step_offsets = list(self.recent_obs), None

        if VLM_AUTO:
            print("  [vlm] auto-confirming collision detection with the VLM "
                  "verifier...")
        else:
            raw = input("  Confirm this collision with OpenAI VLM? [y/N]: ").strip().lower()
            if raw not in ("y", "yes"):
                return None
        try:
            result = vlm.confirm_collision_with_openai(
                sampled_obs, step_offsets=step_offsets, row=row,
                telemetry_history=list(self.logs), env_wrapper=self.env_wrapper,
                task_name=self.task_name, waypoint_index=waypoint,
                target_object=(self.bt_target_objects or {}).get(waypoint),
                stage_context=(self.bt_stage_contexts or {}).get(waypoint),
                model=self.vlm_model, preview_images=False, trace=self.vlm_trace)
        except Exception as exc:
            print(f"  [vlm] confirmation failed: {exc}")
            return None
        verdict = result.get("collision_happened")
        vtext = "YES" if verdict is True else "NO" if verdict is False else "UNKNOWN"
        print(f"  [vlm:{result.get('model')}] collision={vtext}  "
              f"{result.get('explanation', '')}  "
              f"cameras={', '.join(result.get('camera_names', []))}")
        from ._bundle import log_detector_vlm_event
        log_detector_vlm_event("collision", waypoint, verdict, result,
                               getattr(self, "step_counter", None))
        return verdict

    # ---- plot upkeep ------------------------------------------------------
    def _update_plot(self):
        if self.live_plot is None:
            return
        try:
            self.live_plot.update(self.logs, self.frozen_thr, self.paused_steps)
        except Exception as exc:
            print(f"  [detector:collision] live plot update failed ({exc})")
            self.live_plot = None

    def _append_drift_row(self, row, reason):
        """Record one per-step torque telemetry row (plot_trace schema)."""
        if not self._drift_log_path:
            return
        frozen = self.frozen_thr
        holding = bool(row.get("is_holding", False))
        norm_thr = ((frozen.get("tq_norm_hold") if holding
                     else frozen.get("tq_norm_free")) if frozen else "")
        delta_thr = ((frozen.get("tq_delta_hold") if holding
                      else frozen.get("tq_delta_free")) if frozen else "")
        # Method-3 per-joint momentum residual (populated by update_deltas ->
        # update_momentum_observer); empty under methods 1/2 so the torque-spike
        # plot is unaffected.
        mo = row.get("mo_residual")
        resid_cols = {
            f"r{j}": (float(mo[j]) if mo is not None and j < len(mo) else "")
            for j in range(7)
        }
        resid_cols["residual_score"] = row.get("residual_score", "")
        resid_cols["residual_norm"] = row.get("residual_norm", "")
        # The per-joint threshold the gate actually compared against, so a trace
        # can be re-scored later without having to infer which calibration and
        # which multipliers the run used.
        thr_vec = frozen.get("residual_thr_vec") if frozen else None
        for j in range(7):
            resid_cols[f"rthr{j}"] = (
                float(thr_vec[j]) if thr_vec is not None and j < len(thr_vec) else "")
        resid_cols["residual_threshold"] = (
            frozen.get("residual_thr", "") if frozen else "")
        resid_cols["residual_crossed"] = row.get("residual_crossed", "")
        resid_cols["residual_joints_over"] = row.get("residual_joints_over", "")
        self._drift_log_rows.append({
            **resid_cols,
            "step": row.get("step", self.step_counter),
            "torque_norm": row.get("torque_norm", ""),
            "torque_delta": row.get("torque_delta", ""),
            "torque_norm_threshold": norm_thr,
            "torque_delta_threshold": delta_thr,
            "torque_norm_crossed": row.get("torque_norm_crossed", False),
            "torque_delta_crossed": row.get("torque_delta_crossed", False),
            "torque_baseline": row.get("torque_baseline", ""),
            "torque_rise": row.get("torque_rise", ""),
            "torque_rise_frac": row.get("torque_rise_frac", ""),
            "torque_rise_gate": row.get("torque_rise_gate", ""),
            "torque_proportional_spike": row.get("torque_proportional_spike", False),
            "waypoint": row.get("waypoint", ""),
            "collision_raw": row.get("collision", False),
            "consecutive_collision_frames": self.consecutive,
            "collision_reason": reason or row.get("collision_reason", ""),
            "is_holding": holding,
        })
        if len(self._drift_log_rows) >= 100:
            self._flush_drift_log()

    def _flush_drift_log(self):
        if not self._drift_log_path or not self._drift_log_rows:
            return
        import csv as _csv
        write_header = (not os.path.exists(self._drift_log_path)
                        or os.path.getsize(self._drift_log_path) == 0)
        try:
            with open(self._drift_log_path, "a", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=list(_DRIFT_FIELDS))
                if write_header:
                    w.writeheader()
                w.writerows(self._drift_log_rows)
            self._drift_log_rows = []
        except Exception as exc:
            print(f"  [detector:collision] drift log write failed ({exc})")
            self._drift_log_path = None

    def diagnostics_line(self):
        thr = self.frozen_thr or {}
        return (f"      diag: fed {self.fed} obs; max torque_norm="
                f"{self.diag.get('max_tq_norm', 0):.3f} (thr~"
                f"{thr.get('tq_norm_free', 'n/a')}), max torque_delta="
                f"{self.diag.get('max_tq_delta', 0):.3f}")

    def close(self):
        self._flush_drift_log()
        if self.live_plot is not None:
            try:
                self.live_plot.plt.close(self.live_plot.fig)
            except Exception:
                pass
