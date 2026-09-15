"""bt_gui-embedded FREEZING detector.

A copy of aha_scripts/detectors/freezing/interactive.py's real-time detection + live
telemetry plot, adapted to be fed by the bt_gui BT run. Freezing has no VLM
confirmation module and no camera view (matching the standalone harness), so on
detection it prints, force-updates the plot, and pauses for Enter.

Unlike the standalone harness (which only flags inside an injected-freeze
window), here we report whenever the robot actually goes still after warmup,
since any unexpected stall during the BT run is a real freeze.
"""

from aha_publish import paths

import os

from ._bundle import VLM_AUTO, load_detector_bundle

NAME = "freezing"

# A real freezing failure holds still for ~2 s (many frames); brief settles
# between sub-motions are short. Require a longer sustained stillness than the
# standalone default (8) to suppress those false positives. Env-overridable.
FREEZE_CONSECUTIVE_FRAMES = int(
    os.getenv("AHA_FREEZING_CONSECUTIVE_FRAMES", "16"))
# The robot legitimately comes to rest at the final waypoint (task done), which
# looks exactly like a freeze. Skip freezing detection there by default.
SKIP_LAST_WAYPOINT = os.getenv(
    "AHA_FREEZING_SKIP_LAST_WAYPOINT", "1").strip().lower() in (
        "1", "true", "yes", "on")
FAILURE = "freezing"
CONDITION = "not_frozen() == True"
NEEDS_WAYPOINT_POSE = False


class LiveDetector:
    def __init__(self, task_name, *, env_wrapper=None, vlm_enabled=True,
                 vlm_model=None, vlm_trace=False, show_plot=True,
                 n_waypoints=0, failure_waypoint=None):
        self.task_name = task_name
        b = load_detector_bundle(NAME)
        self.det = b.detector
        self.plot_mod = b.plot
        self.WARMUP = int(self.det.WARMUP_STEPS)

        self.settings = self.det.default_detector_settings()
        # Per-task stillness thresholds (calibrated from clean runs, like slip /
        # collision). Falls back to the flat defaults when a task has no stats.
        try:
            self.settings = self.det.apply_task_freezing_stats_thresholds(
                self.settings, task_name)
            info = self.settings.get("task_freezing_stats", {})
            if info.get("status") == "loaded":
                rec = info.get("recommended", {})
                print(f"  [freezing] per-task thresholds for {task_name}: "
                      f"joint_velocity_norm<={rec.get('joint_velocity_norm')}, "
                      f"joint_position_delta<={rec.get('joint_position_delta')}")
        except Exception as exc:
            print(f"  [detector:freezing] task freezing stats unavailable "
                  f"({exc}); using flat default thresholds.")
        # Require longer sustained stillness before declaring a freeze.
        self.settings["consecutive_freeze_frames"] = FREEZE_CONSECUTIVE_FRAMES
        self.camera_names = tuple(self.settings.get("camera_names", ()))
        # Index of the last waypoint; freezing there is the expected end-of-task
        # rest, not a failure.
        self.final_waypoint = (n_waypoints - 1) if n_waypoints else None
        self.skip_last_waypoint = SKIP_LAST_WAYPOINT
        print(
            f"  [freezing] consecutive-frames threshold={FREEZE_CONSECUTIVE_FRAMES}"
            + (f"; skipping final waypoint {self.final_waypoint}"
               if self.skip_last_waypoint and self.final_waypoint is not None
               else "") + "\n")

        self.logs = []
        self.paused_steps = []
        self.step_counter = 0
        self.frozen_thr = None
        self._consec = 0

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
                    task_name, wp, self.WARMUP, camera_names=list(self.camera_names))
            except Exception as exc:
                print(f"  [detector:freezing] live plot disabled ({exc})")

    def step(self, obs, waypoint=None, path_done=False, waypoint_pose=None,
             report=True):
        if self.disabled_reason is not None:
            return False
        self.fed += 1
        if report:
            self.checked += 1
        self.last_waypoint = waypoint
        # Skip when the path is done: the arm is intentionally still while the
        # gripper actuates (open/close) and while settling at the arrived pose.
        # A real freeze happens mid-path (path_done=False), so detection there
        # is preserved; only the expected-stillness phases are ignored. This is
        # what was false-firing at grasp/close waypoints.
        if path_done:
            self._consec = 0
            return False
        # Skip the final waypoint: the robot is expected to come to rest there.
        if (self.skip_last_waypoint and self.final_waypoint is not None
                and waypoint is not None and waypoint >= self.final_waypoint):
            self._consec = 0
            return False
        det = self.det
        try:
            step = self.step_counter
            self.step_counter += 1

            row = det.obs_to_row(obs, step, camera_names=self.camera_names)
            self.logs.append(row)
            if len(self.logs) > 48:
                del self.logs[0]
            det.update_deltas(self.logs)
            for old in self.logs[:-2]:
                if old.get("_camera_images"):
                    old["_camera_images"] = {}

            vel = row.get("joint_velocity_norm")
            if vel is not None:
                cur = self.diag.get("min_velocity")
                self.diag["min_velocity"] = vel if cur is None else min(cur, vel)

            if step < self.WARMUP:
                self._consec = 0
                self._update_plot()
                return False

            if self.frozen_thr is None:
                self.frozen_thr = det.freeze_thresholds(
                    self.logs[:self.WARMUP], self.settings["threshold_overrides"])

            raw, reason = det.check_freezing(
                row, self.frozen_thr,
                min_visible_cameras=self.settings["min_visible_cameras"])
            if raw:
                self._consec += 1
            else:
                self._consec = 0
            required = max(1, int(self.settings["consecutive_freeze_frames"]))
            confirmed = raw and self._consec >= required and report
            if confirmed and waypoint not in self.fired_waypoints:
                self.fire_count += 1
                self.fired_waypoints.add(waypoint)
                if self.first_fire is None:
                    self.first_fire = (step, reason, waypoint)
                self._on_detection(step, reason, waypoint)
                self._update_plot()
                return True
            self._update_plot()
            return False
        except Exception as exc:
            self.disabled_reason = f"runtime error: {exc}"
            print(f"  [detector:freezing] disabled mid-run — {exc}")
            return False

    def _on_detection(self, step, reason, waypoint):
        print(
            f"\n  [DETECTOR] FREEZING detected at waypoint {waypoint} "
            f"(step {step})  reason={reason}\n",
            flush=True,
        )
        self.paused_steps.append(step)
        if self.live_plot is not None:
            try:
                self.live_plot.update(
                    self.logs, self.frozen_thr, self.paused_steps, force=True)
            except Exception:
                pass
        if not VLM_AUTO:
            try:
                input("  [freezing] press Enter to continue the run...")
            except EOFError:
                pass

    def _update_plot(self):
        if self.live_plot is None:
            return
        try:
            self.live_plot.update(self.logs, self.frozen_thr, self.paused_steps)
        except Exception as exc:
            print(f"  [detector:freezing] live plot update failed ({exc})")
            self.live_plot = None

    def diagnostics_line(self):
        thr = self.frozen_thr or {}
        return (f"      diag: fed {self.fed} obs; min joint_velocity="
                f"{self.diag.get('min_velocity', 'n/a')} (still<="
                f"{thr.get('joint_velocity_norm', 'n/a')})")

    def close(self):
        if self.live_plot is not None:
            try:
                self.live_plot.plt.close(self.live_plot.fig)
            except Exception:
                pass
