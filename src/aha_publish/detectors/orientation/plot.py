
from aha_publish import paths
import os
import time

import numpy as np

from detector import WARMUP_STEPS


class LiveCameraView:
    def __init__(
        self,
        camera,
        title,
        refresh_sec=0.05,
        controls=True,
        history_size=5,
    ):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        self.plt = plt
        self.camera = camera
        self.controls = controls
        self.history_size = history_size
        self.frame_history = []
        self.latest_frame = None
        self.refresh_sec = refresh_sec
        self.last_update = 0.0
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8.5, 7.2))
        if controls:
            self.fig.subplots_adjust(bottom=0.42)
        self.fig.canvas.manager.set_window_title(title)
        self.ax.set_title(title)
        self.ax.axis('off')
        self.image_artist = self.ax.imshow(
            np.zeros((10, 10, 3), dtype=np.uint8)
        )

        self.sliders = {}
        if controls:
            position = np.asarray(camera.get_position(), dtype=float)
            orientation_deg = np.degrees(
                np.asarray(camera.get_orientation(), dtype=float)
            )
            specs = [
                ('x', position[0], position[0] - 2.0, position[0] + 2.0),
                ('y', position[1], position[1] - 2.0, position[1] + 2.0),
                ('z', position[2], max(0.05, position[2] - 1.5), position[2] + 1.5),
                ('roll', orientation_deg[0], -180.0, 180.0),
                ('pitch', orientation_deg[1], -180.0, 180.0),
                ('yaw', orientation_deg[2], -180.0, 180.0),
            ]
            for index, (name, value, min_value, max_value) in enumerate(specs):
                slider_ax = self.fig.add_axes(
                    [0.14, 0.32 - index * 0.048, 0.74, 0.026]
                )
                self.sliders[name] = Slider(
                    ax=slider_ax,
                    label=name,
                    valmin=min_value,
                    valmax=max_value,
                    valinit=value,
                    valfmt='%.3f' if name in ('x', 'y', 'z') else '%.1f',
                )
        self.plt.ion()
        self.fig.show()

    def apply_camera_controls(self):
        if not self.controls:
            return
        self.camera.set_position([
            self.sliders['x'].val,
            self.sliders['y'].val,
            self.sliders['z'].val,
        ])
        self.camera.set_orientation(
            np.radians([
                self.sliders['roll'].val,
                self.sliders['pitch'].val,
                self.sliders['yaw'].val,
            ])
        )

    def update(self, force=False):
        now = time.monotonic()
        if not force and now - self.last_update < self.refresh_sec:
            return
        self.last_update = now

        self.apply_camera_controls()
        frame = self.camera.capture_rgb()
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        self.latest_frame = frame.copy()
        self.frame_history.append(frame.copy())
        self.frame_history = self.frame_history[-self.history_size:]

        self.image_artist.set_data(frame)
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)
        return self.latest_frame

    def show_blocking(self):
        self.update(force=True)
        self.plt.ioff()
        self.plt.show()


class LiveTelemetryPlot:
    """Two-panel live view of the orientation detector.

    The detector watches one number: the angle between the gripper and the
    waypoint's target orientation (recalculated from the TTM scene). It flags a
    failure when that angle keeps *growing* and the total rise since it started
    growing crosses a threshold. The two panels mirror that exactly:

      Top    - the watched angle itself (orange), with a faint reference line.
      Bottom - how far the angle has risen *while still worsening* (orange,
               filled), versus the red trigger threshold. A failure fires the
               moment the orange fill crosses the red line.

    Both panels share the same colour for the watched signal so the eye follows
    one story instead of juggling a blue and a green line of equal weight.
    """

    # Single, consistent colour vocabulary (chosen to avoid the old
    # blue-vs-green ambiguity and to read for common colour-blindness).
    C_WATCHED = '#E65100'   # the angle / rise the detector actually judges
    C_REFERENCE = '#9E9E9E'  # informational only, kept faint
    C_TRIGGER = '#D32F2F'   # threshold + detected-failure markers
    C_WAYPOINT = '#7B1FA2'  # waypoint-start guides

    def __init__(
        self,
        task_name,
        failtype,
        waypoints,
        refresh_sec=0.05,
    ):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.refresh_sec = refresh_sec
        self.last_update = 0.0
        self.waypoint_markers = []
        self._rise_fill = None
        self.fig, (self.ax_angle, self.ax_rise) = plt.subplots(
            2, 1, figsize=(14, 7), sharex=True
        )
        self.fig.suptitle(
            f"Orientation detector - task={task_name}  "
            f"failure={failtype}  waypoints={waypoints}",
            fontsize=13,
        )

        # --- Top panel: the angle the detector watches -----------------------
        self.ttm_angle_line, = self.ax_angle.plot(
            [], [], color=self.C_WATCHED, linewidth=1.9,
            label='gripper-to-target angle (what the detector watches)'
        )
        self.live_angle_line, = self.ax_angle.plot(
            [], [], color=self.C_REFERENCE, linewidth=1.0, alpha=0.6,
            label='reference: angle to waypoint-start pose (not judged)'
        )
        self.angle_detected = self.ax_angle.scatter(
            [], [], color=self.C_TRIGGER, s=40, zorder=6,
            edgecolors='black', linewidths=0.6,
            label='orientation FAILURE flagged here'
        )
        self.angle_paused = self.ax_angle.scatter(
            [], [], marker='X', color='black', s=46, zorder=7,
            label='run paused here'
        )

        # --- Bottom panel: the per-frame gradient signal ---------------------
        self.rise_line, = self.ax_rise.plot(
            [], [], color=self.C_WATCHED, linewidth=1.9,
            label='per-frame orientation step (rate of change)'
        )
        self.threshold_line = self.ax_rise.axhline(
            0.0, color=self.C_TRIGGER, linestyle='--', linewidth=1.3,
            label='trigger threshold'
        )
        self.rise_detected = self.ax_rise.scatter(
            [], [], color=self.C_TRIGGER, s=40, zorder=6,
            edgecolors='black', linewidths=0.6,
            label='orientation FAILURE flagged here'
        )

        for ax in (self.ax_angle, self.ax_rise):
            ax.axvspan(0, WARMUP_STEPS, color='#BDBDBD', alpha=0.30,
                       label='warmup (no detection)')
            ax.axvline(WARMUP_STEPS, color='gray', linestyle='--',
                       linewidth=0.9)
            ax.grid(True, linewidth=0.4, alpha=0.5)
            ax.legend(fontsize=8, loc='upper left')

        self.ax_angle.set_title(
            'How tilted is the gripper vs. the target orientation?',
            fontsize=10)
        self.ax_angle.set_ylabel('angle (radians)', fontsize=10)
        self.ax_rise.set_title(
            'How fast is the wrist reorienting? (rate)', fontsize=10)
        self.ax_rise.set_ylabel('orientation step (rad/frame)', fontsize=10)
        self.ax_rise.set_xlabel('step', fontsize=10)
        self.plt.tight_layout()
        self.plt.ion()
        self.fig.show()

    @staticmethod
    def _offsets(steps, values):
        if not steps:
            return np.empty((0, 2))
        return np.column_stack([steps, values])

    def _autoscale(self, axis, steps):
        axis.relim()
        axis.autoscale_view()
        axis.set_xlim(0, max(steps[-1], WARMUP_STEPS) + 5)
        axis.legend(fontsize=8, loc='upper left')

    def _refresh_waypoint_markers(self, logs):
        for artist in self.waypoint_markers:
            artist.remove()
        self.waypoint_markers = []

        starts = [
            (row['step'], row.get('started_waypoint'))
            for row in logs
            if row.get('waypoint_started')
        ]
        seen = set()
        for step, waypoint in starts:
            key = (step, waypoint)
            if key in seen:
                continue
            seen.add(key)
            for ax in (self.ax_angle, self.ax_rise):
                line = ax.axvline(
                    step, color=self.C_WAYPOINT, linestyle=':', linewidth=0.9,
                    alpha=0.75
                )
                label = ax.text(
                    step, 0.98, f'wp{waypoint} start',
                    transform=ax.get_xaxis_transform(),
                    rotation=90,
                    va='top',
                    ha='right',
                    fontsize=8,
                    color=self.C_WAYPOINT,
                )
                self.waypoint_markers.extend([line, label])

    def update(self, logs, thr, paused_steps, force=False):
        now = time.monotonic()
        if not force and now - self.last_update < self.refresh_sec:
            return
        self.last_update = now

        if not logs:
            self.fig.canvas.draw_idle()
            self.plt.pause(0.001)
            return

        steps = [row['step'] for row in logs]
        paused_set = set(paused_steps)
        # The watched signal (top) and the per-frame gradient signal (bottom).
        ttm_angles = [row.get('ttm_waypoint_angle', np.nan) for row in logs]
        live_angles = [row.get('live_waypoint_angle', np.nan) for row in logs]
        rise = [row.get('gripper_orientation_delta', np.nan) for row in logs]

        self.ttm_angle_line.set_data(steps, ttm_angles)
        self.live_angle_line.set_data(steps, live_angles)
        self.rise_line.set_data(steps, rise)

        # Threshold line: the gradient threshold the detector compares the
        # per-frame step against. It is not necessarily in `thr`, so fall back
        # to an instance attr, then the environment; hide the line otherwise.
        threshold = np.nan
        if thr and thr.get('gradient_threshold') is not None:
            threshold = float(thr.get('gradient_threshold'))
        elif getattr(self, '_gradient_threshold', None) is not None:
            threshold = float(self._gradient_threshold)
        else:
            env_thr = os.environ.get('AHA_ORIENTATION_GRADIENT_THRESHOLD')
            if env_thr is not None:
                try:
                    threshold = float(env_thr)
                except ValueError:
                    threshold = np.nan
        if np.isfinite(threshold):
            self.threshold_line.set_visible(True)
            self.threshold_line.set_ydata([threshold, threshold])
            self.threshold_line.set_label(
                f'gradient threshold = {threshold:.4f} rad/frame')
        else:
            self.threshold_line.set_visible(False)

        # Shade the area under the gradient curve. Re-drawn each frame
        # (fill_between has no setter).
        if self._rise_fill is not None:
            self._rise_fill.remove()
        rise_arr = np.asarray(rise, dtype=float)
        self._rise_fill = self.ax_rise.fill_between(
            steps, 0.0, rise_arr,
            where=np.isfinite(rise_arr), interpolate=False,
            color=self.C_WATCHED, alpha=0.18, zorder=1,
        )

        detected_steps = [
            row['step'] for row in logs if row.get('orientation_failure')
        ]
        detected_angles = [
            row.get('ttm_waypoint_angle', np.nan)
            for row in logs if row.get('orientation_failure')
        ]
        detected_rise = [
            row.get('gripper_orientation_delta', np.nan)
            for row in logs if row.get('orientation_failure')
        ]
        self.angle_detected.set_offsets(
            self._offsets(detected_steps, detected_angles)
        )
        self.rise_detected.set_offsets(
            self._offsets(detected_steps, detected_rise)
        )

        pause_steps = [row['step'] for row in logs if row['step'] in paused_set]
        pause_angles = [
            row.get('ttm_waypoint_angle', np.nan)
            for row in logs if row['step'] in paused_set
        ]
        self.angle_paused.set_offsets(
            self._offsets(pause_steps, pause_angles)
        )

        self._refresh_waypoint_markers(logs)
        for axis in (self.ax_angle, self.ax_rise):
            self._autoscale(axis, steps)

        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def show_blocking(self, logs, thr, paused_steps):
        self.update(logs, thr, paused_steps, force=True)
        self.plt.ioff()
        self.plt.show()
