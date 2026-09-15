
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
    def __init__(
        self,
        task_name,
        failtype,
        waypoints,
        refresh_sec=0.05,
        save_path=None,
    ):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.refresh_sec = refresh_sec
        self.save_path = save_path
        self.last_update = 0.0
        self.waypoint_markers = []
        self.fig, (self.ax_distance, self.ax_deviation) = plt.subplots(
            2, 1, figsize=(14, 7), sharex=True
        )
        self.fig.suptitle(
            f"Transition telemetry - task={task_name}  "
            f"failure={failtype}  waypoints={waypoints}",
            fontsize=13,
        )

        self.live_distance_line, = self.ax_distance.plot(
            [], [], color='#1976D2', linewidth=1.3,
            label='distance to waypoint-start target'
        )
        self.ttm_distance_line, = self.ax_distance.plot(
            [], [], color='#2E7D32', linewidth=1.3, linestyle='-.',
            label='distance to TTM recalculated target'
        )
        self.selected_distance_line, = self.ax_distance.plot(
            [], [], color='#5D4037', linewidth=1.0, alpha=0.75,
            label='detector distance'
        )
        self.distance_detected = self.ax_distance.scatter(
            [], [], color='#F44336', s=28, zorder=5,
            label='detected transition failure'
        )
        self.distance_paused = self.ax_distance.scatter(
            [], [], color='black', s=32, zorder=6,
            label='paused'
        )

        self.deviation_line, = self.ax_deviation.plot(
            [], [], color='#5D4037', linewidth=1.6,
            label='per-frame end-effector step (rate of change)'
        )
        self.deviation_zero = self.ax_deviation.axhline(
            0.0, color='red', linestyle='--', linewidth=1.0,
            label='gradient threshold'
        )
        self.deviation_detected = self.ax_deviation.scatter(
            [], [], color='#F44336', s=28, zorder=5,
            label='detected transition failure'
        )

        for ax in (self.ax_distance, self.ax_deviation):
            ax.axvspan(0, WARMUP_STEPS, color='#BDBDBD', alpha=0.35,
                       label='warmup')
            ax.axvline(WARMUP_STEPS, color='gray', linestyle='--',
                       linewidth=0.9)
            ax.grid(True, linewidth=0.4, alpha=0.5)
            ax.legend(fontsize=8, loc='upper right')

        self.ax_distance.set_ylabel('meters', fontsize=10)
        self.ax_deviation.set_title(
            'How fast is the end-effector moving? (rate)', fontsize=10)
        self.ax_deviation.set_ylabel('position step (m/frame)', fontsize=10)
        self.ax_deviation.set_xlabel('step', fontsize=10)
        self.plt.tight_layout()
        self.plt.ion()
        self.fig.show()

    @staticmethod
    def _offsets(steps, values):
        if not steps:
            return np.empty((0, 2))
        return np.column_stack([steps, values])

    @staticmethod
    def _deltas(values):
        deltas = [np.nan]
        for index in range(1, len(values)):
            curr = values[index]
            prev = values[index - 1]
            if np.isfinite(curr) and np.isfinite(prev):
                deltas.append(float(curr) - float(prev))
            else:
                deltas.append(np.nan)
        return deltas

    def _autoscale(self, axis, steps):
        axis.relim()
        axis.autoscale_view()
        axis.set_xlim(0, max(steps[-1], WARMUP_STEPS) + 5)
        axis.legend(fontsize=8, loc='upper right')

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
            for ax in (self.ax_distance, self.ax_deviation):
                line = ax.axvline(
                    step, color='#7B1FA2', linestyle=':', linewidth=0.9,
                    alpha=0.75
                )
                label = ax.text(
                    step, 0.98, f'wp{waypoint} start',
                    transform=ax.get_xaxis_transform(),
                    rotation=90,
                    va='top',
                    ha='right',
                    fontsize=8,
                    color='#7B1FA2',
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
        distances = [row.get('waypoint_distance', np.nan) for row in logs]
        live_distances = [
            row.get('live_waypoint_distance', np.nan) for row in logs
        ]
        ttm_distances = [
            row.get('ttm_waypoint_distance', np.nan) for row in logs
        ]
        # Per-frame gradient signal for the bottom panel.
        gripper_motion = [row.get('gripper_motion', np.nan) for row in logs]

        self.live_distance_line.set_data(steps, live_distances)
        self.ttm_distance_line.set_data(steps, ttm_distances)
        self.selected_distance_line.set_data(steps, distances)
        self.deviation_line.set_data(steps, gripper_motion)

        # Gradient threshold line: not necessarily in `thr`, so fall back to an
        # instance attr, then the environment; hide the line otherwise.
        threshold = np.nan
        if thr and thr.get('gradient_threshold') is not None:
            threshold = float(thr.get('gradient_threshold'))
        elif getattr(self, '_gradient_threshold', None) is not None:
            threshold = float(self._gradient_threshold)
        else:
            env_thr = os.environ.get('AHA_TRANSITION_GRADIENT_THRESHOLD')
            if env_thr is not None:
                try:
                    threshold = float(env_thr)
                except ValueError:
                    threshold = np.nan
        if np.isfinite(threshold):
            self.deviation_zero.set_visible(True)
            self.deviation_zero.set_ydata([threshold, threshold])
            self.deviation_zero.set_label(
                f'gradient threshold = {threshold:.4f} m/frame')
        else:
            self.deviation_zero.set_visible(False)

        detected_steps = [
            row['step'] for row in logs if row.get('transition_failure')
        ]
        detected_distances = [
            row.get('waypoint_distance', np.nan)
            for row in logs if row.get('transition_failure')
        ]
        detected_deltas = [
            row.get('gripper_motion', np.nan)
            for row in logs if row.get('transition_failure')
        ]
        self.distance_detected.set_offsets(
            self._offsets(detected_steps, detected_distances)
        )
        self.deviation_detected.set_offsets(
            self._offsets(detected_steps, detected_deltas)
        )

        pause_steps = [row['step'] for row in logs if row['step'] in paused_set]
        pause_distances = [
            row.get('waypoint_distance', np.nan)
            for row in logs if row['step'] in paused_set
        ]
        self.distance_paused.set_offsets(
            self._offsets(pause_steps, pause_distances)
        )

        self._refresh_waypoint_markers(logs)
        for axis in (self.ax_distance, self.ax_deviation):
            self._autoscale(axis, steps)

        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def show_blocking(self, logs, thr, paused_steps):
        self.update(logs, thr, paused_steps, force=True)
        self.plt.ioff()
        self.plt.show()
