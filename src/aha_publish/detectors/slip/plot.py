
from aha_publish import paths
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
    def __init__(self, task_name, waypoint, warmup_end, refresh_sec=0.05):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.refresh_sec = refresh_sec
        self.last_update = 0.0
        self.phase_spans = []
        self.fig, axes = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True)
        self.fig.suptitle(
            f"Slip telemetry - task={task_name}  waypoint={waypoint}",
            fontsize=13,
        )
        self.series = [
            {
                'axis': axes[0],
                'key': 'grip_force',
                'label': 'grip_force',
                'color': '#2196F3',
                'thr_key': 'grip_force_threshold',
                'threshold_label': 'grip force threshold',
                'cross_keys': ('force_released',),
            },
            {
                'axis': axes[1],
                'key': 'grip_force_drop',
                'label': 'grip_force_drop',
                'color': '#FF9800',
                'thr_key': 'grip_force_drop_threshold',
                'threshold_label': 'drop slip threshold',
                'cross_keys': ('force_drop_crossed',),
            },
        ]

        for item in self.series:
            ax = item['axis']
            line, = ax.plot(
                [], [], color=item['color'], linewidth=1.2,
                label=item['label']
            )
            raw_cross = ax.scatter(
                [], [], color='#8E24AA', s=18, zorder=4,
                label='threshold crossed'
            )
            detected = ax.scatter(
                [], [], color='#F44336', s=26, zorder=5,
                label='detected slip'
            )
            paused = ax.scatter(
                [], [], color='black', s=32, zorder=6,
                label='paused (Enter)'
            )
            threshold = ax.axhline(
                0.0, color='red', linestyle='--', linewidth=1.0,
                visible=False, label=item['threshold_label']
            )
            ax.axvline(warmup_end, color='gray', linestyle='--', linewidth=0.9)
            ax.set_ylabel(item['label'], fontsize=10)
            ax.grid(True, linewidth=0.4, alpha=0.5)
            ax.legend(fontsize=8, loc='upper right')
            item.update(
                line=line,
                raw_cross=raw_cross,
                detected=detected,
                paused=paused,
                threshold=threshold,
            )

        axes[-1].set_xlabel('step', fontsize=10)
        self.plt.tight_layout()
        self.plt.ion()
        self.fig.show()

    @staticmethod
    def _offsets(steps, values):
        if not steps:
            return np.empty((0, 2))
        return np.column_stack([steps, values])

    @staticmethod
    def _is_checked_phase(row):
        return (
            bool(row.get('holding_required_phase', True))
            and row.get('suppression_reason') != 'warmup'
        )

    def _unchecked_intervals(self, logs):
        intervals = []
        start = None
        end = None
        for row in logs:
            step = row['step']
            unchecked = not self._is_checked_phase(row)
            if unchecked and start is None:
                start = step
                end = step
            elif unchecked:
                end = step
            elif start is not None:
                intervals.append((start, end))
                start = None
                end = None
        if start is not None:
            intervals.append((start, end))
        return intervals

    def _draw_phase_background(self, logs):
        for span in self.phase_spans:
            span.remove()
        self.phase_spans = []

        intervals = self._unchecked_intervals(logs)
        if not intervals:
            return

        for item in self.series:
            axis = item['axis']
            for index, (start, end) in enumerate(intervals):
                span = axis.axvspan(
                    start - 0.5,
                    end + 0.5,
                    color='#E0E0E0',
                    alpha=0.55,
                    zorder=-10,
                    label='not checked' if index == 0 else '_nolegend_',
                )
                self.phase_spans.append(span)

    def update(self, logs, thr, paused_steps, force=False):
        now = time.monotonic()
        if not force and now - self.last_update < self.refresh_sec:
            return
        self.last_update = now

        if not logs:
            self.fig.canvas.draw_idle()
            self.plt.pause(0.001)
            return

        steps = [r['step'] for r in logs]
        paused_set = set(paused_steps)
        self._draw_phase_background(logs)

        for item in self.series:
            key = item['key']
            values = [r[key] for r in logs]
            item['line'].set_data(steps, values)

            crossed_steps = [
                r['step'] for r in logs
                if (
                    r.get('threshold_crossed')
                    and any(r.get(cross_key) for cross_key in item['cross_keys'])
                )
            ]
            crossed_values = [
                r[key] for r in logs
                if (
                    r.get('threshold_crossed')
                    and any(r.get(cross_key) for cross_key in item['cross_keys'])
                )
            ]
            item['raw_cross'].set_offsets(
                self._offsets(crossed_steps, crossed_values)
            )

            detected_steps = [r['step'] for r in logs if r.get('slip')]
            detected_values = [r[key] for r in logs if r.get('slip')]
            item['detected'].set_offsets(
                self._offsets(detected_steps, detected_values)
            )

            pause_steps = [r['step'] for r in logs if r['step'] in paused_set]
            pause_values = [r[key] for r in logs if r['step'] in paused_set]
            item['paused'].set_offsets(self._offsets(pause_steps, pause_values))

            if thr is not None:
                threshold = thr[item['thr_key']]
                item['threshold'].set_ydata([threshold, threshold])
                item['threshold'].set_label(
                    f"{item['threshold_label']}={threshold:.3f}"
                )
                item['threshold'].set_visible(True)

            axis = item['axis']
            axis.relim()
            axis.autoscale_view()
            axis.set_xlim(0, max(steps[-1], WARMUP_STEPS) + 5)
            axis.legend(fontsize=8, loc='upper right')

        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def show_blocking(self, logs, thr, paused_steps):
        self.update(logs, thr, paused_steps, force=True)
        self.plt.ioff()
        self.plt.show()
