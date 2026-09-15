
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

    def show_recent_frames(self, title='Last 5 camera frames'):
        if not self.frame_history:
            return

        fig, axes = self.plt.subplots(
            1,
            self.history_size,
            figsize=(3.0 * self.history_size, 3.0),
        )
        if self.history_size == 1:
            axes = [axes]

        empty = np.zeros_like(self.frame_history[-1])
        padded = [empty] * (self.history_size - len(self.frame_history))
        frames = padded + list(self.frame_history)
        for index, (ax, frame) in enumerate(zip(axes, frames)):
            ax.imshow(frame)
            ax.axis('off')
            ax.set_title(
                'current' if index == self.history_size - 1
                else f'-{self.history_size - 1 - index}',
                fontsize=10,
            )

        fig.suptitle(title)
        fig.tight_layout()
        self.plt.show(block=False)
        self.plt.pause(0.1)

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
        self.fig, (self.ax_norm, self.ax_delta) = plt.subplots(
            2, 1, figsize=(14, 7), sharex=True
        )
        self.fig.suptitle(
            f"Collision telemetry - task={task_name}  waypoint={waypoint}",
            fontsize=13,
        )
        self.series = [
            {
                'axis': self.ax_norm,
                'key': 'torque_norm',
                'label': 'torque_norm',
                'color': '#2196F3',
                'free_thr_key': 'tq_norm_free',
                'hold_thr_key': 'tq_norm_hold',
                'cross_key': 'torque_norm_crossed',
            },
            {
                'axis': self.ax_delta,
                'key': 'torque_delta',
                'label': 'torque_delta',
                'color': '#FF9800',
                'free_thr_key': 'tq_delta_free',
                'hold_thr_key': 'tq_delta_hold',
                'cross_key': 'torque_delta_crossed',
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
                label='detected collision'
            )
            paused = ax.scatter(
                [], [], color='black', s=32, zorder=6,
                label='paused (Enter)'
            )
            free_thr = ax.axhline(
                0.0, color='red', linestyle='--', linewidth=1.0,
                visible=False, label='free threshold'
            )
            hold_thr = ax.axhline(
                0.0, color='#D32F2F', linestyle=':', linewidth=1.0,
                visible=False, label='holding threshold'
            )
            ax.axvspan(0, warmup_end, color='#BDBDBD', alpha=0.35,
                       label='warmup')
            ax.axvline(warmup_end, color='gray', linestyle='--', linewidth=0.9)
            ax.set_ylabel(item['label'], fontsize=10)
            ax.grid(True, linewidth=0.4, alpha=0.5)
            ax.legend(fontsize=8, loc='upper right')
            item.update(
                line=line,
                raw_cross=raw_cross,
                detected=detected,
                paused=paused,
                free_thr=free_thr,
                hold_thr=hold_thr,
            )

        self.ax_delta.set_xlabel('step', fontsize=10)
        self.plt.tight_layout()
        self.plt.ion()
        self.fig.show()

    @staticmethod
    def _offsets(steps, values):
        if not steps:
            return np.empty((0, 2))
        return np.column_stack([steps, values])

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

        for item in self.series:
            key = item['key']
            values = [r[key] for r in logs]
            item['line'].set_data(steps, values)

            crossed_steps = [
                r['step'] for r in logs
                if r.get(item['cross_key'])
            ]
            crossed_values = [
                r[key] for r in logs
                if r.get(item['cross_key'])
            ]
            item['raw_cross'].set_offsets(
                self._offsets(crossed_steps, crossed_values)
            )

            detected_steps = [r['step'] for r in logs if r.get('collision')]
            detected_values = [r[key] for r in logs if r.get('collision')]
            item['detected'].set_offsets(
                self._offsets(detected_steps, detected_values)
            )

            pause_steps = [r['step'] for r in logs if r['step'] in paused_set]
            pause_values = [r[key] for r in logs if r['step'] in paused_set]
            item['paused'].set_offsets(self._offsets(pause_steps, pause_values))

            if thr is not None:
                free_val = thr[item['free_thr_key']]
                hold_val = thr[item['hold_thr_key']]
                item['free_thr'].set_ydata([free_val, free_val])
                item['free_thr'].set_label(f"free threshold={free_val:.2f}")
                item['free_thr'].set_visible(True)

                item['hold_thr'].set_ydata([hold_val, hold_val])
                item['hold_thr'].set_label(f"holding threshold={hold_val:.2f}")
                item['hold_thr'].set_visible(abs(hold_val - free_val) > 1e-6)

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
