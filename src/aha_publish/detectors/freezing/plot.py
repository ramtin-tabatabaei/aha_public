
from aha_publish import paths
import time

import numpy as np

from detector import DEFAULT_CAMERA_NAMES, WARMUP_STEPS


class LiveTelemetryPlot:
    def __init__(
        self,
        task_name,
        waypoint,
        warmup_end,
        camera_names=None,
        refresh_sec=0.05,
    ):
        import matplotlib.pyplot as plt

        self.camera_names = list(camera_names or DEFAULT_CAMERA_NAMES)
        self.plt = plt
        self.refresh_sec = refresh_sec
        self.last_update = 0.0
        self.fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        camera_text = ', '.join(self._camera_label(c) for c in self.camera_names)
        self.fig.suptitle(
            f"Freezing telemetry - task={task_name}  waypoint={waypoint}  "
            f"cameras={camera_text}",
            fontsize=13,
        )
        self.series = [
            {
                'axis': axes[0],
                'key': 'joint_position_delta',
                'label': 'joint_position_delta',
                'color': '#2196F3',
                'thr_key': 'joint_position_delta',
            },
            {
                'axis': axes[1],
                'key': 'joint_velocity_norm',
                'label': 'joint_velocity_norm',
                'color': '#FF9800',
                'thr_key': 'joint_velocity_norm',
            },
            {
                'axis': axes[2],
                'key': 'camera_motion',
                'label': 'camera_motion_norm',
                'color': '#4CAF50',
                'thr_key': 'camera_motion',
            },
        ]
        self.camera_delta_lines = {}

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
                label='detected freeze'
            )
            paused = ax.scatter(
                [], [], color='black', s=32, zorder=6,
                label='paused (Enter)'
            )
            threshold = ax.axhline(
                0.0, color='red', linestyle='--', linewidth=1.0,
                visible=False, label='stillness threshold'
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
                threshold=threshold,
            )

        camera_axis = axes[2]
        camera_colors = ['#607D8B', '#795548', '#9C27B0', '#00BCD4', '#CDDC39']
        for index, camera_name in enumerate(self.camera_names):
            line, = camera_axis.plot(
                [],
                [],
                color=camera_colors[index % len(camera_colors)],
                linestyle=':',
                linewidth=1.0,
                alpha=0.8,
                label=f"{self._camera_label(camera_name)} delta",
            )
            self.camera_delta_lines[camera_name] = line

        axes[-1].set_xlabel('step', fontsize=10)
        self.plt.tight_layout()
        self.plt.ion()
        self.fig.show()

    @staticmethod
    def _camera_label(camera_name):
        if camera_name.endswith('_rgb'):
            return camera_name[:-4]
        return camera_name

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

            if key == 'camera_motion':
                for camera_name, line in self.camera_delta_lines.items():
                    delta_key = f'{camera_name}_delta'
                    line.set_data(
                        steps,
                        [r.get(delta_key, np.nan) for r in logs],
                    )

            crossed_steps = [
                r['step'] for r in logs if r.get('threshold_crossed')
            ]
            crossed_values = [
                r[key] for r in logs if r.get('threshold_crossed')
            ]
            item['raw_cross'].set_offsets(
                self._offsets(crossed_steps, crossed_values)
            )

            detected_steps = [r['step'] for r in logs if r.get('freezing')]
            detected_values = [r[key] for r in logs if r.get('freezing')]
            item['detected'].set_offsets(
                self._offsets(detected_steps, detected_values)
            )

            pause_steps = [r['step'] for r in logs if r['step'] in paused_set]
            pause_values = [r[key] for r in logs if r['step'] in paused_set]
            item['paused'].set_offsets(self._offsets(pause_steps, pause_values))

            if thr is not None:
                threshold = thr[item['thr_key']]
                item['threshold'].set_ydata([threshold, threshold])
                item['threshold'].set_label(f"threshold={threshold:.6f}")
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
