"""Per-detector module loader for the bt_gui-embedded detectors.

All five detector folders ship files with the SAME names (detector.py, plot.py,
vlm_confirm.py), so importing them by plain name would collide. This loads each
detector's files by path under namespaced module names. ``plot.py`` does
``from detector import WARMUP_STEPS`` at import time, so we temporarily bind this
detector's already-loaded module as ``detector`` while plot.py executes, and put
the detector folder on ``sys.path`` so its internal lookups resolve — exactly the
environment the standalone interactive.py runs in.
"""

from aha_publish import paths

import importlib.util
import os
import sys
from pathlib import Path

PROJECT_ROOT = (paths.PROJECT_ROOT)
DETECTORS_DIR = (paths.SOURCE_DIR / 'detectors')

# When set, a live-detector detection is confirmed with the detector's VLM
# verifier automatically (no y/N prompt, no pause), and a clear VLM "no
# failure" verdict retracts the detection. YES or UNKNOWN keeps it.
VLM_AUTO = os.getenv(
    "AHA_DETECTOR_VLM_AUTO", "").strip().lower() in ("1", "true", "yes", "on")


def _force_fire_waypoint():
    """Eval-only knob: force a detector to fire at this waypoint regardless of
    telemetry, so a *clean* run produces a deliberate (wrong) detection whose VLM
    confirmation can be measured. Unset -> no forced firing (production default)."""
    raw = os.getenv("AHA_FORCE_FIRE_WAYPOINT", "").strip()
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


FORCE_FIRE_WAYPOINT = _force_fire_waypoint()
FORCE_FIRE_REASON = (
    "FORCED eval false-positive (clean scene; no real failure injected)")


def _exec_module(modname, file_path, inject=None):
    spec = importlib.util.spec_from_file_location(modname, str(file_path))
    module = importlib.util.module_from_spec(spec)
    saved_mods = {}
    if inject:
        for key, mod in inject.items():
            saved_mods[key] = sys.modules.get(key)
            sys.modules[key] = mod
    saved_path = list(sys.path)
    sys.path.insert(0, str(Path(file_path).parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved_path
        for key, old in saved_mods.items():
            if old is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = old
    return module


class DetectorBundle:
    def __init__(self, name):
        self.name = name
        folder = DETECTORS_DIR / name
        self.detector = _exec_module(
            f"aha_live_detector_{name}", folder / "detector.py"
        )
        plot_path = folder / "plot.py"
        self.plot = (
            _exec_module(
                f"aha_live_plot_{name}", plot_path, inject={"detector": self.detector}
            )
            if plot_path.exists()
            else None
        )
        vlm_path = folder / "vlm_confirm.py"
        self.vlm = (
            _exec_module(f"aha_live_vlm_{name}", vlm_path)
            if vlm_path.exists()
            else None
        )


_CACHE = {}


def load_detector_bundle(name):
    if name not in _CACHE:
        _CACHE[name] = DetectorBundle(name)
    return _CACHE[name]


def log_detector_vlm_event(detector, waypoint, verdict, result, step=None):
    """Record one detector fire-verification VLM event to the per-run CSV, if a
    RunLogger is active (no-op otherwise). `result` is the vlm_confirm output
    dict (carries explanation + saved grid path)."""
    try:
        import aha_publish.running.vlm_run_logger as vlm_run_logger
        lg = vlm_run_logger.active()
        if lg is None:
            return
        result = result or {}
        lg.log(
            channel="detector",
            event_kind=detector,
            waypoint=waypoint,
            verdict=verdict,
            explanation=str(result.get("explanation", "")),
            image_path=result.get("grid_path") or result.get("trace_path") or "",
            detection_frame=step,
        )
    except Exception as exc:
        # Never let a logging failure kill the run, but do NOT hide it: a
        # silently dropped detector row makes the batch summary score the case
        # vlm=N/A even though the VLM confirmed the failure in the run log.
        print(f"  [vlm-log] FAILED to record {detector} verdict "
              f"@wp{waypoint}: {type(exc).__name__}: {exc}", flush=True)
