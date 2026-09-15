"""Shared detector-VLM settings used by the batch/eval runners.

Keep these defaults aligned with ``eval_detectors_vlm.py`` so slip/collision
confirmation uses the same prompt path and image input everywhere.
"""

from aha_publish import paths
from pathlib import Path
import os

HERE = (paths.SOURCE_DIR / 'running')
ROOT = (paths.PROJECT_ROOT)

DEFAULT_DETECTOR_VLM_CAMERAS = "side_rgb,wrist_rgb,front_rgb,overhead_rgb"
DEFAULT_DETECTOR_VLM_MODEL = os.getenv(
    "OPENAI_DETECTOR_VLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-5.4")
)
DEFAULT_CONFIRM_DETECTORS = "slip,collision"


def detector_vlm_cameras(value=None):
    value = (value or "").strip()
    if value.lower() == "default":
        return ""
    return value or DEFAULT_DETECTOR_VLM_CAMERAS


def detector_vlm_model(value=None):
    return (value or "").strip() or DEFAULT_DETECTOR_VLM_MODEL


def apply_detector_vlm_env(
    env,
    *,
    cameras=None,
    confirm_detectors=None,
    auto=True,
    immediate=True,
    collision_method=True,
    prefer_local_openai_key=True,
):
    """Apply the detector_vlm_eval-compatible runtime env to ``env`` in place."""
    if auto:
        env["AHA_DETECTOR_VLM_AUTO"] = "1"
    if confirm_detectors is not None:
        env["AHA_VLM_CONFIRM_DETECTORS"] = confirm_detectors
    cams = detector_vlm_cameras(cameras)
    if cams:
        env["AHA_VLM_CONFIRM_CAMERAS"] = cams
    if immediate:
        env["AHA_VLM_CONFIRM_DELAY_FRAMES"] = "0"
    if collision_method:
        env.setdefault("AHA_COLLISION_METHOD", "3")
    return env
