"""Configuration, imports, and shared data structures for task description generation."""

from __future__ import annotations

from aha_publish import paths

import argparse
import base64
from dataclasses import dataclass
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = (paths.PROJECT_ROOT)
sys.path.insert(0, str(paths.SOURCE_DIR))
from aha_publish.common.task_categories import print_task_category
DEFAULT_PHOTO_DIR = (paths.DESCRIPTION_DIR)
DEFAULT_FAILGEN_CONFIG_DIR = (
    (paths.FAILGEN_ROOT / 'failgen/configs')
)
DEFAULT_GRIPPER_SEQUENCE_DIR = (paths.GRIPPER_DIR)

DEFAULT_TTM_CONTEXT_DIR = (paths.TTM_CONTEXT_DIR)


PROVIDER = "openai"
OPENAI_MODEL = "gpt-5.4"
CLAUDE_MODEL = "claude-opus-4-7"

# Object-naming mode for key_scene_objects / waypoint object references.
#   "original"    — copy the exact simulator/TTM names verbatim (default).
#   "descriptive" — name objects for what they visibly are and do in the task.
# Override the default with $AHA_NAMING_MODE, or per-run with the --naming flag.
DEFAULT_NAMING_MODE = os.environ.get("AHA_NAMING_MODE", "descriptive").strip().lower()

SUPPORTED_EXTENSIONS = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

MAX_CHARS_PER_TEXT_FILE = 60000
TEXT_CONTEXT_EXTENSIONS = {
    ".cfg",
    ".csv",
    ".json",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}


@dataclass
class PathEntry:
    label: str
    path: Path


@dataclass
class Usage:
    """Token usage for one or more API calls, with optional cost."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.calls + other.calls,
        )

    def cost(
        self,
        input_price_per_1m: float | None,
        output_price_per_1m: float | None,
    ) -> float | None:
        if input_price_per_1m is None or output_price_per_1m is None:
            return None
        return (
            self.input_tokens / 1_000_000 * input_price_per_1m
            + self.output_tokens / 1_000_000 * output_price_per_1m
        )


@dataclass
class TaskInputs:
    task_name: str
    simulator: str
    image_paths: list[PathEntry]
    scene_context_paths: list[PathEntry]
    task_context_paths: list[PathEntry]
    output_paths: list[PathEntry]
    gripper_sequence_path: Path | None = None

    def first_path(self, group: list[PathEntry], label: str | None = None) -> Path | None:
        if label is not None:
            for entry in group:
                if entry.label == label:
                    return entry.path
        return group[0].path if group else None

    @property
    def grid_image_path(self) -> Path:
        path = self.first_path(self.image_paths, "combined_grid")
        if path is None:
            raise ValueError("No image paths configured.")
        return path

    @property
    def output_json_path(self) -> Path:
        path = self.first_path(self.output_paths, "multimodal_analysis_json")
        if path is None:
            raise ValueError("No output JSON path configured.")
        return path
