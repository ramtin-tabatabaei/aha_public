"""Configuration and shared imports for BT maker."""

from aha_publish import paths

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from functools import lru_cache
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse
import webbrowser

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

SCRIPT_DIR = (paths.SOURCE_DIR / 'behavior_trees')

AHA_SCRIPTS_DIR = (paths.SOURCE_DIR)
sys.path.insert(0, str(paths.SOURCE_DIR))
from aha_publish.common.task_categories import print_task_category, is_two_object_interaction, is_three_object_interaction, is_five_interaction

PROJECT_ROOT = (paths.PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Configure these values, then run:
#     python aha_scripts/BT_maker.py
# ---------------------------------------------------------------------------
DEFAULT_PROVIDER = "openai"  # "openai" or "claude"

DEFAULT_TASK_CONTEXT_PATH = Path(
    os.environ.get(
        "BT_TASK_CONTEXT_PATH",
        str(paths.DESCRIPTION_DIR / 'task_context.json'),
    )
)

DEFAULT_HOST = "127.0.0.1"

DEFAULT_PORT = 8765

PREPARED_BTS_DIR = paths.BT_DIR

AUTO_OPEN_BROWSER = True

AUTO_GENERATE_ON_LOAD = True

# Agent 2 (reviewer) models.
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4")

# Agent 1 (generation) models — decoupled from the reviewer so the generator can
# run a different model. Default to the reviewer model unless overridden.
GENERATOR_CLAUDE_MODEL = os.environ.get("GENERATOR_CLAUDE_MODEL", CLAUDE_MODEL)

GENERATOR_OPENAI_MODEL = os.environ.get("GENERATOR_OPENAI_MODEL", "gpt-5.4")

CONDITION_MAX_OUTPUT_TOKENS = int(os.environ.get("BT_CONDITION_MAX_TOKENS", "16000"))

OPENAI_REQUIRED_PACKAGES = (
    "openai",
    "anyio",
    "distro",
    "httpx",
    "idna",
    "jiter",
    "pydantic",
    "sniffio",
    "tqdm",
    "typing_extensions",
)
