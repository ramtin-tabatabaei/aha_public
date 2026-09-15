"""Shared locations. Set environment variables before starting a stage."""
import os
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIR.parents[1]

def location(name, default):
    return Path(os.environ.get(name) or default).expanduser().resolve()

OUTPUT_DIR = location('AHA_OUTPUT_ROOT', PROJECT_ROOT / 'outputs')
FAILGEN_ROOT = location('AHA_FAILGEN_ROOT', PROJECT_ROOT / 'external/rlbench-failgen')
RLBENCH_ROOT = location('RLBENCH_ROOT', PROJECT_ROOT / 'external/RLBench')
COPPELIASIM_ROOT = location('COPPELIASIM_ROOT', PROJECT_ROOT / 'external/CoppeliaSim')
TTM_CONTEXT_DIR = OUTPUT_DIR / 'ttm_context'
DESCRIPTION_DIR = OUTPUT_DIR / 'descriptions'
GRIPPER_DIR = OUTPUT_DIR / 'gripper_sequences'
BT_DIR = OUTPUT_DIR / 'behavior_trees'
CALIBRATION_DIR = location('AHA_CALIBRATION_ROOT', OUTPUT_DIR / 'calibration')
RUNS_DIR = OUTPUT_DIR / 'runs'
SCORES_DIR = OUTPUT_DIR / 'scores'
BASELINE_DIR = OUTPUT_DIR / 'baseline_grids'
CONFIGS_DIR = FAILGEN_ROOT / 'failgen/configs'
