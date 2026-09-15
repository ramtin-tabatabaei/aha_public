#!/usr/bin/env bash
# Shared by the installer and activation script; paths follow this checkout.
aha_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export COPPELIASIM_ROOT="$aha_project_root/external/CoppeliaSim"
export RLBENCH_ROOT="$aha_project_root/external/RLBench"
export AHA_FAILGEN_ROOT="$aha_project_root/external/rlbench-failgen"
export AHA_OUTPUT_ROOT="$aha_project_root/outputs"
export AHA_CALIBRATION_ROOT="$AHA_OUTPUT_ROOT/calibration"
export PYTHONPATH="$aha_project_root/src:$AHA_FAILGEN_ROOT:$RLBENCH_ROOT"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export CONDA_PKGS_DIRS="$aha_project_root/.conda/pkgs"
export CONDA_ENVS_PATH="$aha_project_root/.conda"
export PIP_CACHE_DIR="$aha_project_root/.cache/pip"
export XDG_CACHE_HOME="$aha_project_root/.cache"
export MPLCONFIGDIR="$aha_project_root/.cache/matplotlib"
export TMPDIR="$aha_project_root/outputs/tmp"
unset CONDITION_RULES_PATH BT_TASK_CONTEXT_PATH
unset AHA_WP_CHAIN_DIR AHA_RLBENCH_TASKS_DIR
unset AHA_TORQUE_STATS_DIR AHA_RESIDUAL_STATS_DIR
unset AHA_TRANSITION_STATS_DIR AHA_ORIENTATION_STATS_DIR AHA_GRIP_FORCE_STATS_PATH
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$MPLCONFIGDIR"
unset aha_project_root
