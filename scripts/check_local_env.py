#!/usr/bin/env python3
"""Verify interpreter, imports, and simulator assets belong to this checkout."""
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def require_local(path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise RuntimeError(f"Outside this project: {resolved}")
    if not resolved.exists():
        raise RuntimeError(f"Missing local dependency: {resolved}")
    return resolved.relative_to(ROOT)


def main():
    print(f"Python: {require_local(sys.executable)}")
    for name in ("aha_publish", "numpy", "PIL", "yaml", "py_trees", "matplotlib",
                 "openai", "pydantic", "pyrep", "pyrep.backend._sim_cffi", "rlbench", "failgen",
                 "failgen.env_wrapper"):
        module = importlib.import_module(name)
        print(f"{name}: {require_local(module.__file__)}")

    from aha_publish import paths
    for path in (paths.FAILGEN_ROOT / "failgen/configs/basketball_in_hoop.yaml",
                 paths.RLBENCH_ROOT / "rlbench/task_ttms/basketball_in_hoop.ttm",
                 paths.RLBENCH_ROOT / "rlbench/task_design.ttt",
                 paths.COPPELIASIM_ROOT / "libcoppeliaSim.so.1"):
        print(f"Asset: {require_local(path)}")
    print("Local environment checks passed.")


if __name__ == "__main__":
    main()
