#!/usr/bin/env bash
# Create a fresh environment, then build the copied simulator bindings in it.
set -euo pipefail
aha_install_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$aha_install_root"
for required in external/PyRep/setup.py external/RLBench/setup.py \
    external/rlbench-failgen/setup.py external/CoppeliaSim/libcoppeliaSim.so; do
    if [[ ! -f "$required" ]]; then
        echo "Missing local dependency: $required (see docs/local_setup.md)" >&2
        exit 2
    fi
done
source scripts/local_paths.sh
aha_install_prefix="$aha_install_root/.conda/aha-publish"
if [[ ! -x "$aha_install_prefix/bin/python" ]]; then
    conda create --prefix "$aha_install_prefix" --copy --yes python=3.10 pip setuptools wheel
fi
"$aha_install_prefix/bin/python" -m pip install -r requirements-simulator.txt
"$aha_install_prefix/bin/python" -m pip install --no-build-isolation --no-deps \
    -e external/PyRep -e external/RLBench -e external/rlbench-failgen
"$aha_install_prefix/bin/python" -m pip check
"$aha_install_prefix/bin/python" scripts/check_local_env.py
echo 'Ready. Activate with: source scripts/activate_local.sh'
