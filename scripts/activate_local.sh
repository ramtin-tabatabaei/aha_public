#!/usr/bin/env bash
# Usage: source scripts/activate_local.sh
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo 'Use: source scripts/activate_local.sh' >&2
    exit 2
fi

aha_activate_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "$aha_activate_root/.conda/aha-publish/bin/python" ]]; then
    echo 'Create the environment first: bash scripts/create_local_env.sh' >&2
    unset aha_activate_root
    return 2
fi
if ! declare -F conda >/dev/null; then
    aha_conda_base="$(conda info --base)" || return
    source "$aha_conda_base/etc/profile.d/conda.sh" || return
    unset aha_conda_base
fi
conda activate "$aha_activate_root/.conda/aha-publish" || return
source "$aha_activate_root/scripts/local_paths.sh"
unset aha_activate_root
