# Local simulator environment

The working folder contains these independent copies:

| Directory | Contents |
| --- | --- |
| `external/CoppeliaSim` | CoppeliaSim Edu 4.1 for Ubuntu 20.04, including shared libraries and plugins |
| `external/PyRep` | PyRep source, CFFI build files, and robot models |
| `external/RLBench` | Modified RLBench source, task scenes, robot models, and assets |
| `external/rlbench-failgen` | AHA's modified failure-generation source and task configurations |

Git metadata, old Python build products, bytecode, and generated research
outputs were omitted. The simulator's `libcoppeliaSim.so.1` link points to its
local `libcoppeliaSim.so`. Backend configuration output paths use
`AHA_OUTPUT_ROOT/backend_data`; workflow scratch outputs also stay in `outputs/`.
Original license files and notices are retained with the dependencies.

## Create and activate

From the project root, with Conda available:

```bash
bash scripts/create_local_env.sh
source scripts/activate_local.sh
python scripts/check_local_env.py
```

This creates `.conda/aha-publish` with Python 3.10 from Conda packages, installs
`requirements-simulator.txt`, then installs the three copied Python backends in
editable mode. It builds PyRep's native extension against `external/CoppeliaSim`.
Downloads require network access. A C compiler is needed to build PyRep.
The existing `aha` environment is not cloned or modified.

Run `source scripts/activate_local.sh` in every new terminal. It selects the
local environment and resets Python, simulator, model, calibration, cache, and
temporary-output paths to this project. It disables user-site Python packages.
API credentials remain environment variables; no key files are copied.

## Verify

```bash
python scripts/check_local_env.py
python -m pip check
python -m unittest discover -s tests
python 01_ttm_context/main.py --task basketball_in_hoop
```

The check imports the actual backend and verifies that Python, imported
packages, and required task/simulator assets resolve inside the project.
The last command launches the simulator and generates
`outputs/ttm_context/basketball_in_hoop.llm_context.json` without an API call.
Later description and behavior-tree generation stages need provider credentials.

Validated in this fresh environment: all 39 regression tests, dependency/import
checks, and the actual `basketball_in_hoop` task inspection, waypoint screenshots,
combined image grid, and four-waypoint gripper replay. Generated artifacts are
under `outputs/`. Paid description/BT generation and a full detector evaluation
were not run as part of this setup check.

The host still supplies Linux, its standard libraries, graphics drivers, and an
X/OpenGL display (including for headless simulator operation). `DISPLAY` is
preserved by activation. Optional wrist-view SAM segmentation requires its own
Torch/SAM installation and model; it is not needed by the standard workflow.

## Copying this folder

`external/`, `.conda/`, `.cache/`, and `outputs/` are ignored by Git. The dependency
copies exist on disk, so include `external/` in a filesystem copy or archive.
A Git clone alone does not include them. Keep their licenses with the files.

Conda prefixes and editable installations contain absolute paths. After moving
the project, omit the old `.conda/` and recreate it with the installer at the
new location. The activation scripts derive all project paths from their own
location.
