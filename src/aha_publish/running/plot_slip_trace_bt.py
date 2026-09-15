"""Record and plot slip-detector telemetry for one task run via the BT main runner.

Unlike plot_slip_trace.py (which runs FailGenEnvWrapper in-process), this script
launches waypoints_interactive_bt_conditions.py as a subprocess — the same path
the real BT pipeline takes — and captures per-step slip telemetry via the
AHA_SLIP_DRIFT_LOG env var written by the embedded slip LiveDetector.

Outputs (same format as plot_slip_trace.py):
  aha_output/slip_plots_bt/<task>.clean.csv/png
  aha_output/slip_plots_bt/<task>.<failtype>.wp<wp>.csv/png

Examples:
  python aha_scripts/main_bt_run/plot_slip_trace_bt.py \
      --task change_clock --failure grasp --waypoint 1
  python aha_scripts/main_bt_run/plot_slip_trace_bt.py \
      --task change_clock --failure none
"""

from aha_publish import paths

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = (paths.SOURCE_DIR / 'running')
ROOT = (paths.PROJECT_ROOT)
FAILGEN_ROOT = (paths.FAILGEN_ROOT)
CONFIGS_PATH = FAILGEN_ROOT / "failgen" / "configs"
OUT_DIR = (paths.OUTPUT_DIR / 'slip_plots_bt')
RUNNER = (paths.SOURCE_DIR / 'running/waypoints_interactive_bt_conditions.py')
COPPELIA = paths.COPPELIASIM_ROOT

DEFAULT_AHA_PY = Path(sys.executable)
PY = os.environ.get(
    "AHA_PYTHON",
    str(DEFAULT_AHA_PY) if DEFAULT_AHA_PY.exists() else sys.executable,
)

for path in (str(FAILGEN_ROOT), str(paths.PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

# Reuse the config/failure helpers from plot_slip_trace.py
from aha_publish.running.plot_slip_trace import load_config, resolve_failure, plot_trace
from aha_publish.common.tasks import available_tasks, choose_task

SLIP_FAILTYPES = ("slip", "grasp")


def _base_env(drift_log_path, grip_force_n=15.0):
    env = dict(os.environ)
    env["COPPELIASIM_ROOT"] = str(COPPELIA)
    env["LD_LIBRARY_PATH"] = env.get("LD_LIBRARY_PATH", "") + ":" + str(COPPELIA)
    env["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(COPPELIA)
    env.pop("QT_QPA_PLATFORM", None)
    env.setdefault("DISPLAY", ":1")
    env["AHA_SLIP_DRIFT_LOG"] = drift_log_path
    env["AHA_GRIP_FORCE_N"] = str(grip_force_n)
    env["AHA_FAIL_DEBUG"] = "1"
    env["AHA_FAIL_EXTREME"] = "1"
    # Disable auto VLM confirm so the plot shows raw detector signal
    env.pop("AHA_DETECTOR_VLM_AUTO", None)
    env["AHA_VLM_CONFIRM_DETECTORS"] = ""
    return env


def run_bt(task, failtype, waypoint, *, headless=True, timeout=1800, grip_force_n=5.0):
    drift_log = tempfile.mktemp(prefix="aha_slip_drift_", suffix=".csv")
    headless_flags = ["--headless"] if headless else []
    cmd = [
        PY, "-u", str(RUNNER),
        "--task", task,
        "--failure", failtype,
        "--mode", "auto",
        "--vlm-checks", "off",
        "--no-side-camera-window",
        "--no-detector-plots",
        *headless_flags,
    ]
    if failtype != "none" and waypoint is not None:
        cmd += ["--failure-waypoint", str(waypoint)]

    env = _base_env(drift_log, grip_force_n=grip_force_n)
    print(f"  running: {' '.join(cmd[-6:])!r}  [drift_log={drift_log}]")
    try:
        p = subprocess.run(
            cmd, env=env, stdin=subprocess.DEVNULL, cwd=str(paths.PROJECT_ROOT),
            capture_output=True, text=True, timeout=timeout,
        )
        out = (p.stdout or "") + "\n" + (p.stderr or "")
        success = p.returncode == 0 or "Live detector summary" in out
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + "\n" + (e.stderr or "")
        success = False
        print("  TIMEOUT")

    fired_steps = _parse_fired_steps(out, waypoint)
    rows = _read_drift_csv(drift_log)
    if os.path.exists(drift_log):
        os.remove(drift_log)
    return rows, success, fired_steps, out


def _parse_fired_steps(out, waypoint):
    """Extract global step numbers where [DETECTOR] SLIP fired."""
    import re
    fired = []
    for m in re.finditer(
        r"\[DETECTOR\] SLIP (?:failure|detected) at waypoint \d+ \(step (\d+)\)", out
    ):
        fired.append(int(m.group(1)))
    return fired


def _read_drift_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    # fix types for plot_trace()
    for r in rows:
        r["holding_required_phase"] = bool(int(r.get("holding_required_phase", 1)))
        r["path_done"] = int(r.get("path_done", 0))
        r["slip_fired"] = bool(int(r.get("slip_fired", 0)))
        r["step"] = int(r.get("step", 0))
        r["waypoint"] = r.get("waypoint", "")
    return rows


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=None)
    parser.add_argument("--failure", choices=("none",) + SLIP_FAILTYPES, default=None)
    parser.add_argument("--waypoint", type=int)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--show-gui", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    parser.add_argument("--grip-force", type=float, default=None,
                        help="gripper squeeze force in N (default 15; 0 = off)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--show", action="store_true",
                        help="open the plot after saving")
    args = parser.parse_args()

    if not args.task:
        tasks = available_tasks()
        args.task = choose_task(tasks)
        if not args.task:
            print("No task chosen; exiting.")
            sys.exit(0)

    config = load_config(args.task)

    # If multiple waypoints are configured and none was specified, ask.
    if args.failure != "none" and args.waypoint is None:
        import yaml as _yaml
        failures = [f for f in config.get("failures", [])
                    if f.get("type") in (args.failure,) + (SLIP_FAILTYPES if args.failure is None else ())]
        if not failures and args.failure is None:
            failures = [f for f in config.get("failures", [])
                        if f.get("type") in SLIP_FAILTYPES]
        wps = []
        for f in failures:
            wps.extend(int(w) for w in f.get("waypoints", []))
        wps = sorted(set(wps))
        if len(wps) > 1:
            print(f"Slip waypoints available: {wps}")
            while True:
                try:
                    raw = input(f"Choose waypoint {wps}: ").strip()
                except EOFError:
                    raw = str(wps[0])
                if raw.isdigit() and int(raw) in wps:
                    args.waypoint = int(raw)
                    break
                print(f"  Enter one of {wps}")

    failtype, waypoint, _failure_cfg = resolve_failure(config, args.failure, args.waypoint)

    if args.grip_force is not None:
        grip_force_n = args.grip_force
    else:
        grip_force_n = float(config.get("grip_force_n",
                             os.environ.get("AHA_GRIP_FORCE_N", "5")))
    print(f"  grip_force_n={grip_force_n} N (from "
          f"{'--grip-force' if args.grip_force is not None else 'config/env'})")
    rows, success, fired_steps, raw_out = run_bt(
        args.task, failtype, waypoint,
        headless=args.headless, timeout=args.timeout,
        grip_force_n=grip_force_n,
    )

    if not rows:
        print("No slip telemetry rows recorded — check AHA_SLIP_DRIFT_LOG and stderr:")
        print(raw_out[-2000:])
        sys.exit(1)

    if failtype == "none":
        stem = f"{args.task}.clean"
        label = "clean"
    else:
        stem = f"{args.task}.{failtype}.wp{waypoint}"
        label = f"{failtype}@wp{waypoint}"

    # fired_steps from drift CSV (step column where slip_fired=True)
    csv_fired = [int(r["step"]) for r in rows if r.get("slip_fired")]
    all_fired = sorted(set(fired_steps + csv_fired))

    csv_path = args.out_dir / f"{stem}.csv"
    png_path = args.out_dir / f"{stem}.png"
    write_csv(csv_path, rows)
    plot_trace(png_path, rows, args.task, label, all_fired)

    print(f"task={args.task} failure={failtype} waypoint={waypoint} success={success}")
    print(f"frames={len(rows)} slip_fired={bool(all_fired)} fired_steps={all_fired[:10]}")
    print(f"csv  -> {csv_path}")
    print(f"plot -> {png_path}")
    if args.show:
        subprocess.Popen(["xdg-open", str(png_path)])


if __name__ == "__main__":
    main()
