"""
Run an RLBench task without injected failures and summarize gripper telemetry.

The statistics are computed only from rows where grip_force is above the
minimum filter value. By default that filter is 0.2, so idle/open-gripper
frames do not dominate the successful-run baseline.

Example:
  python detectors/grip_force_stats/run.py --task basketball_in_hoop --episodes 3
"""

from aha_publish import paths
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = str(paths.PROJECT_ROOT)
LOCAL_FAILGEN_PATH = str(paths.FAILGEN_ROOT)
SLIP_DETECTOR_PATH = str(paths.SOURCE_DIR / 'detectors' / 'slip')
CONFIGS_PATH = os.path.join(LOCAL_FAILGEN_PATH, "failgen/configs")
DEFAULT_COPPELIASIM_ROOT = str(paths.COPPELIASIM_ROOT)

for path in (SLIP_DETECTOR_PATH, LOCAL_FAILGEN_PATH, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)


DEFAULT_SAVE_DIR = str(paths.OUTPUT_DIR / 'grip_force_stats')
DEFAULT_MIN_GRIP_FORCE_FOR_STATS = 0.2
ENV_READY_FLAG = "AHA_GRIP_FORCE_STATS_ENV_READY"
METRICS = ("grip_force", "grip_force_drop")
STAT_KEYS = ("count", "mean", "std", "min", "max")


def configure_coppeliasim_env(headless=True):
    root = os.environ.get("COPPELIASIM_ROOT", DEFAULT_COPPELIASIM_ROOT)
    os.environ["COPPELIASIM_ROOT"] = root
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = root
    if headless:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    ld_parts = [part for part in current_ld_path.split(":") if part]
    if root not in ld_parts:
        ld_parts.append(root)
        os.environ["LD_LIBRARY_PATH"] = ":".join(ld_parts)
        return True
    return False


def ensure_coppeliasim_env_before_pyrep(headless=True):
    changed = configure_coppeliasim_env(headless=headless)
    if changed and os.environ.get(ENV_READY_FLAG) != "1":
        os.environ[ENV_READY_FLAG] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)


def summarize(values):
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def print_stats(title, metric_stats):
    print(f"\n{title}")
    print("-" * len(title))
    print(
        f"{'metric':<18} {'count':>8} {'mean':>12} {'std':>12} "
        f"{'min':>12} {'max':>12}"
    )
    for metric in METRICS:
        stats = metric_stats[metric]
        print(
            f"{metric:<18} {stats['count']:>8d} "
            f"{stats['mean']:>12.6f} {stats['std']:>12.6f} "
            f"{stats['min']:>12.6f} {stats['max']:>12.6f}"
        )


def available_tasks():
    tasks = []
    for filename in os.listdir(CONFIGS_PATH):
        if filename.endswith(".yaml"):
            tasks.append(filename[:-5])
    return sorted(tasks)


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def remove_empty_task_output_dir(save_dir, task_name):
    task_dir = save_dir / task_name
    try:
        task_dir.rmdir()
    except FileNotFoundError:
        return
    except OSError:
        return


def json_safe_stats(stats):
    out = {}
    for key in STAT_KEYS:
        if key not in stats:
            continue
        value = stats[key]
        if key == "count":
            out[key] = int(float(value))
        else:
            out[key] = float(value)
    return out


def stats_rows(task_name, scope, metric_stats):
    rows = []
    for metric, stats in metric_stats.items():
        row = {"task": task_name, "scope": scope, "metric": metric}
        row.update(stats)
        rows.append(row)
    return rows


def task_stats_json(
    task_name,
    episodes_requested,
    summary_rows,
    min_grip_force_for_stats,
):
    metric_data = {}
    for metric in METRICS:
        episode_rows = [
            row for row in summary_rows
            if row["metric"] == metric and row["scope"].startswith("episode_")
        ]
        episode_rows = [
            row for row in episode_rows if row["scope"] != "episode_stat_mean"
        ]
        episode_rows = sorted(
            episode_rows, key=lambda row: int(row["scope"].split("_")[1])
        )
        episodes = []
        for row in episode_rows:
            episodes.append({
                "episode": int(row["scope"].split("_")[1]),
                "stats": json_safe_stats(row),
            })

        average = next(
            (
                json_safe_stats(row) for row in summary_rows
                if row["metric"] == metric and row["scope"] == "episode_stat_mean"
            ),
            None,
        )
        aggregate = next(
            (
                json_safe_stats(row) for row in summary_rows
                if row["metric"] == metric and row["scope"] == "aggregate"
            ),
            None,
        )
        metric_data[metric] = {
            "episodes": episodes,
            "average_of_episode_stats": average,
            "aggregate_filtered_frames": aggregate,
        }

    completed = max(
        (len(metric_data[metric]["episodes"]) for metric in METRICS),
        default=0,
    )
    return {
        "task": task_name,
        "episodes_requested": episodes_requested,
        "episodes_completed": completed,
        "filter": {
            "description": (
                "Only rows with grip_force greater than "
                "min_grip_force_for_stats are included in statistics."
            ),
            "min_grip_force_for_stats": float(min_grip_force_for_stats),
        },
        "metrics": metric_data,
    }


def write_task_stats_json(
    save_dir,
    task_name,
    episodes_requested,
    summary_rows,
    min_grip_force_for_stats,
):
    data = task_stats_json(
        task_name,
        episodes_requested,
        summary_rows,
        min_grip_force_for_stats,
    )
    path = save_dir / f"{task_name}_success_grip_force_stats.json"
    write_json(path, data)
    return path


def mean_of_episode_stats(episode_stats):
    import numpy as np

    if not episode_stats:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": len(episode_stats),
        **{
            key: float(np.mean([stats[key] for stats in episode_stats]))
            for key in ("mean", "std", "min", "max")
        },
    }


def metric_stats_from_rows(rows):
    return {
        metric: summarize([row[metric] for row in rows])
        for metric in METRICS
    }


def collect_task_rows(args, task_name, save_dir, obs_to_row, update_deltas, env_cls):
    env_wrapper = env_cls(
        task_name=task_name,
        headless=not args.show_sim,
        record=False,
        save_data=False,
        no_failures=True,
        save_path=str(save_dir),
        save_keyframes_only=True,
        max_failure_attempts=args.max_attempts,
    )

    all_filtered_rows = []
    summary = []
    episode_metric_stats = {metric: [] for metric in METRICS}

    try:
        for episode in range(args.episodes):
            print(f"\n--- Episode {episode + 1} / {args.episodes} ---")
            logs = []
            step_counter = [0]
            original_step = env_wrapper.on_env_step

            def patched_step(obs):
                original_step(obs)
                step = step_counter[0]
                step_counter[0] += 1
                row = obs_to_row(obs, step)
                row["task"] = task_name
                row["episode"] = episode
                logs.append(row)
                update_deltas(logs)

            env_wrapper.on_env_step = patched_step
            env_wrapper.reset()

            try:
                demo = env_wrapper.get_success()
            finally:
                env_wrapper.on_env_step = original_step

            if demo is None:
                print("No successful demo was returned for this episode.")
                continue

            if not logs:
                for step, obs in enumerate(demo):
                    row = obs_to_row(obs, step)
                    row["task"] = task_name
                    row["episode"] = episode
                    logs.append(row)
                update_deltas(logs)

            filtered_rows = [
                row for row in logs
                if row["grip_force"] > args.min_grip_force_for_stats
            ]
            print(
                f"Filtered rows: {len(filtered_rows)} / {len(logs)} "
                f"(grip_force > {args.min_grip_force_for_stats})"
            )
            metric_stats = metric_stats_from_rows(filtered_rows)
            print_stats(f"Episode {episode} filtered stats", metric_stats)

            summary.extend(
                stats_rows(task_name, f"episode_{episode}", metric_stats)
            )
            for metric in METRICS:
                episode_metric_stats[metric].append(metric_stats[metric])
            all_filtered_rows.extend(filtered_rows)

        if all_filtered_rows:
            mean_stats = {
                metric: mean_of_episode_stats(episode_metric_stats[metric])
                for metric in METRICS
            }
            print_stats(f"{task_name} mean of episode stats", mean_stats)
            summary.extend(
                stats_rows(task_name, "episode_stat_mean", mean_stats)
            )

            aggregate_stats = metric_stats_from_rows(all_filtered_rows)
            print_stats(f"{task_name} aggregate filtered stats", aggregate_stats)
            summary.extend(
                stats_rows(task_name, "aggregate", aggregate_stats)
            )

            json_path = write_task_stats_json(
                save_dir,
                task_name,
                args.episodes,
                summary,
                args.min_grip_force_for_stats,
            )
            print(f"\nSaved JSON: {json_path}")
        else:
            print("\nNo filtered gripper telemetry was collected.")
    finally:
        env_wrapper.shutdown()
        remove_empty_task_output_dir(save_dir, task_name)

    return all_filtered_rows, summary


def read_task_stats_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def combine_aggregate_stats(stats_list):
    import math

    cleaned = [stats for stats in stats_list if stats and stats.get("count", 0) > 0]
    if not cleaned:
        return None

    total_count = sum(int(stats["count"]) for stats in cleaned)
    if total_count <= 0:
        return None

    mean = (
        sum(int(stats["count"]) * float(stats["mean"]) for stats in cleaned)
        / total_count
    )
    second_moment = (
        sum(
            int(stats["count"])
            * (float(stats["std"]) ** 2 + float(stats["mean"]) ** 2)
            for stats in cleaned
        )
        / total_count
    )
    variance = max(0.0, second_moment - mean ** 2)
    return {
        "count": total_count,
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(float(stats["min"]) for stats in cleaned),
        "max": max(float(stats["max"]) for stats in cleaned),
    }


def all_tasks_stats_json(save_dir, task_names, episodes_requested, failed_tasks):
    tasks = []
    completed_tasks = []
    for task_name in task_names:
        data = read_task_stats_json(
            save_dir / f"{task_name}_success_grip_force_stats.json"
        )
        if data is None:
            continue
        tasks.append(data)
        completed_tasks.append(task_name)

    metrics = {}
    for metric in METRICS:
        task_metric_stats = []
        for task in tasks:
            avg_stats = (
                task.get("metrics", {})
                .get(metric, {})
                .get("average_of_episode_stats")
            )
            if avg_stats is not None:
                task_metric_stats.append(avg_stats)

        if task_metric_stats:
            aggregate_stats = [
                task.get("metrics", {})
                .get(metric, {})
                .get("aggregate_filtered_frames")
                for task in tasks
            ]
            metrics[metric] = {
                "average_of_task_episode_averages": {
                    "count": len(task_metric_stats),
                    "mean": sum(row["mean"] for row in task_metric_stats)
                    / len(task_metric_stats),
                    "std": sum(row["std"] for row in task_metric_stats)
                    / len(task_metric_stats),
                    "min": sum(row["min"] for row in task_metric_stats)
                    / len(task_metric_stats),
                    "max": sum(row["max"] for row in task_metric_stats)
                    / len(task_metric_stats),
                },
                "aggregate_filtered_frames": combine_aggregate_stats(
                    aggregate_stats
                ),
            }
        else:
            metrics[metric] = {
                "average_of_task_episode_averages": None,
                "aggregate_filtered_frames": None,
            }

    return {
        "episodes_requested_per_task": episodes_requested,
        "tasks_requested": len(task_names),
        "tasks_completed": len(completed_tasks),
        "metrics": metrics,
        "tasks": tasks,
        "failed_tasks": [
            {"task": task_name, "error": error}
            for task_name, error in failed_tasks
        ],
    }


def task_is_completed(save_dir, task_name, episodes):
    json_path = save_dir / f"{task_name}_success_grip_force_stats.json"
    data = read_task_stats_json(json_path)
    if data is None:
        return False
    return int(data.get("episodes_completed", 0)) >= int(episodes)


def launch_parallel_task_batch(args, task_names, save_dir):
    logs_dir = save_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    configure_coppeliasim_env(headless=not args.show_sim)
    env.update(os.environ)

    queue = [
        task_name for task_name in task_names
        if not (
            args.skip_completed
            and task_is_completed(save_dir, task_name, args.episodes)
        )
    ]
    skipped = len(task_names) - len(queue)
    if skipped:
        print(f"Skipping {skipped} already-completed task(s).")
    total_to_run = len(queue)
    running = []
    failed_tasks = []
    completed = 0
    workers = max(1, int(args.workers))

    def start_task(task_name):
        log_path = logs_dir / f"{task_name}.log"
        log_file = open(log_path, "w")
        cmd = [
            sys.executable,
            str(paths.SOURCE_DIR / 'detectors/grip_force_stats/run.py'),
            "--task",
            task_name,
            "--episodes",
            str(args.episodes),
            "--max-attempts",
            str(args.max_attempts),
            "--save-dir",
            str(save_dir),
            "--min-grip-force-for-stats",
            str(args.min_grip_force_for_stats),
        ]
        if args.show_sim:
            cmd.append("--show-sim")
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=REPO_ROOT,
            env=env,
        )
        return {
            "task": task_name,
            "process": process,
            "log_file": log_file,
            "log_path": log_path,
            "started": time.monotonic(),
        }

    while queue and len(running) < workers:
        task_name = queue.pop(0)
        print(f"Starting {task_name}")
        running.append(start_task(task_name))

    while running:
        time.sleep(1.0)
        for item in list(running):
            return_code = item["process"].poll()
            if return_code is None:
                continue

            running.remove(item)
            item["log_file"].close()
            completed += 1
            elapsed = time.monotonic() - item["started"]
            task_name = item["task"]
            if return_code == 0:
                print(
                    f"Finished {task_name} "
                    f"({completed}/{total_to_run}) "
                    f"in {elapsed:.1f}s"
                )
            else:
                failed_tasks.append((
                    task_name,
                    f"exit={return_code}; log={item['log_path']}",
                ))
                print(
                    f"ERROR: {task_name} failed "
                    f"({completed}/{total_to_run}); "
                    f"see {item['log_path']}"
                )

            if queue:
                next_task = queue.pop(0)
                print(f"Starting {next_task}")
                running.append(start_task(next_task))

    return failed_tasks


def run(args):
    ensure_coppeliasim_env_before_pyrep(headless=not args.show_sim)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    task_names = available_tasks() if args.task == "all" else [args.task]
    if args.task == "all":
        print(
            f"Running {len(task_names)} tasks, {args.episodes} episode(s) each."
        )

    failed_tasks = []
    workers = max(1, int(args.workers))

    if workers > 1 and len(task_names) > 1:
        print(f"Running tasks in parallel with {workers} subprocess workers.")
        failed_tasks.extend(launch_parallel_task_batch(args, task_names, save_dir))
        completed_tasks = [
            task_name for task_name in task_names
            if read_task_stats_json(
                save_dir / f"{task_name}_success_grip_force_stats.json"
            )
        ]
        if completed_tasks:
            missing = sorted(set(task_names) - set(completed_tasks))
            for task_name in missing:
                if not any(task_name == item[0] for item in failed_tasks):
                    failed_tasks.append((task_name, "no completed JSON result"))
            all_json_path = save_dir / "ALL_TASKS_success_grip_force_stats.json"
            write_json(
                all_json_path,
                all_tasks_stats_json(
                    save_dir, task_names, args.episodes, failed_tasks
                ),
            )
            print(
                f"\nSaved all-task JSON: {all_json_path} "
                f"({len(completed_tasks)} completed tasks)"
            )
    else:
        from detector import obs_to_row, update_deltas
        from failgen.env_wrapper import FailGenEnvWrapper

        for task_index, task_name in enumerate(task_names, start=1):
            if args.skip_completed and task_is_completed(
                save_dir, task_name, args.episodes
            ):
                print(f"Skipping completed task: {task_name}")
                continue
            if len(task_names) > 1:
                print(
                    f"\n{'=' * 80}\n"
                    f"Task {task_index} / {len(task_names)}: {task_name}\n"
                    f"{'=' * 80}"
                )
            try:
                collect_task_rows(
                    args,
                    task_name,
                    save_dir,
                    obs_to_row,
                    update_deltas,
                    FailGenEnvWrapper,
                )
            except Exception as exc:
                failed_tasks.append((task_name, str(exc)))
                print(f"\nERROR: task '{task_name}' failed: {exc}")
                continue

        if args.task == "all":
            completed_tasks = [
                task_name for task_name in task_names
                if read_task_stats_json(
                    save_dir / f"{task_name}_success_grip_force_stats.json"
                )
            ]
            if completed_tasks:
                missing = sorted(set(task_names) - set(completed_tasks))
                for task_name in missing:
                    if not any(task_name == item[0] for item in failed_tasks):
                        failed_tasks.append((task_name, "no completed JSON result"))
                all_json_path = save_dir / "ALL_TASKS_success_grip_force_stats.json"
                write_json(
                    all_json_path,
                    all_tasks_stats_json(
                        save_dir, task_names, args.episodes, failed_tasks
                    ),
                )
                print(
                    f"\nSaved all-task JSON: {all_json_path} "
                    f"({len(completed_tasks)} completed tasks)"
                )

    if failed_tasks:
        failed_path = save_dir / "failed_tasks.json"
        write_json(
            failed_path,
            [
                {"task": task_name, "error": error}
                for task_name, error in failed_tasks
            ],
        )
        print(f"\nFailed tasks: {len(failed_tasks)}. Saved: {failed_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        required=True,
        help="RLBench task name, or 'all' to run every failgen config task.",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument(
        "--min-grip-force-for-stats",
        type=float,
        default=DEFAULT_MIN_GRIP_FORCE_FOR_STATS,
        help=(
            "Only include rows with grip_force above this value in stats. "
            "Default: 0.2."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of tasks to run in parallel. Use 2-4 carefully; each worker "
            "launches its own CoppeliaSim instance."
        ),
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip tasks that already have a JSON result with enough episodes.",
    )
    parser.add_argument(
        "--show-sim",
        action="store_true",
        help="Run with the simulator window visible instead of headless.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
