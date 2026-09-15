#!/usr/bin/env python3
"""Detection rate and delay per method and failure type, stated as sentences.

    A1: Condition VLM detects 16.7% of collision episodes (4 of 24),
        delay mean 89.1, std 34.2 steps.

Reads the same two files score_detection_samples.py does -- detection_samples.csv
and its _frames.csv sidecar -- so nothing has to be re-simulated.

What counts as a detection
--------------------------
The unit here is the EPISODE, not the checkpoint: "detects X% of collisions"
means X% of the collision episodes raised an alarm, which is the question this
report answers.  An episode counts as detected when the method alarms at any
checkpoint at or after the injection waypoint.

That is deliberately the same rule the frames sidecar uses to pick an alarm
frame, so the rate and the delay describe ONE set of episodes.  score_detection
_samples.py's `--window to-injection` is stricter -- it credits only an alarm
exactly at the injection waypoint -- so its recall is lower than the rate here,
and its Lat. column is averaged over a different (larger) set than its recall.
Here they always agree, which is what makes "X% with delay mean y" a sentence
about one population.

Delay: waypoints or frames
--------------------------
Two units, `--delay waypoint` (the DEFAULT), `frame`, or `both`.

WAYPOINT delay is how many execution checkpoints pass between the injection
waypoint and the one the method alarms at -- 0 means it caught the failure at
the very waypoint it was injected.  It comes straight from the sample table, it
is the unit a reactive BT actually acts in (it stops at a checkpoint), and it is
sound for every method.

FRAME delay is control steps, read from the sidecar.  It is finer, but it is
only a real measurement for the DETECTOR-driven methods: vlm_events.csv carries
a detection_frame on detector rows and never on condition rows, so A1's and B2's
alarm frames are filled in from the END of the waypoint's step span.  Their frame
delays are therefore waypoint boundaries with decimals attached -- they do not
say when inside the waypoint the VLM noticed, because nothing recorded that.
A4 mixes the two: it takes whichever channel fired first.  So compare frame
delays among A2/A3 freely, and never read a frame gap between A1/B2 and a
detector method as a real latency difference -- use the waypoint column there.

Either way the delay is over the detected episodes only -- a miss has no delay --
and `n` is how many went into the mean.  For frames `n` can be below the detected
count when the sidecar has no alarm frame for an episode; those are reported at
the end rather than silently dropped.  std is the sample standard deviation and
needs at least two episodes.

Blind combinations
------------------
A method with no way to express a failure type -- the condition VLM has no
collision predicate, the detector bank has no grasp detector -- is marked
(blind).  It can still detect through downstream effects, so the rate is real;
it just cannot name what it found.

Usage
-----
    python aha_scripts/report_detection_delay.py
    python aha_scripts/report_detection_delay.py --methods A1 A4
    python aha_scripts/report_detection_delay.py --failures collision Slip
    python aha_scripts/report_detection_delay.py --table
"""

from __future__ import annotations

from aha_publish import paths

import argparse
import csv
import statistics
import sys
from pathlib import Path

from aha_publish.scoring.score_detection_samples import BLIND, DEFAULT_SAMPLES, FAILURE_ROWS, METHOD_LABELS, METHODS, _int, is_alarm, load_frames, load_samples, print_table

DEFAULT_OUT = DEFAULT_SAMPLES.with_name("detection_delay.csv")
FAILURE_LABELS = dict(FAILURE_ROWS)


def episode_detected(episode: dict, method: str, unc: bool) -> bool | None:
    """True/False for a failure episode; None when the method has no result."""
    preds = episode["preds"][method]
    if preds is None:
        return None
    return any(is_alarm(pred, unc)
               for (verdict, _), pred in zip(episode["gt"], preds)
               if verdict == "failure")


# vlm_events.csv has a detection_frame on detector rows only, so these methods'
# alarm frames are filled in from the end of the waypoint's step span rather than
# measured.  Their frame delays quantize to waypoint boundaries.
FRAME_IS_BOUNDARY = ("B2", "A1")


def first_crediting_alarm(episode: dict, method: str,
                          unc: bool) -> tuple[int, str | None] | None:
    """The first alarm at or after injection, as (waypoint delay, named type).

    A waypoint delay of 0 means the method alarmed at the very waypoint the
    failure was injected at.  Straight from the sample table -- no sidecar, no
    frame reconstruction.  The FIRST alarm is the one that matters: a reactive
    system stops there, so its type is the diagnosis the system would report.
    """
    preds = episode["preds"][method]
    if preds is None:
        return None
    flags = [verdict == "failure" for verdict, _ in episode["gt"]]
    if not any(flags):
        return None                                   # clean run: no injection
    injected = episode["waypoints"][flags.index(True)]
    for waypoint, failed, pred in zip(episode["waypoints"], flags, preds):
        if failed and is_alarm(pred, unc):
            return waypoint - injected, pred[1]
    return None


def names_type(named: str | None, truth: str) -> bool:
    """Whether the alarm's label covers the true type.

    A fused label ('Wrongtransition|Grasp') counts when the truth is among the
    names it lists -- the same convention score_detection_samples.py uses.  An
    alarm with no type at all ('uncertain') never counts.
    """
    return named is not None and truth in str(named).split("|")


def episode_delay(episode: dict, frames: dict, method: str) -> int | None:
    row = frames.get((episode["task"], episode["case"]))
    if not row:
        return None
    alarm = _int(row.get(f"{method}_alarm_frame"))
    injection = _int(row.get("injection_frame"))
    if alarm is None or injection is None:
        return None
    return max(0, alarm - injection)


def summarise(episodes: list[dict], method: str, truth: str, frames: dict,
              unc: bool) -> dict:
    """Detection rates over `episodes`, plus the delay spread of the ones caught.

    Two rates, because raising an alarm and knowing what went wrong are separate
    abilities.  ANY is binary -- the method flagged the episode as a failure,
    whatever it called it.  TYPED is the subset whose first alarm also named the
    right failure, which is what a system needs to react correctly rather than
    merely stop.  A structurally blind method scores on the first and zero on the
    second, which is exactly the distinction the pair is there to show.
    """
    covered = [e for e in episodes if e["preds"][method] is not None]
    caught, typed = [], []
    frame_delays, waypoint_delays, typed_waypoints, no_frame = [], [], [], []
    for episode in covered:
        alarm = first_crediting_alarm(episode, method, unc)
        if alarm is None:
            continue
        hops, named = alarm
        caught.append(episode)
        waypoint_delays.append(hops)
        delay = episode_delay(episode, frames, method)
        if delay is None:
            no_frame.append(episode)
        else:
            frame_delays.append(delay)
        if names_type(named, truth):
            typed.append(episode)
            typed_waypoints.append(hops)

    def spread(values, prefix):
        return {
            f"{prefix}mean": statistics.fmean(values) if values else None,
            f"{prefix}std": statistics.stdev(values) if len(values) > 1 else None,
            f"{prefix}median": statistics.median(values) if values else None,
            f"{prefix}n": len(values),
        }

    return {
        "episodes": len(covered),
        "uncovered": len(episodes) - len(covered),
        "detected": len(caught),
        "rate": (100.0 * len(caught) / len(covered)) if covered else None,
        "typed": len(typed),
        "typed_rate": (100.0 * len(typed) / len(covered)) if covered else None,
        "no_frame": no_frame,
        **spread(frame_delays, "frame_"),
        **spread(waypoint_delays, "wp_"),
        **spread(typed_waypoints, "typedwp_"),
    }


def fmt(value, digits=1) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--frames", type=Path, default=None,
                        help="delay sidecar (default: <samples>_frames.csv)")
    parser.add_argument("--methods", nargs="+", default=list(METHODS),
                        metavar="M", help=f"any of {', '.join(METHODS)}, or 'all'")
    parser.add_argument("--failures", nargs="+", default=None, metavar="F",
                        help="ground-truth type names to report; default: all "
                             "present in the sample table")
    parser.add_argument("--uncertain-as", choices=("success", "failure"),
                        default="success",
                        help="whether the baseline's 'uncertain' counts as an "
                             "alarm (default: it does not)")
    parser.add_argument("--delay", choices=("waypoint", "frame", "both"),
                        default="waypoint",
                        help="unit for the delay: 'waypoint' (default) counts "
                             "checkpoints and is sound for every method; "
                             "'frame' counts control steps but is only measured "
                             "for the detector-driven methods (A1/B2 frames are "
                             "waypoint-boundary fallbacks); 'both' shows each")
    parser.add_argument("--table", action="store_true",
                        help="print a compact table instead of sentences")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    if any(m.lower() == "all" for m in args.methods):
        args.methods = list(METHODS)
    unknown = [m for m in args.methods if m not in METHODS]
    if unknown:
        sys.exit(f"unknown method(s): {', '.join(unknown)}")
    selected = [m for m in METHODS if m in args.methods]

    episodes = load_samples(args.samples)
    frames = load_frames(args.frames or
                         args.samples.with_name(args.samples.stem + "_frames.csv"))
    if not frames:
        print("  (no delay sidecar found -- every delay will print '--')")

    present = [f for f, _ in FAILURE_ROWS
               if any(e["failure"] == f for e in episodes)]
    failures = args.failures or present
    unknown = [f for f in failures if f not in present]
    if unknown:
        sys.exit(f"no episode of type(s): {', '.join(unknown)}; "
                 f"the table has {', '.join(present)}")

    unc = args.uncertain_as == "failure"
    clean = [e for e in episodes if e["failure"] is None]
    stats, gaps = {}, []
    for method in selected:
        for failure in failures:
            subset = [e for e in episodes if e["failure"] == failure]
            entry = summarise(subset, method, failure, frames, unc)
            entry["blind"] = failure in BLIND.get(method, set())
            stats[(method, failure)] = entry
            gaps += [(method, failure, e["task"], e["case"])
                     for e in entry["no_frame"]]

    WAYPOINT, FRAME = ("wp_", "waypoints", "wp"), ("frame_", "steps", "step")
    units = ((WAYPOINT, FRAME) if args.delay == "both"
             else (WAYPOINT,) if args.delay == "waypoint" else (FRAME,))

    def clause(entry: dict, prefix: str, unit: str, _short: str) -> str:
        if not entry[f"{prefix}n"]:
            return f"no {unit} delay available"
        return (f"delay mean {fmt(entry[prefix + 'mean'])}, "
                f"std {fmt(entry[prefix + 'std'])} {unit} "
                f"(n={entry[prefix + 'n']})")

    if args.table:
        heads, cols = [], []
        for prefix, _, short in units:
            heads += [f"Mean({short})", f"Std({short})", f"Med({short})",
                      f"n({short})"]
            cols += [prefix + k for k in ("mean", "std", "median", "n")]
        print_table(
            f"Detection rate and delay (episode unit; delay in "
            f"{' and '.join(u for _, u, _ in units)})",
            ["Method", "Failure", "Any", "Typed", "of", "Any%", "Typed%"]
            + heads,
            [[METHOD_LABELS[m], FAILURE_LABELS.get(f, f)
              + (" (blind)" if stats[(m, f)]["blind"] else ""),
              stats[(m, f)]["detected"], stats[(m, f)]["typed"],
              stats[(m, f)]["episodes"],
              fmt(stats[(m, f)]["rate"]), fmt(stats[(m, f)]["typed_rate"])]
             + [stats[(m, f)][c] if c.endswith("_n") else fmt(stats[(m, f)][c])
                for c in cols]
             for m in selected for f in failures])
    else:
        for method in selected:
            print(f"\n{METHOD_LABELS[method]}")
            for failure in failures:
                s = stats[(method, failure)]
                label = FAILURE_LABELS.get(failure, failure)
                blind = " [blind: cannot name this type]" if s["blind"] else ""
                if not s["episodes"]:
                    print(f"  no {label} episode it has a result for{blind}")
                    continue
                delay = "; ".join(clause(s, *u) for u in units)
                print(f"  detects {fmt(s['rate'])}% of {label} episodes "
                      f"({s['detected']} of {s['episodes']}), {delay}{blind}")
                typed = (clause(s, "typedwp_", "waypoints", "wp")
                         if s["typedwp_n"] else "never named correctly")
                print(f"      of those, named correctly "
                      f"{fmt(s['typed_rate'])}% ({s['typed']} of "
                      f"{s['episodes']}), {typed}")
            false_alarm = [e for e in clean
                           if e["preds"][method] is not None
                           and any(is_alarm(p, unc) for p in e["preds"][method])]
            covered_clean = [e for e in clean if e["preds"][method] is not None]
            if covered_clean:
                print(f"  false alarm on "
                      f"{100.0 * len(false_alarm) / len(covered_clean):.1f}% of "
                      f"clean episodes ({len(false_alarm)} of "
                      f"{len(covered_clean)})")

    if args.delay in ("frame", "both"):
        boundary = [m for m in selected if m in FRAME_IS_BOUNDARY]
        if boundary:
            print(f"\n  note: {', '.join(boundary)} frame delays are NOT "
                  f"measured -- vlm_events.csv carries no detection_frame on "
                  f"condition rows, so they fall back to the end of the "
                  f"waypoint. Use the waypoint delay to compare them with the "
                  f"detector methods.")

    uncovered = {(m, s["uncovered"]) for (m, _), s in stats.items()
                 if s["uncovered"]}
    for method, count in sorted(uncovered):
        print(f"\n  note: {METHOD_LABELS[method]} has no result for {count} "
              f"episode(s) of some type; its rates use only what it covers")
    if gaps:
        print(f"\n  note: {len(gaps)} detected episode(s) carry no alarm frame "
              f"in the sidecar, so they count in the rate but not the delay:")
        for method, failure, task, case in gaps[:8]:
            print(f"    {method}  {failure:<22} {task}/{case}")
        if len(gaps) > 8:
            print(f"    ... and {len(gaps) - 8} more")

    if args.no_write:
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "failure", "can_name_type", "episodes",
                         "detected_any", "rate_any",
                         "detected_typed", "rate_typed",
                         "delay_mean_wp", "delay_std_wp", "delay_median_wp",
                         "n_delay_wp",
                         "typed_delay_mean_wp", "typed_delay_std_wp",
                         "typed_delay_median_wp", "n_typed_delay_wp",
                         "delay_mean_steps", "delay_std_steps",
                         "delay_median_steps", "n_delay_steps",
                         "steps_are_measured"])
        for method in selected:
            for failure in failures:
                s = stats[(method, failure)]
                writer.writerow(
                    [method, failure, not s["blind"], s["episodes"],
                     s["detected"], s["rate"], s["typed"], s["typed_rate"]]
                    + [s["wp_" + k] for k in ("mean", "std", "median", "n")]
                    + [s["typedwp_" + k] for k in ("mean", "std", "median", "n")]
                    + [s["frame_" + k] for k in ("mean", "std", "median", "n")]
                    + [method not in FRAME_IS_BOUNDARY])
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
