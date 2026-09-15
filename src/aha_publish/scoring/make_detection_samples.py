#!/usr/bin/env python3
"""Stage 1 of 2 -- build the per-checkpoint detection sample table.

Walks the BT run directories and the GPT-baseline grids and flattens every
execution checkpoint into one CSV row:

    task,failure_condition,waypoint,gt,B2,A1,A2,A3,A4

    task              RLBench task name.
    failure_condition the injected case, e.g. 'clean', 'collision_wp3',
                      'wrong_sequence_v2'.  Together with `task` it identifies
                      one episode; the trailing _wpN is the injection waypoint.
    waypoint          the execution checkpoint inside that episode.
    gt                'success' before the injection waypoint and on clean runs;
                      'failure,<Type>' from the injection waypoint onwards.  Same
                      rule AND same vocabulary as the GPT baseline's own ground
                      truth (assess_reactive_gpt.ground_truth / FAILTYPE_TO_TYPES),
                      so a row here is directly comparable with the `gt` column of
                      out_grids/<task>/<case>/vlm_result.csv.
    B2 A1 A2 A3 A4    each method's verdict at this checkpoint, in the same
                      vocabulary as `gt`: 'success' when it raised no alarm,
                      'failure,<Type>' with the type it called, and (B2 only)
                      'uncertain'.  EMPTY when the method has no result for this
                      episode at all, e.g. an ungraded baseline case.
                      score_detection_samples.py decides what to do with the
                      empty and 'uncertain' cells.

Where each method's predicted TYPE comes from
---------------------------------------------
B2  the baseline's own `gpt` column, copied verbatim.
A1  the pre/post conditions that failed, and nothing else.  A lone
    object_in_gripper violation is Grasp at the grasp primitive and Slip during
    transport; several conditions failing at once, a robot-alignment condition or
    a gripper open/closed condition is Wrong Sequence; otherwise the BT's own tag
    on the violated predicate.
    Failed selected_object(...) or on(...) predicates take precedence and map
    to WrongObjectSelection.
A2  the detector that fired: collision -> collision, slip -> Slip, freezing ->
    freezing, transition -> Wrongtransition, orientation -> WrongOrientation.
A3  the same, restricted to confirmed fires.
    For injected wrong_object episodes, transition and orientation detector
    alarms are excluded from A2/A3/A4 and their latency frames: these deviations
    reflect the wrong target selection rather than separate pose failures.
A4  A3's type wherever a detector fired, since the sensor measures the fault
    directly and a condition violated at the same moment is usually its symptom;
    A1's type only at checkpoints no detector flagged.

Two type names are then folded away in every column (see TYPE_REMAP): a
`freezing` call becomes `success`, because no freezing case is injected in these
runs and the class cannot be true anywhere in the table, and `ObjectNotFound`
becomes `Wrong Sequence`, the injected class it corresponds to here.

When several detectors fire at one waypoint the EARLIEST fire names the
checkpoint -- a reactive system stops on the first alarm, so that alarm's type is
what it would report.

Where each method's alarms come from
------------------------------------
B2  GPT-5.6 reactive baseline .. out_grids/<task>/<case>/, one row per stage,
                                 verdict success / failure,<type> / uncertain.
                                 assess_reactive_gpt.py's --csv names the file, so
                                 it is not always vlm_result.csv; any CSV in the
                                 case folder carrying the assessor's schema counts,
                                 most recently written wins.
A1  Condition VLM only ......... vlm_events.csv channel=condition rows with a
                                 FAIL verdict.  A check is attributed to the
                                 waypoint that had just completed: prepost@w
                                 covers the boundary after w-1, post@w after w.
A2  Detectors only ............. raw '[DETECTOR] X ... at waypoint N (step S)'
                                 lines in run.log -- no VLM in the loop.
A3  Detectors + confirmation ... A2, but a fire from a detector that HAS a VLM
                                 confirmation channel (collision, slip) only
                                 counts when the VLM answered YES.  Detectors
                                 with no confirmation channel pass through.
A4  Full ....................... A1 OR A3, alarm-wise; where both fire the type
                                 is A3's alone (see above), and the alarm frame
                                 is still the earlier of the two.

Also writes a sidecar `<name>_frames.csv` with the injection frame and each
method's first alarm frame per episode, on the same (task, failure_condition)
key.  Only the latency rows of the failure-specific table need it; the scorer
runs fine without it.

Episodes whose injected failure could not physically manifest (planner skipped
the waypoint, the slip never released, freezing at the final waypoint) are
dropped, matching run_all_tasks_all_failures.score()'s N/A rules.  --keep-na
keeps them.

Usage
-----
    python aha_scripts/make_detection_samples.py           # every task under --bt-root
    python aha_scripts/make_detection_samples.py --tasks close_jar --keep-na

The baseline's 'uncertain' verdict is carried through as-is; whether it counts as
an alarm is score_detection_samples.py's --uncertain-as flag.
"""

from __future__ import annotations

from aha_publish import paths

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = (paths.PROJECT_ROOT)

DEFAULT_BT_ROOT = (paths.RUNS_DIR)
DEFAULT_GRID_ROOT = (paths.BASELINE_DIR)
DEFAULT_OUT = (paths.SCORES_DIR / 'detection_samples.csv')

METHODS = ("B2", "A1", "A2", "A3", "A4")
DETECTORS = ("collision", "freezing", "orientation", "slip", "transition")

# Injected failgen failure -> the live detector that owns it ('' = none; the
# failure is only reachable through the VLM condition checks).
RESPONSIBLE = {
    "slip": "slip",
    "collision": "collision",
    "freezing": "freezing",
    "transition": "transition",
    "combined_transition": "transition",
    "orientation": "orientation",
    "combined_orientation": "orientation",
    "rotation": "orientation",
    "translation": "transition",
    "grasp": "",
    "wrong_object": "",
    "wrong_sequence": "",
    "wrong_sequence_v2": "",
}

# Only these detectors are put to a VLM confirmation vote (channel=detector in
# vlm_events.csv), so A3 can only differ from A2 on their failures.
CONFIRMED_DETECTORS = ("collision", "slip")

# Injected failgen failure -> the canonical ground-truth type name.  Kept
# identical to assess_reactive_gpt.FAILTYPE_TO_TYPES (first entry of each tuple)
# so both pipelines label the same episode the same way; the trailing entries are
# aliases for failgen names the baseline map does not list.
FAILTYPE_TO_TYPE = {
    "slip": "Slip",
    "grasp": "Grasp",
    "collision": "collision",
    "freezing": "freezing",
    "wrong_sequence_v2": "Wrong Sequence",
    "combined_transition": "Wrongtransition",
    "combined_orientation": "WrongOrientation",
    "wrong_object": "WrongObjectSelection",
    "wrong_sequence": "Wrong Sequence",
    "transition": "Wrongtransition",
    "translation": "Wrongtransition",
    "orientation": "WrongOrientation",
    "rotation": "WrongOrientation",
}

# Type names that no ground-truth row in this benchmark can carry, folded into
# the vocabulary it does score.  Applied to EVERY column, gt and predictions
# alike, so the table keeps one shared vocabulary.
#
#   freezing        no freezing case is injected in these runs, so a freezing
#                   call names a class that cannot be true anywhere in the
#                   table.  It is folded into 'success': the method is treated
#                   as having raised no alarm at that checkpoint rather than an
#                   alarm of a type the benchmark cannot confirm or refute.
#   ObjectNotFound  has no injected counterpart either.  The BT raises it when
#                   the object the stage names is not where the stage expects
#                   it, which in this benchmark is what a mis-ordered execution
#                   looks like, so it scores as Wrong Sequence.
#
# A None target drops the type; when a multi-type label loses every component
# the checkpoint becomes 'success'.
TYPE_REMAP = {
    "freezing": None,
    "ObjectNotFound": "Wrong Sequence",
}


# Root-cause precedence for detectors that fire at the SAME waypoint.  A collision
# shoves the arm and the held object, so the transition/orientation deviation it
# produces is a downstream effect, not a second independent failure; the same goes
# for a slip or a stall.  When several fire together the lowest rank here names the
# checkpoint, and the earliest fire of any of them still sets the alarm frame (that
# is when a reactive system would stop).  This is a property of the detectors, not
# of the label -- it never consults the ground truth.
DETECTOR_PRECEDENCE = {
    "collision": 0,
    "slip": 1,
    "freezing": 2,
    "transition": 3,
    "orientation": 3,
}

# Live detector -> the canonical type name it predicts, same vocabulary as the
# ground truth above.
DETECTOR_TO_TYPE = {
    "collision": "collision",
    "slip": "Slip",
    "freezing": "freezing",
    "transition": "Wrongtransition",
    "orientation": "WrongOrientation",
}

# Everything the detector bank can name.  A1 type outside this set has no
# detector counterpart, so A3 priority has nothing to override and the name
# survives into A4.
DETECTOR_VOCABULARY = frozenset(DETECTOR_TO_TYPE.values())

# A1 sees pre/post conditions and nothing else, so its predicted type is the tag
# the BT already attaches to the violated predicate -- printed in run.log as
# '  - object_in_gripper(ball) == True  [GraspStateFailure]'.  These are the BT's
# own tag names; the map converts them to the shared type vocabulary.
#
# object_in_gripper cannot separate "never grasped" from "grasped then dropped",
# so GraspStateFailure -> Grasp: a slip episode is genuinely indistinguishable
# from a no-grasp episode through conditions alone.
CONDITION_TAG_TO_TYPE = {
    "ObjectNotFound": "ObjectNotFound",
    "WrongObjectSelection": "WrongObjectSelection",
    "WrongPosition": "Wrongtransition",
    "WrongOrientation": "WrongOrientation",
    "GraspStateFailure": "Grasp",
    "ReleaseStateFailure": "Wrong Sequence",
    "ExecutionSequenceMismatch": "Wrong Sequence",
    "ContactStateFailure": "Wrongtransition",
}

# 'PRE-CHECK before waypoint 1: grasp the ball' / 'POST-CHECK after waypoint 0: ...'
CHECK_HEADER_RE = re.compile(
    r"^(?P<kind>PRE|POST)-CHECK\s+(?:before|after)\s+waypoint\s+(?P<wp>\d+):",
    re.MULTILINE)
GRIP_PREDICATE = "object_in_gripper"
ALIGNMENT_PREDICATE = "end_effector_aligned_with"
# selected_object(X) is the one condition that names WHICH object the robot went
# for, so unlike the state predicates its violation identifies the fault outright.
SELECTION_PREDICATE = "selected_object"

# '  VLM PRE+POST CHECK: FAIL' -- one line per condition check the VLM ran.
CHECK_VERDICT_RE = re.compile(
    r"^\s+VLM\s+(?:PRE\+POST|PRE|POST)\s+CHECK:\s+(?P<verdict>\w+)\s*$",
    re.MULTILINE)
# '  - <predicate>  [Tag]' in a PRE-CHECK / POST-CHECK condition listing.
CONDITION_TAG_RE = re.compile(
    r"^\s+-\s+(?P<pred>.+?)\s+\[(?P<tag>[A-Za-z_]+)\]\s*$", re.MULTILINE)
# '    - not_satisfied: pre waypoint 2: object_in_gripper(ball) == True'
NOT_SATISFIED_RE = re.compile(
    r"^\s+-\s+not_satisfied:\s+(?:(?:pre|post)\s+waypoint\s+\d+:\s*)?"
    r"(?P<pred>.+?)\s*$", re.MULTILINE)

DETECTOR_FIRE_RE = re.compile(
    r"\[DETECTOR\]\s+(ORIENTATION|TRANSITION|SLIP|COLLISION|FREEZING)\s+"
    r"(?:failure|detected)\s+at waypoint\s+(\d+)\s+\(step\s+(\d+)\)",
    re.IGNORECASE,
)
DETECTOR_SUMMARY_RE = re.compile(
    r"^\s*(orientation|transition|slip|collision|freezing)\s+"
    r"DETECTED\s+\1\s+x(\d+)\s+\(first:\s+step\s+(\d+),\s+waypoint\s+(\d+),",
    re.IGNORECASE,
)
WAYPOINT_OF_RE = re.compile(r"^Waypoint\s+(\d+)\s+of\s+(\d+)\s*$", re.MULTILINE)
FREEZE_SKIP_RE = re.compile(r"skipping final waypoint (\d+)")

# Only these logs carry a GLOBAL simulation step in their `step` column, so only
# they can define waypoint frame boundaries.  transition_/orientation_drift.csv
# restart `step` at every waypoint.
GLOBAL_STEP_LOGS = ("slip_trace.csv", "slip_drift.csv", "collision_drift.csv")

# ... and for the same reason, these detectors report a per-waypoint step in
# their '[DETECTOR] ... (step N)' line, which has to be rebased onto the global
# frame axis before it can be compared with an injection frame.
LOCAL_STEP_DETECTORS = frozenset({"orientation", "transition"})


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _int(value):
    try:
        if value in ("", None):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


# assess_reactive_gpt.py writes its per-case results under whatever --csv names,
# so a case folder may hold vlm_result.csv, slip_results_gpt54.csv, ... . These
# columns identify the file as that assessor's output whatever it is called.
GRID_RESULT_COLUMNS = {"case", "image", "stage", "gt", "gpt"}


def find_grid_result(case_dir: Path, preferred: str = "") -> Path | None:
    """The baseline's result CSV in one case folder, newest first.

    `preferred` pins an exact filename; otherwise every CSV with the assessor's
    schema is a candidate and the most recently written one is used.
    """
    if not case_dir.is_dir():
        return None
    if preferred:
        path = case_dir / preferred
        return path if path.exists() else None
    candidates = []
    for path in case_dir.glob("*.csv"):
        try:
            with path.open(newline="") as handle:
                header = next(csv.reader(handle), [])
        except OSError:
            continue
        if GRID_RESULT_COLUMNS <= set(header):
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def split_case(case: str) -> tuple[str, int | None]:
    """'collision_wp3' -> ('collision', 3); 'wrong_sequence_v2' -> (name, None)."""
    if case == "clean":
        return "clean", None
    match = re.match(r"(.+)_wp(\d+)$", case)
    if match:
        return match.group(1), int(match.group(2))
    return case, None


def add_alarm(alarms: dict, wp: int, frame: int | None, ftype: str,
              rank: int = 99) -> None:
    """Record one alarm at `wp`, keeping every distinct type that fired there.

    The alarm frame is the earliest fire at that waypoint -- when a reactive
    system would actually stop.  Types accumulate instead of overwriting: nothing
    that fired is thrown away, and `alarm_type` orders them by DETECTOR_PRECEDENCE
    so the root cause is named first.
    """
    alarm = alarms.setdefault(wp, {"frame": frame, "types": []})
    if frame is not None and (alarm["frame"] is None or frame < alarm["frame"]):
        alarm["frame"] = frame
    if not any(t["type"] == ftype for t in alarm["types"]):
        alarm["types"].append({"type": ftype, "rank": rank,
                               "frame": frame if frame is not None else 1 << 30})


def alarm_type(alarm: dict) -> str:
    """The alarm's label: every type it named, root cause first.

    Ordered by DETECTOR_PRECEDENCE, then by which fired first.  A collision that
    fires alongside the transition/orientation deviation it caused therefore leads
    the label, and the downstream types stay visible behind it.
    """
    ordered = sorted(alarm["types"], key=lambda t: (t["rank"], t["frame"]))
    return "|".join(t["type"] for t in ordered)


# --------------------------------------------------------------------------- #
# telemetry
# --------------------------------------------------------------------------- #
def waypoint_bounds(run_dir: Path, n_wp: int = 0) -> dict[int, tuple[int, int]]:
    """waypoint -> (first_global_step, last_global_step).

    Waypoints the planner skipped have no telemetry; they inherit a zero-length
    span at the previous waypoint's end so a checkpoint alarm there still gets a
    monotone frame.
    """
    bounds: dict[int, list[int]] = {}
    for name in GLOBAL_STEP_LOGS:
        for row in read_csv(run_dir / name):
            wp, step = _int(row.get("waypoint")), _int(row.get("step"))
            if wp is None or step is None:
                continue
            span = bounds.setdefault(wp, [step, step])
            span[0] = min(span[0], step)
            span[1] = max(span[1], step)
    out = {wp: (lo, hi) for wp, (lo, hi) in bounds.items()}
    if not out:
        return out
    last = 0
    for wp in range(max(n_wp, max(out) + 1)):
        if wp in out:
            last = out[wp][1]
        else:
            out[wp] = (last, last)
    return out


def parse_detector_fires(text: str) -> dict[str, dict[int, int]]:
    """detector -> {waypoint: first firing step}."""
    fires: dict[str, dict[int, int]] = {name: {} for name in DETECTORS}

    def add(name, waypoint, step):
        name = name.lower()
        wp, st = _int(waypoint), _int(step)
        if name not in fires or wp is None:
            return
        if wp not in fires[name] or (st is not None and st < fires[name][wp]):
            fires[name][wp] = st if st is not None else fires[name].get(wp)

    for match in DETECTOR_FIRE_RE.finditer(text or ""):
        add(match.group(1), match.group(2), match.group(3))
    for line in (text or "").splitlines():
        match = DETECTOR_SUMMARY_RE.search(line)
        if match:
            add(match.group(1), match.group(4), match.group(3))
    return fires


def condition_stage(row: dict) -> int | None:
    """The waypoint a condition check reports on.

    'pre'     @ w -> checked before w ran, so it reports on w-1 (-1 at w=0);
    'prepost' @ w -> post-conditions of w-1 and pre-conditions of w  -> w-1;
    'post'    @ w -> the final check after w finished                -> w.
    """
    wp = _int(row.get("waypoint"))
    if wp is None:
        return None
    kind = (row.get("event_kind") or "").strip().lower()
    return wp if kind == "post" else wp - 1


def condition_alarm(row: dict) -> bool:
    return (str(row.get("failure_detected") or "").strip().lower() == "true"
            or str(row.get("verdict") or "").strip().upper() in ("FAIL", "FAILED"))


# --------------------------------------------------------------------------- #
# N/A rules
# --------------------------------------------------------------------------- #
def slip_holding_at_injection(run_dir: Path, inj_wp: int) -> bool | None:
    rows = read_csv(run_dir / "slip_trace.csv") or read_csv(run_dir / "slip_drift.csv")
    if not rows:
        return None
    first = next((i for i, r in enumerate(rows) if _int(r.get("waypoint")) == inj_wp),
                 None)
    if first is None:
        return None
    window = rows[max(0, first - 10):first + 3]
    return any(_int(r.get("is_holding")) or _int(r.get("prior_holding_streak"))
               for r in window)


def episode_na_reason(run_text: str, run_dir: Path, failure: str, inj_wp,
                      responsible: str, fires: dict) -> str:
    """Why this episode cannot be scored, '' when it is scorable.

    Mirrors run_all_tasks_all_failures.score(): a failure that physically could
    not manifest is Not-Applicable, not a missed detection.
    """
    if failure == "clean" or inj_wp is None:
        return ""
    if any(wp in (inj_wp, inj_wp + 1) for wp in fires.get(responsible, {})):
        return ""                      # a genuine catch is never downgraded
    if responsible == "freezing":
        match = FREEZE_SKIP_RE.search(run_text)
        if match and int(match.group(1)) == inj_wp:
            return "NA_final_wp"
    if responsible == "slip":
        if "SLIPPING NOW" not in run_text:
            return "NA_no_slip"
        if slip_holding_at_injection(run_dir, inj_wp) is False:
            return "NA_no_grasp"
    if responsible in ("orientation", "transition"):
        if not re.search(rf"finished waypoint {inj_wp} path", run_text):
            return "NA_unreached"
    elif responsible:
        bounds = waypoint_bounds(run_dir)
        if bounds and max(bounds) < inj_wp:
            return "NA_unreached"
    return ""


# --------------------------------------------------------------------------- #
# one episode
# --------------------------------------------------------------------------- #
def _norm_predicate(text: str) -> str:
    """Predicate text as a comparison key: no case, no spaces, no trailing 'OK'."""
    return re.sub(r"\s+", "", re.sub(r"\s+OK$", "", text or "")).lower()


def condition_tags(run_text: str) -> dict[str, str]:
    """Predicate -> the BT's failure tag, from the PRE/POST-CHECK listings.

    The detector hold-conditions carry lowercase tags ([collision], [slip], ...)
    and are excluded: they belong to A2, not to A1.
    """
    tags = {}
    for match in CONDITION_TAG_RE.finditer(run_text or ""):
        tag = match.group("tag")
        if tag in CONDITION_TAG_TO_TYPE:
            tags.setdefault(_norm_predicate(match.group("pred")), tag)
    return tags


def grasp_and_transport_stages(run_text: str) -> tuple[int | None, set[int]]:
    """(grasp stage, transport stages), from where object_in_gripper is required.

    The BT asserts object_in_gripper as a POSTcondition of the stage that closes
    the gripper and as a PREcondition of every stage that carries the object.  So
    the grasp primitive is the stage that must end holding without having started
    holding, and the transport primitives are the ones that must start holding.
    """
    pre: dict[int, set[str]] = defaultdict(set)
    post: dict[int, set[str]] = defaultdict(set)
    headers = list(CHECK_HEADER_RE.finditer(run_text or ""))
    for index, match in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(run_text)
        block = run_text[match.end():end]
        wp = int(match.group("wp"))
        target = pre if match.group("kind") == "PRE" else post
        for line in block.splitlines():
            hit = CONDITION_TAG_RE.match(line)
            if hit:
                target[wp].add(_norm_predicate(hit.group("pred")))
            elif line.strip().startswith("[vlm]") or line.strip().startswith("VLM"):
                break                       # past the listing, into the check itself

    def holds(bucket, wp):
        return any(p.startswith(GRIP_PREDICATE) for p in bucket.get(wp, ()))

    stages = set(pre) | set(post)
    grasp = next((wp for wp in sorted(stages)
                  if holds(post, wp) and not holds(pre, wp)), None)
    transport = {wp for wp in stages if holds(pre, wp)}
    return grasp, transport


def condition_failure_type(violated: list[str], stage: int | None,
                           tags: dict[str, str], grasp_stage: int | None,
                           transport: set[int]) -> str:
    """The type A1 reports for one FAIL check, from the predicates that failed.

    A violated selected_object(X) or on(X, Y) is decisive and maps to
    WrongObjectSelection under the dataset labeling policy.
    A lone object_in_gripper violation is a grasp problem or a
    slip depending on which primitive it lands in: failing to be holding at the
    grasp stage means the object was never picked up, while failing during
    transport means it was picked up and lost.  Anything else -- several
    conditions failing together, a robot-alignment condition, or a gripper
    open/closed condition -- is an execution-order problem, so it is labelled a
    sequence violation.
    """
    if not violated:
        return "unknown"
    # A violated selected_object(X) says the robot went for the wrong object, and
    # it says so outright -- so it outranks the several-conditions-failed rule
    # below.  What fails alongside it is downstream of that one choice: the arm
    # aligns to the wrong object, then fails to hold the right one.  Same
    # root-cause precedence the detectors get from DETECTOR_PRECEDENCE.
    if any(p.startswith((SELECTION_PREDICATE, "on(")) for p in violated):
        return "WrongObjectSelection"
    only_grip = (len(violated) == 1 and violated[0].startswith(GRIP_PREDICATE))
    if only_grip:
        if stage is not None and stage in transport and stage != grasp_stage:
            return "Slip"
        return "Grasp"
    if len(violated) > 1:
        return "Wrong Sequence"
    predicate = violated[0]
    tag = tags.get(predicate)
    if predicate.startswith(ALIGNMENT_PREDICATE) or tag == "ExecutionSequenceMismatch":
        return "Wrong Sequence"
    return CONDITION_TAG_TO_TYPE.get(tag, "unknown")


def failed_check_predicates(run_text: str) -> list[list[str]]:
    """Every FAIL check's violated predicates, in the order run.log printed them."""
    checks = list(CHECK_VERDICT_RE.finditer(run_text or ""))
    out = []
    for index, match in enumerate(checks):
        if match.group("verdict").strip().upper() not in ("FAIL", "FAILED"):
            continue
        end = checks[index + 1].start() if index + 1 < len(checks) else len(run_text)
        block = run_text[match.end():end]
        out.append([_norm_predicate(h.group("pred"))
                    for h in NOT_SATISFIED_RE.finditer(block)])
    return out


def failed_check_types(run_text: str) -> list[str]:
    """Type of every FAIL condition check, in the order run.log printed them.

    Each check's block runs to the next check, so the not_satisfied lines it
    contains are exactly its own.  The order matches the condition rows of
    vlm_events.csv, which is how the type is attached to a waypoint.
    """
    tags = condition_tags(run_text)
    checks = list(CHECK_VERDICT_RE.finditer(run_text or ""))
    types = []
    for index, match in enumerate(checks):
        if match.group("verdict").strip().upper() not in ("FAIL", "FAILED"):
            continue
        end = checks[index + 1].start() if index + 1 < len(checks) else len(run_text)
        block = run_text[match.end():end]
        found = "unknown"
        for hit in NOT_SATISFIED_RE.finditer(block):
            tag = tags.get(_norm_predicate(hit.group("pred")))
            if tag:
                found = CONDITION_TAG_TO_TYPE[tag]
                break
        types.append(found)
    return types


def load_episode(task: str, case_dir: Path, grid_root: Path,
                 grid_csv_name: str = "") -> dict | None:
    run_log = case_dir / "run.log"
    if not run_log.exists():
        return None
    case = case_dir.name
    failure, inj_wp = split_case(case)
    responsible = RESPONSIBLE.get(failure, "")

    run_text = run_log.read_text(errors="ignore")
    stdout = case_dir / "batch_stdout.log"
    if stdout.exists():
        run_text += "\n" + stdout.read_text(errors="ignore")

    n_wp = max((int(b) + 1 for _, b in WAYPOINT_OF_RE.findall(run_text)), default=0)
    bounds = waypoint_bounds(case_dir, n_wp)
    if not n_wp:
        n_wp = (max(bounds) + 1) if bounds else 0
    if not n_wp:
        return None

    events = read_csv(case_dir / "vlm_events.csv")
    fires = parse_detector_fires(run_text)

    inj_frame = next((_int(r.get("injection_frame")) for r in events
                      if _int(r.get("injection_frame")) is not None), None)
    if inj_frame is None and inj_wp is not None and inj_wp in bounds:
        inj_frame = bounds[inj_wp][0]
    if inj_frame is None:
        inj_frame = 0

    # ---- A1: condition-VLM alarms ---------------------------------------- #
    # vlm_events.csv says a check FAILED but not which predicate did it, so the
    # type is read from run.log's FAIL blocks, paired with the FAIL event rows in
    # order.  A1 therefore only ever names a failure its own pre/post conditions
    # can express -- no detector, no episode-level diagnosis.
    tags = condition_tags(run_text)
    grasp_stage, transport = grasp_and_transport_stages(run_text)
    fail_predicates = failed_check_predicates(run_text)
    a1: dict[int, dict] = {}
    seen_fail = 0
    no_grasp_predicates: set[str] = set()
    for row in events:
        if row.get("channel") != "condition":
            continue
        if not condition_alarm(row):
            # A passing holding checkpoint is evidence that the grasp recovered.
            wp = _int(row.get("waypoint"))
            kind = (row.get("event_kind") or "").strip().lower()
            if (str(row.get("verdict") or "").strip().upper() == "PASS"
                    and ((kind in ("pre", "prepost") and wp in transport)
                         or (kind == "post" and wp == grasp_stage))):
                no_grasp_predicates.clear()
            continue
        violated = (fail_predicates[seen_fail]
                    if seen_fail < len(fail_predicates) else [])
        seen_fail += 1
        stage = condition_stage(row)
        if stage is None or not 0 <= stage < n_wp:
            continue
        ftype = condition_failure_type(violated, stage, tags, grasp_stage, transport)
        holding_failures = {p for p in violated if p.startswith(GRIP_PREDICATE)}
        if stage == grasp_stage:
            no_grasp_predicates.update(holding_failures)
        if ftype == "Slip" and holding_failures & no_grasp_predicates:
            # The same object was never acquired: a later failed holding check
            # is still No grasp (CSV: Grasp), not Grasp loss (CSV: Slip).
            ftype = "Grasp"
        # A pre-check's STATE predicates report on the waypoint that just ended --
        # 'pre waypoint 4: object_in_gripper' means the object was lost during
        # waypoint 3 -- which is what condition_stage() encodes.  selected_object
        # is not a state predicate: it names the TARGET of the waypoint the check
        # gates, so 'pre waypoint 1: selected_object(chicken)' says waypoint 1 is
        # aimed at the wrong object, and that is evidence about waypoint 1.  The
        # predicate that decides the type therefore also decides the attribution;
        # every other predicate keeps the just-ended-waypoint rule.
        kind = (row.get("event_kind") or "").strip().lower()
        if (any(p.startswith(SELECTION_PREDICATE) for p in violated)
                and kind in ("pre", "prepost")
                and stage + 1 < n_wp):
            stage += 1
        frame = _int(row.get("detection_frame"))
        if frame is None:
            frame = bounds.get(stage, (None, None))[1]
        add_alarm(a1, stage, frame, ftype)

    # ---- A2 / A3: detector alarms ---------------------------------------- #
    confirmed_yes: dict[str, dict[int, int]] = defaultdict(dict)
    for row in events:
        if row.get("channel") != "detector":
            continue
        name = (row.get("event_kind") or "").strip().lower()
        wp = _int(row.get("waypoint"))
        if name in DETECTORS and wp is not None \
                and str(row.get("verdict") or "").strip().upper() == "YES":
            confirmed_yes[name].setdefault(wp, _int(row.get("detection_frame")))

    def fire_frame(name: str, wp: int, step: int | None) -> int | None:
        if step is None:
            return bounds.get(wp, (None, None))[1]
        if name in LOCAL_STEP_DETECTORS:
            start = bounds.get(wp, (None, None))[0]
            return None if start is None else start + step
        return step

    a2: dict[int, dict] = {}
    a3: dict[int, dict] = {}
    for name, per_wp in fires.items():
        # Dataset policy: wrong-object episodes can deviate from the intended
        # pose simply by targeting another object. Do not label those detector
        # fires as independent translation/orientation failures, or use their
        # frames for detection latency. Keep condition-VLM predictions intact.
        if failure == "wrong_object" and name in LOCAL_STEP_DETECTORS:
            continue
        ftype = DETECTOR_TO_TYPE.get(name, name)
        for wp, step in per_wp.items():
            if not 0 <= wp < n_wp:
                continue
            frame = fire_frame(name, wp, step)
            rank = DETECTOR_PRECEDENCE.get(name, 99)
            add_alarm(a2, wp, frame, ftype, rank)
            if name in CONFIRMED_DETECTORS:
                if wp in confirmed_yes.get(name, {}):
                    yes = confirmed_yes[name][wp]
                    add_alarm(a3, wp, yes if yes is not None else frame, ftype, rank)
            else:
                add_alarm(a3, wp, frame, ftype, rank)

    # A4 fuses the two channels with A3 taking priority: where a detector fired,
    # its type IS A4's verdict and the condition VLM's reading of that same
    # checkpoint is dropped.  The sensor measures the fault directly, while a
    # condition violated at that moment is usually its downstream symptom -- the
    # object leaves the gripper BECAUSE the arm hit something -- so keeping both
    # names would credit A4 for a type it only reached through the effect.
    #
    # That reasoning only holds for types a detector could have produced.  When
    # A1 names something OUTSIDE the detector vocabulary -- wrong object, grasp,
    # sequence -- the detector fire is not a competing diagnosis of the same
    # event but the deviation that failure caused, so A1's name is the only
    # account of the root cause and survives alongside it (behind it, since it
    # keeps A1's rank).  Priority resolves disagreements; it does not silence the
    # one channel that can express the failure.
    #
    # The alarm FRAME still comes from whichever channel fired first: a reactive
    # system stops at the first alarm no matter which channel raised it, so
    # priority decides the label, not the latency.
    a4: dict[int, dict] = {}
    for wp, alarm in a3.items():
        for entry in alarm["types"]:
            add_alarm(a4, wp, alarm["frame"], entry["type"], entry["rank"])
    for wp, alarm in a1.items():
        if wp in a4:
            frame = alarm["frame"]
            if frame is not None and (a4[wp]["frame"] is None
                                      or frame < a4[wp]["frame"]):
                a4[wp]["frame"] = frame
            for entry in alarm["types"]:
                if entry["type"] not in DETECTOR_VOCABULARY \
                        and entry["type"] != "unknown":
                    add_alarm(a4, wp, alarm["frame"], entry["type"],
                              entry["rank"])
            continue
        for entry in alarm["types"]:
            add_alarm(a4, wp, alarm["frame"], entry["type"], entry["rank"])

    # ---- B2: GPT baseline grids ------------------------------------------ #
    # The baseline already writes 'success' / 'uncertain' / 'failure,<Type>', so
    # its verdict is copied through unchanged; only the failure verdicts become
    # alarms (and so feed the latency sidecar).
    b2_verdicts: dict[int, str] | None = None
    b2: dict[int, dict] = {}
    grid_csv = find_grid_result(grid_root / task / case, grid_csv_name)
    grid_rows = read_csv(grid_csv) if grid_csv else []
    if grid_rows:
        b2_verdicts = {}
        for row in grid_rows:
            stage = _int(row.get("stage"))
            if stage is None or not 0 <= stage < n_wp:
                continue
            verdict = (row.get("gpt") or "").strip()
            b2_verdicts[stage] = verdict or "success"
            if verdict.lower().startswith("failure"):
                _, _, ftype = verdict.partition(",")
                add_alarm(b2, stage, bounds.get(stage, (None, None))[1],
                          ftype.strip() or "?")

    return {
        "task": task,
        "case": case,
        "failure": failure,
        "inj_wp": inj_wp,
        "inj_frame": inj_frame,
        "n_wp": n_wp,
        "na": episode_na_reason(run_text, case_dir, failure, inj_wp,
                                responsible, fires),
        "alarms": {"A1": a1, "A2": a2, "A3": a3, "A4": a4,
                   "B2": None if b2_verdicts is None else b2},
        "b2_verdicts": b2_verdicts,
        "b2_source": grid_csv.name if grid_csv and grid_rows else "",
    }


def discover_tasks(bt_root: Path) -> list[str]:
    """Every task under `bt_root`, i.e. every subdirectory holding an episode.

    The run root also carries loose bookkeeping files (plan.csv, summary.csv,
    vlm_events_all.csv) and can hold half-started task folders, so a directory
    only counts as a task once at least one of its case folders has a run.log.
    """
    if not bt_root.is_dir():
        return []
    return sorted(p.name for p in bt_root.iterdir()
                  if p.is_dir() and any(c.is_dir() and (c / "run.log").exists()
                                        for c in p.iterdir()))


def load_all(bt_root: Path, grid_root: Path, tasks, grid_csv_name: str = ""):
    episodes = []
    for task in tasks:
        task_dir = bt_root / task
        if not task_dir.is_dir():
            print(f"!! no run directory for task {task} under {bt_root}",
                  file=sys.stderr)
            continue
        for case_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            episode = load_episode(task, case_dir, grid_root, grid_csv_name)
            if episode:
                episodes.append(episode)
    return episodes


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def gt_label(episode: dict, wp: int) -> str:
    """'success', or 'failure,<Type>' once the injected failure is present.

    Mirrors assess_reactive_gpt.ground_truth(): stage k of case <failtype>_wpN is
    ground-truth failure iff k >= N, and a case with no _wpN (wrong_sequence_v2)
    is corrupted from the first waypoint.
    """
    if episode["failure"] == "clean":
        return "success"
    inj = episode["inj_wp"]
    if wp < (0 if inj is None else inj):
        return "success"
    return f"failure,{FAILTYPE_TO_TYPE.get(episode['failure'], '?')}"


def remap_types(label: str) -> str:
    """Apply TYPE_REMAP to one 'failure,<Type>[|<Type>...]' cell.

    'success', 'uncertain' and the empty cell pass through untouched.
    """
    if not label.startswith("failure,"):
        return label
    kept: list[str] = []
    for name in label[len("failure,"):].split("|"):
        target = TYPE_REMAP.get(name, name)
        if target and target not in kept:
            kept.append(target)
    return f"failure,{'|'.join(kept)}" if kept else "success"


def method_label(episode: dict, method: str, wp: int) -> str:
    """One method's verdict at one checkpoint, in the `gt` vocabulary."""
    alarms = episode["alarms"][method]
    if alarms is None:
        return ""                                   # no result for this episode
    if method == "B2":
        # The baseline's verdict is authoritative, including 'uncertain'.
        return episode["b2_verdicts"].get(wp, "success")
    alarm = alarms.get(wp)
    return f"failure,{alarm_type(alarm)}" if alarm else "success"


# Explicit user-requested checkpoint exclusions from the exported dataset.
EXCLUDED_CHECKPOINTS = {
    ("pick_up_cup", "wrong_object_wp1"): frozenset({1, 2}),
}


def checkpoint_excluded(episode: dict, wp: int) -> bool:
    return wp in EXCLUDED_CHECKPOINTS.get((episode["task"], episode["case"]), ())


def write_samples(episodes, out: Path) -> int:
    rows = 0
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "failure_condition", "waypoint", "gt", *METHODS])
        for episode in episodes:
            for wp in range(episode["n_wp"]):
                if checkpoint_excluded(episode, wp):
                    continue
                writer.writerow(
                    [episode["task"], episode["case"], wp,
                     remap_types(gt_label(episode, wp))]
                    + [remap_types(method_label(episode, m, wp))
                       for m in METHODS])
                rows += 1
    return rows


def write_frames(episodes, out: Path) -> None:
    """Injection frame + each method's first crediting alarm frame, per episode.

    'Crediting' means at or after the injection waypoint -- the same alarm the
    scorer counts as the detection, so the difference is the latency.
    """
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "failure_condition", "injection_frame"]
                        + [f"{m}_alarm_frame" for m in METHODS])
        for episode in episodes:
            if episode["failure"] == "clean":
                continue
            floor = 0 if episode["inj_wp"] is None else episode["inj_wp"]
            cells = []
            for method in METHODS:
                alarms = episode["alarms"][method]
                frames = [] if not alarms else [
                    a["frame"] for wp, a in alarms.items()
                    if wp >= floor and not checkpoint_excluded(episode, wp)
                    and a["frame"] is not None
                    # an alarm whose every type is dropped by TYPE_REMAP is
                    # scored as 'success', so it sets no latency frame either
                    and remap_types(f"failure,{alarm_type(a)}") != "success"]
                cells.append(min(frames) if frames else "")
            writer.writerow([episode["task"], episode["case"],
                             episode["inj_frame"], *cells])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bt-root", type=Path, default=DEFAULT_BT_ROOT,
                        help="root of the A1..A4 BT runs")
    parser.add_argument("--grid-root", type=Path, default=DEFAULT_GRID_ROOT,
                        help="root of the B2 GPT baseline grids")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="tasks to include; default: every task directory "
                             "found under --bt-root")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--grid-csv", default="",
                        help="exact baseline result filename to use inside each "
                             "case folder; default: newest CSV with the "
                             "assessor's schema, whatever it is named")
    parser.add_argument("--keep-na", action="store_true",
                        help="keep episodes whose injection could not manifest")
    args = parser.parse_args()

    tasks = args.tasks
    if tasks is None:
        tasks = discover_tasks(args.bt_root)
        if not tasks:
            sys.exit(f"no task directories under {args.bt_root}")
        print(f"   {len(tasks)} task(s) under {args.bt_root}: {', '.join(tasks)}")

    episodes = load_all(args.bt_root, args.grid_root, tasks, args.grid_csv)
    if not episodes:
        sys.exit(f"no episodes found under {args.bt_root}")

    dropped = [e for e in episodes if e["na"]]
    if not args.keep_na:
        episodes = [e for e in episodes if not e["na"]]
        for episode in dropped:
            print(f"   dropped {episode['task']}/{episode['case']}: {episode['na']}")

    missing = defaultdict(list)
    for episode in episodes:
        for method in METHODS:
            if episode["alarms"][method] is None:
                missing[method].append(f"{episode['task']}/{episode['case']}")
    odd = sorted({e["b2_source"] for e in episodes
                  if e["b2_source"] and e["b2_source"] != "vlm_result.csv"})
    if odd:
        print(f"   B2 read from non-default result file(s): {', '.join(odd)}")
    for method, cases in missing.items():
        print(f"   {method}: no result for {len(cases)} episode(s) "
              f"-> empty cells ({', '.join(cases[:4])}"
              f"{', ...' if len(cases) > 4 else ''})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = write_samples(episodes, args.out)
    frames = args.out.with_name(args.out.stem + "_frames.csv")
    write_frames(episodes, frames)

    clean = sum(1 for e in episodes if e["failure"] == "clean")
    print(f"\n{len(episodes)} episodes ({clean} clean), {rows} checkpoints")
    print(f"wrote {args.out}")
    print(f"wrote {frames}  (latency sidecar)")


if __name__ == "__main__":
    main()
